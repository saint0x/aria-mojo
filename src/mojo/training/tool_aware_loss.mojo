"""
Tool-Aware Loss Functions for LLaMA3.1 Training

Specialized loss functions designed for maximizing early <tool> token
prediction accuracy and deterministic tool-calling flow. Implements
weighted loss schemes, multi-token sequence optimization, and 
tool classification objectives.
"""

from sys import simdwidthof
from memory import memset_zero, memcpy
from algorithm import vectorize, parallelize
from builtin import int32, float32
from math import exp, log, sqrt
from tensor import Tensor
from collections import Dict, List


# Special token IDs for tool-calling (matching our tokenizer)
alias TOOL_START_ID = 50001      # <tool>
alias TOOL_END_ID = 50002        # </tool>
alias TOOL_RESPONSE_ID = 50003   # <tool_response>
alias RESPONSE_ID = 50004        # <response>
alias THINKING_ID = 50005        # <thinking>
alias EOS_ID = 50006            # <eos>
alias BOS_ID = 50007            # <bos>

alias SIMD_WIDTH = simdwidthof[DType.float32]()


struct ToolTokenWeights:
    """Configurable weights for different tool-related tokens"""
    var tool_start_weight: Float32      # Weight for <tool> prediction
    var tool_response_weight: Float32   # Weight for <tool_response> prediction  
    var thinking_weight: Float32        # Weight for <thinking> prediction
    var response_weight: Float32        # Weight for <response> prediction
    var regular_token_weight: Float32   # Weight for regular vocabulary tokens
    var sequence_boundary_weight: Float32  # Extra weight at sequence boundaries
    
    fn __init__(inout self):
        # Universal Thinking Prefix weights - all examples start with <thinking>
        self.thinking_weight = 4.0        # HIGHEST weight - all examples start with thinking
        self.tool_start_weight = 3.5      # High weight for tool decision after thinking
        self.tool_response_weight = 2.5   # Important for tool flow continuation
        self.response_weight = 2.0        # Enhanced weight for final response
        self.regular_token_weight = 1.0   # Baseline weight
        self.sequence_boundary_weight = 1.2  # Slightly higher at boundaries
    
    fn get_token_weight(self, token_id: Int32) -> Float32:
        """Get appropriate loss weight for token ID"""
        if token_id == TOOL_START_ID:
            return self.tool_start_weight
        elif token_id == TOOL_RESPONSE_ID:
            return self.tool_response_weight
        elif token_id == THINKING_ID:
            return self.thinking_weight
        elif token_id == RESPONSE_ID:
            return self.response_weight
        elif token_id == BOS_ID or token_id == EOS_ID:
            return self.sequence_boundary_weight
        else:
            return self.regular_token_weight


struct ToolClassificationHead:
    """Specialized classification head for early tool vs thinking decision"""
    var weight: Tensor[DType.float32]    # [hidden_dim, num_classes]
    var bias: Tensor[DType.float32]      # [num_classes]
    var hidden_dim: Int
    var num_classes: Int                 # tool, thinking, response, etc.
    
    fn __init__(inout self, hidden_dim: Int, num_classes: Int = 3):
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes  # <tool>, <thinking>, <response>
        
        # Initialize classification weights
        self.weight = Tensor[DType.float32](hidden_dim, num_classes)
        self.bias = Tensor[DType.float32](num_classes)
        
        # Xavier initialization for classification head
        let xavier_std = sqrt(2.0 / (hidden_dim + num_classes))
        for i in range(self.weight.num_elements()):
            self.weight.data()[i] = (Float32(i % 200) - 100.0) / 100.0 * xavier_std
        
        # Zero bias initialization
        memset_zero(self.bias.data(), num_classes * 4)
    
    fn forward(self, hidden_states: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass for tool classification"""
        let batch_size = hidden_states.shape()[0]
        let seq_len = hidden_states.shape()[1]
        
        # Apply classification head: [batch, seq, hidden] @ [hidden, classes] -> [batch, seq, classes]
        var logits = Tensor[DType.float32](batch_size, seq_len, self.num_classes)
        
        @parameter
        fn classify_batch_seq(idx: Int):
            let batch_idx = idx // seq_len
            let seq_idx = idx % seq_len
            
            let hidden_offset = (batch_idx * seq_len + seq_idx) * self.hidden_dim
            let logits_offset = (batch_idx * seq_len + seq_idx) * self.num_classes
            
            # Compute logits for each class
            for class_idx in range(self.num_classes):
                var logit_sum = SIMD[DType.float32, SIMD_WIDTH](0.0)
                
                @parameter
                fn dot_product_simd(hidden_idx: Int):
                    let hidden_vals = hidden_states.load[width=SIMD_WIDTH](hidden_offset + hidden_idx)
                    let weight_vals = self.weight.load[width=SIMD_WIDTH](class_idx * self.hidden_dim + hidden_idx)
                    logit_sum = logit_sum + hidden_vals * weight_vals
                
                vectorize[dot_product_simd, SIMD_WIDTH](self.hidden_dim)
                
                # Reduce and add bias
                var final_logit: Float32 = self.bias[class_idx]
                for i in range(SIMD_WIDTH):
                    final_logit = final_logit + logit_sum[i]
                
                logits[logits_offset + class_idx] = final_logit
        
        parallelize[classify_batch_seq](batch_size * seq_len)
        return logits


struct ToolAwareLoss:
    """Comprehensive tool-aware loss function with multiple objectives"""
    var token_weights: ToolTokenWeights
    var classification_head: ToolClassificationHead
    var vocab_size: Int
    var use_auxiliary_loss: Bool
    var auxiliary_loss_weight: Float32
    
    fn __init__(
        inout self, 
        vocab_size: Int, 
        hidden_dim: Int,
        use_auxiliary_loss: Bool = True,
        auxiliary_loss_weight: Float32 = 0.1
    ):
        self.vocab_size = vocab_size
        self.token_weights = ToolTokenWeights()
        self.classification_head = ToolClassificationHead(hidden_dim, 3)
        self.use_auxiliary_loss = use_auxiliary_loss
        self.auxiliary_loss_weight = auxiliary_loss_weight
    
    fn compute_loss(
        self,
        logits: Tensor[DType.float32],        # [batch, seq, vocab]
        targets: Tensor[DType.int32],         # [batch, seq]
        hidden_states: Tensor[DType.float32], # [batch, seq, hidden] - for auxiliary loss
        attention_mask: Tensor[DType.float32] # [batch, seq] - 1.0 for valid tokens
    ) -> Tuple[Float32, Tensor[DType.float32]]:
        """
        Compute tool-aware loss with optional auxiliary classification objective.
        Returns (total_loss, grad_logits)
        """
        let batch_size = logits.shape()[0]
        let seq_len = logits.shape()[1]
        
        # Initialize gradient tensor
        var grad_logits = Tensor[DType.float32](logits.shape())
        memset_zero(grad_logits.data(), grad_logits.num_elements() * 4)
        
        # Compute primary weighted cross-entropy loss
        let primary_loss = self._compute_weighted_cross_entropy(
            logits, targets, attention_mask, grad_logits
        )
        
        var total_loss = primary_loss
        
        # Add auxiliary tool classification loss
        if self.use_auxiliary_loss:
            let aux_loss = self._compute_auxiliary_classification_loss(
                hidden_states, targets, attention_mask
            )
            total_loss = total_loss + self.auxiliary_loss_weight * aux_loss
        
        return (total_loss, grad_logits)
    
    fn _compute_weighted_cross_entropy(
        self,
        logits: Tensor[DType.float32],
        targets: Tensor[DType.int32],
        attention_mask: Tensor[DType.float32],
        inout grad_logits: Tensor[DType.float32]
    ) -> Float32:
        """Compute weighted cross-entropy with tool-aware token weighting"""
        let batch_size = logits.shape()[0]
        let seq_len = logits.shape()[1]
        var total_loss: Float32 = 0.0
        var total_weight: Float32 = 0.0
        
        @parameter
        fn weighted_cross_entropy_batch(batch_idx: Int):
            for seq_idx in range(seq_len):
                let mask_val = attention_mask[batch_idx * seq_len + seq_idx]
                if mask_val < 0.5:  # Skip padded tokens
                    continue
                
                let target_id = targets[batch_idx * seq_len + seq_idx]
                let token_weight = self.token_weights.get_token_weight(target_id)
                let logits_offset = (batch_idx * seq_len + seq_idx) * self.vocab_size
                
                # Find max logit for numerical stability
                var max_logit: Float32 = logits[logits_offset]
                
                @parameter
                fn find_max_simd(vocab_idx: Int):
                    let vals = logits.load[width=SIMD_WIDTH](logits_offset + vocab_idx)
                    for i in range(SIMD_WIDTH):
                        if vals[i] > max_logit:
                            max_logit = vals[i]
                
                vectorize[find_max_simd, SIMD_WIDTH](self.vocab_size)
                
                # Compute log-sum-exp with stability
                var sum_exp = SIMD[DType.float32, SIMD_WIDTH](0.0)
                
                @parameter
                fn log_sum_exp_simd(vocab_idx: Int):
                    let logit_vals = logits.load[width=SIMD_WIDTH](logits_offset + vocab_idx)
                    let exp_vals = exp(logit_vals - max_logit)
                    sum_exp = sum_exp + exp_vals
                
                vectorize[log_sum_exp_simd, SIMD_WIDTH](self.vocab_size)
                
                # Reduce sum_exp
                var total_sum_exp: Float32 = 0.0
                for i in range(SIMD_WIDTH):
                    total_sum_exp = total_sum_exp + sum_exp[i]
                
                let log_sum_exp = log(total_sum_exp) + max_logit
                let target_logit = logits[logits_offset + Int(target_id)]
                let sample_loss = log_sum_exp - target_logit
                
                # Weight the loss
                let weighted_loss = sample_loss * token_weight
                total_loss = total_loss + weighted_loss
                total_weight = total_weight + token_weight
                
                # Compute weighted gradients
                @parameter
                fn compute_gradients_simd(vocab_idx: Int):
                    let logit_vals = logits.load[width=SIMD_WIDTH](logits_offset + vocab_idx)
                    let softmax_vals = exp(logit_vals - log_sum_exp)
                    
                    var grad_vals = softmax_vals * token_weight
                    for i in range(SIMD_WIDTH):
                        if vocab_idx + i == target_id:
                            grad_vals[i] = grad_vals[i] - token_weight
                    
                    grad_logits.store[width=SIMD_WIDTH](logits_offset + vocab_idx, grad_vals)
                
                vectorize[compute_gradients_simd, SIMD_WIDTH](self.vocab_size)
        
        parallelize[weighted_cross_entropy_batch](batch_size)
        
        return total_loss / total_weight if total_weight > 0.0 else 0.0
    
    fn _compute_auxiliary_classification_loss(
        self,
        hidden_states: Tensor[DType.float32],
        targets: Tensor[DType.int32],
        attention_mask: Tensor[DType.float32]
    ) -> Float32:
        """Compute auxiliary loss for early tool vs thinking classification"""
        # Get classification logits
        let class_logits = self.classification_head.forward(hidden_states)
        
        # Create classification targets from sequence targets
        var class_targets = self._create_classification_targets(targets)
        
        # Compute classification loss
        let batch_size = class_logits.shape()[0]
        let seq_len = class_logits.shape()[1]
        let num_classes = class_logits.shape()[2]
        
        var total_loss: Float32 = 0.0
        var valid_samples: Float32 = 0.0
        
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                let mask_val = attention_mask[batch_idx * seq_len + seq_idx]
                if mask_val < 0.5:
                    continue
                
                let class_target = class_targets[batch_idx * seq_len + seq_idx]
                if class_target < 0:  # Skip non-classification positions
                    continue
                
                let logits_offset = (batch_idx * seq_len + seq_idx) * num_classes
                
                # Compute cross-entropy for this sample
                var max_logit = class_logits[logits_offset]
                for i in range(1, num_classes):
                    if class_logits[logits_offset + i] > max_logit:
                        max_logit = class_logits[logits_offset + i]
                
                var sum_exp: Float32 = 0.0
                for i in range(num_classes):
                    sum_exp = sum_exp + exp(class_logits[logits_offset + i] - max_logit)
                
                let log_sum_exp = log(sum_exp) + max_logit
                let target_logit = class_logits[logits_offset + class_target]
                let sample_loss = log_sum_exp - target_logit
                
                total_loss = total_loss + sample_loss
                valid_samples = valid_samples + 1.0
        
        return total_loss / valid_samples if valid_samples > 0.0 else 0.0
    
    fn _create_classification_targets(self, targets: Tensor[DType.int32]) -> Tensor[DType.int32]:
        """Create classification targets from sequence targets"""
        let batch_size = targets.shape()[0]
        let seq_len = targets.shape()[1]
        var class_targets = Tensor[DType.int32](batch_size, seq_len)
        
        # Initialize with -1 (ignore class)
        for i in range(class_targets.num_elements()):
            class_targets.data()[i] = -1
        
        # Map special tokens to classification classes
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                let target_id = targets[batch_idx * seq_len + seq_idx]
                let class_idx = batch_idx * seq_len + seq_idx
                
                if target_id == TOOL_START_ID:
                    class_targets[class_idx] = 0  # Tool class
                elif target_id == THINKING_ID:
                    class_targets[class_idx] = 1  # Thinking class
                elif target_id == RESPONSE_ID:
                    class_targets[class_idx] = 2  # Response class
                # else: keep as -1 (ignore)
        
        return class_targets
    
    fn update_token_weights(inout self, new_weights: ToolTokenWeights):
        """Update token weights for curriculum learning or adaptation"""
        self.token_weights = new_weights
    
    fn get_loss_statistics(self, targets: Tensor[DType.int32]) -> Dict[String, Float32]:
        """Analyze token distribution in batch for loss balancing"""
        var stats = Dict[String, Float32]()
        
        let total_tokens = targets.num_elements()
        var tool_tokens: Float32 = 0.0
        var thinking_tokens: Float32 = 0.0
        var response_tokens: Float32 = 0.0
        
        for i in range(total_tokens):
            let token_id = targets[i]
            if token_id == TOOL_START_ID or token_id == TOOL_RESPONSE_ID:
                tool_tokens = tool_tokens + 1.0
            elif token_id == THINKING_ID:
                thinking_tokens = thinking_tokens + 1.0
            elif token_id == RESPONSE_ID:
                response_tokens = response_tokens + 1.0
        
        stats["tool_token_ratio"] = tool_tokens / Float32(total_tokens)
        stats["thinking_token_ratio"] = thinking_tokens / Float32(total_tokens)
        stats["response_token_ratio"] = response_tokens / Float32(total_tokens)
        stats["total_special_tokens"] = (tool_tokens + thinking_tokens + response_tokens) / Float32(total_tokens)
        
        return stats


struct SequenceLevelLoss:
    """Sequence-level loss for complete tool-calling flows"""
    var sequence_weights: Dict[String, Float32]
    var min_sequence_length: Int
    
    fn __init__(inout self):
        self.sequence_weights = Dict[String, Float32]()
        self.min_sequence_length = 3  # Minimum tokens for a valid sequence
        
        # Configure Universal Thinking Prefix sequence pattern weights
        self.sequence_weights["thinking_tool_sequence"] = 3.0    # <thinking>...<tool>...<tool_response>...<response>
        self.sequence_weights["thinking_direct_sequence"] = 2.5  # <thinking>...<response>
        self.sequence_weights["multi_tool_sequence"] = 3.5      # Multiple <thinking>...<tool> patterns
        self.sequence_weights["incomplete_sequence"] = 0.3      # Stronger penalty for incomplete sequences
    
    fn compute_sequence_loss(
        self,
        logits: Tensor[DType.float32],
        targets: Tensor[DType.int32],
        attention_mask: Tensor[DType.float32]
    ) -> Float32:
        """Compute loss based on complete sequence patterns"""
        let batch_size = logits.shape()[0]
        let seq_len = logits.shape()[1]
        
        var total_sequence_loss: Float32 = 0.0
        var num_sequences: Float32 = 0.0
        
        # Analyze each sequence in the batch
        for batch_idx in range(batch_size):
            let sequence_loss = self._analyze_sequence_pattern(
                targets, batch_idx, seq_len, attention_mask
            )
            total_sequence_loss = total_sequence_loss + sequence_loss
            num_sequences = num_sequences + 1.0
        
        return total_sequence_loss / num_sequences if num_sequences > 0.0 else 0.0
    
    fn _analyze_sequence_pattern(
        self,
        targets: Tensor[DType.int32],
        batch_idx: Int,
        seq_len: Int,
        attention_mask: Tensor[DType.float32]
    ) -> Float32:
        """Analyze single sequence for Universal Thinking Prefix pattern completeness"""
        var sequence_score: Float32 = 1.0  # Start with neutral score
        var thinking_count = 0
        var tool_count = 0
        var tool_response_count = 0
        var response_count = 0
        var has_thinking_first = False
        var in_tool_flow = False
        
        for seq_idx in range(seq_len):
            let mask_val = attention_mask[batch_idx * seq_len + seq_idx]
            if mask_val < 0.5:
                continue
            
            let token_id = targets[batch_idx * seq_len + seq_idx]
            
            # Track Universal Thinking Prefix patterns
            if token_id == THINKING_ID:
                thinking_count += 1
                if thinking_count == 1 and seq_idx < 3:  # First thinking token should be early
                    has_thinking_first = True
            elif token_id == TOOL_START_ID:
                tool_count += 1
                if thinking_count > 0:  # Tool after thinking - good pattern
                    in_tool_flow = True
            elif token_id == TOOL_RESPONSE_ID:
                tool_response_count += 1
                if in_tool_flow and tool_count > 0:
                    # Complete thinking→tool→tool_response pattern
                    sequence_score = sequence_score * self.sequence_weights["thinking_tool_sequence"]
            elif token_id == RESPONSE_ID:
                response_count += 1
                if thinking_count > 0 and tool_count == 0:
                    # thinking→response pattern (no tool)
                    sequence_score = sequence_score * self.sequence_weights["thinking_direct_sequence"]
                elif thinking_count > 1 and tool_count > 1:
                    # Multi-step thinking→tool pattern
                    sequence_score = sequence_score * self.sequence_weights["multi_tool_sequence"]
        
        # Validate Universal Thinking Prefix requirements
        if not has_thinking_first:
            # STRONG penalty for not starting with thinking
            sequence_score = sequence_score * 0.1
        
        # Penalize incomplete sequences
        if tool_count > tool_response_count:
            # Tools without responses
            sequence_score = sequence_score * self.sequence_weights["incomplete_sequence"]
        if thinking_count > 0 and response_count == 0:
            # Thinking without final response
            sequence_score = sequence_score * self.sequence_weights["incomplete_sequence"]
        
        return sequence_score


# Factory functions
fn create_tool_aware_loss(vocab_size: Int, hidden_dim: Int) -> ToolAwareLoss:
    """Create tool-aware loss with default configuration"""
    return ToolAwareLoss(vocab_size, hidden_dim, use_auxiliary_loss=True)

fn create_sequence_level_loss() -> SequenceLevelLoss:
    """Create sequence-level loss for flow validation"""
    return SequenceLevelLoss()

fn create_tool_token_weights() -> ToolTokenWeights:
    """Create default tool token weights"""
    return ToolTokenWeights()