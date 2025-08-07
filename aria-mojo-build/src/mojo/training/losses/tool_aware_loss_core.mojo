"""
Core Tool-Aware Loss Functions

Implements the foundational loss calculations for tool-calling training
with Universal Thinking Prefix support and weighted token objectives.
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

struct TokenWeights:
    """Manages token weights for Universal Thinking Prefix training"""
    var weights: Dict[Int, Float32]
    var default_weight: Float32
    
    fn __init__(inout self):
        self.weights = Dict[Int, Float32]()
        self.default_weight = 1.0
        self._initialize_universal_thinking_weights()
    
    fn _initialize_universal_thinking_weights(inout self):
        """Initialize weights optimized for Universal Thinking Prefix"""
        # Universal Thinking Prefix - ALL examples start with <thinking>
        self.weights[THINKING_ID] = 4.0      # HIGHEST weight - all examples start here
        self.weights[TOOL_START_ID] = 3.0    # High weight for tool decision after thinking
        self.weights[TOOL_RESPONSE_ID] = 2.0 # Boost tool responses
        self.weights[RESPONSE_ID] = 1.5      # Important for final output
        self.weights[TOOL_END_ID] = 1.0      # Standard weight for closing tags
        self.weights[EOS_ID] = 1.2           # Slightly boost end-of-sequence
        self.weights[BOS_ID] = 0.8           # Lower weight for start tokens
    
    fn get_weight(self, token_id: Int) -> Float32:
        """Get weight for a specific token"""
        if token_id in self.weights:
            return self.weights[token_id]
        return self.default_weight
    
    fn update_weight(inout self, token_id: Int, weight: Float32):
        """Update weight for a specific token"""
        self.weights[token_id] = weight

struct ToolAwareCrossEntropy:
    """Cross-entropy loss with tool-aware token weighting"""
    var token_weights: TokenWeights
    var sequence_weight_decay: Float32
    var temperature: Float32
    
    fn __init__(inout self, temperature: Float32 = 1.0):
        self.token_weights = TokenWeights()
        self.sequence_weight_decay = 0.95
        self.temperature = temperature
    
    fn forward(
        self,
        logits: Tensor[DType.float32],  # [batch_size, seq_len, vocab_size]
        targets: Tensor[DType.int32],   # [batch_size, seq_len]
        attention_mask: Tensor[DType.float32]  # [batch_size, seq_len]
    ) -> Float32:
        """Compute weighted cross-entropy loss"""
        let batch_size = logits.shape()[0]
        let seq_len = logits.shape()[1]
        let vocab_size = logits.shape()[2]
        
        var total_loss: Float32 = 0.0
        var total_weight: Float32 = 0.0
        
        @parameter
        fn compute_sample_loss(batch_idx: Int):
            var sample_loss: Float32 = 0.0
            var sample_weight: Float32 = 0.0
            
            for seq_idx in range(seq_len):
                let mask_value = attention_mask[batch_idx * seq_len + seq_idx]
                if mask_value > 0.0:  # Only compute loss for non-masked positions
                    let target_id = targets[batch_idx * seq_len + seq_idx]
                    let logits_offset = (batch_idx * seq_len + seq_idx) * vocab_size
                    
                    # Apply temperature scaling
                    var max_logit: Float32 = logits[logits_offset]
                    for v in range(vocab_size):
                        if logits[logits_offset + v] > max_logit:
                            max_logit = logits[logits_offset + v]
                    
                    # Compute softmax with temperature
                    var sum_exp: Float32 = 0.0
                    for v in range(vocab_size):
                        let scaled_logit = (logits[logits_offset + v] - max_logit) / self.temperature
                        sum_exp = sum_exp + exp(scaled_logit)
                    
                    let target_logit = (logits[logits_offset + target_id] - max_logit) / self.temperature
                    let log_softmax = target_logit - log(sum_exp)
                    
                    # Apply token-specific weighting
                    let token_weight = self.token_weights.get_weight(target_id)
                    let position_weight = pow(self.sequence_weight_decay, seq_idx)
                    let final_weight = token_weight * position_weight * mask_value
                    
                    sample_loss = sample_loss + (-log_softmax * final_weight)
                    sample_weight = sample_weight + final_weight
            
            total_loss = total_loss + sample_loss
            total_weight = total_weight + sample_weight
        
        parallelize[compute_sample_loss](batch_size)
        
        return total_loss / total_weight if total_weight > 0.0 else 0.0
    
    fn backward(
        self,
        logits: Tensor[DType.float32],
        targets: Tensor[DType.int32],
        attention_mask: Tensor[DType.float32],
        inout grad_logits: Tensor[DType.float32]
    ) -> None:
        """Compute gradients for weighted cross-entropy loss"""
        let batch_size = logits.shape()[0]
        let seq_len = logits.shape()[1]
        let vocab_size = logits.shape()[2]
        
        # Zero out gradients
        memset_zero(grad_logits.data(), grad_logits.num_elements() * 4)
        
        @parameter
        fn compute_sample_gradients(batch_idx: Int):
            for seq_idx in range(seq_len):
                let mask_value = attention_mask[batch_idx * seq_len + seq_idx]
                if mask_value > 0.0:
                    let target_id = targets[batch_idx * seq_len + seq_idx]
                    let logits_offset = (batch_idx * seq_len + seq_idx) * vocab_size
                    
                    # Compute softmax
                    var max_logit: Float32 = logits[logits_offset]
                    for v in range(vocab_size):
                        if logits[logits_offset + v] > max_logit:
                            max_logit = logits[logits_offset + v]
                    
                    var sum_exp: Float32 = 0.0
                    for v in range(vocab_size):
                        let scaled_logit = (logits[logits_offset + v] - max_logit) / self.temperature
                        sum_exp = sum_exp + exp(scaled_logit)
                    
                    # Compute gradients
                    let token_weight = self.token_weights.get_weight(target_id)
                    let position_weight = pow(self.sequence_weight_decay, seq_idx)
                    let final_weight = token_weight * position_weight * mask_value
                    
                    for v in range(vocab_size):
                        let scaled_logit = (logits[logits_offset + v] - max_logit) / self.temperature
                        let softmax_val = exp(scaled_logit) / sum_exp
                        
                        var grad_val = softmax_val / self.temperature
                        if v == target_id:
                            grad_val = grad_val - (1.0 / self.temperature)
                        
                        grad_logits[logits_offset + v] = grad_val * final_weight
        
        parallelize[compute_sample_gradients](batch_size)

struct SequenceClassificationLoss:
    """Loss function for sequence-level tool classification"""
    var class_weights: List[Float32]
    
    fn __init__(inout self):
        self.class_weights = List[Float32]()
        # Tool usage classes: no_tool, math_tool, text_tool, conversion_tool, reasoning_tool
        self.class_weights.append(1.0)  # no_tool
        self.class_weights.append(2.0)  # math_tool - higher weight
        self.class_weights.append(1.5)  # text_tool
        self.class_weights.append(1.5)  # conversion_tool
        self.class_weights.append(2.0)  # reasoning_tool - higher weight
    
    fn forward(
        self,
        sequence_logits: Tensor[DType.float32],  # [batch_size, num_classes]
        sequence_targets: Tensor[DType.int32]    # [batch_size]
    ) -> Float32:
        """Compute weighted sequence classification loss"""
        let batch_size = sequence_logits.shape()[0]
        let num_classes = sequence_logits.shape()[1]
        
        var total_loss: Float32 = 0.0
        
        for batch_idx in range(batch_size):
            let target_class = sequence_targets[batch_idx]
            let logits_offset = batch_idx * num_classes
            
            # Compute log-softmax
            var max_logit: Float32 = sequence_logits[logits_offset]
            for c in range(num_classes):
                if sequence_logits[logits_offset + c] > max_logit:
                    max_logit = sequence_logits[logits_offset + c]
            
            var sum_exp: Float32 = 0.0
            for c in range(num_classes):
                sum_exp = sum_exp + exp(sequence_logits[logits_offset + c] - max_logit)
            
            let target_logit = sequence_logits[logits_offset + target_class] - max_logit
            let log_softmax = target_logit - log(sum_exp)
            
            # Apply class weighting
            let class_weight = self.class_weights[target_class]
            total_loss = total_loss + (-log_softmax * class_weight)
        
        return total_loss / batch_size

struct EarlyToolPredictionLoss:
    """Specialized loss for early tool prediction in Universal Thinking Prefix"""
    var thinking_boost: Float32
    var tool_decision_boost: Float32
    
    fn __init__(inout self, thinking_boost: Float32 = 5.0, tool_decision_boost: Float32 = 3.0):
        self.thinking_boost = thinking_boost
        self.tool_decision_boost = tool_decision_boost
    
    fn forward(
        self,
        logits: Tensor[DType.float32],
        targets: Tensor[DType.int32],
        sequence_positions: Tensor[DType.int32]  # Position of each token in sequence
    ) -> Float32:
        """Compute loss with heavy emphasis on early thinking and tool predictions"""
        let batch_size = logits.shape()[0]
        let seq_len = logits.shape()[1]
        let vocab_size = logits.shape()[2]
        
        var total_loss: Float32 = 0.0
        var total_weight: Float32 = 0.0
        
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                let target_id = targets[batch_idx * seq_len + seq_idx]
                let position = sequence_positions[batch_idx * seq_len + seq_idx]
                let logits_offset = (batch_idx * seq_len + seq_idx) * vocab_size
                
                # Compute cross-entropy
                var max_logit: Float32 = logits[logits_offset]
                for v in range(vocab_size):
                    if logits[logits_offset + v] > max_logit:
                        max_logit = logits[logits_offset + v]
                
                var sum_exp: Float32 = 0.0
                for v in range(vocab_size):
                    sum_exp = sum_exp + exp(logits[logits_offset + v] - max_logit)
                
                let target_logit = logits[logits_offset + target_id] - max_logit
                let log_softmax = target_logit - log(sum_exp)
                
                # Apply Universal Thinking Prefix weighting
                var weight: Float32 = 1.0
                
                if target_id == THINKING_ID:
                    # Massively boost <thinking> tokens, especially early ones
                    weight = self.thinking_boost * (1.0 + 10.0 / (position + 1))
                elif target_id == TOOL_START_ID:
                    # Boost <tool> tokens that come after thinking
                    weight = self.tool_decision_boost * (1.0 + 5.0 / (position + 1))
                elif position < 10:
                    # Boost early tokens in general
                    weight = 2.0
                
                total_loss = total_loss + (-log_softmax * weight)
                total_weight = total_weight + weight
        
        return total_loss / total_weight if total_weight > 0.0 else 0.0