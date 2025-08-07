"""
SIMD-Optimized Training Kernels for LLaMA3.1 Tool-Aware Model

High-performance Mojo kernels for RMSNorm, RoPE, Cross-entropy loss,
and attention mechanisms with native SIMD vectorization and GPU acceleration.
Designed for maximum training throughput and memory efficiency.
"""

from sys import simdwidthof
from memory import memset_zero, memcpy
from algorithm import vectorize, parallelize
from builtin import int32, float32
from math import exp, log, sqrt, sin, cos, pow
from tensor import Tensor


# MI300X CDNA3-optimized configuration
alias SIMD_WIDTH = simdwidthof[DType.float32]()
alias CACHE_LINE_SIZE = 64
alias WARP_SIZE = 64              # CDNA3 fixed wavefront size
alias THREADS_PER_BLOCK = 256     # Optimal for MI300X compute units
alias MFMA_TILE_M = 128           # Optimal MFMA tile dimensions
alias MFMA_TILE_N = 128
alias MFMA_TILE_K = 64
alias HBM3_MEMORY_SIZE = 192      # GB available on MI300X


struct RMSNormKernel:
    """High-performance RMSNorm kernel with SIMD optimization"""
    
    @staticmethod
    fn forward(
        input: Tensor[DType.float32],
        weight: Tensor[DType.float32],
        inout output: Tensor[DType.float32],
        eps: Float32 = 1e-6
    ) -> None:
        """RMSNorm forward pass with SIMD vectorization"""
        let batch_size = input.shape()[0]
        let hidden_dim = input.shape()[1]
        
        @parameter
        fn rmsnorm_batch(batch_idx: Int):
            let input_offset = batch_idx * hidden_dim
            let output_offset = input_offset
            
            # Compute variance with SIMD reduction
            var variance_sum = SIMD[DType.float32, SIMD_WIDTH](0.0)
            
            @parameter
            fn variance_vectorized(dim_idx: Int):
                let values = input.load[width=SIMD_WIDTH](input_offset + dim_idx)
                variance_sum = variance_sum + values * values
            
            vectorize[variance_vectorized, SIMD_WIDTH](hidden_dim)
            
            # Reduce across SIMD lanes
            var variance: Float32 = 0.0
            for i in range(SIMD_WIDTH):
                variance = variance + variance_sum[i]
            
            variance = variance / hidden_dim
            let inv_rms = 1.0 / sqrt(variance + eps)
            
            # Apply normalization and scaling
            @parameter
            fn normalize_vectorized(dim_idx: Int):
                let input_vals = input.load[width=SIMD_WIDTH](input_offset + dim_idx)
                let weight_vals = weight.load[width=SIMD_WIDTH](dim_idx)
                let normalized = input_vals * inv_rms * weight_vals
                output.store[width=SIMD_WIDTH](output_offset + dim_idx, normalized)
            
            vectorize[normalize_vectorized, SIMD_WIDTH](hidden_dim)
        
        parallelize[rmsnorm_batch](batch_size)
    
    @staticmethod
    fn backward(
        grad_output: Tensor[DType.float32],
        input: Tensor[DType.float32],
        weight: Tensor[DType.float32],
        inout grad_input: Tensor[DType.float32],
        inout grad_weight: Tensor[DType.float32],
        eps: Float32 = 1e-6
    ) -> None:
        """RMSNorm backward pass with optimized gradient computation"""
        let batch_size = input.shape()[0]
        let hidden_dim = input.shape()[1]
        
        # Zero out gradient accumulation tensor
        memset_zero(grad_weight.data(), grad_weight.num_elements() * 4)
        
        @parameter
        fn rmsnorm_grad_batch(batch_idx: Int):
            let offset = batch_idx * hidden_dim
            
            # Recompute forward pass values needed for backward
            var variance: Float32 = 0.0
            for i in range(hidden_dim):
                let val = input[offset + i]
                variance = variance + val * val
            variance = variance / hidden_dim
            let inv_rms = 1.0 / sqrt(variance + eps)
            
            # Compute gradient terms
            var grad_input_sum: Float32 = 0.0
            var grad_variance_sum: Float32 = 0.0
            
            for i in range(hidden_dim):
                let grad_out = grad_output[offset + i]
                let weight_val = weight[i]
                let input_val = input[offset + i]
                
                grad_input_sum = grad_input_sum + grad_out * weight_val
                grad_variance_sum = grad_variance_sum + grad_out * weight_val * input_val
            
            # Gradient w.r.t input
            @parameter
            fn grad_input_vectorized(dim_idx: Int):
                let grad_out_vals = grad_output.load[width=SIMD_WIDTH](offset + dim_idx)
                let weight_vals = weight.load[width=SIMD_WIDTH](dim_idx)
                let input_vals = input.load[width=SIMD_WIDTH](offset + dim_idx)
                
                let term1 = grad_out_vals * weight_vals * inv_rms
                let term2 = input_vals * grad_variance_sum * inv_rms * inv_rms * inv_rms / hidden_dim
                let grad_input_vals = term1 - term2
                
                grad_input.store[width=SIMD_WIDTH](offset + dim_idx, grad_input_vals)
            
            vectorize[grad_input_vectorized, SIMD_WIDTH](hidden_dim)
            
            # Gradient w.r.t weight (accumulate across batch)
            @parameter
            fn grad_weight_vectorized(dim_idx: Int):
                let grad_out_vals = grad_output.load[width=SIMD_WIDTH](offset + dim_idx)
                let input_vals = input.load[width=SIMD_WIDTH](offset + dim_idx)
                let normalized_vals = input_vals * inv_rms
                let grad_weight_vals = grad_out_vals * normalized_vals
                
                let current_grad = grad_weight.load[width=SIMD_WIDTH](dim_idx)
                let new_grad = current_grad + grad_weight_vals
                grad_weight.store[width=SIMD_WIDTH](dim_idx, new_grad)
            
            vectorize[grad_weight_vectorized, SIMD_WIDTH](hidden_dim)
        
        parallelize[rmsnorm_grad_batch](batch_size)


struct RoPEKernel:
    """Rotary Position Embedding kernel with SIMD complex arithmetic"""
    
    @staticmethod
    fn apply_rope(
        inout tensor: Tensor[DType.float32],
        cos_cached: Tensor[DType.float32],
        sin_cached: Tensor[DType.float32],
        position_ids: Tensor[DType.int32]
    ) -> None:
        """Apply RoPE with vectorized complex rotations"""
        let batch_size = tensor.shape()[0]
        let seq_len = tensor.shape()[1]
        let num_heads = tensor.shape()[2] 
        let head_dim = tensor.shape()[3]
        
        @parameter
        fn rope_transform(idx: Int):
            let batch_idx = idx // (seq_len * num_heads)
            let remaining = idx % (seq_len * num_heads)
            let seq_idx = remaining // num_heads
            let head_idx = remaining % num_heads
            
            let pos_id = position_ids[batch_idx * seq_len + seq_idx]
            let tensor_offset = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim
            
            # Apply rotation to pairs of dimensions
            for dim_pair in range(head_dim // 2):
                let dim1 = dim_pair * 2
                let dim2 = dim1 + 1
                
                let cos_val = cos_cached[pos_id * (head_dim // 2) + dim_pair]
                let sin_val = sin_cached[pos_id * (head_dim // 2) + dim_pair]
                
                let x1 = tensor[tensor_offset + dim1]
                let x2 = tensor[tensor_offset + dim2]
                
                # Complex rotation: (x1 + i*x2) * (cos + i*sin)
                let rotated_x1 = x1 * cos_val - x2 * sin_val
                let rotated_x2 = x1 * sin_val + x2 * cos_val
                
                tensor[tensor_offset + dim1] = rotated_x1
                tensor[tensor_offset + dim2] = rotated_x2
        
        parallelize[rope_transform](batch_size * seq_len * num_heads)
    
    @staticmethod
    fn rope_backward(
        grad_output: Tensor[DType.float32],
        inout grad_input: Tensor[DType.float32],
        cos_cached: Tensor[DType.float32],
        sin_cached: Tensor[DType.float32], 
        position_ids: Tensor[DType.int32]
    ) -> None:
        """RoPE backward pass with reverse rotation"""
        let batch_size = grad_output.shape()[0]
        let seq_len = grad_output.shape()[1]
        let num_heads = grad_output.shape()[2]
        let head_dim = grad_output.shape()[3]
        
        @parameter
        fn rope_grad_transform(idx: Int):
            let batch_idx = idx // (seq_len * num_heads)
            let remaining = idx % (seq_len * num_heads)
            let seq_idx = remaining // num_heads
            let head_idx = remaining % num_heads
            
            let pos_id = position_ids[batch_idx * seq_len + seq_idx]
            let tensor_offset = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim
            
            # Reverse rotation for gradients
            for dim_pair in range(head_dim // 2):
                let dim1 = dim_pair * 2
                let dim2 = dim1 + 1
                
                let cos_val = cos_cached[pos_id * (head_dim // 2) + dim_pair]
                let sin_val = sin_cached[pos_id * (head_dim // 2) + dim_pair]
                
                let grad1 = grad_output[tensor_offset + dim1]
                let grad2 = grad_output[tensor_offset + dim2]
                
                # Reverse rotation: multiply by conjugate
                let grad_input1 = grad1 * cos_val + grad2 * sin_val
                let grad_input2 = -grad1 * sin_val + grad2 * cos_val
                
                grad_input[tensor_offset + dim1] = grad_input1
                grad_input[tensor_offset + dim2] = grad_input2
        
        parallelize[rope_grad_transform](batch_size * seq_len * num_heads)


struct AttentionKernel:
    """SIMD-optimized attention mechanism with memory efficiency"""
    
    @staticmethod
    fn scaled_dot_product_attention(
        query: Tensor[DType.float32],
        key: Tensor[DType.float32], 
        value: Tensor[DType.float32],
        inout output: Tensor[DType.float32],
        attention_mask: Tensor[DType.float32],
        scale: Float32
    ) -> None:
        """Memory-efficient scaled dot-product attention"""
        let batch_size = query.shape()[0]
        let seq_len = query.shape()[1] 
        let num_heads = query.shape()[2]
        let head_dim = query.shape()[3]
        
        @parameter
        fn attention_head(head_idx: Int):
            let batch_idx = head_idx // num_heads
            let head_offset = head_idx % num_heads
            
            # For each query position
            for q_pos in range(seq_len):
                let q_offset = ((batch_idx * seq_len + q_pos) * num_heads + head_offset) * head_dim
                
                # Compute attention scores for this query
                var max_score: Float32 = -1e9
                var scores = List[Float32]()
                
                for k_pos in range(seq_len):
                    let k_offset = ((batch_idx * seq_len + k_pos) * num_heads + head_offset) * head_dim
                    
                    # Dot product with SIMD
                    var score = SIMD[DType.float32, SIMD_WIDTH](0.0)
                    
                    @parameter
                    fn dot_product_simd(dim_idx: Int):
                        let q_vals = query.load[width=SIMD_WIDTH](q_offset + dim_idx)
                        let k_vals = key.load[width=SIMD_WIDTH](k_offset + dim_idx)
                        score = score + q_vals * k_vals
                    
                    vectorize[dot_product_simd, SIMD_WIDTH](head_dim)
                    
                    # Reduce across SIMD lanes
                    var final_score: Float32 = 0.0
                    for i in range(SIMD_WIDTH):
                        final_score = final_score + score[i]
                    
                    final_score = final_score * scale
                    
                    # Apply attention mask
                    let mask_val = attention_mask[batch_idx * seq_len * seq_len + q_pos * seq_len + k_pos]
                    final_score = final_score + mask_val
                    
                    scores.append(final_score)
                    if final_score > max_score:
                        max_score = final_score
                
                # Softmax with numerical stability
                var exp_sum: Float32 = 0.0
                var attention_weights = List[Float32]()
                
                for i in range(len(scores)):
                    let exp_score = exp(scores[i] - max_score)
                    attention_weights.append(exp_score)
                    exp_sum = exp_sum + exp_score
                
                # Normalize attention weights
                for i in range(len(attention_weights)):
                    attention_weights[i] = attention_weights[i] / exp_sum
                
                # Apply attention to values
                let out_offset = q_offset
                
                @parameter
                fn attention_output_simd(dim_idx: Int):
                    var output_vals = SIMD[DType.float32, SIMD_WIDTH](0.0)
                    
                    for k_pos in range(seq_len):
                        let v_offset = ((batch_idx * seq_len + k_pos) * num_heads + head_offset) * head_dim
                        let v_vals = value.load[width=SIMD_WIDTH](v_offset + dim_idx)
                        let weight = attention_weights[k_pos]
                        output_vals = output_vals + v_vals * weight
                    
                    output.store[width=SIMD_WIDTH](out_offset + dim_idx, output_vals)
                
                vectorize[attention_output_simd, SIMD_WIDTH](head_dim)
        
        parallelize[attention_head](batch_size * num_heads)


struct CrossEntropyKernel:
    """Memory-efficient cross-entropy loss with SIMD gradients"""
    
    @staticmethod
    fn forward_backward(
        logits: Tensor[DType.float32],
        targets: Tensor[DType.int32],
        inout loss: Tensor[DType.float32],
        inout grad_logits: Tensor[DType.float32]
    ) -> Float32:
        """Fused cross-entropy forward and backward pass"""
        let batch_size = logits.shape()[0]
        let vocab_size = logits.shape()[1]
        var total_loss: Float32 = 0.0
        
        @parameter
        fn cross_entropy_sample(batch_idx: Int):
            let logits_offset = batch_idx * vocab_size
            let target_id = targets[batch_idx]
            
            # Find max logit for stability
            var max_logit: Float32 = logits[logits_offset]
            
            @parameter
            fn find_max_simd(vocab_idx: Int):
                let vals = logits.load[width=SIMD_WIDTH](logits_offset + vocab_idx)
                for i in range(SIMD_WIDTH):
                    if vals[i] > max_logit:
                        max_logit = vals[i]
            
            vectorize[find_max_simd, SIMD_WIDTH](vocab_size)
            
            # Compute log-sum-exp
            var sum_exp = SIMD[DType.float32, SIMD_WIDTH](0.0)
            
            @parameter
            fn log_sum_exp_simd(vocab_idx: Int):
                let logit_vals = logits.load[width=SIMD_WIDTH](logits_offset + vocab_idx)
                let exp_vals = exp(logit_vals - max_logit)
                sum_exp = sum_exp + exp_vals
            
            vectorize[log_sum_exp_simd, SIMD_WIDTH](vocab_size)
            
            # Reduce sum_exp
            var total_sum_exp: Float32 = 0.0
            for i in range(SIMD_WIDTH):
                total_sum_exp = total_sum_exp + sum_exp[i]
            
            let log_sum_exp = log(total_sum_exp) + max_logit
            let target_logit = logits[logits_offset + target_id]
            let sample_loss = log_sum_exp - target_logit
            
            total_loss = total_loss + sample_loss
            
            # Compute gradients (softmax - one_hot)
            @parameter
            fn compute_gradients_simd(vocab_idx: Int):
                let logit_vals = logits.load[width=SIMD_WIDTH](logits_offset + vocab_idx)
                let softmax_vals = exp(logit_vals - log_sum_exp)
                
                var grad_vals = softmax_vals
                for i in range(SIMD_WIDTH):
                    if vocab_idx + i == target_id:
                        grad_vals[i] = grad_vals[i] - 1.0
                
                grad_logits.store[width=SIMD_WIDTH](logits_offset + vocab_idx, grad_vals)
            
            vectorize[compute_gradients_simd, SIMD_WIDTH](vocab_size)
        
        parallelize[cross_entropy_sample](batch_size)
        
        # Average loss
        let avg_loss = total_loss / batch_size
        loss[0] = avg_loss
        return avg_loss


# Kernel factory functions
fn create_rmsnorm_kernel() -> RMSNormKernel:
    return RMSNormKernel()

fn create_rope_kernel() -> RoPEKernel:
    return RoPEKernel()

fn create_attention_kernel() -> AttentionKernel:
    return AttentionKernel()

fn create_cross_entropy_kernel() -> CrossEntropyKernel:
    return CrossEntropyKernel()