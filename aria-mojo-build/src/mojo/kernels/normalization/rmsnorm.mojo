"""
RMSNorm Kernel

High-performance RMSNorm implementation with SIMD optimization for MI300X.
Includes both forward and backward passes with vectorization.
"""

from sys import simdwidthof
from memory import memset_zero, memcpy
from algorithm import vectorize, parallelize
from builtin import int32, float32
from math import sqrt
from tensor import Tensor

alias SIMD_WIDTH = simdwidthof[DType.float32]()
alias CACHE_LINE_SIZE = 64

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

struct LayerNormKernel:
    """Standard LayerNorm implementation for comparison"""
    
    @staticmethod
    fn forward(
        input: Tensor[DType.float32],
        weight: Tensor[DType.float32],
        bias: Tensor[DType.float32],
        inout output: Tensor[DType.float32],
        eps: Float32 = 1e-6
    ) -> None:
        """LayerNorm forward pass"""
        let batch_size = input.shape()[0]
        let hidden_dim = input.shape()[1]
        
        @parameter
        fn layernorm_batch(batch_idx: Int):
            let offset = batch_idx * hidden_dim
            
            # Compute mean
            var mean: Float32 = 0.0
            for i in range(hidden_dim):
                mean = mean + input[offset + i]
            mean = mean / hidden_dim
            
            # Compute variance
            var variance: Float32 = 0.0
            for i in range(hidden_dim):
                let diff = input[offset + i] - mean
                variance = variance + diff * diff
            variance = variance / hidden_dim
            
            let inv_std = 1.0 / sqrt(variance + eps)
            
            # Normalize
            @parameter
            fn normalize_vectorized(dim_idx: Int):
                let input_vals = input.load[width=SIMD_WIDTH](offset + dim_idx)
                let weight_vals = weight.load[width=SIMD_WIDTH](dim_idx)
                let bias_vals = bias.load[width=SIMD_WIDTH](dim_idx)
                
                let centered = input_vals - mean
                let normalized = centered * inv_std * weight_vals + bias_vals
                output.store[width=SIMD_WIDTH](offset + dim_idx, normalized)
            
            vectorize[normalize_vectorized, SIMD_WIDTH](hidden_dim)
        
        parallelize[layernorm_batch](batch_size)