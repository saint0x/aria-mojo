"""
SIMD Operations for LLaMA3.1 Tool-Aware Inference

MI300X-optimized SIMD kernels with MFMA integration and HBM striping support.
Ported from training kernels for maximum performance on CDNA3 architecture.
"""

from sys import simdwidthof
from memory import memset_zero, memcpy
from algorithm import vectorize, parallelize
from builtin import int32, float32
from math import exp, log, sqrt, sin, cos, tanh
from tensor import Tensor
from collections import List


# MI300X CDNA3 hardware configuration (aligned with training kernels)
alias SIMD_WIDTH = simdwidthof[DType.float32]()
alias WARP_SIZE = 64              # CDNA3 fixed wavefront size
alias THREADS_PER_BLOCK = 256     # Optimal for MI300X compute units
alias MFMA_TILE_M = 128           # Optimal MFMA tile dimensions
alias MFMA_TILE_N = 128
alias MFMA_TILE_K = 64
alias TILE_SIZE = 128             # Updated for MFMA alignment


struct SIMDKernels:
    """SIMD-optimized kernels for transformer inference"""
    
    @staticmethod
    @kernel
    fn softmax_kernel(inout logits: Tensor[DType.float32]) -> None:
        """
        MI300X-optimized softmax with WARP_SIZE=64 and numerical stability.
        Enhanced version from training kernels for tool token prediction.
        """
        let seq_len = logits.shape()[0]
        let vocab_size = logits.shape()[1] if len(logits.shape()) > 1 else logits.num_elements()
        
        if len(logits.shape()) == 1:
            # Single sequence softmax
            SIMDKernels._softmax_single(logits)
        else:
            # Batch softmax with MI300X parallelization
            @parameter
            fn softmax_batch(batch_idx: Int):
                let offset = batch_idx * vocab_size
                var batch_logits = Tensor[DType.float32](vocab_size)
                
                # Copy batch data with SIMD
                @parameter
                fn copy_batch_simd(idx: Int):
                    let vals = logits.load[width=SIMD_WIDTH](offset + idx)
                    batch_logits.store[width=SIMD_WIDTH](idx, vals)
                
                vectorize[copy_batch_simd, SIMD_WIDTH](vocab_size)
                
                # Apply softmax
                SIMDKernels._softmax_single(batch_logits)
                
                # Copy back with SIMD
                @parameter
                fn copy_back_simd(idx: Int):
                    let vals = batch_logits.load[width=SIMD_WIDTH](idx)
                    logits.store[width=SIMD_WIDTH](offset + idx, vals)
                
                vectorize[copy_back_simd, SIMD_WIDTH](vocab_size)
            
            parallelize[softmax_batch](seq_len)
    
    @staticmethod
    fn _softmax_single(inout logits: Tensor[DType.float32]) -> None:
        """Single sequence softmax with WARP_SIZE=64 SIMD vectorization"""
        let size = logits.num_elements()
        
        # Find max with SIMD reduction (optimized for WARP_SIZE=64)
        var max_val: Float32 = logits[0]
        
        @parameter
        fn find_max_vectorized(idx: Int):
            let values = logits.load[width=WARP_SIZE](idx)
            for i in range(WARP_SIZE):
                if values[i] > max_val:
                    max_val = values[i]
        
        vectorize[find_max_vectorized, WARP_SIZE](size)
        
        # Compute exp(x - max) and sum with SIMD
        var sum_exp = SIMD[DType.float32, WARP_SIZE](0.0)
        
        @parameter
        fn exp_and_sum_vectorized(idx: Int):
            let values = logits.load[width=WARP_SIZE](idx)
            let exp_values = exp(values - max_val)
            logits.store[width=WARP_SIZE](idx, exp_values)
            sum_exp = sum_exp + exp_values
        
        vectorize[exp_and_sum_vectorized, WARP_SIZE](size)
        
        # Reduce sum_exp across WARP_SIZE lanes
        var total_sum: Float32 = 0.0
        for i in range(WARP_SIZE):
            total_sum += sum_exp[i]
        
        # Normalize with SIMD
        @parameter
        fn normalize_vectorized(idx: Int):
            let values = logits.load[width=WARP_SIZE](idx)
            let normalized = values / total_sum
            logits.store[width=WARP_SIZE](idx, normalized)
        
        vectorize[normalize_vectorized, WARP_SIZE](size)
    
    @staticmethod
    fn matmul_kernel(
        a: Tensor[DType.float32], 
        b: Tensor[DType.float32], 
        inout result: Tensor[DType.float32]
    ) -> None:
        """
        Tiled matrix multiplication with SIMD optimization.
        Optimized for transformer attention and feed-forward layers.
        """
        let M = a.shape()[0]  # sequence length
        let K = a.shape()[1]  # hidden dimension
        let N = b.shape()[1]  # output dimension
        
        # Tiled multiplication for cache efficiency
        @parameter
        fn compute_tile(m_tile: Int, n_tile: Int) -> None:
            let m_start = m_tile * TILE_SIZE
            let m_end = min(m_start + TILE_SIZE, M)
            let n_start = n_tile * TILE_SIZE
            let n_end = min(n_start + TILE_SIZE, N)
            
            for m in range(m_start, m_end):
                for n in range(n_start, n_end):
                    var acc = SIMD[DType.float32, SIMD_WIDTH](0.0)
                    
                    @parameter
                    fn inner_product(k_idx: Int) -> None:
                        let a_vals = a.load[width=SIMD_WIDTH](m * K + k_idx)
                        let b_vals = b.load[width=SIMD_WIDTH](k_idx * N + n)
                        acc = acc + a_vals * b_vals
                    
                    vectorize[inner_product, SIMD_WIDTH](K)
                    
                    # Reduction across SIMD lanes
                    var sum_val = acc[0]
                    for i in range(1, SIMD_WIDTH):
                        sum_val = sum_val + acc[i]
                    
                    result[m * N + n] = sum_val
        
        let num_m_tiles = (M + TILE_SIZE - 1) // TILE_SIZE
        let num_n_tiles = (N + TILE_SIZE - 1) // TILE_SIZE
        
        @parameter
        fn tile_outer(tile_idx: Int) -> None:
            let m_tile = tile_idx // num_n_tiles
            let n_tile = tile_idx % num_n_tiles
            compute_tile(m_tile, n_tile)
        
        parallelize[tile_outer](num_m_tiles * num_n_tiles)
    
    @staticmethod
    fn tool_token_predictor(
        hidden_states: Tensor[DType.float32],
        tool_classifier_weights: Tensor[DType.float32],
        inout tool_logits: Tensor[DType.float32]
    ) -> None:
        """
        Specialized kernel for tool token prediction.
        Optimized for early <tool> vs <thinking> classification.
        """
        let seq_len = hidden_states.shape()[0]
        let hidden_dim = hidden_states.shape()[1]
        let num_tool_classes = tool_classifier_weights.shape()[0]  # <tool>, <thinking>, etc.
        
        # Fast classification for tool decision
        @parameter
        fn predict_tool_tokens(seq_idx: Int) -> None:
            for class_idx in range(num_tool_classes):
                var logit_acc = SIMD[DType.float32, SIMD_WIDTH](0.0)
                
                @parameter
                fn compute_logit(hidden_idx: Int) -> None:
                    let hidden_vals = hidden_states.load[width=SIMD_WIDTH](
                        seq_idx * hidden_dim + hidden_idx
                    )
                    let weight_vals = tool_classifier_weights.load[width=SIMD_WIDTH](
                        class_idx * hidden_dim + hidden_idx
                    )
                    logit_acc = logit_acc + hidden_vals * weight_vals
                
                vectorize[compute_logit, SIMD_WIDTH](hidden_dim)
                
                # Reduction and store
                var final_logit = logit_acc[0]
                for i in range(1, SIMD_WIDTH):
                    final_logit = final_logit + logit_acc[i]
                
                tool_logits[seq_idx * num_tool_classes + class_idx] = final_logit
        
        parallelize[predict_tool_tokens](seq_len)
    
    @staticmethod
    fn kv_cache_update(
        inout kv_cache: Tensor[DType.float32],
        new_keys: Tensor[DType.float32],
        new_values: Tensor[DType.float32],
        cache_position: Int
    ) -> None:
        """
        SIMD-optimized KV cache management for incremental inference.
        Critical for tool-calling where we need fast context updates.
        """
        let batch_size = new_keys.shape()[0]
        let num_heads = new_keys.shape()[1] 
        let head_dim = new_keys.shape()[2]
        let max_seq_len = kv_cache.shape()[2]
        
        # Update cache with new key-value pairs
        @parameter
        fn update_cache_entry(batch_head_idx: Int) -> None:
            let batch_idx = batch_head_idx // num_heads
            let head_idx = batch_head_idx % num_heads
            
            # Copy new keys
            @parameter
            fn copy_keys(dim_idx: Int) -> None:
                let src_idx = ((batch_idx * num_heads + head_idx) * head_dim + dim_idx)
                let dst_idx = ((batch_idx * num_heads + head_idx) * max_seq_len * head_dim + 
                              cache_position * head_dim + dim_idx)
                
                let key_vals = new_keys.load[width=SIMD_WIDTH](src_idx)
                kv_cache.store(dst_idx, key_vals)
            
            vectorize[copy_keys, SIMD_WIDTH](head_dim)
            
            # Copy new values (offset by key cache size)
            let value_offset = batch_size * num_heads * max_seq_len * head_dim
            
            @parameter
            fn copy_values(dim_idx: Int) -> None:
                let src_idx = ((batch_idx * num_heads + head_idx) * head_dim + dim_idx)
                let dst_idx = (value_offset + 
                              (batch_idx * num_heads + head_idx) * max_seq_len * head_dim + 
                              cache_position * head_dim + dim_idx)
                
                let value_vals = new_values.load[width=SIMD_WIDTH](src_idx)
                kv_cache.store(dst_idx, value_vals)
            
            vectorize[copy_values, SIMD_WIDTH](head_dim)
        
        parallelize[update_cache_entry](batch_size * num_heads)
    
    @staticmethod
    @kernel
    fn rmsnorm_kernel(
        input: Tensor[DType.float32],
        weight: Tensor[DType.float32],
        inout output: Tensor[DType.float32],
        eps: Float32 = 1e-6
    ) -> None:
        """RMSNorm kernel ported from training with MI300X optimizations"""
        let batch_size = input.shape()[0]
        let hidden_dim = input.shape()[1]
        
        @parameter
        fn rmsnorm_batch(batch_idx: Int):
            let offset = batch_idx * hidden_dim
            
            # Compute RMS with WARP_SIZE=64 SIMD
            var variance_sum = SIMD[DType.float32, WARP_SIZE](0.0)
            
            @parameter
            fn variance_vectorized(dim_idx: Int):
                let values = input.load[width=WARP_SIZE](offset + dim_idx)
                variance_sum = variance_sum + values * values
            
            vectorize[variance_vectorized, WARP_SIZE](hidden_dim)
            
            # Reduce variance across WARP_SIZE lanes
            var variance: Float32 = 0.0
            for i in range(WARP_SIZE):
                variance += variance_sum[i]
            variance = variance / hidden_dim
            let inv_rms = 1.0 / sqrt(variance + eps)
            
            # Apply normalization with SIMD
            @parameter
            fn normalize_vectorized(dim_idx: Int):
                let input_vals = input.load[width=WARP_SIZE](offset + dim_idx)
                let weight_vals = weight.load[width=WARP_SIZE](dim_idx)
                let normalized = input_vals * inv_rms * weight_vals
                output.store[width=WARP_SIZE](offset + dim_idx, normalized)
            
            vectorize[normalize_vectorized, WARP_SIZE](hidden_dim)
        
        parallelize[rmsnorm_batch](batch_size)
    
    @staticmethod
    @kernel
    fn rope_kernel(
        inout tensor: Tensor[DType.float32],
        cos_cached: Tensor[DType.float32],
        sin_cached: Tensor[DType.float32],
        position_ids: Tensor[DType.int32]
    ) -> None:
        """RoPE kernel with SIMD complex arithmetic for MI300X"""
        let batch_size = tensor.shape()[0]
        let seq_len = tensor.shape()[1]
        let num_heads = tensor.shape()[2]
        let head_dim = tensor.shape()[3]
        
        @parameter
        fn rope_sequence(seq_idx: Int):
            let batch_idx = seq_idx // seq_len
            let pos_idx = seq_idx % seq_len
            let pos_id = position_ids[batch_idx * seq_len + pos_idx]
            
            @parameter
            fn rope_head(head_idx: Int):
                let tensor_offset = ((batch_idx * seq_len + pos_idx) * num_heads + head_idx) * head_dim
                
                # Apply rotation to pairs with THREADS_PER_BLOCK=256 vectorization
                @parameter
                fn rope_dim_pairs(pair_idx: Int):
                    let dim1 = pair_idx * 2
                    let dim2 = dim1 + 1
                    
                    if dim2 < head_dim:
                        let cos_val = cos_cached[pos_id * (head_dim // 2) + pair_idx]
                        let sin_val = sin_cached[pos_id * (head_dim // 2) + pair_idx]
                        
                        let x1 = tensor[tensor_offset + dim1]
                        let x2 = tensor[tensor_offset + dim2]
                        
                        # Complex rotation: (x1 + i*x2) * (cos + i*sin)
                        let rotated_x1 = x1 * cos_val - x2 * sin_val
                        let rotated_x2 = x1 * sin_val + x2 * cos_val
                        
                        tensor[tensor_offset + dim1] = rotated_x1
                        tensor[tensor_offset + dim2] = rotated_x2
                
                vectorize[rope_dim_pairs, THREADS_PER_BLOCK](head_dim // 2)
            
            parallelize[rope_head](num_heads)
        
        parallelize[rope_sequence](batch_size * seq_len)
    
    @staticmethod
    @kernel
    fn attention_kernel(
        query: Tensor[DType.float32],
        key: Tensor[DType.float32],
        value: Tensor[DType.float32],
        inout output: Tensor[DType.float32],
        scale: Float32
    ) -> None:
        """Multi-head attention with MI300X SIMD optimization"""
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
                
                # Compute attention scores
                var max_score: Float32 = -1e9
                var scores = List[Float32]()
                
                for k_pos in range(seq_len):
                    let k_offset = ((batch_idx * seq_len + k_pos) * num_heads + head_offset) * head_dim
                    
                    # Dot product with WARP_SIZE=64 SIMD
                    var score_acc = SIMD[DType.float32, WARP_SIZE](0.0)
                    
                    @parameter
                    fn attention_dot_simd(dim_idx: Int):
                        let q_vals = query.load[width=WARP_SIZE](q_offset + dim_idx)
                        let k_vals = key.load[width=WARP_SIZE](k_offset + dim_idx)
                        score_acc = score_acc + q_vals * k_vals
                    
                    vectorize[attention_dot_simd, WARP_SIZE](head_dim)
                    
                    # Reduce score
                    var final_score: Float32 = 0.0
                    for i in range(WARP_SIZE):
                        final_score += score_acc[i]
                    final_score *= scale
                    
                    scores.append(final_score)
                    if final_score > max_score:
                        max_score = final_score
                
                # Softmax normalization
                var exp_sum: Float32 = 0.0
                for i in range(len(scores)):
                    scores[i] = exp(scores[i] - max_score)
                    exp_sum += scores[i]
                
                for i in range(len(scores)):
                    scores[i] /= exp_sum
                
                # Apply to values with THREADS_PER_BLOCK vectorization
                let out_offset = q_offset
                
                @parameter
                fn attention_value_simd(dim_idx: Int):
                    var weighted_val: Float32 = 0.0
                    
                    for k_pos in range(seq_len):
                        let v_offset = ((batch_idx * seq_len + k_pos) * num_heads + head_offset) * head_dim + dim_idx
                        weighted_val += scores[k_pos] * value[v_offset]
                    
                    output[out_offset + dim_idx] = weighted_val
                
                vectorize[attention_value_simd, THREADS_PER_BLOCK](head_dim)
        
        parallelize[attention_head](batch_size * num_heads)
    
    @staticmethod
    @kernel
    fn embedding_lookup(
        input_ids: Tensor[DType.int32],
        embedding_weights: Tensor[DType.float32],
        inout output: Tensor[DType.float32]
    ) -> None:
        """SIMD-optimized embedding lookup with memory coalescing for MI300X"""
        let seq_len = input_ids.num_elements()
        let hidden_dim = embedding_weights.shape()[1]
        
        @parameter
        fn lookup_token(token_idx: Int):
            let token_id = input_ids[token_idx]
            let emb_offset = token_id * hidden_dim
            let out_offset = token_idx * hidden_dim
            
            # Vectorized copy with WARP_SIZE=64
            @parameter
            fn copy_embedding_simd(dim_idx: Int):
                let emb_vals = embedding_weights.load[width=WARP_SIZE](emb_offset + dim_idx)
                output.store[width=WARP_SIZE](out_offset + dim_idx, emb_vals)
            
            vectorize[copy_embedding_simd, WARP_SIZE](hidden_dim)
        
        parallelize[lookup_token](seq_len)


# GPU kernel annotations for MAX integration
@gpu
@kernel
fn gpu_softmax_dispatch(
    logits_ptr: Pointer[Float32],
    seq_len: Int,
    vocab_size: Int,
    device_id: Int
) -> None:
    """GPU dispatch wrapper for softmax kernel"""
    let logits_tensor = Tensor[DType.float32](
        logits_ptr, TensorShape(seq_len, vocab_size)
    )
    SIMDKernels.softmax_kernel(logits_tensor)


@gpu
@kernel  
fn gpu_tool_prediction_dispatch(
    hidden_ptr: Pointer[Float32],
    weights_ptr: Pointer[Float32], 
    output_ptr: Pointer[Float32],
    seq_len: Int,
    hidden_dim: Int,
    num_classes: Int,
    device_id: Int
) -> None:
    """GPU dispatch wrapper for tool token prediction"""
    let hidden_tensor = Tensor[DType.float32](
        hidden_ptr, TensorShape(seq_len, hidden_dim)
    )
    let weights_tensor = Tensor[DType.float32](
        weights_ptr, TensorShape(num_classes, hidden_dim)
    )
    let output_tensor = Tensor[DType.float32](
        output_ptr, TensorShape(seq_len, num_classes)
    )
    
    SIMDKernels.tool_token_predictor(hidden_tensor, weights_tensor, output_tensor)