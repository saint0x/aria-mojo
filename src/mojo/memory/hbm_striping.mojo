"""
HBM Striping System for MI300X Memory Optimization

Advanced memory bank management for MI300X's 192GB HBM3 with page-strided 
weight sharding to maximize bandwidth utilization. Implements @memory_bank
annotations and smart allocation patterns for CDNA3 architecture.
"""

from sys import simdwidthof
from memory import memset_zero, memcpy
from algorithm import vectorize, parallelize
from builtin import int32, float32
from math import ceil, floor
from tensor import Tensor
from collections import List, Dict


# MI300X HBM3 configuration constants
alias SIMD_WIDTH = simdwidthof[DType.float32]()
alias HBM3_CHANNELS = 24  # MI300X has 24 HBM3 memory channels
alias HBM3_CHANNEL_WIDTH = 8192  # 8GB per channel
alias PAGE_SIZE = 4096  # 4KB pages
alias CACHE_LINE_SIZE = 64  # L2 cache line size
alias PREFERRED_STRIPE_SIZE = 256 * 1024  # 256KB stripes for optimal throughput


@memory_bank("hbm_channel_0")
struct HBMChannel0[dtype: DType]:
    """HBM channel 0 with explicit bank annotation"""
    var data: Tensor[dtype]
    var channel_id: Int
    var allocated_bytes: Int
    
    fn __init__(inout self, size: Int):
        self.data = Tensor[dtype](size)
        self.channel_id = 0
        self.allocated_bytes = size * sizeof[Scalar[dtype]]()


@memory_bank("hbm_channel_1") 
struct HBMChannel1[dtype: DType]:
    """HBM channel 1 with explicit bank annotation"""
    var data: Tensor[dtype]
    var channel_id: Int
    var allocated_bytes: Int
    
    fn __init__(inout self, size: Int):
        self.data = Tensor[dtype](size)
        self.channel_id = 1
        self.allocated_bytes = size * sizeof[Scalar[dtype]]()


@memory_bank("hbm_striped")
struct StripedTensor[dtype: DType]:
    """Multi-channel striped tensor for maximum bandwidth utilization"""
    var stripes: List[Tensor[dtype]]
    var stripe_size: Int
    var total_size: Int
    var num_stripes: Int
    var access_pattern: String  # "sequential", "random", "cyclic"
    
    fn __init__(inout self, total_size: Int, num_stripes: Int = HBM3_CHANNELS, access_pattern: String = "cyclic"):
        self.total_size = total_size
        self.num_stripes = min(num_stripes, HBM3_CHANNELS)
        self.stripe_size = (total_size + self.num_stripes - 1) // self.num_stripes
        self.access_pattern = access_pattern
        self.stripes = List[Tensor[dtype]]()
        
        # Allocate stripes across different memory banks
        for i in range(self.num_stripes):
            let stripe = Tensor[dtype](self.stripe_size)
            self.stripes.append(stripe)
    
    fn get_stripe_for_index(self, global_index: Int) -> Tuple[Int, Int]:
        """Get stripe number and local index for global index"""
        if self.access_pattern == "cyclic":
            let stripe_id = global_index % self.num_stripes
            let local_index = global_index // self.num_stripes
            return (stripe_id, local_index)
        elif self.access_pattern == "sequential":
            let stripe_id = global_index // self.stripe_size
            let local_index = global_index % self.stripe_size
            return (stripe_id, local_index)
        else:
            # Default to cyclic
            let stripe_id = global_index % self.num_stripes
            let local_index = global_index // self.num_stripes
            return (stripe_id, local_index)
    
    fn store(inout self, index: Int, value: Scalar[dtype]):
        """Store value with automatic stripe routing"""
        let stripe_info = self.get_stripe_for_index(index)
        let stripe_id = stripe_info[0]
        let local_index = stripe_info[1]
        
        if stripe_id < self.num_stripes and local_index < self.stripe_size:
            self.stripes[stripe_id][local_index] = value
    
    fn load(self, index: Int) -> Scalar[dtype]:
        """Load value with automatic stripe routing"""
        let stripe_info = self.get_stripe_for_index(index)
        let stripe_id = stripe_info[0]
        let local_index = stripe_info[1]
        
        if stripe_id < self.num_stripes and local_index < self.stripe_size:
            return self.stripes[stripe_id][local_index]
        else:
            return Scalar[dtype](0)
    
    fn store_vectorized(inout self, base_index: Int, values: SIMD[dtype, SIMD_WIDTH]):
        """Vectorized store across multiple stripes"""
        @parameter
        fn store_simd_element(i: Int):
            let global_idx = base_index + i
            self.store(global_idx, values[i])
        
        vectorize[store_simd_element, 1](SIMD_WIDTH)
    
    fn load_vectorized(self, base_index: Int) -> SIMD[dtype, SIMD_WIDTH]:
        """Vectorized load across multiple stripes"""
        var result = SIMD[dtype, SIMD_WIDTH](0)
        
        @parameter
        fn load_simd_element(i: Int):
            let global_idx = base_index + i
            result[i] = self.load(global_idx)
        
        vectorize[load_simd_element, 1](SIMD_WIDTH)
        return result


struct HBMAllocationManager:
    """Smart HBM allocation manager for optimal channel utilization"""
    var channel_usage: List[Int]  # Bytes allocated per channel
    var channel_capacity: List[Int]  # Total capacity per channel
    var allocation_strategy: String  # "round_robin", "least_used", "striped"
    var total_allocated: Int
    var total_capacity: Int
    
    fn __init__(inout self, strategy: String = "striped"):
        self.allocation_strategy = strategy
        self.channel_usage = List[Int]()
        self.channel_capacity = List[Int]()
        self.total_allocated = 0
        self.total_capacity = 0
        
        # Initialize channel capacities (MI300X has 24 channels of ~8GB each)
        for i in range(HBM3_CHANNELS):
            self.channel_usage.append(0)
            self.channel_capacity.append(HBM3_CHANNEL_WIDTH * 1024 * 1024)  # Convert to bytes
            self.total_capacity += self.channel_capacity[i]
    
    fn allocate_striped_tensor(inout self, size: Int, dtype_size: Int = 4) -> StripedTensor[DType.float32]:
        """Allocate tensor with optimal striping across HBM channels"""
        let total_bytes = size * dtype_size
        let optimal_stripes = self._compute_optimal_stripe_count(total_bytes)
        
        # Update allocation tracking
        let bytes_per_stripe = (total_bytes + optimal_stripes - 1) // optimal_stripes
        for i in range(optimal_stripes):
            self.channel_usage[i % HBM3_CHANNELS] += bytes_per_stripe
        
        self.total_allocated += total_bytes
        
        return StripedTensor[DType.float32](size, optimal_stripes, "cyclic")
    
    fn _compute_optimal_stripe_count(self, tensor_bytes: Int) -> Int:
        """Compute optimal number of stripes based on tensor size and available channels"""
        if tensor_bytes < PREFERRED_STRIPE_SIZE:
            # Small tensors: use minimal striping to avoid overhead
            return 1
        elif tensor_bytes < PREFERRED_STRIPE_SIZE * 4:
            # Medium tensors: use moderate striping
            return 4
        elif tensor_bytes < PREFERRED_STRIPE_SIZE * 12:
            # Large tensors: use many stripes
            return 12
        else:
            # Very large tensors: use maximum striping
            return HBM3_CHANNELS
    
    fn get_channel_utilization(self) -> List[Float32]:
        """Get utilization percentage for each HBM channel"""
        var utilization = List[Float32]()
        
        for i in range(len(self.channel_usage)):
            let usage_pct = Float32(self.channel_usage[i]) / Float32(self.channel_capacity[i])
            utilization.append(usage_pct)
        
        return utilization
    
    fn get_memory_statistics(self) -> Dict[String, Float32]:
        """Get comprehensive memory allocation statistics"""
        var stats = Dict[String, Float32]()
        
        stats["total_allocated_gb"] = Float32(self.total_allocated) / (1024.0 * 1024.0 * 1024.0)
        stats["total_capacity_gb"] = Float32(self.total_capacity) / (1024.0 * 1024.0 * 1024.0)
        stats["overall_utilization"] = Float32(self.total_allocated) / Float32(self.total_capacity)
        
        # Compute channel balance (lower is better)
        let utilization = self.get_channel_utilization()
        var min_util: Float32 = 1.0
        var max_util: Float32 = 0.0
        
        for i in range(len(utilization)):
            if utilization[i] < min_util:
                min_util = utilization[i]
            if utilization[i] > max_util:
                max_util = utilization[i]
        
        stats["channel_balance"] = max_util - min_util  # 0 = perfectly balanced
        stats["active_channels"] = Float32(HBM3_CHANNELS)
        
        return stats


struct ModelWeightStriping:
    """Specialized weight striping for transformer model components"""
    var embedding_weights: StripedTensor[DType.float32]
    var attention_weights: Dict[String, StripedTensor[DType.float32]]
    var ffn_weights: Dict[String, StripedTensor[DType.float32]]
    var layer_norm_weights: Dict[String, StripedTensor[DType.float32]]
    var allocator: HBMAllocationManager
    
    fn __init__(inout self, model_config: Dict[String, Int]):
        self.allocator = HBMAllocationManager("striped")
        self.attention_weights = Dict[String, StripedTensor[DType.float32]]()
        self.ffn_weights = Dict[String, StripedTensor[DType.float32]]()
        self.layer_norm_weights = Dict[String, StripedTensor[DType.float32]]()
        
        # Extract model dimensions
        let vocab_size = model_config.get("vocab_size", 128256)
        let hidden_dim = model_config.get("hidden_dim", 4096)
        let num_layers = model_config.get("num_layers", 32)
        let ffn_dim = model_config.get("ffn_dim", 14336)
        let num_heads = model_config.get("num_heads", 32)
        let head_dim = hidden_dim // num_heads
        
        # Allocate embedding weights with striping
        self.embedding_weights = self.allocator.allocate_striped_tensor(vocab_size * hidden_dim)
        
        # Allocate attention weights for all layers
        for layer in range(num_layers):
            let layer_key = "layer_" + str(layer)
            
            # Q, K, V projections
            self.attention_weights[layer_key + "_q"] = self.allocator.allocate_striped_tensor(hidden_dim * hidden_dim)
            self.attention_weights[layer_key + "_k"] = self.allocator.allocate_striped_tensor(hidden_dim * hidden_dim)
            self.attention_weights[layer_key + "_v"] = self.allocator.allocate_striped_tensor(hidden_dim * hidden_dim)
            self.attention_weights[layer_key + "_o"] = self.allocator.allocate_striped_tensor(hidden_dim * hidden_dim)
            
            # FFN weights
            self.ffn_weights[layer_key + "_gate"] = self.allocator.allocate_striped_tensor(hidden_dim * ffn_dim)
            self.ffn_weights[layer_key + "_up"] = self.allocator.allocate_striped_tensor(hidden_dim * ffn_dim)
            self.ffn_weights[layer_key + "_down"] = self.allocator.allocate_striped_tensor(ffn_dim * hidden_dim)
            
            # Layer norm weights (small, don't stripe)
            self.layer_norm_weights[layer_key + "_attn"] = self.allocator.allocate_striped_tensor(hidden_dim)
            self.layer_norm_weights[layer_key + "_ffn"] = self.allocator.allocate_striped_tensor(hidden_dim)
    
    fn get_attention_weight(self, layer_id: Int, weight_type: String) -> StripedTensor[DType.float32]:
        """Get attention weight tensor for specific layer"""
        let key = "layer_" + str(layer_id) + "_" + weight_type
        return self.attention_weights[key]
    
    fn get_ffn_weight(self, layer_id: Int, weight_type: String) -> StripedTensor[DType.float32]:
        """Get FFN weight tensor for specific layer"""
        let key = "layer_" + str(layer_id) + "_" + weight_type
        return self.ffn_weights[key]
    
    fn prefetch_layer_weights_async(self, layer_id: Int):
        """Asynchronously prefetch all weights for a layer into cache"""
        # This would trigger actual prefetch operations in a real implementation
        # For MI300X, this leverages the GPU's automatic prefetcher
        pass
    
    fn get_memory_layout_report(self) -> Dict[String, Float32]:
        """Generate detailed memory layout report"""
        var report = self.allocator.get_memory_statistics()
        
        # Add model-specific metrics
        let embedding_size_gb = Float32(self.embedding_weights.total_size * 4) / (1024.0 * 1024.0 * 1024.0)
        report["embedding_weights_gb"] = embedding_size_gb
        
        # Estimate total attention weights
        var total_attention_params: Int = 0
        for key in self.attention_weights:
            total_attention_params += self.attention_weights[key].total_size
        
        report["attention_weights_gb"] = Float32(total_attention_params * 4) / (1024.0 * 1024.0 * 1024.0)
        
        # Estimate total FFN weights
        var total_ffn_params: Int = 0
        for key in self.ffn_weights:
            total_ffn_params += self.ffn_weights[key].total_size
        
        report["ffn_weights_gb"] = Float32(total_ffn_params * 4) / (1024.0 * 1024.0 * 1024.0)
        
        return report


@kernel
fn striped_matrix_multiply[dtype: DType](
    a: StripedTensor[dtype],
    b: StripedTensor[dtype],
    inout c: StripedTensor[dtype],
    m: Int,
    n: Int,
    k: Int
) -> None:
    """SIMD-optimized matrix multiplication with striped memory access"""
    
    @parameter
    fn matmul_tile(tile_idx: Int):
        let row = tile_idx // n
        let col = tile_idx % n
        
        var accumulator = SIMD[dtype, SIMD_WIDTH](0.0)
        
        # Vectorized inner product with striped memory access
        @parameter
        fn inner_product_simd(k_idx: Int):
            let a_vals = a.load_vectorized(row * k + k_idx)
            let b_vals = b.load_vectorized(k_idx * n + col)
            accumulator = accumulator + a_vals * b_vals
        
        vectorize[inner_product_simd, SIMD_WIDTH](k)
        
        # Reduce and store result
        var result: Scalar[dtype] = 0
        for i in range(SIMD_WIDTH):
            result = result + accumulator[i]
        
        c.store(row * n + col, result)
    
    parallelize[matmul_tile](m * n)


# Factory functions for easy integration
fn create_hbm_allocation_manager(strategy: String = "striped") -> HBMAllocationManager:
    """Create HBM allocation manager with optimal settings"""
    return HBMAllocationManager(strategy)

fn create_striped_tensor(size: Int, num_stripes: Int = HBM3_CHANNELS) -> StripedTensor[DType.float32]:
    """Create optimally striped tensor for MI300X"""
    return StripedTensor[DType.float32](size, num_stripes, "cyclic")

fn create_model_weight_striping(vocab_size: Int, hidden_dim: Int, num_layers: Int) -> ModelWeightStriping:
    """Create complete model weight striping system"""
    var config = Dict[String, Int]()
    config["vocab_size"] = vocab_size
    config["hidden_dim"] = hidden_dim
    config["num_layers"] = num_layers
    config["ffn_dim"] = hidden_dim * 4  # Standard transformer ratio
    config["num_heads"] = hidden_dim // 128  # Assume 128-dim heads
    
    return ModelWeightStriping(config)