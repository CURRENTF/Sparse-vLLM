import torch
import cpu_gpu_transfer_cuda

class EfficientTransfer:
    def __init__(self, batch_size, num_heads, map_size=256):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.map_size = map_size
        self.block_num = batch_size * num_heads
        
        # Internal buffers for the kernel
        self.offsets = torch.zeros(self.block_num * map_size, device='cuda', dtype=torch.int32)
        self.cnts = torch.zeros(self.block_num, device='cuda', dtype=torch.int32)
        self.signals = torch.zeros(self.block_num, device='cuda', dtype=torch.int32)
        
        # This will hold the "state" of what's currently in GPU buffer to optimize D2D vs H2D
        self.cached_pos_ids = torch.zeros(batch_size, num_heads, map_size, device='cuda', dtype=torch.int64).fill_(-1)

    def gather_from_cpu(self, cpu_tensor, gpu_buffer, temp_buffer, query_pos_ids, 
                        cpu_v_length, gpu_v_length, gpu_v_offset, gpu_v_stride):
        """
        High efficiency gather from CPU (pinned memory) to GPU buffer.
        
        Args:
            cpu_tensor: Pinned memory tensor on CPU [batch, heads, seq_len, dim_packed]
            gpu_buffer: Destination tensor on GPU
            temp_buffer: Temporary GPU buffer for intermediate copies
            query_pos_ids: The indices to gather [batch, heads, map_size]
            cpu_v_length: Length of CPU sequence (in chunks/elements)
            gpu_v_length: Length of GPU sequence buffer
            gpu_v_offset: Offset in GPU buffer
            gpu_v_stride: Stride in GPU buffer
        """
        if not cpu_tensor.is_pinned():
            import warnings
            warnings.warn("cpu_tensor is not pinned. Performance will be significantly degraded.")
            
        # 1. Reorder keys and compute which ones are in cache (D2D) vs need to be fetched (H2D)
        cpu_gpu_transfer_cuda.reorder_keys_and_compute_offsets(
            self.cached_pos_ids,
            query_pos_ids.to(torch.int64),
            self.offsets,
            self.cnts,
            self.batch_size,
            self.num_heads,
            self.map_size
        )
        
        # 2. Perform the actual gather-copy
        cpu_gpu_transfer_cuda.gather_copy_with_offsets(
            cpu_tensor,
            gpu_buffer,
            temp_buffer,
            self.offsets,
            self.cnts,
            self.signals,
            self.batch_size,
            self.num_heads,
            int(cpu_v_length),
            int(gpu_v_length),
            int(gpu_v_offset),
            int(gpu_v_stride),
            self.map_size
        )
        
        return gpu_buffer

def simple_gather(cpu_tensor, position_ids, gpu_v_length):
    """
    Simplified API for one-off gather.
    Note: For better performance, use EfficientTransfer class to reuse buffers.
    """
    batch_size, num_heads = position_ids.shape[:2]
    map_size = position_ids.shape[2]
    
    # Calculate dim from the last dimension of cpu_tensor
    # Note: the kernel expects certain data layout, typically [B, H, L, D]
    # cpu_tensor shape: [batch, heads, length, dim]
    dim = cpu_tensor.shape[-1]
    
    gpu_buffer = torch.empty((batch_size, num_heads, map_size, dim), device='cuda', dtype=cpu_tensor.dtype)
    
    # For a simple one-off, we use the basic gather_copy kernel
    cpu_gpu_transfer_cuda.gather_copy(
        cpu_tensor,
        gpu_buffer,
        position_ids.to(torch.int64),
        batch_size,
        num_heads,
        cpu_tensor.shape[2] * dim, # cpu_v_length
        map_size * dim,            # gpu_v_length
        map_size
    )
    
    return gpu_buffer
