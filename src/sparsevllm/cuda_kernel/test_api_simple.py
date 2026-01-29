import torch
import numpy as np
# Assumes the extension is compiled and installed
try:
    from api import simple_gather
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False
    print("Extension not found. Please compile it first using 'pip install -e .'")

def test_simple_gather():
    if not HAS_EXTENSION:
        return

    batch_size = 2
    num_heads = 4
    seq_len = 1000
    dim = 128
    map_size = 16

    # Create random data on CPU and pin it
    cpu_data = torch.randn(batch_size, num_heads, seq_len, dim, dtype=torch.bfloat16).pin_memory()
    
    # Random position ids
    pos_ids = torch.randint(0, seq_len, (batch_size, num_heads, map_size), device='cuda', dtype=torch.int32)
    
    # Native torch gather for ground truth
    # We need to expand indices for torch.gather
    expanded_pos_ids = pos_ids.unsqueeze(-1).expand(-1, -1, -1, dim).long()
    expected = torch.gather(cpu_data.cuda(), 2, expanded_pos_ids)
    
    # Our optimized gather
    # The kernel expects flattened dim or specific layout, let's test based on the implementation
    # Based on ShadowKV, it often flattens the last dimension or uses it as part of the copy size
    actual = simple_gather(cpu_data, pos_ids, map_size)
    
    # Compare
    diff = (actual - expected).abs().max()
    print(f"Max difference: {diff}")
    if diff < 1e-3:
        print("Success!")
    else:
        print("Failure!")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_simple_gather()
    else:
        print("CUDA not available, skipping test.")
