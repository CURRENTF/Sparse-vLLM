#include <torch/extension.h>

void gather_copy(
    torch::Tensor values, torch::Tensor v_cache_buffer, torch::Tensor position_ids,
    int batch_size, int heads, int cpu_v_length, int gpu_v_length, int map_size);

void gather_copy_d2d_with_offsets(
    torch::Tensor keys,             // gpu keys
    torch::Tensor offsets,          // input, offsets computed from reorder_keys_and_compute_offsets, size as elements (numBlocks*256)
    torch::Tensor cnts,             // input, counts computed from reorder_keys_and_compute_offsets, size as numBlocks
    int batch_size, int heads, 
    int gpu_k_length, 
    int gpu_k_offset, 
    int gpu_k_stride, 
    int map_size);

void reorder_keys_and_compute_offsets(
    torch::Tensor cached_pos_ids, // inout, as cached previous position id as input, also reordered position ids, int64_t type
    torch::Tensor cur_pos_ids,    // input, incoming position id, int64_t type
    torch::Tensor offsets,        // output, offsets for gather_copy_with_offsets, size as numBlocks
    torch::Tensor cnts,           // output, counts to separate d2d and h2d, size as numBlocks
    int batch_size, int heads, int map_size);

void gather_copy_with_offsets(
    torch::Tensor values,           // input, cpu values
    torch::Tensor v_cache_buffer,   // inout, gpu values
    torch::Tensor temp,             // a temp gpu memory for copy, size same as single layer v_cache_buffer 
    torch::Tensor offsets,          // input, offsets computed from reorder_keys_and_compute_offsets, size as numBlocks, 
    torch::Tensor cnts,             // input, counts computed from reorder_keys_and_compute_offsets, size as numBlocks
    torch::Tensor signals,          // extra internal signals, all zeros sizes as numBlocks, size as numBlocks
    int batch_size, int heads, int cpu_v_length, int gpu_v_length, int gpu_v_offset, int gpu_v_stride, int map_size);
