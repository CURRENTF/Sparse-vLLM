/*
################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
*/


#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda_bf16.h>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#define SORT_OFFSET 1

#ifndef BLOCK_SIZE_CP
#define BLOCK_SIZE_CP 128
#endif

#include "copy.cuh"
#include "functions.h"
#include "map.cuh"

#if BLOCK_SIZE_CP == 128
#define PTYPE int4
#endif

#if BLOCK_SIZE_CP == 256
#define PTYPE int2
#endif

void gather_copy(
    torch::Tensor values, torch::Tensor v_cache_buffer, torch::Tensor position_ids,
    int batch_size, int heads, int cpu_v_length, int gpu_v_length, int map_size = 256)
{
    int blockSize = BLOCK_SIZE_CP;
    int numBlocks = batch_size * heads;
    int maxSMBytes = CPY_SIZE*2*1024 + map_size*4; // must less than 160 KB

    // Cast bf16 data pointers to int2 or int4
    PTYPE* values_ptr = reinterpret_cast<PTYPE*>(values.data_ptr<at::BFloat16>());
    PTYPE* v_cache_buffer_ptr = reinterpret_cast<PTYPE*>(v_cache_buffer.data_ptr<at::BFloat16>());

    // this only needs to run once
    if(map_size == 256) {
        // this only needs to run once
        cudaFuncSetAttribute(gahter_copy_fixed_start_end<PTYPE, int64_t, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gahter_copy_fixed_start_end<PTYPE, int64_t, 256><<<numBlocks, blockSize, maxSMBytes>>>(
        values_ptr,
        v_cache_buffer_ptr,
        cpu_v_length, 
        gpu_v_length, 
        position_ids.data_ptr<int64_t>(),
        0/*assume no hit*/, 
        map_size);
    }  else if (map_size == 128) {
        // this only needs to run once
        cudaFuncSetAttribute(gahter_copy_fixed_start_end<PTYPE, int64_t, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gahter_copy_fixed_start_end<PTYPE, int64_t, 128><<<numBlocks, blockSize, maxSMBytes>>>(
        values_ptr,
        v_cache_buffer_ptr,
        cpu_v_length, 
        gpu_v_length, 
        position_ids.data_ptr<int64_t>(),
        0/*assume no hit*/, 
        map_size);
    } else if (map_size == 512) {
        // this only needs to run once
        cudaFuncSetAttribute(gahter_copy_fixed_start_end<PTYPE, int64_t, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gahter_copy_fixed_start_end<PTYPE, int64_t, 512><<<numBlocks, blockSize, maxSMBytes>>>(
        values_ptr,
        v_cache_buffer_ptr,
        cpu_v_length, 
        gpu_v_length, 
        position_ids.data_ptr<int64_t>(),
        0/*assume no hit*/, 
        map_size);
    } else if (map_size == 1024) {
        // this only needs to run once
        cudaFuncSetAttribute(gahter_copy_fixed_start_end<PTYPE, int64_t, 1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gahter_copy_fixed_start_end<PTYPE, int64_t, 1024><<<numBlocks, blockSize, maxSMBytes>>>(
        values_ptr,
        v_cache_buffer_ptr,
        cpu_v_length, 
        gpu_v_length, 
        position_ids.data_ptr<int64_t>(),
        0/*assume no hit*/, 
        map_size);
    }
}

/// for keys
void gather_copy_d2d_with_offsets(
    torch::Tensor keys, 
    torch::Tensor offsets,          // input, offsets computed from reorder_keys_and_compute_offsets, size as elements (numBlocks*256)
    torch::Tensor cnts,             // input, counts computed from reorder_keys_and_compute_offsets, size as numBlocks
    int batch_size, int heads, 
    int gpu_v_length, 
    int gpu_v_offset, 
    int gpu_v_stride, 
    int map_size = 256)
{
    int blockSize = BLOCK_SIZE_CP;
    int numBlocks = batch_size * heads;
    int maxSMBytes = CPY_SIZE*2*1024 + map_size*4 + sizeof(PTYPE); // must less than 160 KB

    // Cast bf16 data pointers to int2 or int4
    PTYPE* keys_ptr = reinterpret_cast<PTYPE*>(keys.data_ptr<at::BFloat16>());
    int* offsets_ptr = reinterpret_cast<int*>(offsets.data_ptr<int32_t>());
    int* cnts_ptr = reinterpret_cast<int*>(cnts.data_ptr<int32_t>());

    // this only needs to run once
    if(map_size == 256) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_d2d<PTYPE, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_d2d<PTYPE, 256><<<numBlocks, blockSize, maxSMBytes>>>(
        keys_ptr, 
        nullptr,
        gpu_v_length, 
        gpu_v_offset,
        gpu_v_stride,
        offsets_ptr,
        0 /*start*/, 
        cnts_ptr);
    } else if (map_size == 128) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_d2d<PTYPE, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_d2d<PTYPE, 128><<<numBlocks, blockSize, maxSMBytes>>>(
        keys_ptr, 
        nullptr,
        gpu_v_length, 
        gpu_v_offset,
        gpu_v_stride,
        offsets_ptr,
        0 /*start*/, 
        cnts_ptr);    
    } else if (map_size == 512) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_d2d<PTYPE, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_d2d<PTYPE, 512><<<numBlocks, blockSize, maxSMBytes>>>(
        keys_ptr, 
        nullptr,
        gpu_v_length, 
        gpu_v_offset,
        gpu_v_stride,
        offsets_ptr,
        0 /*start*/, 
        cnts_ptr);
    } else if (map_size == 1024) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_d2d<PTYPE, 1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_d2d<PTYPE, 1024><<<numBlocks, blockSize, maxSMBytes>>>(
        keys_ptr, 
        nullptr,
        gpu_v_length, 
        gpu_v_offset,
        gpu_v_stride,
        offsets_ptr,
        0, /*start*/
        cnts_ptr);
    }
}

// reorder position ids, and compute offsets and cnts by computing cache hits/misses
// call it before gather_copy_with_offsets
void reorder_keys_and_compute_offsets(
    torch::Tensor cached_pos_ids, // inout, as cached previous position id as input, also reordered position ids, int64_t type
    torch::Tensor cur_pos_ids,    // input, incoming position id, int64_t type
    torch::Tensor offsets,        // output, offsets for gather_copy_with_offsets, size as numBlocks, int type
    torch::Tensor cnts,           // output, counts to separate d2d and h2d, size as numBlocks, int type
    int batch_size, int heads, 
    int map_size = 256)
{
    int blockSize = map_size;
    int numBlocks = batch_size * heads;

    int64_t* cached_pos = cached_pos_ids.data_ptr<int64_t>();
    int64_t* cur_pos = cur_pos_ids.data_ptr<int64_t>();
    int* offsets_ptr = reinterpret_cast<int*>(offsets.data_ptr<int32_t>());
    int* cnts_ptr = reinterpret_cast<int*>(cnts.data_ptr<int32_t>());


    if(map_size == 256) {
        reorder_keys_and_mixed_offsets<int64_t, 256, 1024><<<numBlocks, blockSize>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    } else if (map_size == 128) {
        reorder_keys_and_mixed_offsets<int64_t, 128, 1024><<<numBlocks, blockSize>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    } else if (map_size == 512) {
        reorder_keys_and_mixed_offsets<int64_t, 512, 2048><<<numBlocks, blockSize>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    } else if (map_size == 1024) {
        reorder_keys_and_mixed_offsets<int64_t, 1024, 4096><<<numBlocks, blockSize>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    }
}

// gather copy with offsets
// call it after reorder_keys_and_compute_offsets
void gather_copy_with_offsets(
    torch::Tensor values,           // input, cpu values
    torch::Tensor v_cache_buffer,   // inout, gpu values
    torch::Tensor temp,             // a temp gpu memory for copy, size same as single layer v_cache_buffer 
    torch::Tensor offsets,          // input, offsets computed from reorder_keys_and_compute_offsets, size as elements (numBlocks*256)
    torch::Tensor cnts,             // input, counts computed from reorder_keys_and_compute_offsets, size as numBlocks
    torch::Tensor signals,          // extra internal signals, all zeros sizes as numBlocks, size as numBlocks
    int batch_size, int heads, int cpu_v_length, int gpu_v_length, int gpu_v_offset, int gpu_v_stride, int map_size = 256) // input, torch stream
{
    int blockSize = BLOCK_SIZE_CP;
    int numBlocks = batch_size * heads * 2; // numBlocksBP = 2 * BLOCK_NUM
    int maxSMBytes = CPY_SIZE * 2 * 1024 + map_size * 4 + sizeof(PTYPE); // must less than 160 KB

    // Cast bf16 data pointers to int2 or int4
    PTYPE* values_ptr = reinterpret_cast<PTYPE*>(values.data_ptr<at::BFloat16>());
    PTYPE* v_cache_buffer_ptr = reinterpret_cast<PTYPE*>(v_cache_buffer.data_ptr<at::BFloat16>());
    PTYPE* temp_ptr = reinterpret_cast<PTYPE*>(temp.data_ptr<at::BFloat16>());
    int* offsets_ptr = reinterpret_cast<int*>(offsets.data_ptr<int32_t>());
    int* cnts_ptr = reinterpret_cast<int*>(cnts.data_ptr<int32_t>());
    unsigned int* signals_ptr = reinterpret_cast<unsigned int*>(signals.data_ptr<int32_t>());
    
    // Get cudaStream_t from torch::cuda::Stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if(map_size == 256) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 256><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            values_ptr, 
            temp_ptr, 
            v_cache_buffer_ptr, 
            cpu_v_length, 
            gpu_v_length,
            gpu_v_offset,
            gpu_v_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr);
    } else if(map_size == 128) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 128><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            values_ptr, 
            temp_ptr, 
            v_cache_buffer_ptr, 
            cpu_v_length, 
            gpu_v_length,
            gpu_v_offset,
            gpu_v_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr);
    }else if(map_size == 512) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 512><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            values_ptr, 
            temp_ptr, 
            v_cache_buffer_ptr, 
            cpu_v_length, 
            gpu_v_length,
            gpu_v_offset,
            gpu_v_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr);
    } else if(map_size == 1024) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 1024><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            values_ptr, 
            temp_ptr, 
            v_cache_buffer_ptr, 
            cpu_v_length, 
            gpu_v_length,
            gpu_v_offset,
            gpu_v_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_copy", &gather_copy, "Gather bf16 rows from pinned CPU to GPU buffer.");
    m.def(
        "reorder_keys_and_compute_offsets",
        &reorder_keys_and_compute_offsets,
        "Reorder query position ids and compute offsets/cnts for gather_copy_with_offsets."
    );
    m.def(
        "gather_copy_with_offsets",
        &gather_copy_with_offsets,
        "Gather bf16 rows from pinned CPU to GPU buffer using precomputed offsets/cnts."
    );
    m.def(
        "gather_copy_d2d_with_offsets",
        &gather_copy_d2d_with_offsets,
        "Device-to-device copy using precomputed offsets/cnts (for cached hits)."
    );
}
