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


#ifndef COPY_CUH
#define COPY_CUH

#ifndef CPY_SIZE
#define CPY_SIZE 32 // 64+ for A100, cannt large than somewhere 80, 32 for smaller cards
#endif

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 8
#endif

#define COPY_UNROLL_OUTER #pragma unroll 1

#define COPY_UNROLL_INNER #pragma unroll UNROLL_FACTOR

#ifndef BLOCK_SIZE_CP
#define BLOCK_SIZE_CP 128
#endif

#ifndef BLOCK_SIZE_MAP
#define BLOCK_SIZE_MAP 256
#endif

#ifndef SORT_OFFSET
#define SORT_OFFSET 1
#endif

namespace
{
    // code from cutlass, with some modification

    __forceinline__ __device__ void red_release(unsigned int *ptr)
    {
#if (__CUDA_ARCH__ >= 700)
        asm volatile("fence.acq_rel.gpu;\n");
        // asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(ptr), "r"(val));
        asm volatile("red.relaxed.gpu.global.inc.u32 [%0], 1;\n" : : "l"(ptr));
        //    __threadfence();
        // atomicInc(ptr, 1);
#else
        __threadfence();
        atomicInc(ptr, 1);
#endif // (__CUDA_ARCH__ >= 700)
    }

    __forceinline__ __device__ void arrive_inc(unsigned int *ptr)
    {
        __syncthreads();
        if (threadIdx.x == 0)
        {
            red_release(ptr);
        }
    }

    __forceinline__ __device__ int ld_acquire(unsigned int *ptr)
    {
        int state = 0;
#if (__CUDA_ARCH__ >= 700)
        asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#else
        asm volatile("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#endif // (__CUDA_ARCH__ >= 700)
        return state;
    }

    __forceinline__ __device__ void wait_eq(unsigned int *ptr, unsigned int val = 0)
    {

        if (threadIdx.x == 0)
        {
// Spin-loop
#pragma unroll 1
            while (ld_acquire(ptr) != val)
            {
            }
        }
        __syncthreads();
    }

    __forceinline__ __device__ void wait_eq_reset(unsigned int *ptr, unsigned int val = 0)
    {

        if (threadIdx.x == 0)
        {
// Spin-loop
#pragma unroll 1
            while (atomicCAS(ptr, val, 0) != val)
            {
                //  printf("ptr %d\n", ptr[0]);
            }
        }
        __syncthreads();
    }

}

// A gather-copy based on s_offsets
// assume only 128 or 256 threads in a thread block
// 2KB = 128 threads x sizeof(int4) or 256 threads x sizeof(int2)
// each thread loads 16 or 8 bytes
template <typename T>
__device__ void gather_copy(
    T *src,
    T *dst,
    T *s_data,
    int *s_offsets,      // gather offsets
    int src_buffer_offset,
    int src_buffer_size, // per block
    int dst_buffer_offset,
    int dst_buffer_size, // per block
    int start,
    int end,
    int dst_start,
    int bid)
{

    int64_t bid_64 = bid; // ned to cast to int64_t to make sure  bid_64 * src_buffer_size not overflow.
    // assume using src_buffer_size * sizeof(bf16) Byte / sizeof(T) Byte
    int64_t src_base = (bid_64 * src_buffer_size + src_buffer_offset) * 2 / sizeof(T); // int64_t to avoid overflow
    int64_t dst_base = (bid_64 * dst_buffer_size + dst_buffer_offset) * 2 / sizeof(T); // int64_t to avoid overflow
    int64_t dst_offset = dst_base + dst_start * BLOCK_SIZE_CP + threadIdx.x;           // int64_t to avoid overflow

    int64_t offset_index = start;
    int64_t offset_end = end;

    COPY_UNROLL_OUTER
    while (offset_index < offset_end)
    {

        // read
        int64_t iter = 0;
        int64_t l_offset = threadIdx.x;

        COPY_UNROLL_INNER
        while ((iter < CPY_SIZE) && (offset_index < offset_end))
        {
            iter++;
            int64_t offset = s_offsets[offset_index]; // int64_t to avoid overflow
            offset_index++;
            int64_t src_offset = src_base + offset * BLOCK_SIZE_CP + threadIdx.x; // int64_t to avoid overflow
            s_data[l_offset] = src[src_offset];
            l_offset += BLOCK_SIZE_CP;
        }

        // write
        int64_t max_iter = iter;
        iter = 0;
        l_offset = threadIdx.x;

        COPY_UNROLL_INNER
        while (iter < max_iter)
        {
            iter++;
            dst[dst_offset] = s_data[l_offset];
            l_offset += BLOCK_SIZE_CP;
            dst_offset += BLOCK_SIZE_CP;
        }
        __syncthreads(); // must, avoid data racing
    }
}

// A regular-copy
// assume only 128 or 256 threads in a thread block
// 2KB = 128 threads x sizeof(int4) or 256 threads x sizeof(int2)
// each thread loads 16 or 8 bytes
template <typename T>
__device__ void copy(
    T *src,
    T *dst,
    T *s_data,
    int src_buffer_offset,
    int src_buffer_size, // per block
    int dst_buffer_offset,
    int dst_buffer_size, // per block
    int start,
    int end,
    int dst_start,
    int bid = blockIdx.x)
{
    int64_t bid_64 = bid;
    // assume using src_buffer_size * sizeof(bf16) Byte / sizeof(T) Byte
    int64_t src_base = (bid_64 * src_buffer_size + src_buffer_offset) * 2 / sizeof(T); // int64_t to avoid overflow
    int64_t dst_base = (bid_64 * dst_buffer_size + dst_buffer_offset) * 2 / sizeof(T); // int64_t to avoid overflow
    int64_t src_offset = src_base + start * BLOCK_SIZE_CP + threadIdx.x;               // int64_t to avoid overflow
    int64_t dst_offset = dst_base + dst_start * BLOCK_SIZE_CP + threadIdx.x;           // int64_t to avoid overflow

    int64_t offset_index = start;
    int64_t offset_end = end;

    COPY_UNROLL_OUTER
    while (offset_index < offset_end)
    {
        // read
        int64_t iter = 0;
        int64_t l_offset = threadIdx.x;

        COPY_UNROLL_INNER
        while (iter < CPY_SIZE && offset_index < offset_end)
        {
            iter++;
            offset_index++;
            s_data[l_offset] = src[src_offset];
            src_offset += BLOCK_SIZE_CP;
            l_offset += BLOCK_SIZE_CP;
        }

        // here, not need __syncthreads

        // write
        int64_t max_iter = iter;
        iter = 0;
        l_offset = threadIdx.x;

        COPY_UNROLL_INNER
        while (iter < max_iter)
        {
            iter++;
            dst[dst_offset] = s_data[l_offset];
            l_offset += BLOCK_SIZE_CP;
            dst_offset += BLOCK_SIZE_CP;
        }
        __syncthreads(); // must, avoid data racing
    }
}

// a gahter-copy with fixed start and end
template <typename T, typename TKEYS, int MAP_SIZE = BLOCK_SIZE_MAP>
__global__ void gahter_copy_fixed_start_end(
    T *src, T *dst,
    int src_buffer_size,
    int dst_buffer_size,
    TKEYS *keys,
    int start,
    int end)
{

    extern __shared__ int s[];
    int *s_offsets = s; // MAP_SIZE
    T *s_data = (T *)(s + MAP_SIZE);

    int key_offset = blockIdx.x * MAP_SIZE + threadIdx.x;

    int idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < MAP_SIZE / BLOCK_SIZE_CP; i++)
    {
        s_offsets[idx] = keys[key_offset];
        idx += BLOCK_SIZE_CP;
        key_offset += BLOCK_SIZE_CP;
    }
    __syncthreads();

    gather_copy(src, dst, s_data, s_offsets, 0, src_buffer_size, 0, dst_buffer_size, start, end, 0, blockIdx.x);
}

// a gahter-copy with variable start and a fixed end
template <typename T, int MAP_SIZE = BLOCK_SIZE_MAP>
__global__ void gather_copy_var_start_fixed_end(
    T *src, T *dst,
    int src_buffer_size,
    int dst_buffer_size,
    int *offsets,
    int *start_cnts,
    int end)
{

    extern __shared__ int s[];
    int *s_offsets = s;                  // BLOCK_SIZE_MAP
    int *start_cnt = s + MAP_SIZE; // 1, but occupy 2 to avoid alignment issue
    T *s_data = (T *)(start_cnt + sizeof(T) / 4);

    int key_offset = blockIdx.x * MAP_SIZE + threadIdx.x;

    int idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < MAP_SIZE / BLOCK_SIZE_CP; i++)
    {
        s_offsets[idx] = offsets[key_offset];
        idx += BLOCK_SIZE_CP;
        key_offset += BLOCK_SIZE_CP;
    }

    if (threadIdx.x == 0)
        start_cnt[0] = start_cnts[blockIdx.x];
    __syncthreads();

    gather_copy(src, dst, s_data, s_offsets, 0, src_buffer_size, 0, dst_buffer_size, start_cnt[0], end, start_cnt[0], blockIdx.x);
}

// a gahter-copy with a fixed start and a variable end
// and using a temp buffer to store partial movement
template <typename T, int MAP_SIZE = BLOCK_SIZE_MAP>
__global__ void gahter_copy_fixed_start_var_end_with_temp(
    T *src, T *temp, T *dst,
    int src_buffer_size,
    int dst_buffer_size,
    int *offsets,
    int start,
    int *end_cnts)
{

    extern __shared__ int s[];
    int *s_offsets = s;                // BLOCK_SIZE_MAP
    int *end_cnt = s + MAP_SIZE; // 1, but occupy 2 to avoid alignment issue
    T *s_data = (T *)(end_cnt + sizeof(T) / 4);

    int key_offset = blockIdx.x * MAP_SIZE + threadIdx.x;

    int idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < MAP_SIZE / BLOCK_SIZE_CP; i++)
    {
        s_offsets[idx] = offsets[key_offset];
        idx += BLOCK_SIZE_CP;
        key_offset += BLOCK_SIZE_CP;
    }

    if (threadIdx.x == 0)
        end_cnt[0] = end_cnts[blockIdx.x];
    __syncthreads();

#if SORT_OFFSET
    gather_copy(src, dst, s_data, s_offsets, 0, src_buffer_size, 0, dst_buffer_size, start, end_cnt[0], start, blockIdx.x);
#else
    gather_copy(src, temp, s_data, s_offsets, 0, src_buffer_size, 0, dst_buffer_size, start, end_cnt[0], start, blockIdx.x);
    copy(temp, dst, s_data, 0, src_buffer_size, 0, dst_buffer_size, start, end_cnt[0], start);
#endif
}

// a gahter-copy with a fixed start and a variable end
// and using a temp buffer to store partial movement
template <typename T, int MAP_SIZE = BLOCK_SIZE_MAP>
__global__ void gather_copy_d2d(
    T *d_values, 
    T *temp, //optinal
    int d_buffer_size,
    int d_buffer_offset,
    int d_buffer_stride,
    int *offsets,
    int start,
    int *end_cnts)
{

    extern __shared__ int s[];
    int *s_offsets = s;                // BLOCK_SIZE_MAP
    int *end_cnt = s + MAP_SIZE; // 1, but occupy 2 to avoid alignment issue
    T *s_data = (T *)(end_cnt + sizeof(T) / 4);

    int key_offset = blockIdx.x * MAP_SIZE + threadIdx.x;

    int idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < MAP_SIZE / BLOCK_SIZE_CP; i++)
    {
        s_offsets[idx] = offsets[key_offset];
        idx += BLOCK_SIZE_CP;
        key_offset += BLOCK_SIZE_CP;
    }

    if (threadIdx.x == 0)
        end_cnt[0] = end_cnts[blockIdx.x];
    __syncthreads();

#if SORT_OFFSET
    gather_copy(d_values, d_values, s_data, s_offsets, d_buffer_offset, d_buffer_stride, d_buffer_offset, d_buffer_stride, start, end_cnt[0], start, blockIdx.x);
#else
    gather_copy(d_values, temp, s_data, s_offsets, d_buffer_offset, d_buffer_stride, 0, d_buffer_size, start, end_cnt[0], start, blockIdx.x);
    copy(temp, d_values, s_data, 0, d_buffer_size, d_buffer_offset, d_buffer_stride, start, end_cnt[0], start);
#endif
}

// a gather copy with variable mid point (input 'cnts')
template <typename T, int MAP_SIZE = BLOCK_SIZE_MAP>
__global__ void gather_copy_var_midpoint(
    T *h_values, T *temp, T *d_values,
    int h_buffer_size,
    int d_buffer_size,
    int d_buffer_offset,
    int d_buffer_stride,
    int *offsets,
    int *cnts)
{

    extern __shared__ int s[];
    int *s_offsets = s;                    // BLOCK_SIZE_MAP
    int *cnt = s_offsets + MAP_SIZE; // 1, but occupy 2 to avoid alignment issue
    T *s_data = (T *)(cnt + sizeof(T) / 4);

    int key_offset = blockIdx.x * MAP_SIZE + threadIdx.x;
    int idx = threadIdx.x;

#pragma unroll
    for (int i = 0; i < MAP_SIZE / BLOCK_SIZE_CP; i++)
    {
        s_offsets[idx] = offsets[key_offset];
        idx += BLOCK_SIZE_CP;
        key_offset += BLOCK_SIZE_CP;
    }

    if (threadIdx.x == 0)
    {
        cnt[0] = cnts[blockIdx.x];
    }
    __syncthreads();
#if SORT_OFFSET
    gather_copy(d_values, d_values, s_data, s_offsets, d_buffer_offset, d_buffer_stride, d_buffer_offset, d_buffer_stride, 0, cnt[0], 0, blockIdx.x);
    gather_copy(h_values, d_values, s_data, s_offsets, 0, h_buffer_size, d_buffer_offset, d_buffer_stride, cnt[0], MAP_SIZE, cnt[0], blockIdx.x);
#else
    gather_copy(d_values, temp, s_data, s_offsets, d_buffer_offset, d_buffer_stride, 0, d_buffer_size, 0, cnt[0], 0, blockIdx.x);
    copy(temp, d_values, s_data, 0, d_buffer_size, d_buffer_offset, d_buffer_stride, 0, cnt[0], 0, blockIdx.x);
    gather_copy(h_values, d_values, s_data, s_offsets, 0, h_buffer_size, d_buffer_offset, d_buffer_stride, cnt[0], MAP_SIZE, cnt[0], blockIdx.x);
#endif
}

// a gather copy with variable mid point (input 'cnts')
// it uses block-specialization
template <typename T, int MAP_SIZE = BLOCK_SIZE_MAP> 
__global__ void gather_copy_var_midpoint_BP(
    T *h_values, T *temp, T *d_values,
    int h_buffer_size,
    int d_buffer_size,
    int d_buffer_offset,
    int d_buffer_stride,
    int *offsets,
    int *cnts,
    unsigned int *signals)
{

    extern __shared__ int s[];
    int *s_offsets = s;                    // BLOCK_SIZE_MAP
    int *cnt = s_offsets + MAP_SIZE; // 1, but occupy 2 to avoid alignment issue
    T *s_data = (T *)(cnt + sizeof(T) / 4);

    int bid = blockIdx.x >> 1;
    int bs_task = blockIdx.x & 1; // 0 or 1

    int key_offset = bid * MAP_SIZE + threadIdx.x;

    int idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < MAP_SIZE / BLOCK_SIZE_CP; i++)
    {
        s_offsets[idx] = offsets[key_offset];
        idx += BLOCK_SIZE_CP;
        key_offset += BLOCK_SIZE_CP;
    }

    if (threadIdx.x == 0)
        cnt[0] = cnts[bid];
    __syncthreads();

    signals += bid;

    if (bs_task)
    {
#if SORT_OFFSET
        // d2d: from d_values to d_values
        gather_copy(d_values, d_values, s_data, s_offsets, d_buffer_offset, d_buffer_stride, d_buffer_offset, d_buffer_stride, 0, cnt[0], 0, bid);
        arrive_inc(signals);
#else 
        // d2d: to temp
        gather_copy(d_values, temp, s_data, s_offsets, d_buffer_offset, d_buffer_stride, 0, d_buffer_size, 0, cnt[0], 0, bid);
        arrive_inc(signals);
        wait_eq(signals);
        // d2d: from temp
        copy(temp, d_values, s_data, 0, d_buffer_size, d_buffer_offset, d_buffer_stride, 0, cnt[0], 0, bid);
#endif
    }
    else
    {
        // h2d: to temp
        gather_copy(h_values, temp, s_data, s_offsets, 0, h_buffer_size, 0, d_buffer_size, cnt[0], MAP_SIZE, cnt[0], bid);
        arrive_inc(signals);
        wait_eq_reset(signals);
        // d2d: from temp
        copy(temp, d_values, s_data, 0, d_buffer_size, d_buffer_offset, d_buffer_stride, cnt[0], MAP_SIZE, cnt[0], bid);
    }
}

#endif // COPY_CUH