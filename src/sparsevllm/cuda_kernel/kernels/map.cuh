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


#ifndef MAP_CUH
#define MAP_CUH

#include <limits.h>

#define EMPTY_KEY -1

#ifndef TABLE_SIZE
#define TABLE_SIZE 1024 // Chosen as a power of 2 greater than 256 for efficiency
#endif

#ifndef SORT_OFFSET
#define SORT_OFFSET 1
#endif

#ifndef BLOCK_SIZE_MAP
#define BLOCK_SIZE_MAP 256
#endif

// assuming TABLE_SIZE is a power of 2
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__device__ unsigned int fast_hash(int key)
{
    return key & (LUT_SIZE - 1); // Simple mask for hash calculation

    // complicated has key
#if 0
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = (key >> 16) ^ key;
    return key & (LUT_SIZE - 1);
#endif
}

// insert (key, value) into a map
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__device__ void insert_map(int key, int value, int *map_keys, int *map_values)
{
    unsigned int pos = fast_hash<MAP_SIZE, LUT_SIZE>(key);
    while (true)
    {
        int existing_key = atomicCAS(&map_keys[pos], EMPTY_KEY, key);
        if (existing_key == EMPTY_KEY || existing_key == key)
        {
            map_values[pos] = value;
            break;
        }
        pos = (pos + 1) & (LUT_SIZE - 1);
    }
}

// clear all keys
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__device__ void reset_map(int *map_keys)
{
    int id = threadIdx.x;
#pragma unroll
    for (int i = 0; i < LUT_SIZE; i += MAP_SIZE)
    {
        map_keys[id] = EMPTY_KEY;
        id += MAP_SIZE;
    }
}

// init a map based on per-thead key and threadIdx.x as a value
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__device__ void init_map(int key, int *map_keys, int *map_values)
{
    reset_map<MAP_SIZE, LUT_SIZE>(map_keys);
    __syncthreads();
    insert_map<MAP_SIZE, LUT_SIZE>(key, threadIdx.x, map_keys, map_values);
}

// write a map from shared memory to global memory
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__device__ void write_back_map(int *s_keys, int *s_values, int *g_keys, int *g_values)
{
    int id = threadIdx.x;
    int offset = blockIdx.x * LUT_SIZE + threadIdx.x;
#pragma unroll
    for (int i = 0; i < LUT_SIZE; i += MAP_SIZE)
    {
        g_keys[offset] = s_keys[id];
        g_values[offset] = s_values[id];
        id += MAP_SIZE;
        offset += MAP_SIZE;
    }
}

// load a map from global memory to shared memory
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__device__ void load_map(int *s_keys, int *s_values, int *g_keys, int *g_values)
{
    int id = threadIdx.x;
    int offset = blockIdx.x * LUT_SIZE + threadIdx.x;

#pragma unroll
    for (int i = 0; i < LUT_SIZE; i += MAP_SIZE)
    {
        s_keys[id] = g_keys[offset];
        s_values[id] = g_values[offset];
        id += MAP_SIZE;
        offset += MAP_SIZE;
    }
}

// get a value of a key in a map
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__device__ int lookup_map(int key, int *map_keys, int *map_values)
{
    unsigned int pos = fast_hash<MAP_SIZE, LUT_SIZE>(key);

    while (true)
    {
        if (map_keys[pos] == key)
        {
            // value = map_values[pos];
            return map_values[pos];
        }
        if (map_keys[pos] == EMPTY_KEY)
        {
            // value = -1; // Key not found
            return -1;
        }
        pos = (pos + 1) & (LUT_SIZE - 1); // Linear probing
    }
}

// debug only
__device__ void warp_bitonic_sort(int &key, int &value) {
   // Bitonic sort within a warp
    int lane = threadIdx.x & 31;
    for ( int k = 2; k <= 32; k = 2 * k) {
        for ( int j = k >> 1; j > 0; j = j >> 1) {
            bool dir = (( lane & j) != 0) ^ (( lane & k) != 0) ;
            int partnerKey = __shfl_xor_sync(0xFFFFFFF , key , j);
            int partnerValue = __shfl_xor_sync(0xFFFFFFFF, value, j);

            if((key > partnerKey) ^ dir) {
              key = partnerKey;
              value = partnerValue;
            } 
         }
    }
}

// bitonic sort within a warp using shuffle instructions
__device__ void warp_bitonic_sort2(int &key, int &value, int &key2, int &value2) {
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int k = 2; k <= 32; k = 2 * k) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j = j >> 1) {
            bool dir = (( lane & j) != 0) ^ (( lane & k) != 0) ;
            int partnerKey = __shfl_xor_sync(0xFFFFFFF , key , j);
            int partnerValue = __shfl_xor_sync(0xFFFFFFFF, value, j);

            int partnerKey2 = __shfl_xor_sync(0xFFFFFFF , key2 , j);
            int partnerValue2 = __shfl_xor_sync(0xFFFFFFFF, value2, j);


            if((key > partnerKey) ^ dir) {
              key = partnerKey;
              value = partnerValue;
            } 

            if((key2 > partnerKey2) ^ dir) {
              key2 = partnerKey2;
              value2 = partnerValue2;
            } 
         }
    }
}

template<int MAP_SIZE = BLOCK_SIZE_MAP>
__device__ void merge_sort(int *keys, int *values) {
    int localIdx = threadIdx.x;
    #pragma unroll
    for (int size = 2; size <= MAP_SIZE; size *= 2) {
        #pragma unroll
        for (int stride = size / 2; stride > 0; stride /= 2) {
            int ixj = localIdx ^ stride;

            if (ixj > localIdx) {
                // Determine the direction of sorting (ascending or descending)
                bool ascending = ((localIdx & size) == 0);

                // Compare and possibly swap
                int key1 = keys[localIdx];
                int key2 = keys[ixj];

                if ((key1 > key2) == ascending) {
                    // Swap elements
                    keys[localIdx] = key2;
                    keys[ixj] = key1;

                    int temp = values[localIdx] ;
                    values[localIdx]  = values[ixj];
                    values[ixj] = temp;
                }
            }
            __syncthreads(); // Synchronize threads within the block
        }
    }
}

// block-level sort 
template<int MAP_SIZE = BLOCK_SIZE_MAP>
__device__ void block_sort2(int *keys, int *values, int mp) {
    int tid = threadIdx.x;

    int key1 = tid < mp ? keys[tid] : INT_MAX;
    int value1 = values[tid];
    int key2 = tid < mp ? -1 : keys[tid];
    int value2 = values[tid];

    //warp_bitonic_sort2(key1, value1, key2, value2);
    //__syncthreads();

    keys[tid] = key1;
    values[tid] = value1;
    __syncthreads();
    merge_sort<MAP_SIZE>(keys, values);
     // hidden sync

    key1 = keys[tid];
    value1 = values[tid];
    // no need sync 
    keys[tid] = key2;
    values[tid] = value2;

    __syncthreads();
    merge_sort<MAP_SIZE>(keys, values);
    // hidden sync

    keys[tid] = tid < mp ? key1 : keys[tid];
    values[tid] = tid < mp ? value1 : values[tid];
    __syncthreads();
}

// given hits and keys of a thread-block
// reorder keys, and values
// all values with true hits are packed together.
// then value with false hits are assigned by counts (from 0, 1, 2,...) and packed together.
// and return how many true hits
// e.g. hits:      [false, true, false, true],
//      keys:      [23,    40,   52,    99],
//      values:    [345,   455,  544,   24],
//      reorders:  [40,    99,   23,    52],
//      new_values:[455,   24,    0,    1],
//      return: 2
template<int MAP_SIZE = BLOCK_SIZE_MAP>
__device__ int update_keys_and_offsets(bool hit, int key, int value, int *reorder_key, int *new_values, int *warp_sums)
{
    int thid = threadIdx.x;
    int laneId = thid % 32; // Warp lane index
    int warpId = thid / 32; // Warp index within the block

    if (warpId == 0 && thid < 32)
    {
        warp_sums[thid] = 0;
    }
    __syncthreads();

    // Step 1: Compute the ballot mask for the warp
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, hit);
    unsigned int ballot2 = __ballot_sync(0xFFFFFFFF, !hit);
    // Step 2: Compute the exclusive prefix sum within the warp
    unsigned int prefix = __popc(ballot & ((1U << laneId) - 1));
    unsigned int prefix2 = __popc(ballot2 & ((1U << laneId) - 1));

    unsigned int result = prefix;
    unsigned int result2 = prefix2;
    if (laneId == 31)
    {
        warp_sums[warpId] = prefix + hit;
        warp_sums[MAP_SIZE / 32 + warpId] = prefix2 + !hit;
    }
    __syncthreads();

    // Perform a warp-level scan to propagate the sum across warps
    if (warpId == 0 && thid < 32)
    {
        int warp_sum = warp_sums[thid];

#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            int n = __shfl_up_sync(0xFFFFFFFF, warp_sum, offset);
            if (laneId >= offset)
                warp_sum += n;
        }
        warp_sums[thid] = warp_sum;
    }
    __syncthreads();

    if (warpId > 0)
    {
        result += warp_sums[warpId - 1];
    }
    result2 += warp_sums[MAP_SIZE / 32 + warpId - 1];
    int idx = hit ? result : result2;
    int value_new = hit ? value : result2 - warp_sums[MAP_SIZE / 32 - 1];
    new_values[idx] = value_new;
    reorder_key[idx] = key;
    __syncthreads();
    return warp_sums[MAP_SIZE / 32 - 1];
}

// given hits and keys of a thread-block
// reorder keys, and values
// all values with true hits are packed together.
// then value with false hits are assigned by counts (from 0, 1, 2,...) and packed together.
// and return how many true hits
// e.g. hits:      [false, true, false, true],
//      keys:      [23,    40,   52,    99],
//      values:    [345,   455,  544,   24],
//      reorders:  [40,    99,   23,    52],
//      new_values:[455,   24,   23,    52],
//      return: 2
template<int MAP_SIZE = BLOCK_SIZE_MAP>
__device__ int update_keys_and_mixed_offsets(bool hit, int key, int value, int *reorder_key, int *new_values, int *warp_sums)
{
    int thid = threadIdx.x;
    int laneId = thid & 31; // Warp lane index
    int warpId = thid >> 5; // Warp index within the block

    if (thid < 64)
    {
        warp_sums[thid] = 0;
    }
    __syncthreads();

    // Step 1: Compute the ballot mask for the warp
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, hit);
    unsigned int ballot2 = __ballot_sync(0xFFFFFFFF, !hit);
    // Step 2: Compute the exclusive prefix sum within the warp
    unsigned int prefix = __popc(ballot & ((1U << laneId) - 1));
    unsigned int prefix2 = __popc(ballot2 & ((1U << laneId) - 1));

    unsigned int result = prefix;
    unsigned int result2 = prefix2;
    if (laneId == 31)
    {
        warp_sums[warpId] = prefix + hit;
        warp_sums[MAP_SIZE / 32 + warpId] = prefix2 + !hit; // max id is 64
    }
    __syncthreads();

    // Perform a warp-level scan to propagate the sum across warps
    if (thid < 64)
    {
        int warp_sum = warp_sums[thid];

        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            int n = __shfl_up_sync(0xFFFFFFFF, warp_sum, offset);
            if (laneId >= offset)
                warp_sum += n;
        }
        warp_sums[thid] = warp_sum;
    }
    __syncthreads();

    if(warpId == 0) {
        warp_sums[32 + laneId] += warp_sums[31];
    }

    __syncthreads();

    if (warpId > 0)
    {
        result += warp_sums[warpId - 1];
    }
    result2 += warp_sums[MAP_SIZE / 32 + warpId - 1];
    int idx = hit ? result : result2;
    int value_new = hit ? value : key;
    new_values[idx] = value_new;
    reorder_key[idx] = key;
    __syncthreads();  
    return warp_sums[MAP_SIZE / 32 - 1];
}


// insert keys with values threadblock (not used in Shadow-KV)
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__global__ void map_insert(int *keys, int *map_keys, int *map_values)
{
    __shared__ int s_values[LUT_SIZE];
    __shared__ int s_keys[LUT_SIZE];

    reset_map<MAP_SIZE, LUT_SIZE>(s_keys);
    __syncthreads();

    int offset = blockIdx.x * MAP_SIZE + threadIdx.x;
    int key = keys[offset];
    int value = threadIdx.x;

    insert_map<MAP_SIZE, LUT_SIZE>(key, value, s_keys, s_values);
    __syncthreads();

    // write back map
    write_back_map<MAP_SIZE, LUT_SIZE>(s_keys, s_values, map_keys, map_values);
}


// Not used in Shadow-KV
// create a shared memory map using orig_keys and threadIdx.x as values
// then look up values based on query_keys
// then reorder keys and update values
// Note: query_keys and g_reorder_keys can use the same pointer
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__global__ void reorder_keys_and_offsets(
    int *orig_keys,
    int *query_keys,
    int *g_reorder_keys,
    int *g_offsets,
    int *g_hit_cnt)
{
    __shared__ int s_values[LUT_SIZE];
    __shared__ int s_keys[LUT_SIZE];
    __shared__ int warp_sums[32]; // Assuming a maximum of 32 warps per block
    __shared__ int offsets[MAP_SIZE];
    __shared__ int reorder_keys[MAP_SIZE];

    int offset = blockIdx.x * MAP_SIZE + threadIdx.x;

    // create a shared memory map
    int old_key = orig_keys[offset];
    init_map<MAP_SIZE, LUT_SIZE>(old_key, s_keys, s_values);
    __syncthreads();

    // creat a new key as value (tid) for lookup
    int new_key = query_keys[offset];
    int value = lookup_map<MAP_SIZE, LUT_SIZE>(new_key, s_keys, s_values);
    bool hit = value != EMPTY_KEY;

    int hit_cnt = update_keys_and_offsets<MAP_SIZE>(hit, new_key, value, reorder_keys, offsets, warp_sums);
    // a hidden sync above
    // write out
    g_reorder_keys[offset] = reorder_keys[threadIdx.x];
    g_offsets[offset] = offsets[threadIdx.x];
    if (threadIdx.x == 0)
    {
        g_hit_cnt[blockIdx.x] = hit_cnt;
    }
}

// create a shared memory map using orig_keys and threadIdx.x as values
// then look up values based on query_keys
// then reorder keys and update values
// Note: query_keys and g_reorder_keys can use the same pointer
template<typename T, int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__global__ void reorder_keys_and_mixed_offsets(
    T *orig_keys,
    T *query_keys,
    T *g_reorder_keys,
    int *g_offsets,
    int *g_hit_cnt)
{
    __shared__ int s_values[LUT_SIZE];
    __shared__ int s_keys[LUT_SIZE];
    __shared__ int warp_sums[64]; // Assuming a maximum of 32 warps per block
    __shared__ int offsets[MAP_SIZE];
    __shared__ int reorder_keys[MAP_SIZE];

    int offset = blockIdx.x * MAP_SIZE + threadIdx.x;

    // create a shared memory map
    int old_key = (T) (orig_keys[offset]);  // might be a cast
    init_map<MAP_SIZE, LUT_SIZE>(old_key, s_keys, s_values);
    __syncthreads();

    // creat a new key as value (tid) for lookup
    int new_key = (T) (query_keys[offset]); // might be a cast

    int value = lookup_map<MAP_SIZE, LUT_SIZE>(new_key, s_keys, s_values);
    bool hit = value != EMPTY_KEY;

    int hit_cnt = update_keys_and_mixed_offsets<MAP_SIZE>(hit, new_key, value, reorder_keys, offsets, warp_sums);

#if SORT_OFFSET
    //sort offset as keys and reorder_keys as values
    block_sort2<MAP_SIZE>(offsets, reorder_keys, hit_cnt);
#endif

    // a hidden sync above
    // write out
    g_reorder_keys[offset] = (T) (reorder_keys[threadIdx.x]); // might be a cast
    g_offsets[offset] = offsets[threadIdx.x];
    if (threadIdx.x == 0)
    {
        g_hit_cnt[blockIdx.x] = hit_cnt;
    }
}

#endif // MAP_CUH