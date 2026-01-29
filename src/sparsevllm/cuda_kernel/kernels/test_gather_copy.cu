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


#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdlib.h> /* srand, rand */
#include <time.h>   /* time */
#include <thread>
#include <map>
#include <algorithm>
#include <vector>
#include <unordered_set>

#define CPY_SIZE 40 // local test

#define FUSE_MEM 1
#define USE_SPLIT 1

// allow change BLOCK_SIZE_CP here
#ifndef BLOCK_SIZE_CP
#define BLOCK_SIZE_CP 128
#endif

#include "copy.cuh"
#include "map.cuh"
#include "functions.h"

#if BLOCK_SIZE_CP == 128
#define PTYPE int4
#endif

#if BLOCK_SIZE_CP == 256
#define PTYPE int2
#endif


#define BLOCK_NUM (8*20)
#define CPU_V_LENGTH (128 * 1024) // 128K without multiply 128
#define GPU_V_LENGTH (256 * 8)    //
#define FAKE_HIT_CNT 0

// when hit is very high,FUSE_MEM =1, USE_SPLIT = 0
// when hit is 50%-80%, , FUSE_MEM =1, USE_SPLIT = 1


using namespace std;

bool check_cached_copy(int *h_v, int *h_d_v, int64_t *d_offsets, int *d_cnt, int *d_compute)
{
    int *h_golden = (int *)malloc((long int)BLOCK_NUM * GPU_V_LENGTH * 128 * 2 );
    int *h_compute = (int *)malloc((long int)BLOCK_NUM * GPU_V_LENGTH * 128 * 2 );
    cudaMemcpy(h_compute, d_compute, (long int)BLOCK_NUM * GPU_V_LENGTH * 128 * 2, cudaMemcpyDeviceToHost);
    int64_t *h_offsets = (int64_t *)malloc(BLOCK_NUM * BLOCK_SIZE_MAP * sizeof(int64_t));
    cudaMemcpy(h_offsets, d_offsets, BLOCK_NUM * BLOCK_SIZE_MAP * sizeof(int64_t), cudaMemcpyDeviceToHost);
    int *h_cnt = (int *)malloc(BLOCK_NUM  * sizeof(int));
    cudaMemcpy(h_cnt, d_cnt, BLOCK_NUM * sizeof(int), cudaMemcpyDeviceToHost);

    int64_t idx = 0;
    for (int64_t k = 0; k < BLOCK_NUM; ++k)
    {
        for (int64_t i = 0; i < BLOCK_SIZE_MAP; ++i)
        {
            int *ptr;
            ptr = h_v + k * CPU_V_LENGTH * 128 * 2 / 4 + h_offsets[idx] * 64 * 8; // 4 is sizeof(int)

            std::memcpy(h_golden + idx * 128 * 4, ptr, 128 * 8 * 2);
            idx++;
        }
    }

    idx = 0;
    int cnt = 0;

    for (int k = 0; k < BLOCK_NUM; ++k)
    {
        for (int i = 0; i < BLOCK_SIZE_MAP; ++i)
        {

            // 128x4 == 128*8*2/4
            for (int j = 0; j < 128 * 4; ++j)
            {
                if (h_compute[idx] != h_golden[idx] && cnt < 10)
                {
                    cnt++;
                    std::cout << "h_compute " << h_compute[idx] << ",h_golden " << h_golden[idx]
                              << " at (" << k << "," << i << "," << j << ")" << std::endl;
                }
                idx++;
            }
        }
    }

    free(h_golden);
    free(h_compute);

    return cnt == 0;
}

int main()
{
    int deviceCnt;
    cudaGetDeviceCount(&deviceCnt);
    cudaSetDevice(deviceCnt - 1);
    srand(time(NULL));
    //srand(12345);

    // Allocate host memory for keys and initialize them
    int num_elements = BLOCK_NUM * BLOCK_SIZE_MAP;
    int64_t *h_keys;
    int64_t *h_reorder_keys;
    cudaMallocHost(&h_keys, num_elements * sizeof(int64_t));
    cudaMallocHost(&h_reorder_keys, num_elements * sizeof(int64_t));

    // initialization of keys
    int idx = 0;
    for (int k = 0; k < BLOCK_NUM; ++k)
    {
        std::unordered_set<int> previous_keys;
        std::unordered_set<int> previous_reorders;
        for (int i = 0; i < BLOCK_SIZE_MAP; ++i)
        {
            idx = k * BLOCK_SIZE_MAP + i;
            int key;
            do {
                key = rand() % (CPU_V_LENGTH / 8);
            } while(previous_keys.count((int)key) > 0);
            previous_keys.insert((int)key);
            h_keys[idx] =  key;
        }

        for (int i = 0; i < BLOCK_SIZE_MAP; ++i)
        {
            idx = k * BLOCK_SIZE_MAP + i;
            int hit = rand() % 256;
            
            if (hit < FAKE_HIT_CNT)
            {
                int ref_idx = k * BLOCK_SIZE_MAP + rand() % 256;
                h_reorder_keys[idx] = h_keys[ref_idx];
                
            }
            else
            {   
                int reorder;
                do {
                    reorder = rand() % (CPU_V_LENGTH / 8);
                } while(previous_reorders.count(reorder) > 0);
                h_reorder_keys[idx] = reorder; 
            }
            previous_reorders.insert((int)h_reorder_keys[idx]);
        }
    }

    // Allocate device memory for keys
    int64_t *d_keys;
    cudaMalloc(&d_keys, num_elements * sizeof(int64_t));
    int64_t *d_reorder_keys;
    cudaMalloc(&d_reorder_keys, num_elements * sizeof(int64_t));
    int64_t *d_reorder_keys_dst;
    cudaMalloc(&d_reorder_keys_dst, num_elements * sizeof(int64_t));

    // Copy keys to device
    cudaMemcpy(d_keys, h_keys, num_elements * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reorder_keys, h_reorder_keys, num_elements * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Allocate host v and initialize it
    int *h_v;
    // 128 is hidden, 2 as sizeof(bf16)
    std::cout << "malloc h_v " << (long int)BLOCK_NUM * CPU_V_LENGTH * 128 * 2 / 1024 / 1024 << " MB" << std::endl;
    cudaMallocHost(&h_v, (long int)BLOCK_NUM * CPU_V_LENGTH * 128 * 2);

    idx = 0;
    for (int k = 0; k < BLOCK_NUM; ++k)
    {
        for (int i = 0; i < CPU_V_LENGTH; ++i)  
        {
            for (int j = 0; j < 128 * 2 / 4; ++j)  // 64's int
            {
                h_v[idx] = rand() % 10000;
                idx++;
            }
        }
    }

    // Allocate device v
    int *host_d_v = (int *)malloc((long int)BLOCK_NUM * GPU_V_LENGTH * 128 * 2);
    // initialize
    idx = 0;
    for (int64_t k = 0; k < BLOCK_NUM; ++k)
    {
        
        for (int64_t i = 0; i < BLOCK_SIZE_MAP; ++i)
        {
            int64_t id = k * BLOCK_SIZE_MAP + i;
            int64_t key = h_keys[id];
            int64_t base = k*CPU_V_LENGTH*128*2/4 + key * 8*64;
        
            for (int j = 0; j < (8 * 128 * 2 / 4); ++j)   // 8 x 64 
            {
                host_d_v[idx] = h_v[base + j];
                idx++;
            }
        }
    }

    int *d_v;
    int *d_vtemp;
    std::cout << "malloc d_v " << (long int)BLOCK_NUM * GPU_V_LENGTH * 128 * 2 / 1024 / 1024 << " MB" << std::endl;
    cudaMalloc(&d_v, (long int)BLOCK_NUM * GPU_V_LENGTH * 128 * 2);
    cudaMemcpy(d_v, host_d_v, (long int)BLOCK_NUM * GPU_V_LENGTH * 128 * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&d_vtemp, (long int)BLOCK_NUM * GPU_V_LENGTH * 128 * 2);

    // Allocate offsets/cnt
    int *d_offsets;
    cudaMalloc(&d_offsets, num_elements * sizeof(int));
    int *d_cnt;
    cudaMalloc(&d_cnt, BLOCK_NUM * sizeof(int));
    cudaMemset(d_cnt, 0x0, BLOCK_NUM * sizeof(int));

    unsigned int *d_signals;
    cudaMalloc(&d_signals, BLOCK_NUM * sizeof(int));
    cudaMemset(d_signals, 0x0, BLOCK_NUM * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    int test_iteration = 1000;
    int warmup = 10;

    // Insert elements into the hash map
    int blockSizeCopy = BLOCK_SIZE_CP;
    int numBlocks = BLOCK_NUM;

    int maxbytes = CPY_SIZE * 2 * 1024 + BLOCK_SIZE_MAP * 4 ; // < 160 KB
    std::cout << "gather_copy maxbytes shared memory size (KB) " << maxbytes / 1024 << std::endl;
    cudaFuncSetAttribute(gahter_copy_fixed_start_end<PTYPE, int64_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

    cudaStream_t stream;
    cudaStream_t streamCopy1;
    cudaStream_t streamCopy2;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&streamCopy1);
    cudaStreamCreate(&streamCopy2);

    cudaEvent_t trigger;
    cudaEventCreate(&trigger);



    bool pass = false;

    gahter_copy_fixed_start_end<<<numBlocks, blockSizeCopy, maxbytes>>>((PTYPE *)h_v, (PTYPE *)d_v, CPU_V_LENGTH *128 , GPU_V_LENGTH *128, d_keys, 0, 256);

    pass = check_cached_copy(h_v, host_d_v, d_keys, d_cnt, d_v);
    if(pass) {
        std::cout << "check_cached_copy check pass" << std::endl;
    } else {
        std::cout << "check_cached_copy check fail" << std::endl;
        return 0;
    }


    float BW = 0.0;
    int cnt = 0;
    for (int i = 0; i < test_iteration + warmup; ++i)
    {
        cudaEventRecord(start);
        gahter_copy_fixed_start_end<<<numBlocks, blockSizeCopy, maxbytes>>>((PTYPE *)h_v, (PTYPE *)d_v, CPU_V_LENGTH *128 , GPU_V_LENGTH *128, d_keys, 0, 256);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0.0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        float each_BW = (256) * 2048 * BLOCK_NUM / (milliseconds / 1000.0) / 1024 / 1024 / 1024;
        cnt++;
        // print first 10 for checking
        if (cnt < 10)
        {
            std::cout << "milliseconds: " << milliseconds << std::endl;
            std::cout << "BW: " << each_BW << std::endl;
        }
        if( i >= warmup)
            BW += each_BW;
    }

    std::cout << "avg BW: " << BW / test_iteration << std::endl;

    // Free device memory
    cudaFree(d_keys);
    cudaFree(d_reorder_keys);
    cudaFree(d_cnt);
    cudaFree(d_v);
    cudaFree(d_vtemp);
    cudaFree(d_reorder_keys_dst);

    return 0;
}
