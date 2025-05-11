#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
template<const int BLOCK_SIZE>
__global__ void mysgemm_v2(int M, int N, int K, float *A, float *B, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    // Fix: Use both threadIdx.x and threadIdx.y for proper 2D indexing
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column indices
    int row = by * BM + ty;
    int col = bx * BN + tx;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float tmp = 0.0f;

    // Point to the starting position of the sub-matrices
    float *A_start = &A[by * BM * K];
    float *B_start = &B[0 * N + bx * BN];

    // Loop over all sub-matrices needed for the result
    for (int k = 0; k < K; k+=BK) {
        // Load data into shared memory with bounds checking
        if (row < M && k  + tx < K) {
            As[ty * BK + tx] = A_start[ty * K + tx];
        } else {
            As[ty * BK + tx] = 0.0f;
        }

        if (col < N && k  + ty < K) {
            Bs[ty * BN + tx] = B_start[ty * N + tx];
        } else {
            Bs[ty * BN + tx] = 0.0f;
        }

        __syncthreads();

        // Move to next sub-matrices
        A_start += BK;
        B_start += BK * N;

        // Calculate dot product for this block
        for (int i = 0; i < BK; i++) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }

        __syncthreads();
    }

    // Write the result with bounds checking
    if (row < M && col < N) {
        C[row * N + col] = tmp;
    }
}


void test_mysgemm_v2(int M, int N, int K,  float *A, float *B, float *C) {
    dim3 blockDim(32, 32); // 32 x 32 = 1024 threads per block
    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    mysgemm_v2<32><<<gridDim, blockDim>>>(M, N, K, A, B, C);
  }