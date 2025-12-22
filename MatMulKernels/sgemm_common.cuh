#pragma once
#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void sgemm_naive(int M, int N, int K,
                            float alpha, const float *A,
                            const float *B, float beta, float *C);
__global__ void sgemm_coalescing(int M, int N, int K, float alpha, const float *A,
                                  const float *B, float beta, float *C);
__global__ void sgemm_shared_mem_block(int M, int N, int K, 
                                       float alpha, 
                                       const float *A, 
                                       const float *B, 
                                       float beta, 
                                       float *C) ;
