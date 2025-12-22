#include "sgemm_common.cuh"
#define BLOCKSIZE 32

__global__ void sgemm_shared_mem_block(int M, int N, int K, 
                                       float alpha, 
                                       const float *A, 
                                       const float *B, 
                                       float beta, 
                                       float *C) {
    // Shared memory for caching tiles of A and B
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];
    
    // Advance pointers to the starting positions for this block
    A += blockIdx.y * BLOCKSIZE * K;                    // row=blockIdx.y, col=0
    B += blockIdx.x * BLOCKSIZE;                        // row=0, col=blockIdx.x
    C += blockIdx.y * BLOCKSIZE * N + blockIdx.x * BLOCKSIZE; // row=blockIdx.y, col=blockIdx.x
    
    float tmp = 0.0;
    
    // Outer loop: advance A along columns and B along rows
    // Process K dimension in chunks of BLOCKSIZE
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Load one element of A and B into shared memory
        // Each thread loads one element from A and one from B
        // threadIdx.x is consecutive for memory coalescing
        As[threadIdx.y * BLOCKSIZE + threadIdx.x] = A[threadIdx.y * K + threadIdx.x];
        Bs[threadIdx.y * BLOCKSIZE + threadIdx.x] = B[threadIdx.y * N + threadIdx.x];
        
        // Synchronize to ensure all threads have loaded their data
        // before any thread starts computing
        __syncthreads();
        
        // Advance pointers to next chunk
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
        
        // Compute dot product on the cached block
        // Each thread computes one element of the C tile
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadIdx.y * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadIdx.x];
        }
        
        // Synchronize again to ensure all threads finish computing
        // before the next iteration loads new data into shared memory
        __syncthreads();
    }
    
    // Write final result to global memory
    C[threadIdx.y * N + threadIdx.x] = 
        alpha * tmp + beta * C[threadIdx.y * N + threadIdx.x];
}
