#include "sgemm_common.cuh"
#define BLOCKSIZE 32

__global__ void sgemm_shared_mem_block(int M, int N, int K, 
                                       float alpha, 
                                       const float *A, 
                                       const float *B, 
                                       float beta, 
                                       float *C) {
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];
    
    // Global row and column for this thread
    int globalRow = blockIdx.y * BLOCKSIZE + threadIdx.y;
    int globalCol = blockIdx.x * BLOCKSIZE + threadIdx.x;
    
    float tmp = 0.0;
    
    // Process K dimension in chunks of BLOCKSIZE
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Load tile from A with bounds checking
        int aRow = globalRow;
        int aCol = bkIdx + threadIdx.x;
        As[threadIdx.y * BLOCKSIZE + threadIdx.x] = 
            (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        
        // Load tile from B with bounds checking
        int bRow = bkIdx + threadIdx.y;
        int bCol = globalCol;
        Bs[threadIdx.y * BLOCKSIZE + threadIdx.x] = 
            (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        
        __syncthreads();
        
        // Compute dot product on the cached block
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadIdx.y * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result with bounds checking
    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = 
            alpha * tmp + beta * C[globalRow * N + globalCol];
    }
}
