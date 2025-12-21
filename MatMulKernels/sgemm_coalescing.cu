#include "sgemm_common.cuh"
# define BLOCKSIZE 32
__global__ void sgemm_coalescing(int M, int N, int K, float alpha, const float *A,
                                  const float *B, float beta, float *C) {
  // Changed: thread-to-output mapping for coalesced access
  const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // Boundary check (same as before)
  if (x < M && y < N) {
    float tmp = 0.0;
    
    // Same computation loop as kernel 1
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    
    // Same final computation
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

