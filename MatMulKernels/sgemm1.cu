#include "sgemm_common.cuh"
__global__ void sgemm_naive(int M, int N, int K,
                            float alpha, const float *A,
                            const float *B, float beta, float *C) {

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i)
            tmp += A[x * K + i] * B[i * N + y];

        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
