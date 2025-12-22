#include "sgemm_common.cuh"
__global__ void sgemmKernel4_simple(
    int M, int N, int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
) {
    // Tile sizes
    const int BM = 64;   // rows per block
    const int BN = 64;   // cols per block
    const int BK = 8;    // K tile
    const int TM = 8;    // outputs per thread

    // Block indices
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread mapping (1D block)
    int threadCol = threadIdx.x % BN;   // column in C tile
    int threadRow = threadIdx.x / BN;   // which TM-row group

    // Shared memory
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Global base pointers
    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    // Register accumulation
    float threadResults[TM];
    for (int i = 0; i < TM; i++) {
        threadResults[i] = 0.0f;
    }

    // Indices for loading shared memory
    int innerColA = threadIdx.x % BK;
    int innerRowA = threadIdx.x / BK;

    int innerColB = threadIdx.x % BN;
    int innerRowB = threadIdx.x / BN;

    // Loop over K tiles
    for (int bk = 0; bk < K; bk += BK) {

        // Load A tile
        int aRow = innerRowA;
        int aCol = innerColA;
        if (aRow < BM && (bk + aCol) < K) {
            As[aRow * BK + aCol] =
                A[aRow * K + (bk + aCol)];
        } else {
            As[aRow * BK + aCol] = 0.0f;
        }

        // Load B tile
        int bRow = innerRowB;
        int bCol = innerColB;
        if (bRow < BK && bCol < BN && (bk + bRow) < K) {
            Bs[bRow * BN + bCol] =
                B[(bk + bRow) * N + bCol];
        } else {
            Bs[bRow * BN + bCol] = 0.0f;
        }

        __syncthreads();

        // Compute
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            float bVal = Bs[dotIdx * BN + threadCol];
            for (int r = 0; r < TM; r++) {
                int aRowIdx = threadRow * TM + r;
                threadResults[r] +=
                    As[aRowIdx * BK + dotIdx] * bVal;
            }
        }

        __syncthreads();
    }

    // Write results
    for (int r = 0; r < TM; r++) {
        int row = threadRow * TM + r;
        if (row < BM && threadCol < BN) {
            C[row * N + threadCol] =
                alpha * threadResults[r] +
                beta * C[row * N + threadCol];
        }
    }
}
