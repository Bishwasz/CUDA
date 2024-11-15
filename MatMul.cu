#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 16

__global__ void matrixMultiplyTiled(
    float* A, float* B, float* C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    for (int p = 0; p < (numAColumns - 1) / TILE_WIDTH + 1; ++p) {
        if (Row < numARows && p * TILE_WIDTH + tx < numAColumns)
            ds_A[ty][tx] = A[Row * numAColumns + p * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (p * TILE_WIDTH + ty < numBRows && Col < numBColumns)
            ds_B[ty][tx] = B[(p * TILE_WIDTH + ty) * numBColumns + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            Pvalue += ds_A[ty][i] * ds_B[i][tx];

        __syncthreads();
    }

    if (Row < numCRows && Col < numCColumns)
        C[Row * numCColumns + Col] = Pvalue;
}

void launchMatrixMultiplyTiled(
    float* deviceA, float* deviceB, float* deviceC,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns) {

    // Define grid and block dimensions
    dim3 gridDim((numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the kernel
    matrixMultiplyTiled<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC,
        numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Error: " << cudaGetErrorString(err) << std::endl;
    }
}
