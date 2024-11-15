#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define Mask_width 5
#define Mask_radius (Mask_width / 2)
#define w (TILE_WIDTH + Mask_width - 1)
__global__ void convolution_tiled(float *I, const float *M, float *P, int channels, int width, int height) {

	  __shared__ float N_ds[w][w];

    for (int k = 0; k < channels; k++) {

        // First phase: Main tile loading
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;// tile index
        int tileY = dest / w;     //row in shared Memory
        int tileX = dest % w;     //col in shared Memory

        int row = blockIdx.y * TILE_WIDTH + tileY - Mask_radius; // index of pixel in input image
        int col = blockIdx.x * TILE_WIDTH + tileX - Mask_radius; // index of pixel in input image

        if (row >= 0 && row < height && col >= 0 && col < width) {
            N_ds[tileY][tileX] = I[(row * width + col) * channels + k];
        } else {
            N_ds[tileY][tileX] = 0.0f;
        }

        // Second phase: Loading edge pixels or halo regions
        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        tileY = dest / w;
        tileX = dest % w;

        row = blockIdx.y * TILE_WIDTH + tileY - Mask_radius;
        col = blockIdx.x * TILE_WIDTH + tileX - Mask_radius;

        if (tileY < w) {
            if (row >= 0 && row < height && col >= 0 && col < width) {
                N_ds[tileY][tileX] = I[(row * width + col) * channels + k];
            } else {
                N_ds[tileY][tileX] = 0.0f;
            }
        }
        __syncthreads();

        // Convolution: Multiply and accumulate
        float accum = 0.0f;
        for (int y = 0; y < Mask_width; y++) {
            for (int x = 0; x < Mask_width; x++) {
                accum += N_ds[threadIdx.y + y][threadIdx.x + x] * devMask[y * Mask_width + x];
            }
        }

        // Store result to output
        int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (y < height && x < width) {
            P[(y * width + x) * channels + k] = accum;
        }
        __syncthreads();
    }
  }
__constant__ float devMask[Mask_width * Mask_width];

int main() {
    // Input dimensions
    int channels = 3, width = 1024, height = 1024;

    // Allocate host memory
    float *h_I = (float *)malloc(width * height * channels * sizeof(float));
    float *h_M = (float *)malloc(Mask_width * Mask_width * sizeof(float));
    float *h_P = (float *)malloc(width * height * channels * sizeof(float));

    // Initialize data
    // (Fill h_I and h_M with values)

    // Allocate device memory
    float *d_I, *d_P;
    cudaMalloc((void **)&d_I, width * height * channels * sizeof(float));
    cudaMalloc((void **)&d_P, width * height * channels * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_I, h_I, width * height * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(devMask, h_M, Mask_width * Mask_width * sizeof(float));

    // Kernel launch configuration
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    // Launch kernel
    convolution_tiled<<<dimGrid, dimBlock>>>(d_I, devMask, d_P, channels, width, height);

    // Copy result back to host
    cudaMemcpy(h_P, d_P, width * height * channels * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    free(h_I);
    free(h_M);
    free(h_P);
    cudaFree(d_I);
    cudaFree(d_P);

    return 0;
}

