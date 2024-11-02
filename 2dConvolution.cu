#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

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