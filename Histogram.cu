#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void histogram_kernel_shared(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins) {

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
  extern __shared__ unsigned int histo_s[];

	for(int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
		histo_s[i] = 0;
		}
	__syncthreads();

	// Main loop to compute per-block sub-histograms
	for(int i = tid; i < num_elements ; i += stride) {
		unsigned int val = input[i];
		atomicAdd(&histo_s[val], 1);
	}
	__syncthreads();
	// Merge per-block sub-histograms and write to global memory
	for(int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
		atomicAdd(&bins[i], histo_s[i]);
	}

}