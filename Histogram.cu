#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define NUM_BINS 4096
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
int main() {
    int N = 1 << 24;  // arbitary number of elements
    unsigned int num_bins = NUM_BINS;
    size_t size = N * sizeof(unsigned int);
    size_t hist_size = num_bins * sizeof(unsigned int);

    // Allocate host memory
    unsigned int *h_input = (unsigned int*)malloc(size);
    unsigned int *h_bins = (unsigned int*)malloc(hist_size);

    // Initialize input data
    for(int i = 0; i < N; i++) {
        h_input[i] = rand() % num_bins;  // Random values between 0 and NUM_BINS-1
    }
    memset(h_bins, 0, hist_size);

    // Allocate device memory
    unsigned int *d_input, *d_bins;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_bins, hist_size);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins, h_bins, hist_size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(512);
    dim3 gridDim(30);
    
    histogram_kernel_shared<<<gridDim, blockDim, num_bins * sizeof(unsigned int)>>>(d_input, d_bins, N, num_bins);



    // Copy results back to host
    cudaMemcpy(h_bins, d_bins, hist_size, cudaMemcpyDeviceToHost);



    // Cleanup
    free(h_input);
    free(h_bins);
    cudaFree(d_input);
	cudaFree(d_bins);

    return 0;
}