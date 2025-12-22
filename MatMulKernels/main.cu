#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

#include "sgemm_common.cuh"

template <typename Kernel>
void time_kernel(const char* name,
                 Kernel kernel,
                 dim3 gridDim,
                 dim3 blockDim,
                 int M, int N, int K,
                 float alpha,
                 const float* d_A,
                 const float* d_B,
                 float beta,
                 float* d_C,
                 int iters = 10)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up 
    kernel<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();

    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        kernel<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;  // average per run

    double flops = 2.0 * M * N * K;
    double gflops = flops / (ms * 1e6);

    std::cout << name << ":\n"
              << "  Time: " << ms << " ms\n"
              << "  Perf: " << gflops << " GFLOP/s\n\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // ---------------- Matrix sizes ----------------
    const int M = 4092;
    const int N = 4092;
    const int K = 4092;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // ---------------- Host memory ----------------
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& x : h_A) x = dist(gen);
    for (auto& x : h_B) x = dist(gen);

    // ---------------- Device memory ----------------
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), sizeC, cudaMemcpyHostToDevice);

    // ---------------- Launch config ----------------
     dim3 blockDim(32, 32);
    //dim3 blockDim(32*32);
    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));

    std::cout << "Matrix sizes: "
              << M << " x " << N << " x " << K << "\n\n";

    // ---------------- Time kernels ----------------
    time_kernel("Sgemm Shared Memory",  sgemm_shared_mem_block,gridDim,  blockDim,M, N, K, alpha, d_A, d_B, beta, d_C); 
// time_kernel("SGEMM Naive",sgemm_naive,gridDim, blockDim,M, N, K, alpha, d_A, d_B, beta, d_C);
   // time_kernel("SGEMM Coalescing", sgemm_coalescing, gridDim, blockDim, M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

