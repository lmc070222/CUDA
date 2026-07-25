#include <cassert>
#include <chrono>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <random>

// You may increase this value to test larger matrices
// But it will be slow on CPU
constexpr int MAXN = 2048;

/**
 * @brief A naive implementation of matrix multiplication on CPU.
 * Perform C = A * B, where A is M x K, B is K x N, and C is M x N.
 */
void naiveSgemm(float *a, float *b, float *c, const int M, const int N,
                const int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += a[m * K + k] * b[k * N + n];
      }
      c[m * N + n] = sum;
    }
  }
}

/**
 * @brief A naive implementation of matrix multiplication on GPU.
 * Perform C = A * B, where A is M x K, B is K x N, and C is M x N.
 */
__global__ void naiveSgemm2D(float *a, float *b, float *c, const int M,
                             const int N, const int K) {
  int m = blockIdx.x * blockDim.x + threadIdx.x; // Row index
  int n = blockIdx.y * blockDim.y + threadIdx.y; // Column index
  if (m < M && n < N) {
    float sum = 0.0;
    for (int k = 0; k < K; ++k) {
      sum += a[m * K + k] * b[k * N + n];
    }
    c[m * N + n] = sum;
  }
}

/**
 * @brief Launch naiveSgemm2D kernel.
 */
void launchSgemm2D(float *a, float *b, float *c, const int M, const int N,
                   const int K) {
  dim3 block(16, 16); // 256 threads per block (16 * 16 = 256)
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  naiveSgemm2D<<<grid, block>>>(a, b, c, M, N, K);
}

#define BLOCK_SIZE 16

__global__ void tiledSgemm2D(float *a, float *b, float *c, const int M,
                             const int N, const int K) {
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
    if (row < M && (t * BLOCK_SIZE + threadIdx.x) < K)
      As[threadIdx.y][threadIdx.x] = a[row * K + t * BLOCK_SIZE + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0f;

    if ((t * BLOCK_SIZE + threadIdx.y) < K && col < N)
      Bs[threadIdx.y][threadIdx.x] =
          b[(t * BLOCK_SIZE + threadIdx.y) * N + col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; ++k)
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

    __syncthreads();
  }

  if (row < M && col < N)
    c[row * N + col] = sum;
}

void launchTiledSgemm2D(float *a, float *b, float *c, const int M, const int N,
                         const int K) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  tiledSgemm2D<<<grid, block>>>(a, b, c, M, N, K);
}

void initialize(float *a, float *b, float *c, const int M, const int N,
                const int K) {
  auto gen = std::mt19937(2024);
  auto dis = std::uniform_real_distribution<float>(-1.0, 1.0);
  for (int i = 0; i < M * K; ++i) {
    a[i] = dis(gen);
  }
  for (int i = 0; i < K * N; ++i) {
    b[i] = dis(gen);
  }
  for (int i = 0; i < M * N; ++i) {
    c[i] = 0.0;
  }
}

/** 
 * @brief Launch sgemm using cuBLAS
 */
void launchCublasSgemm(float *a, float *b, float *c, const int M, const int N,
                        const int K) {
  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    cublasCreate(&handle);
  }
  float alpha = 1.0;
  float beta = 0.0;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N, a, K,
              &beta, c, N);
}

bool compare(float *a, float *b, const int N) {
  for (int i = 0; i < N; ++i) {
    if (std::abs(a[i] - b[i]) > 1e-2) {
      printf("Mismatch at index %d: %f vs %f\n", i, a[i], b[i]);
      return false;
    }
  }
  printf("Results match\n");
  return true;
}

int main() {
  const int size = MAXN * MAXN;
  float *a, *b, *c, *h_ref;
  a = new float[size];
  b = new float[size];
  c = new float[size];
  h_ref = new float[size];
  initialize(a, b, c, MAXN, MAXN, MAXN);

  auto start = std::chrono::high_resolution_clock::now();
  naiveSgemm(a, b, h_ref, MAXN, MAXN, MAXN);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  printf("CPU time: %.3fs\n", elapsed.count());

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size * sizeof(float));
  cudaMalloc(&d_b, size * sizeof(float));
  cudaMalloc(&d_c, size * sizeof(float));
  cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_c, c, size * sizeof(float), cudaMemcpyHostToDevice);
  start = std::chrono::high_resolution_clock::now();
  launchSgemm2D(d_a, d_b, d_c, MAXN, MAXN, MAXN);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  printf("GPU time: %.3fs\n", elapsed.count());

  launchCublasSgemm(d_a, d_b, d_c, MAXN, MAXN, MAXN);
  cudaDeviceSynchronize();
  cudaMemset(d_c, 0, size * sizeof(float));
  start = std::chrono::high_resolution_clock::now();
  launchCublasSgemm(d_a, d_b, d_c, MAXN, MAXN, MAXN);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  printf("cuBLAS time: %.3fs\n", elapsed.count());

  cudaMemset(d_c, 0, size * sizeof(float));
  start = std::chrono::high_resolution_clock::now();
  launchTiledSgemm2D(d_a, d_b, d_c, MAXN, MAXN, MAXN);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  printf("Tiled GPU time: %.3fs\n", elapsed.count());

  cudaMemcpy(c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);
  compare(h_ref, c, size);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] h_ref;
  return 0;
}
