#include <cstdio>

#include "vector_vector_ops.hpp"
#include "vector_vector_ops_cuda.hpp"
#include "matrix_vector_ops_cuda.hpp"
#include <cublas.h>
#include "cuda_check_errors.hpp"

void sumVectors(cublasHandle_t& handle_cublas,
                int n,
                const double* x,
                double* y,
                const double* alpha)
{
  checkCudaErrors(cublasDaxpy(handle_cublas,
                              n,
                              alpha,
                              x, 1,
                              y, 1));
}

void dotProduct(cublasHandle_t& handle_cublas,
                int n,
                const double* x,
                const double* y,
                double* r)
{
  checkCudaErrors(cublasDdot(handle_cublas,
                             n,
                             x, 1,
                             y, 1,
                             r));
}
__device__ int mutex = 0;

void deviceDotProduct(int n,
                const double* x,
                const double* y,
                double* r)
{
  int num_blocks;
  int block_size = BLOCK_SIZE;
  num_blocks = (n + block_size - 1) / block_size;

  deviceDotProductKernel<<<num_blocks, block_size>>>(n, x, y, r);
}

void deviceDotProduct(int n,
  const double* x,
  const double* y,
  double* r,
  cudaStream_t stream)
{
int num_blocks;
int block_size = BLOCK_SIZE;
num_blocks = (n + block_size - 1) / block_size;

deviceDotProductKernel<<<num_blocks, block_size, 0, stream>>>(n, x, y, r);
}

void scaleVector(cublasHandle_t& handle_cublas,
                 int n,
                 const double* alpha,
                 double* x)
{
  checkCudaErrors(cublasDscal(handle_cublas,
                              n,
                              alpha,
                              x, 1));
}

__global__ void deviceDotProductKernel(int n, const double* x, const double* y, double* r) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  __shared__ double cache[BLOCK_SIZE];
  double temp = 0.0;

  if (index == 0) {
    *r = 0.0;
  }

  while (index < n) {
    temp += x[index] * y[index];
    index += stride;
  }

  cache[threadIdx.x] = temp;
  __syncthreads();

  int i = blockDim.x / 2;
  while (i >= warpSize) {
    if (threadIdx.x < i) {
      cache[threadIdx.x] += cache[threadIdx.x + i];
    }
    __syncthreads();
    i /= 2;
  }

  double local_sum = cache[threadIdx.x];

  if (threadIdx.x < warpSize) {
    while (i != 0) {
      local_sum += __shfl_down_sync(FULL_MASK, local_sum, i);
      i /= 2;
    }
  }

  if (threadIdx.x == 0) {
    atomicAdd(r, local_sum);
  }
}

void fun_divide(const double* x, const double* y, double* z) {
  divide<<<1, 1>>>(x, y, z);
}

__global__ void divide(const double* x, const double* y, double* z) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index == 0) {
    *z = *x / *y;
  }
}

void fun_mem_copy(const double* x, double* y) {
  mem_copy<<<1, 1>>>(x, y);
}

__global__ void mem_copy(const double* x, double* y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0) {
    *y = *x;
  }
}

