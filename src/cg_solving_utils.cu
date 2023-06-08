#include "cg_solving_utils.hpp"
#include "cg_solving_utils_cuda.hpp"
#include "constants.hpp"

__global__ void
spmv_vector_kernel (int num_rows,
    int num_cols,
    int* __restrict__ col_ids,
    int* __restrict__ row_ptr,
    double* __restrict__ data,
    double* __restrict__ x,
    double* y)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    if( row < num_rows ){
        double dot = 0;
        int row_start = row_ptr [ row ];
        int row_end = row_ptr [ row +1];
        for (int jj = row_start ; jj < row_end ; jj ++)
        dot += data [ jj ] * x[ col_ids [ jj ]];
        y[ row ] = dot ;
    }
}

__global__ void 
add(int n, double alpha, double beta, double* arr1, double* arr2)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    arr1[i] = alpha * arr1[i] + beta * arr2[i];
  }
}

__global__ void
copy(int n, double* v1, double* v2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        v1[i] = v2[i];
    }
}

__global__ void dot(int n, const double* x, const double* y, double* r) {
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

__device__ void cg_solve_kernel(int itmax, 
    double tol, 
    int n, 
    int* a_i, 
    int* a_j, 
    double* a_v, 
    int* a_t_i,
    int* a_t_j, 
    double* a_t_v, 
    double* b, 
    double* c,
    double* x, 
    double* r, 
    double* z, 
    double* p, 
    double* A_p,
    double* d_r_dot_z)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index != 0) return;
    //apply operator r = Ax
    add<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>> (n, MINUS_ONE, ONE, r, b);
    //preconditioner solve
    copy<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>> (n, p, z);
    //dot<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>> (n, r, z, d_r_dot_z);
    spmv_vector_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(n, n, a_j, a_i, a_v, x, r);
}