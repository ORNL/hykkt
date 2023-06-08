#include "log_barrier_utils_cuda.hpp"
#include "log_barrier_utils.hpp"
#include "cuda_memory_utils.hpp"
#include "cuda_check_errors.hpp"
#include "cub/cub.cuh"
#include "constants.hpp"


/**
 * @brief Compute the log barrier objective for a qaudratic problem and add it to out (out = f(x))
*/
void fun_lb_objective(LogBarrierInfo& problem_info, double* x, double* Q_x, double* A_x, double t, double* out)
{
    int num_blocks;
    int block_size = BLOCK_SIZE;
    num_blocks = (problem_info.m_ + block_size - 1) / block_size;
    
    lb_objective<<<num_blocks, block_size>>>(problem_info.m_, problem_info.n_, t, x, problem_info.b_, problem_info.c_, Q_x, A_x, out);
}

__global__ void lb_objective(int m, int n, double t, double* __restrict__ x, double* __restrict__ b, double* __restrict__ c, double* __restrict__ Q_x, double* __restrict__ A_x, double* out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ double cache[BLOCK_SIZE];

    if (index == 0) {
        *out = 0.0;
    }

    double temp = 0.0;
    #pragma unroll
    while(index < m) {
        if (index < n)
        {
            temp += t * x[index] * (0.5 * Q_x[index] + c[index]) - log(b[index] - A_x[index]);
        }
        else if (index < m)
        {
            temp += - log(b[index] - A_x[index]);
        }
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
      atomicAdd(out, local_sum);
    }
}

//***************REPEATED CODE - FIX!!!***************//
/**
 * @brief Compute the log barrier objective for a linear problem and add it to out (out = f(x))
*/
void fun_lb_objective(LogBarrierInfo& problem_info, double* x, double* A_x, double t, double* out)
{
    int num_blocks;
    int block_size = BLOCK_SIZE;
    num_blocks = (problem_info.m_ + block_size - 1) / block_size;

    lb_objective<<<num_blocks, block_size>>>(problem_info.m_, problem_info.n_, t, x, problem_info.b_, problem_info.c_, A_x, out);
}

__global__ void lb_objective(int m, int n, double t, double* __restrict__ x, double* __restrict__ b, double* __restrict__ c, double*__restrict__ A_x, double* __restrict__ out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ double cache[BLOCK_SIZE];

    if (index == 0) {
        *out = 0.0;
    }

    double temp = 0.0;
    #pragma unroll
    while(index < m) {
        if (index < n)
        {
            temp += t * x[index] * c[index] - log(b[index] - A_x[index]);
        }
        else if (index < m)
        {
            temp += - log(b[index] - A_x[index]);
        }
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
      atomicAdd(out, local_sum);
    }
}
//****************************************************//

/**
 * @brief Compute the log barrier gradient for a qaudratic problem
*/
void fun_lb_gradient(LogBarrierInfo& info, double* Q_x, double* A_x, double* out, double t, double scale)
{
    int num_blocks;
    int block_size = BLOCK_SIZE;
    num_blocks = (info.n_ + block_size - 1) / block_size;
    lb_gradient<<<num_blocks, block_size>>>(info.n_, t, scale, info.b_, info.c_, Q_x, A_x, info.a_t_i_, info.a_t_j_, info.a_t_v_, out);
}

//TODO: device helper function to calculate A_t*(1.0 ./ (pf.b - A * x))
//manual implementation of sparse matrix vector multiplication to reduce number of kernel calls
__global__ void lb_gradient(int n, double t, double scale, double* __restrict__ b, double* __restrict__ c, double* __restrict__ Q_x, double* __restrict__ A_x, int* __restrict__ a_t_i, int* __restrict__ a_t_j, double* __restrict__ a_t_v, double* out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int i, j;
    #pragma unroll
    for (i = index; i < n; i+= stride) {
        out[i] = 0.0;
        int start = a_t_i[i];
        int end = a_t_i[i+1];
        for (j = start; j < end; j++) {
            int col_index = a_t_j[j];
            out[i] += a_t_v[j] / (b[col_index] - A_x[col_index]); //A_t*(1.0 ./ (pf.b - A * x))
        }
        out[i] += t * (Q_x[i] + c[i]);
        out[i] *= scale;
    }
}
//***************REPEATED CODE - FIX!!!***************//
void fun_lb_gradient(LogBarrierInfo& info, double* A_x, double* out, double t, double scale)
{
    int num_blocks;
    int block_size = BLOCK_SIZE;
    num_blocks = (info.n_ + block_size - 1) / block_size;
    lb_gradient<<<num_blocks, block_size>>>(info.n_, t, scale, info.b_, info.c_, A_x, info.a_t_i_, info.a_t_j_, info.a_t_v_, out);
}

__global__ void lb_gradient(int n, double t, double scale, double* __restrict__ b, double* __restrict__ c, double* __restrict__ A_x, int* __restrict__ a_t_i, int* __restrict__ a_t_j, double* __restrict__ a_t_v, double* out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int i, j;
    #pragma unroll
    for (i = index; i < n; i+= stride) {
        out[i] = 0.0;
        int start = a_t_i[i];
        int end = a_t_i[i+1];
        for (j = start; j < end; j++) {
            int col_index = a_t_j[j];
            out[i] += a_t_v[j] / (b[col_index] - A_x[col_index]); //A_t*(1.0 ./ (pf.b - A * x))
        }
        out[i] += t * c[i];
        out[i] *= scale;
    }
}
//****************************************************//

/**
 * @brief Compute the log barrier hessian
*/
void fun_lb_hessian(LogBarrierInfo& info, double* A_x, double* out)
{
    int num_blocks;
    int block_size = BLOCK_SIZE;
    num_blocks = (info.m_ + block_size - 1) / block_size;
    lb_hessian<<<num_blocks, block_size>>>(info.m_, info.b_, A_x, out);
}

__global__ void lb_hessian(int m, double* b, double* __restrict__ A_x, double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        out[i] = 1.0 / pow(b[i] - A_x[i], 2);
    }
}