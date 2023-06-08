#define FULL_WARP_MASK 0xFFFFFFFF

#include "sparse_mat_mul.hpp"
#include "sparse_mat_mul_cuda.hpp"
#include "constants.hpp"
#include "cuda.h"

//https://medium.com/analytics-vidhya/sparse-matrix-vector-multiplication-with-cuda-42d191878e8f
//https://code.google.com/archive/p/cusp-library/downloads

texture<int2,1>  tex_x_double;

template <bool UseCache>
__inline__ __device__ double fetch_x(const int& i, const double * x)
{
#if __CUDA_ARCH__ >= 130
    // double requires Compute Capability 1.3 or greater
    //if (UseCache)
   /// {
    //    int2 v = tex1Dfetch(tex_x_double, i);
    //    return __hiloint2double(v.y, v.x);
   // }
    //else
   // {
        return x[i];
    //}
#else
    return 1.0/0.0; // should never be called
#endif
}

void fun_csr_spmv_kernel(
    int n_rows,
    int n_cols,
    int* col_ids,
    int* row_ptr,
    double* data,
    double* x,
    double* y) {
    
    int num_blocks;
    int block_size = BLOCK_SIZE;
    num_blocks = (n_rows + block_size - 1) / block_size;
    csr_spmv_vector_kernel<<<num_blocks, block_size>>>(n_rows, n_cols, col_ids, row_ptr, data, x, y);
}

__global__ void
csr_spmv_vector_kernel (int num_rows,
    int num_cols,
    int* __restrict__ col_ids,
    int* __restrict__ row_ptr,
    double* __restrict__ data,
    double* __restrict__ x,
    double* y)
{
#if 1
    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    if( row < num_rows ){
        double dot = 0;
        int row_start = row_ptr [ row ];
        int row_end = row_ptr [ row +1];
        for (int jj = row_start ; jj < row_end ; jj ++)
        dot += data [ jj ] * x[ col_ids [ jj ]];
        y[ row ] = dot ;
    }
#elif 1
    __shared__ double sdata[BLOCK_SIZE + 16];                          // padded to avoid reduction ifs
    __shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
    
    const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
    const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
    const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
    const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
    const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

    for(int row = warp_id; row < num_rows; row += num_warps){
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[warp_lane][thread_lane] = row_ptr[row + thread_lane];
        const int row_start = ptrs[warp_lane][0];                   //same as: row_start = Ap[row];
        const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = Ap[row+1];

        // compute local sum
        double sum = 0;
        for(int jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
            sum += data[jj] * fetch_x<true>(col_ids[jj], x);

        // reduce local sums to row sum (ASSUME: warpsize 32)
        sdata[threadIdx.x] = sum;
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
       
        if (thread_lane == 0)
            y[row] = sdata[threadIdx.x];
    }
#else
    __shared__ double shared_elements[BLOCK_SIZE];

    const unsigned int id_in_row = threadIdx.x % WARP_SIZE;
    const unsigned int block_increment = blockDim.x * ((num_cols - 1) / (gridDim.x * blockDim.x) + 1);
    const unsigned int block_start = blockIdx.x * block_increment;
    const unsigned int block_stop  = min(block_start + block_increment, num_cols);

    for (unsigned int row = block_start + threadIdx.x / WARP_SIZE; row < block_stop; row += blockDim.x / WARP_SIZE) {
        double total = 0;
        unsigned int row_end = row_ptr[row + 1];
        for (unsigned int i = row_ptr[row] + id_in_row; i < row_end; i += WARP_SIZE) {
            total += data[i] * x[col_ids[i]];
        }

        shared_elements[threadIdx.x] = total;
        if (1  < WARP_SIZE) shared_elements[threadIdx.x] += shared_elements[threadIdx.x ^  1];
        if (2  < WARP_SIZE) shared_elements[threadIdx.x] += shared_elements[threadIdx.x ^  2];
        if (4  < WARP_SIZE) shared_elements[threadIdx.x] += shared_elements[threadIdx.x ^  4];
        if (8  < WARP_SIZE) shared_elements[threadIdx.x] += shared_elements[threadIdx.x ^  8];
        if (16 < WARP_SIZE) shared_elements[threadIdx.x] += shared_elements[threadIdx.x ^ 16];

        if (id_in_row == 0) {
            y[row] = shared_elements[threadIdx.x];
        }
    }
#endif
}