#pragma once

template <class T>
__device__ T warp_reduce (T val);

__global__ void csr_spmv_vector_kernel (
    int n_rows,
    int n_cols,
    int* col_ids,
    int* row_ptr,
    double* data,
    double* x,
    double* y);