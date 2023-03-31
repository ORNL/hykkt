#include "matrix_vector_ops_cuda.hpp"
#include "matrix_vector_ops.hpp"
#include "cuda_memory_utils.hpp"
#include <assert.h>
#include <stdio.h>
#include "cusparse_params.hpp"

#include "cuda_check_errors.hpp"


void SpMV_product_reuse(cusparseHandle_t handle,
    double alpha,
    cusparseSpMatDescr_t a_desc_sp,
    cusparseDnVecDescr_t b_desc_dn,
    double beta,
    cusparseDnVecDescr_t c_desc_dn,
    void** buffer,
    bool allocated)
{
    if(!allocated){
      size_t buffer_size = 0;
      fun_SpMV_buffer(handle,
          alpha,
          a_desc_sp,
          b_desc_dn,
          beta,
          c_desc_dn,
          &buffer_size);

      allocateBufferOnDevice(buffer, buffer_size);
    }
    fun_SpMV_product(handle,
        alpha,
        a_desc_sp,
        b_desc_dn,
        beta,
        c_desc_dn,
        *buffer);
}

void fun_SpMV_full(cusparseHandle_t handle, 
                   double alpha, 
                   cusparseSpMatDescr_t a_desc_sp, 
                   cusparseDnVecDescr_t b_desc_dn, 
                   double beta, 
                   cusparseDnVecDescr_t c_desc_dn)
{
  size_t buffer_size = 0;
  void* buffer = nullptr;
  fun_SpMV_buffer(handle, 
                  alpha, 
                  a_desc_sp, 
                  b_desc_dn, 
                  beta, 
                  c_desc_dn, 
                  &buffer_size);
  
  allocateBufferOnDevice(&buffer, buffer_size);  
  fun_SpMV_product(handle, 
                   alpha, 
                   a_desc_sp, 
                   b_desc_dn, 
                   beta, 
                   c_desc_dn, 
                   buffer);

  deleteOnDevice(buffer);
}

void fun_SpMV_buffer(cusparseHandle_t handle, 
                     double alpha, 
                     cusparseSpMatDescr_t a_desc_sp, 
                     cusparseDnVecDescr_t b_desc_dn, 
                     double beta, 
                     cusparseDnVecDescr_t c_desc_dn, 
                     size_t* buffer_size)
{
  checkCudaErrors(cusparseSpMV_bufferSize(handle, 
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                          &alpha, 
                                          a_desc_sp, 
                                          b_desc_dn,
                                          &beta, 
                                          c_desc_dn, 
                                          COMPUTE_TYPE, 
                                          CUSPARSE_MV_ALG_DEFAULT, 
                                          buffer_size));
}

void fun_SpMV_product(cusparseHandle_t handle, 
                      double alpha, 
                      cusparseSpMatDescr_t a_desc_sp, 
                      cusparseDnVecDescr_t b_desc_dn, 
                      double beta, 
                      cusparseDnVecDescr_t c_desc_dn, 
                      void* buffer)
{
  checkCudaErrors(cusparseSpMV(handle, 
                               CUSPARSE_OPERATION_NON_TRANSPOSE, 
                               &alpha, 
                               a_desc_sp, 
                               b_desc_dn,
                               &beta, 
                               c_desc_dn, 
                               COMPUTE_TYPE, 
                               CUSPARSE_MV_ALG_DEFAULT, 
                               buffer));
}

void fun_adapt_diag_scale(int n, 
                          int m, 
                          double* a_v, 
                          int* a_i, 
                          int* a_j,
                          double* b_v, 
                          int* b_i, 
                          int* b_j, 
                          double* bt_v, 
                          int* bt_i, 
                          int* bt_j, 
                          double* scale, 
                          double* d_rhs1, 
                          double* d_rhs2, 
                          double* max_d)
{
  int num_blocks;
  int block_size = 512;
  num_blocks = (m + block_size - 1) / block_size;
  adapt_diag_scale<<<num_blocks, block_size>>>(n,
                                               m, 
                                               a_v, 
                                               a_i, 
                                               a_j, 
                                               b_v, 
                                               b_i, 
                                               b_j,
                                               bt_v, 
                                               bt_i, 
                                               bt_j, 
                                               scale, 
                                               d_rhs1, 
                                               d_rhs2, 
                                               max_d);    
}

__global__ void adapt_diag_scale(int n, 
                                 int m, 
                                 double* a_v, 
                                 int* a_i, 
                                 int* a_j, 
                                 double* b_v,
                                 int* b_i, 
                                 int* b_j, 
                                 double* bt_v, 
                                 int* bt_i, 
                                 int* bt_j, 
                                 double* scale, 
                                 double* d_rhs1,
                                 double* d_rhs2, 
                                 double* max_d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j;
  if(i < n)
  {
    for(j = a_i[i]; j < a_i[i + 1]; j++)
    {
      a_v[j] *= scale[i] * scale[a_j[j]];
    }
    d_rhs1[i] *= scale[i];
    max_d[i] *= scale[i];
    for(j = bt_i[i]; j < bt_i[i + 1]; j++)
    {
      bt_v[j] *= scale[i] * scale[n + bt_j[j]];
    }
  }
  if(i >= n && i < m)
  {
    for(j = b_i[i - n]; j < b_i[i - n + 1]; j++)
    {
      b_v[j] *= scale[i] * scale[b_j[j]];
    }
    d_rhs2[i - n] *= scale[i];
    max_d[i] *= scale[i];
  }
}

void fun_adapt_row_max(int n, 
                       int m, 
                       double* a_v, 
                       int* a_i, 
                       int* a_j, 
                       double* b_v, 
                       int* b_i, 
                       int* b_j, 
                       double* bt_v, 
                       int* bt_i, 
                       int* bt_j, 
                       double* scale)
{
  int num_blocks;
  int block_size = 512;
  num_blocks = (m + block_size - 1) / block_size;
  adapt_row_max<<<num_blocks, block_size>>>(n,
                                            m, 
                                            a_v, 
                                            a_i, 
                                            a_j, 
                                            b_v, 
                                            b_i, 
                                            b_j,
                                            bt_v, 
                                            bt_i, 
                                            bt_j, 
                                            scale);    
}

__global__ void adapt_row_max(int n, int m, double* A_v, int* A_i, int* A_j,
    double* B_v, int* B_i, int* B_j, double* bt_v, int* bt_i, int* bt_j, 
    double* scale)
{
  double max_l = 0;
  double max_u = 0;
  int    i = blockIdx.x * blockDim.x + threadIdx.x;
  int    j;
  double entry;
  if(i < n)
  {
    for(j = A_i[i]; j < A_i[i + 1]; j++)
    {
      entry = fabs(A_v[j]);
      if(entry > max_l)
      {
        max_l = entry;
      }
    }
    for(j = bt_i[i]; j < bt_i[i + 1]; j++)
    {
      entry = fabs(bt_v[j]);
      if(entry > max_u)
      {
        max_u = entry;
      }
    }
    if(max_l > max_u)
    {
      scale[i] = 1.0 / sqrt(max_l);
    }
    else
    {
      scale[i] = 1.0 / sqrt(max_u);
    }
  }
  if(i >= n && i < m)
  {
    for(j = B_i[i - n]; j < B_i[i - n + 1]; j++)
    {
      entry = fabs(B_v[j]);
      if(entry > max_l)
      {
        max_l = entry;
      }
    }
    scale[i] = 1.0 / sqrt(max_l);
  }
}

void fun_set_const(int n, double val, double* arr)
{
  int num_blocks;
  int block_size = 512;
  num_blocks = (n + block_size - 1) / block_size;
  set_const<<<num_blocks, block_size>>>(n, val, arr);
}

__global__ void set_const(int n, double val, double* arr)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    arr[i] = val;
  }
}

void fun_add_const(int n, int val, int* arr)
{
  int num_blocks;
  int block_size = 512;
  num_blocks = (n + block_size - 1) / block_size;
  add_const<<<num_blocks, block_size>>>(n, val, arr);
}

__global__ void add_const(int n, int val, int* arr)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    arr[i] += val;
  }
}

void fun_add_vecs(int n, double* arr1, double alp, double* arr2)
{
  int num_blocks;
  int block_size = 512;
  num_blocks = (n + block_size - 1) / block_size;
  add_vecs<<<num_blocks, block_size>>>(n, arr1, alp, arr2);
}

__global__ void add_vecs(int n, double* arr1, double alp, double* arr2)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    arr1[i] += alp * arr2[i];
  }
}

void fun_mult_const(int n, double val, double* arr)
{
  int num_blocks;
  int block_size = 512;
  num_blocks = (n + block_size - 1) / block_size;
  mult_const<<<num_blocks, block_size>>>(n, val, arr);
}

__global__ void mult_const(int n, double val, double* arr)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    arr[i] *= val;
  }
}

void fun_add_diag(int n, double val, int* a_i, int* a_j, double* a_v)
{
  int num_blocks;
  int block_size = 512;
  num_blocks = (n + block_size - 1) / block_size;
  add_diag<<<num_blocks, block_size>>>(n, val, a_i, a_j, a_v);
}

__global__ void add_diag(int n, double val, int* a_i, int* a_j, double* a_v)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    for(int j = a_i[i]; j < a_i[i + 1]; j++)
    {
      if(i == a_j[j])
      {
        a_v[j] += val;
        break;
      }
    }
  }
}

void fun_inv_vec_scale(int n, double* d_rhs, double* ds)
{
  int num_blocks;
  int block_size = 512;
  num_blocks = (n + block_size - 1) / block_size;
  inv_vec_scale<<<num_blocks, block_size>>>(n, d_rhs, ds);
}

__global__ void inv_vec_scale(int n, double* d_rhs, double* ds)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    d_rhs[i] /= ds[i];
  }
}

void fun_vec_scale(int n, double* d_rhs, double* ds)
{
  int num_blocks;
  int block_size = 512;
  num_blocks = (n + block_size - 1) / block_size;
  vec_scale<<<num_blocks, block_size>>>(n, d_rhs, ds);
}

__global__ void vec_scale(int n, double* d_rhs, double* ds)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    d_rhs[i] *= ds[i];
  }
}

void fun_concatenate(int n, 
                     int m, 
                     int nnz_a, 
                     int nnz_b, 
                     double* a_v, 
                     int* a_i, 
                     int* a_j, 
                     double* b_v, 
                     int* b_i, 
                     int* b_j, 
                     double* c_v, 
                     int* c_i, 
                     int* c_j)
{
  int num_blocks;
  int block_size = 512;
  num_blocks = (n + m + block_size - 1) / block_size;
  concatenate<<<num_blocks, block_size>>>(n, 
                                          m, 
                                          nnz_a, 
                                          nnz_b, 
                                          a_v, 
                                          a_i, 
                                          a_j, 
                                          b_v,
                                          b_i, 
                                          b_j, 
                                          c_v, 
                                          c_i, 
                                          c_j);
}

__global__ void concatenate(int n, 
                            int m, 
                            int nnz_a, 
                            int nnz_b, 
                            double* a_v, 
                            int* a_i, 
                            int* a_j,
                            double* b_v, 
                            int* b_i, 
                            int* b_j, 
                            double* c_v, 
                            int* c_i, 
                            int* c_j)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {   
    /* Matrix A is copied as is 
       except last entry of row starts to signal end of matrix) */

    for(int j = a_i[i]; j < a_i[i + 1]; j++)
    {
      c_v[j] = a_v[j];
      c_j[j] = a_j[j];
    }
    c_i[i] = a_i[i];
  }
  if(i >= n && i < n + m)
  {   
    // Matrix b is copied after, with modified row starts
    
    for(int j = b_i[i - n]; j < b_i[i - n + 1]; j++)
    {
      c_v[j + nnz_a] = b_v[j];
      c_j[j + nnz_a] = b_j[j];
    }
    c_i[i] = b_i[i - n] + nnz_a;
  }
  if(i == (n + m))
    c_i[i] = nnz_a + nnz_b;   // end of matrix
}

void fun_row_scale(int n, 
                   double* a_v, 
                   int* a_i, 
                   int* a_j, 
                   double* a_vs, 
                   double* d_rhs, 
                   double* d_rhs_s, 
                   double* ds)
{
  int num_blocks;
  int block_size = 512;
  num_blocks = (n + 1 + block_size - 1) / block_size;
  row_scale<<<num_blocks, block_size>>>(n, 
                                        a_v, 
                                        a_i, 
                                        a_j, 
                                        a_vs, 
                                        d_rhs, 
                                        d_rhs_s, 
                                        ds);
}

__global__ void row_scale(int n, 
                          double* a_v, 
                          int* a_i, 
                          int* a_j, 
                          double* a_vs, 
                          double* d_rhs, 
                          double* d_rhs_s, 
                          double* ds)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j;
  if(i < n)
  {
    for(j = a_i[i]; j < a_i[i + 1]; j++)
    {
      a_vs[j] = a_v[j] * ds[i];
    }
    d_rhs_s[i] = d_rhs[i] * ds[i];
  }
}

void fun_diag_scale(int n, 
                    int m, 
                    double* a_v, 
                    int* a_i, 
                    int* a_j, 
                    double* at_v,
                    int* at_i,
                    int* at_j, 
                    double* scale, 
                    double* d_rhs, 
                    double* max_d, 
                    int flag)
{
  int num_blocks;
  int block_size = 512;
  num_blocks = (n + m + block_size - 1) / block_size;
  diag_scale<<<num_blocks, block_size>>>(n, 
                                         m, 
                                         a_v, 
                                         a_i, 
                                         a_j, 
                                         at_v, 
                                         at_i, 
                                         at_j, 
                                         scale, 
                                         d_rhs, 
                                         max_d, 
                                         flag);
}

__global__ void diag_scale(int n, 
                           int m, 
                           double* a_v, 
                           int* a_i, 
                           int* a_j, 
                           double* at_v, 
                           int* at_i,
                           int* at_j, 
                           double* scale, 
                           double* d_rhs, 
                           double* max_d, 
                           int flag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j;
  if(i < n && flag)
  {
    for(j = at_i[i]; j < at_i[i + 1]; j++)
    {
      at_v[j] *= scale[i] * scale[n + at_j[j]];
    }
  }
  if(i < m)
  {
    for(j = a_i[i]; j < a_i[i + 1]; j++)
    {
      a_v[j] *= scale[i] * scale[a_j[j]];
    }
    d_rhs[i] *= scale[i];
    max_d[i] *= scale[i];
  }
}

void fun_row_max(int n, 
                 int m, 
                 double* a_v, 
                 int* a_i, 
                 int* a_j, 
                 double* at_v, 
                 int* at_i, 
                 int* at_j, 
                 double* scale)
{
  int num_blocks;
  int block_size = 512;
  num_blocks = (n + m + block_size - 1) / block_size;
  row_max<<<num_blocks, block_size>>>(n, 
                                      m, 
                                      a_v, 
                                      a_i, 
                                      a_j, 
                                      at_v, 
                                      at_i, 
                                      at_j, 
                                      scale);
}

__global__ void row_max(int n, 
                        int m, 
                        double* a_v, 
                        int* a_i, 
                        int* a_j, 
                        double* at_v, 
                        int* at_i, 
                        int* at_j, 
                        double* scale)
{
  double max_l = 0;
  double max_u = 0;
  int    i = blockIdx.x * blockDim.x + threadIdx.x;
  int    j;
  double entry;
  if(i < n)
  {
    for(j = a_i[i]; j < a_i[i + 1]; j++)
    {
      entry = fabs(a_v[j]);
      if(entry > max_l)
      {
        max_l = entry;
      }
    }
    for(j = at_i[i]; j < at_i[i + 1]; j++)
    {
      entry = fabs(at_v[j]);
      if(entry > max_u)
      {
        max_u = entry;
      }
    }
    if(max_l > max_u)
    {
      scale[i] = 1.0 / sqrt(max_l);
    }
    else
    {
      scale[i] = 1.0 / sqrt(max_u);
    }
  }
  if(i >= n && i < m)
  {
    for(j = a_i[i]; j < a_i[i + 1]; j++)
    {
      entry = fabs(a_v[j]);
      if(entry > max_l)
      {
        max_l = entry;
      }
    }
    scale[i] = 1.0 / sqrt(max_l);
  }
}
