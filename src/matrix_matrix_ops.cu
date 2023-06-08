#include "matrix_matrix_ops_cuda.hpp"
#include "matrix_matrix_ops.hpp"
#include "constants.hpp"

void fun_q_sparse_product(int n, 
int q_nnz, 
int* q_i, 
int* q_j, 
double* q_v, 
int a_nnz, 
int* a_t_i, 
int* a_t_j, 
double* a_t_v, 
double* h_v, 
double* out) 
{
    int num_blocks;
    int block_size = BLOCK_SIZE;
    num_blocks = (n + block_size - 1) / block_size;
    q_sparse_product<<<num_blocks, block_size>>>(n, 
      q_nnz, 
      q_i, 
      q_j, 
      q_v, 
      a_nnz, 
      a_t_i, 
      a_t_j, 
      a_t_v, 
      h_v, 
      out);
}

__global__ void q_sparse_product(int n,
int q_nnz, 
int* q_i, 
int* q_j, 
double* q_v, 
int a_nnz, 
int* a_i, 
int* a_j, 
double* a_v, 
double* h_v, 
double* out) 
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int ptr1;
  int ptr2;
  int start, end;
  int k;
  int row_offset;
  double total = 0.0;
  #pragma unroll
  for (int i = index; i < n; i+= stride) {
    start = q_i[i]; //first index in q_j array for row i
    end = q_i[i + 1]; //last index + 1 in q_j array for row i
    row_offset = 0; //initially no offset
    for (int j = start; j < end; j++) { //indeces of column pointer of nonzero row i elements
      k = q_j[j]; //column k for out_i,k
      total = 0;
      ptr1 = a_i[i]; //pointer for a_i,j
      ptr2 = a_i[k]; //pointer for a_k,j
      while (ptr1 < a_i[i + 1] && ptr2 < a_i[k + 1]) {
        if (a_j[ptr1] == a_j[ptr2]) {
          total += a_v[ptr1] * a_v[ptr2] * h_v[a_j[ptr1]]; //a_i,j * a_k,j * h_j,j
          ptr1++;
          ptr2++;
        }
        else if (a_j[ptr1] < a_j[ptr2]) {
          ptr1++;
        }
        else {
          ptr2++;
        }
      }
      double val = q_v[q_i[i] + row_offset] + total;
      out[q_i[i] + row_offset] = val; //out_i,k
      row_offset++;
    }
  }
}

void fun_inv_diagonal_product(int n, int* a_i, int* a_j, double* a_v, double* h_v, double* out) {
  int num_blocks;
  int block_size = BLOCK_SIZE;
  num_blocks = (n + block_size - 1) / block_size;
  inv_diagonal_product<<<num_blocks, block_size>>>(n, 
      a_i, 
      a_j, 
      a_v, 
      h_v, 
      out);
}

__global__ void inv_diagonal_product(int n, int* a_i, int* a_j, double* a_v, double* h_v, double* out) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  #pragma unroll
  for (int i = index; i < n; i+= stride) {
    int start = a_i[i];
    int end = a_i[i + 1];
    double total = 0.0;
    for (int j = start; j < end; j++) {
      total += a_v[j] * a_v[j] * h_v[a_j[j]];
    }
    out[i] = total != 0.0 ? 1.0 / total : 0.0;
  }
}