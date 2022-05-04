#include <matrix_vector_ops.hpp>

/*
@brief: wrapper for CUDA matrix-vector product

@inputs: matrix A, vectors b and c, scalars alpha and beta

@outputs: c = alpha*Ab+beta*c
 */
void fun_SpMV(double alpha, cusparseSpMatDescr_t A, cusparseDnVecDescr_t b,
    double beta, cusparseDnVecDescr_t c){
  double zero=0.0;
  cusparseHandle_t handle            = NULL;
  cusparseCreate(&handle);
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, b,
     &beta, c, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &zero);
}
/*
@brief: diagonally scales a matrix from the left and right, and
diagonally scales rhs (from the left)

@inputs: Size n of the matrix, the lower half of the matrix stored in csr format,
the upper half of the matrix stored in csr format,
a scaling vector representing a diagonal matrix,
a rhs (vector) of the equation, max_d a vector that aggregates scaling,
a flag to determine whether to scale the second matrix (not necessary in last iteration)

@outputs: The value arrays of the matrices (A_v, At_v) are scaled along with
D_rhs the rhs vector. max_d is updated to include the aggregate scaling
 */

void fun_adapt_diag_scale(int n, int m, double* A_v, int* A_i, int* A_j,
    double* B_v, int* B_i, int* B_j, double* Bt_v, int* Bt_i, int* Bt_j, 
    double* scale, double* D_rhs1, double* D_rhs2, double* max_d)
{
  int numBlocks, blockSize=512;
  numBlocks = (m + blockSize - 1) / blockSize;
  adapt_diag_scale<<<numBlocks, blockSize>>>(n,m, A_v, A_i, A_j, B_v, B_i, B_j,
      Bt_v, Bt_i, Bt_j, scale, D_rhs1, D_rhs2, max_d);    
}
__global__ void adapt_diag_scale(int n, int m, double* A_v, int* A_i, int* A_j, double* B_v,
  int* B_i, int* B_j, double* Bt_v, int* Bt_i, int* Bt_j, double* scale, double* D_rhs1,
  double* D_rhs2, double* max_d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j;
  if(i < n)
  {
    for(j = A_i[i]; j < A_i[i + 1]; j++)
    {
      A_v[j] *= scale[i] * scale[A_j[j]];
    }
    D_rhs1[i] *= scale[i];
    max_d[i] *= scale[i];
    for(j = Bt_i[i]; j < Bt_i[i + 1]; j++)
    {
      Bt_v[j] *= scale[i] * scale[n + Bt_j[j]];
    }
  }
  if(i >= n && i < m)
  {
    for(j = B_i[i - n]; j < B_i[i - n + 1]; j++)
    {
      B_v[j] *= scale[i] * scale[B_j[j]];
    }
    D_rhs2[i - n] *= scale[i];
    max_d[i] *= scale[i];
  }
}

/*
@brief: Determines the correct scaling
(corresponding to one iteration of Ruiz scaling)
to be used by the adapt_diag_scale kernel on matrix (1) with form [A B^T; B 0]

@inputs: Size n of A, m - total rows in matrix (1), matrices A, B, B^T, in
csr format (works also if the third matrix is not B^T provided dimensions
are correct), an empty scaling vector representing a diagonal matrix,

@outputs: The scaling vector scale which is updated entry-wise with
1/sqrt(the maximum of each row) of matrix (1)
 */
void fun_adapt_row_max(int n, int m, double* A_v, int* A_i, int* A_j, 
    double* B_v, int* B_i, int* B_j, double* Bt_v, int* Bt_i, int* Bt_j, 
    double* scale)
{
  int numBlocks, blockSize=512;
  numBlocks = (m + blockSize - 1) / blockSize;
  adapt_row_max<<<numBlocks, blockSize>>>(n,m, A_v, A_i, A_j, B_v, B_i, B_j,
      Bt_v, Bt_i, Bt_j, scale);    
}
__global__ void adapt_row_max(int n, int m, double* A_v, int* A_i, int* A_j,
    double* B_v, int* B_i, int* B_j, double* Bt_v, int* Bt_i, int* Bt_j, 
    double* scale)
{
  double max_l = 0, max_u = 0;
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
    for(j = Bt_i[i]; j < Bt_i[i + 1]; j++)
    {
      entry = fabs(Bt_v[j]);
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


/*
@brief: adds a constant to an array

@inputs: Length of array n, val - the value to be added,
arr - a pointer to the array the constant is added to

@outputs: arr with entries increased by val
 */
void fun_add_const(int n, int val, int* arr)
{
  int numBlocks, blockSize=512;
  numBlocks = (n + blockSize - 1) / blockSize;
  add_const<<<numBlocks, blockSize>>>(n,val,arr);
}
__global__ void add_const(int n, int val, int* arr)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    arr[i] += val;
  }
}
/*
@brief: add arrays arr1, arr2 such that arr1 = arr1+ alp*arr2

@inputs: Length of array n, arr1, arr2 - arrays to be added,
alp - scaling constant

@outputs: arr1 += alp*arr2
 */
void fun_add_vecs(int n, double* arr1, double alp, double* arr2)
{
  int numBlocks, blockSize=512;
  numBlocks = (n + blockSize - 1) / blockSize;
  add_vecs<<<numBlocks, blockSize>>>(n, arr1, alp, arr2);
}
__global__ void add_vecs(int n, double* arr1, double alp, double* arr2)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    arr1[i] += alp * arr2[i];
  }
}
/*
@brief: multiplies an array by a constant

@inputs: Length of array n, val - the value to multiply,
arr - a pointer to the array the constant is added to

@outputs: Each entry in arr is scaled by val
 */
void fun_mult_const(int n, double val, double* arr)
{
  int numBlocks, blockSize=512;
  numBlocks = (n + blockSize - 1) / blockSize;
  mult_const<<<numBlocks, blockSize>>>(n,val,arr);
}
__global__ void mult_const(int n, double val, double* arr)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    arr[i] *= val;
  }
}
/*
@brief: adds a multiple of I to a CSR matrix A

@inputs: Length of array n, val - the value to add,
and A_i, A_j, A_v - pointers for rows, columns and values

@outputs: A[i][i]+=val for all i
 */
void fun_add_diag(int n, double val, int* A_i, int* A_j, double* A_v)
{
  int numBlocks, blockSize=512;
  numBlocks = (n + blockSize - 1) / blockSize;
  add_diag<<<numBlocks, blockSize>>>(n,val,A_i, A_j, A_v);
}
__global__ void add_diag(int n, double val, int* A_i, int* A_j, double* A_v)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    for(int j = A_i[i]; j < A_i[i + 1]; j++)
    {
      if(i == A_j[j])
      {
        A_v[j]+=val;
        break;
      }
    }
  }
}
/*
@brief: Applies the inverse of a diagonal matrix (stored as a vector) on the left to a vector

@inputs: Size n of the matrix,
D_rhs a (dense) vector, Ds a dense vector represting a diagonal matrix

@outputs: D_rhs=D_rhs./Ds (elementwise)
 */
void fun_inv_vec_scale(int n, double* D_rhs, double* Ds)
{
  int numBlocks, blockSize=512;
  numBlocks = (n + blockSize - 1) / blockSize;
  inv_vec_scale<<<numBlocks, blockSize>>>(n, D_rhs, Ds);
}

__global__ void inv_vec_scale(int n, double* D_rhs, double* Ds)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    D_rhs[i] /= Ds[i];
  }
}
/*
@brief: Applies a diagonal matrix (stored as a vector) on the left to a vector

@inputs: Size n of the matrix,
D_rhs a (dense) vector, Ds a dense vector represting a diagonal matrix

@outputs: D_rhs=Ds*D_rhs (elementwise)
 */
void fun_vec_scale(int n, double* D_rhs, double* Ds)
{
  int numBlocks, blockSize=512;
  numBlocks = (n + blockSize - 1) / blockSize;
  vec_scale<<<numBlocks, blockSize>>>(n, D_rhs, Ds);
}

__global__ void vec_scale(int n, double* D_rhs, double* Ds)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    D_rhs[i] *= Ds[i];
  }
}
/*
@brief: concatenates 2 matrices into a third matrix (one under the other)

@inputs: Row count n the matrix A and row count m of matrix B,
number of non zeros (nnz) of matrices A and B
matrices A and B in CSR format, an empty matrix C to be overwritten

@outputs: Matrix C in CSR format [A' B']'
 */
void fun_concatenate(int n, int m, int nnzA, int nnzB, double* A_v, int* A_i, 
    int* A_j, double* B_v, int* B_i, int* B_j, double* C_v, int* C_i, int* C_j)
{
  int numBlocks, blockSize=512;
  numBlocks = (n + m + blockSize - 1) / blockSize;
  concatenate<<<numBlocks, blockSize>>>(n, m, nnzA, nnzB, A_v, A_i, A_j, B_v,
      B_i, B_j, C_v, C_i, C_j);
}
__global__ void concatenate(int n, int m, int nnzA, int nnzB, double* A_v, int* A_i, int* A_j,
  double* B_v, int* B_i, int* B_j, double* C_v, int* C_i, int* C_j)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {   // Matrix A is copied as is (except last entry of row starts to signal end of matrix)
    for(int j = A_i[i]; j < A_i[i + 1]; j++)
    {
      C_v[j] = A_v[j];
      C_j[j] = A_j[j];
    }
    C_i[i] = A_i[i];
  }
  if(i >= n && i < n + m)
  {   // Matrix b is copied after, with modified row starts
    for(int j = B_i[i - n]; j < B_i[i - n + 1]; j++)
    {
      C_v[j + nnzA] = B_v[j];
      C_j[j + nnzA] = B_j[j];
    }
    C_i[i] = B_i[i - n] + nnzA;
  }
  if(i == (n + m))
    C_i[i] = nnzA + nnzB;   // end of matrix
}

/*
@brief: Applies a diagonal matrix (stored as a vector) on the
left to a matrix and a vector and stores the result in separate arrays

@inputs: Size n of the matrix,
Matrix A in csr storage format,
D_rhs a (dense) vector, Ds a dense vector represting a diagonal matrix

@outputs: The value array of the matrix (A_v) is scaled along with D_rhs.
 */
void fun_row_scale(
  int n, double* A_v, int* A_i, int* A_j, double* A_vs, double* D_rhs, double* D_rhs_s, double* Ds)
{
  int numBlocks, blockSize=512;
  numBlocks = (n + 1 + blockSize - 1) / blockSize;
  row_scale<<<numBlocks, blockSize>>>(n, A_v, A_i, A_j, A_vs, D_rhs, D_rhs_s, Ds);
}
__global__ void row_scale(
  int n, double* A_v, int* A_i, int* A_j, double* A_vs, double* D_rhs, double* D_rhs_s, double* Ds)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j;
  if(i < n)
  {
    for(j = A_i[i]; j < A_i[i + 1]; j++)
    {
      A_vs[j] = A_v[j] * Ds[i];
    }
    D_rhs_s[i] = D_rhs[i] * Ds[i];
  }
}
/*
@brief: diagonally scales a matrix from the left and right, and
diagonally scales rhs (from the left)

@inputs: Size n of the matrix, the lower half of the matrix stored in csr format,
the upper half of the matrix stored in csr format,
a scaling vector representing a diagonal matrix,
a rhs (vector) of the equation, max_d a vector that aggregates scaling,
a flag to determine whether to scale the second matrix (not necessary in last iteration)

@outputs: The value arrays of the matrices (A_v, At_v) are scaled along with
D_rhs the rhs vector. max_d is updated to include the aggregate scaling
 */
void fun_diag_scale(int n, int m, double* A_v, int* A_i, int* A_j, double* At_v,
  int* At_i,int* At_j, double* scale, double* D_rhs, double* max_d, int flag)
{
  int numBlocks, blockSize=512;
  numBlocks = (n + m + blockSize - 1) / blockSize;
  diag_scale<<<numBlocks, blockSize>>>(n, m, A_v, A_i, A_j, At_v, At_i, At_j, 
      scale, D_rhs, max_d, flag);
}
__global__ void diag_scale(int n, int m, double* A_v, int* A_i, int* A_j, double* At_v, int* At_i,
  int* At_j, double* scale, double* D_rhs, double* max_d, int flag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j;
  if(i < n && flag)
  {
    for(j = At_i[i]; j < At_i[i + 1]; j++)
    {
      At_v[j] *= scale[i] * scale[n + At_j[j]];
    }
  }
  if(i < m)
  {
    for(j = A_i[i]; j < A_i[i + 1]; j++)
    {
      A_v[j] *= scale[i] * scale[A_j[j]];
    }
    D_rhs[i] *= scale[i];
    max_d[i] *= scale[i];
  }
}

/*
@brief: Determines the correct scaling
(corresponding to one iteration of Ruiz scaling)
to be used by the diag_scale kernel

@inputs: Size n of the matrix, the lower half of the matrix stored in csr format,
the upper half of the matrix stored in csr format,
a scaling vector representing a diagonal matrix,

@outputs: The scaling vector scale which is updated entry-wise with
1/sqrt(the maximum of each row)
 */
void fun_row_max(int n, int m, double* A_v, int* A_i, int* A_j, double* At_v, 
  int* At_i, int* At_j, double* scale)
{
  int numBlocks, blockSize=512;
  numBlocks = (n + m + blockSize - 1) / blockSize;
  row_max<<<numBlocks, blockSize>>>(n, m, A_v, A_i, A_j, At_v, At_i, At_j, scale);
}
__global__ void row_max(
  int n, int m, double* A_v, int* A_i, int* A_j, double* At_v, int* At_i, int* At_j, double* scale)
{
  double max_l = 0, max_u = 0;
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
    for(j = At_i[i]; j < At_i[i + 1]; j++)
    {
      entry = fabs(At_v[j]);
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
    for(j = A_i[i]; j < A_i[i + 1]; j++)
    {
      entry = fabs(A_v[j]);
      if(entry > max_l)
      {
        max_l = entry;
      }
    }
    scale[i] = 1.0 / sqrt(max_l);
  }
}
