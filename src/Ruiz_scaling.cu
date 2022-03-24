#include <ruiz_scaling.hpp>
#define ruiz_its 2
/*
@brief: adds a constant to an array

@inputs: Length of array n, val - the value to be added,
arr - a pointer to the array the constant is added to

@outputs: arr with entries increased by val
 */
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
__global__ void mult_const(int n, double val, double* arr)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    arr[i] *= val;
  }
}
/*
@brief: Applies the inverse of a diagonal matrix (stored as a vector) on the left to a vector

@inputs: Size n of the matrix,
D_rhs a (dense) vector, Ds a dense vector represting a diagonal matrix

@outputs: D_rhs=D_rhs./Ds (elementwise)
 */
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
