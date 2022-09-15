#include <ruiz_scaling.hpp>
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
__global__ void adapt_row_max(int n, int m, double* A_v, int* A_i, int* A_j, double* B_v, int* B_i,
  int* B_j, double* Bt_v, int* Bt_i, int* Bt_j, double* scale)
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
