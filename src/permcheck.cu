#include "permcheck_cuda.hpp"
#include "permcheck.hpp"

void fun_map_idx(int n, int* perm, double* old_val, double* new_val)
{
  int numBlocks;
  int blockSize = 512;
  numBlocks = (n + blockSize - 1) / blockSize;
  map_idx<<<numBlocks, blockSize>>>(n, perm, old_val, new_val);
}

__global__ void map_idx(int n, int* perm, double* old_val, double* new_val)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    new_val[i] = old_val[perm[i]];
  }
}

void selection_sort2(int len, int* arr1, int* arr2)
{
  int min_ind;
  int temp;
  for(int i = 0; i < len - 1; i++)
  {
    min_ind = i;
    for(int j = i + 1; j < len; j++)
    {
      if(arr1[j] < arr1[min_ind])
      {
        min_ind = j;
      }
    }
    if(i != min_ind)
    {
      temp          = arr1[i];
      arr1[i]       = arr1[min_ind];
      arr1[min_ind] = temp;
      temp          = arr2[i];
      arr2[i]       = arr2[min_ind];
      arr2[min_ind] = temp;
    }
  }
}

void make_vec_map_c(int n, 
    int* rows, 
    int* cols, 
    int* rev_perm, 
    int* perm_cols, 
    int* perm_map)
{
  int row_s;
  int rowlen;
  for(int i = 0; i < n; i++)
  {
    row_s  = rows[i];
    rowlen = rows[i + 1] - row_s;
    for(int j = 0; j < rowlen; j++)
    {
      perm_map[row_s + j]  = row_s + j;
      perm_cols[row_s + j] = rev_perm[cols[row_s + j]];
    }
    selection_sort2(rowlen, &perm_cols[row_s], &perm_map[row_s]);
  }
}

void reverse_perm(int n, int* perm, int* rev_perm)
{
  for(int i = 0; i < n; i++)
  {
    rev_perm[perm[i]] = i;
  }
}

void make_vec_map_r(int n, 
    int* rows, 
    int* cols, 
    int* perm, 
    int* perm_rows, 
    int* perm_cols, 
    int* perm_map)
{
  perm_rows[0] = 0;
  int count    = 0;
  int idx;
  int row_s;
  int rowlen;
  for(int i = 0; i < n; i++)
  {
    idx              = perm[i];
    row_s            = rows[idx];
    rowlen           = rows[idx + 1] - row_s;
    perm_rows[i + 1] = perm_rows[i] + rowlen;
    for(int j = 0; j < rowlen; j++)
    {
      perm_map[count + j]  = row_s + j;
      perm_cols[count + j] = cols[row_s + j];
    }
    count += rowlen;
  }
}

void make_vec_map_rc(int n, 
    int* rows, 
    int* cols, 
    int* perm, 
    int* rev_perm, 
    int* perm_rows,
    int* perm_cols, 
    int* perm_map)
{
  perm_rows[0] = 0;
  int count    = 0;
  int idx;
  int row_s;
  int rowlen;
  for(int i = 0; i < n; i++)
  {
    idx              = perm[i];
    row_s            = rows[idx];
    rowlen           = rows[idx + 1] - row_s;
    perm_rows[i + 1] = perm_rows[i] + rowlen;
    for(int j = 0; j < rowlen; j++)
    {
      perm_map[count + j]  = row_s + j;
      perm_cols[count + j] = rev_perm[cols[row_s + j]];
    }
    selection_sort2(rowlen, &perm_cols[count], &perm_map[count]);
    count += rowlen;
  }
}
