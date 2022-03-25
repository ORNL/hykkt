/* written by SR based on a code by KS
         How to compile:
         nvcc -lcusparse -lcusolver -lcublas cuSolver_driver_chol.cu
 */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <ctype.h>
#include <string.h>
#include <cusolver_common.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include <algorithm>
#include "cusolverSp.h"
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverRf.h>
#include <iostream>
#include <memory>
#include <string>
#include <input_functions.hpp>

static int indexPlusValue_comp(const void* a, const void* b)
{
  const struct indexPlusValue* da = (indexPlusValue*)a;
  const struct indexPlusValue* db = (indexPlusValue*)b;

  return da->idx < db->idx ? -1 : da->idx > db->idx;
}

/*
Brief: reads a matrix stored in matrixFileName into a structure A of COO format
   */
void read_mm_file_into_coo(const char* matrixFileName, mmatrix* A, int lines)
{
  // this reads triangular matrix but expands into full as it goes (important)
  // lines indicates the number of the first useful line
  FILE* fpm = fopen(matrixFileName, "r");

  assert(fpm != NULL);
  char lineBuffer[256];
  for(int i=0; i<lines; i++) fgets(lineBuffer, sizeof(lineBuffer), fpm);
  // first line is size and nnz, need this info to allocate memory
  sscanf(lineBuffer, "%ld %ld %ld", &(A->n), &(A->m), &(A->nnz));
  // allocate
  printf("allocating COO structures %d %d %d\n", A->n, A->m, A->nnz);

  A->coo_vals = (double*)calloc(A->nnz, sizeof(double));
  A->coo_rows = (int*)calloc(A->nnz, sizeof(int));
  A->coo_cols = (int*)calloc(A->nnz, sizeof(int));

  // read
  int    r, c;
  double val;
  int    i = 0;
  while(fgets(lineBuffer, sizeof(lineBuffer), fpm) != NULL)
  {

    sscanf(lineBuffer, "%d %d %lf", &r, &c, &val);
    A->coo_rows[i] = r - 1;
    A->coo_cols[i] = c - 1;
    A->coo_vals[i] = val;
    i++;
  }
  fclose(fpm);
}
/*
Brief: Takes a matrix stored in implicit symmetric format COO in A
and fills out the entries and converts it to csr format within the structure
   */
void sym_coo_to_csr(mmatrix* A)
{
  // this is diffucult
  // first, decide how many nnz we have in each row
  int* nnz_counts;
  nnz_counts       = (int*)calloc(A->n, sizeof(int));
  int nnz_unpacked = 0;
  for(int i = 0; i < A->nnz; ++i)
  {
    nnz_counts[A->coo_rows[i]]++;
    nnz_unpacked++;
    if(A->coo_rows[i] != A->coo_cols[i])
    {
      nnz_counts[A->coo_cols[i]]++;
      nnz_unpacked++;
    }
  }
  // allocate full CSR structure
  A->nnz_unpacked     = nnz_unpacked;
  A->csr_vals         = (double*)calloc(A->nnz_unpacked, sizeof(double));
  A->csr_ja           = (int*)calloc(A->nnz_unpacked, sizeof(int));
  A->csr_ia           = (int*)calloc((A->n) + 1, sizeof(int));
  indexPlusValue* tmp = (indexPlusValue*)calloc(A->nnz_unpacked, sizeof(indexPlusValue));
  // create IA (row starts)
  A->csr_ia[0] = 0;
  for(int i = 1; i < A->n + 1; ++i)
  {
    A->csr_ia[i] = A->csr_ia[i - 1] + nnz_counts[i - 1];
  }

  int* nnz_shifts = (int*)calloc(A->n, sizeof(int));
  int  r, start;

  for(int i = 0; i < A->nnz; ++i)
  {
    // which row
    r     = A->coo_rows[i];
    start = A->csr_ia[r];
    if((start + nnz_shifts[r]) > A->nnz_unpacked)
      printf("index out of boubds\n");
    tmp[start + nnz_shifts[r]].idx   = A->coo_cols[i];
    tmp[start + nnz_shifts[r]].value = A->coo_vals[i];

    nnz_shifts[r]++;

    if(A->coo_rows[i] != A->coo_cols[i])
    {

      r     = A->coo_cols[i];
      start = A->csr_ia[r];

      if((start + nnz_shifts[r]) > A->nnz_unpacked)
        printf("index out of boubds 2\n");
      tmp[start + nnz_shifts[r]].idx   = A->coo_rows[i];
      tmp[start + nnz_shifts[r]].value = A->coo_vals[i];
      nnz_shifts[r]++;
    }
  }
  for(int i = 0; i < A->n; ++i)
  {
    int colStart = A->csr_ia[i];
    int colEnd   = A->csr_ia[i + 1];
    int length   = colEnd - colStart;

    qsort(&tmp[colStart], length, sizeof(indexPlusValue), indexPlusValue_comp);
  }
  for(int i = 0; i < A->nnz_unpacked; ++i)
  {
    A->csr_ja[i]   = tmp[i].idx;
    A->csr_vals[i] = tmp[i].value;
  }
  A->nnz = A->nnz_unpacked;
}

/*
Brief: Takes a matrix stored in COO format in A
and fills out the entries and converts it to csr format within the structure
   */
void coo_to_csr(mmatrix* A)
{
  // this is diffucult
  // first, decide how many nnz we have in each row
  int* nnz_counts;
  nnz_counts = (int*)calloc(A->n, sizeof(int));
  for(int i = 0; i < A->nnz; ++i)
  {
    nnz_counts[A->coo_rows[i]]++;
  }
  // allocate full CSR structure
  // A->csr_vals =(double*)  calloc(A->nnz, sizeof(double));
  // A->csr_ja =(int*)  calloc(A->nnz, sizeof(int));
  A->csr_ia           = (int*)calloc((A->n) + 1, sizeof(int));
  indexPlusValue* tmp = (indexPlusValue*)calloc(A->nnz, sizeof(indexPlusValue));
  // create IA (row starts)
  A->csr_ia[0] = 0;
  for(int i = 1; i < A->n + 1; ++i)
  {
    A->csr_ia[i] = A->csr_ia[i - 1] + nnz_counts[i - 1];
  }

  int* nnz_shifts = (int*)calloc(A->n, sizeof(int));
  int  r, start;

  for(int i = 0; i < A->nnz; ++i)
  {
    // which row
    r     = A->coo_rows[i];
    start = A->csr_ia[r];
    if((start + nnz_shifts[r]) > A->nnz)
      printf("index out of boubds\n");
    tmp[start + nnz_shifts[r]].idx   = A->coo_cols[i];
    tmp[start + nnz_shifts[r]].value = A->coo_vals[i];

    nnz_shifts[r]++;
  }
  // now sort whatever is inside rows

  for(int i = 0; i < A->n; ++i)
  {

    // now sorting (and adding 1)
    int colStart = A->csr_ia[i];
    int colEnd   = A->csr_ia[i + 1];
    int length   = colEnd - colStart;

    qsort(&tmp[colStart], length, sizeof(indexPlusValue), indexPlusValue_comp);
  }

  // and copy
  for(int i = 0; i < A->nnz; ++i)
  {
    A->coo_cols[i] = tmp[i].idx;
    A->coo_vals[i] = tmp[i].value;
  }
}

/*
Brief: reads a 1-index based permutation array stored in rhsFileName
into an array rhs that is 0-index based
   */
void read_1idx_perm(const char* rhsFileName, int* rhs)
{
  FILE* fpr = fopen(rhsFileName, "r");
  char  lineBuffer[256];
  fgets(lineBuffer, sizeof(lineBuffer), fpr);
  fgets(lineBuffer, sizeof(lineBuffer), fpr);
  int N, m;
  sscanf(lineBuffer, "%ld %ld", &N, &m);
  printf("N = %d, m=%d\n", N, m);
  int i = 0;
  int val;

  while(fgets(lineBuffer, sizeof(lineBuffer), fpr) != NULL)
  {
    sscanf(lineBuffer, "%ld", &val);
    rhs[i] = val - 1;
    i++;
  }
  fclose(fpr);
}
/*
Brief: reads vector stored in rhsFileName into an array rhs
   */
void read_rhs(const char* rhsFileName, double* rhs)
{
  FILE* fpr = fopen(rhsFileName, "r");
  char  lineBuffer[256];

  fgets(lineBuffer, sizeof(lineBuffer), fpr);
  fgets(lineBuffer, sizeof(lineBuffer), fpr);
  int N, m;
  sscanf(lineBuffer, "%ld %ld", &N, &m);
  int    i = 0;
  double val;
  // allocate

  while(fgets(lineBuffer, sizeof(lineBuffer), fpr) != NULL)
  {
    sscanf(lineBuffer, "%lf", &val);
    rhs[i] = val;
    // printf("%16.16f \n", val);
    i++;
  }
  fclose(fpr);
}
/*
Brief: Checks used and available GPU memory
   */
void checkGpuMem()
{
  size_t avail;
  size_t total;
  cudaMemGetInfo(&avail, &total);
  size_t used = total - avail;
  printf("Available memory of a : %zu\n", avail);
  printf("Total memory of a : %zu\n", total);
  printf("Used memory of a : %zu\n", used);
}
