#ifndef INPUT__H__
#define INPUT__H__

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

struct indexPlusValue
{
  double value;
  int    idx;
};

static int indexPlusValue_comp(const void*, const void*);

typedef struct
{
  int*    coo_rows;
  int*    coo_cols;
  double* coo_vals;

  int*    csr_ia;
  int*    csr_ja;
  double* csr_vals;

  int n;
  int m;
  int nnz;
  int nnz_unpacked;
} mmatrix;

void read_mm_file_into_coo(const char*, mmatrix*, int);

void sym_coo_to_csr(mmatrix*);

void coo_to_csr(mmatrix*);

void read_1idx_perm(const char*, int*);

void read_rhs(const char*, double*);

void checkGpuMem();

#endif
