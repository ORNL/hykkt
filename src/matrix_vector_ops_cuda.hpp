#pragma once

#include <cusolver_common.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include "cusolverSp.h"
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverRf.h>

template <typename T>
void check(T result, 
           char const *const func, 
           const char *const file,
           int const line) 
{
  if (result) {
    printf("CUDA error at %s:%d, error# %d\n", file, line, result);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

//***************************************************************************//
//****See matrix_vector_ops.hpp for kernel wrapper function documentation****//
//***************************************************************************//

__global__ void adapt_diag_scale(int, int, double*, int*, int*, double*, int*,
   int*, double*, int*, int*, double*, double*, double*, double*);

__global__ void adapt_row_max(int, int, double*, int*, int*, double*, int*,
   int*, double*, int*, int*, double*);

__global__ void add_const(int, int, int*);

__global__ void set_const(int n, double val, double* arr);

  __global__ void add_vecs(int, double*, double, double*);

__global__ void mult_const(int, double, double*);

__global__ void add_diag(int, double, int*, int*, double*);

__global__ void inv_vec_scale(int, double*, double*);

__global__ void vec_scale(int, double*, double*);

__global__ void concatenate(int, int, int, int, double*, int*, int*,
  double*, int*, int*, double*, int*, int*);

__global__ void row_scale(int, double*, int*, int*, double*, double*,
    double*, double*);

__global__ void diag_scale(int, int, double*, int*, int*, double*, int*,
  int*, double*, double*, double*, int);

__global__ void row_max(int, int, double*, int*, int*, double*, int*, int*,
   double* scale);

