#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

//***************************************************************************//
//****See matrix_vector_ops.hpp for kernel wrapper function documentation****//
//***************************************************************************//

__global__ void adapt_diag_scale(int, int, double*, int*, int*, double*, int*,
   int*, double*, int*, int*, double*, double*, double*, double*);

__global__ void adapt_row_max(int, int, double*, int*, int*, double*, int*,
   int*, double*, int*, int*, double*);

__global__ void add_const(int, int, int*);

__global__ void set_const(int n, double val, double* arr);

__global__ void add_vecs_scaled(int n, double alpha, double beta, double* arr1, double* arr2);

__global__ void add_vecs_scaled(int n, double* alpha, double* arr1, double* arr2);

__global__ void add_vecs(int, double*, double, double*);

__global__ void add_vecs2(int n, double alpha, double* a1, double* b1, double beta, double* a2, double* b2);

__global__ void add_vecs(int, double, double, double*, double*, double*);

__global__ void add_vecs(int n, double* arr1, double* alpha, double* arr2);

__global__ void sub_vecs(int n, double* arr1, double* alpha, double* arr2);

__global__ void cg_helper1(int n, double* r_dot_z, double* p_Ap, double* x, double* r, double* p, double* A_p);

__global__ void cg_helper2(int n, double* d_r_dot_z, double* d_r_dot_z_prev, double* p_, double* z_);

__global__ void mult_const(int, double, double*);

__global__ void add_diag(int, double, int*, int*, double*);

__global__ void inv_vec_scale(int, double*, double*);

__global__ void vec_scale(int, double*, double*);

__global__ void vec_scale(int, double*, double*, double*);

__global__ void concatenate(int, int, int, int, double*, int*, int*,
  double*, int*, int*, double*, int*, int*);

__global__ void row_scale(int, double*, int*, int*, double*, double*,
    double*, double*);

__global__ void diag_scale(int, int, double*, int*, int*, double*, int*,
  int*, double*, double*, double*, int);

__global__ void row_max(int, int, double*, int*, int*, double*, int*, int*,
   double* scale);

