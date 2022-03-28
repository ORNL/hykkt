#ifndef MVO__H__
#define MVO__H__

void fun_adapt_diag_scale(int, int, double*, int*, int*, double*, int*,
   int*, double*, int*, int*, double*, double*, double*, double*);

__global__ void adapt_diag_scale(int, int, double*, int*, int*, double*, int*,
   int*, double*, int*, int*, double*, double*, double*, double*);

void fun_adapt_row_max(int, int, double*, int*, int*, double*, int*,
   int*, double*, int*, int*, double*);

__global__ void adapt_row_max(int, int, double*, int*, int*, double*, int*,
   int*, double*, int*, int*, double*);

void fun_add_const(int, int, int*);

__global__ void add_const(int, int, int*);

void fun_add_vecs(int, double*, double, double*);

__global__ void add_vecs(int, double*, double, double*);

void fun_mult_const(int, double, double*);

__global__ void mult_const(int, double, double*);

void fun_add_diag(int, double, int*, int*, double*);

__global__ void add_diag(int, double, int*, int*, double*);

void fun_inv_vec_scale(int, double*, double*);

__global__ void inv_vec_scale(int, double*, double*);

void fun_vec_scale(int, double*, double*);

__global__ void vec_scale(int, double*, double*);

void fun_concatenate(int, int, int, int, double*, int*, int*,
  double*, int*, int*, double*, int*, int*);

__global__ void concatenate(int, int, int, int, double*, int*, int*,
  double*, int*, int*, double*, int*, int*);

void fun_row_scale(int, double*, int*, int*, double*, double*,
    double*, double*);

__global__ void row_scale(int, double*, int*, int*, double*, double*,
    double*, double*);

void fun_diag_scale(int, int, double*, int*, int*, double*, int*,
  int*, double*, double*, double*, int);

__global__ void diag_scale(int, int, double*, int*, int*, double*, int*,
  int*, double*, double*, double*, int);

void fun_row_max(int, int, double*, int*, int*, double*, int*, int*,
   double* scale);

__global__ void row_max(int, int, double*, int*, int*, double*, int*, int*,
   double* scale);

#endif
