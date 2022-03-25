#ifndef RUIZ__H__
#define RUIZ__H__

__global__ void adapt_diag_scale(int, int, double*, int*, int*, double*, int*,
   int*, double*, int*, int*, double*, double*, double*, double*);

__global__ void adapt_row_max(int, int, double*, int*, int*, double*, int*,
   int*, double*, int*, int*, double*);

__global__ void add_const(int, int, int*);

__global__ void add_vecs(int, double*, double, double*);

__global__ void mult_const(int, double, double*);

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

#endif
