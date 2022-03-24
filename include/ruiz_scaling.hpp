#ifndef RUIZ__H__
#define RUIZ__H__

__global__ void adapt_diag_scale(int, int, double*, int*, int*, double*, int*, int*, double*, int*, int*, double*, double*, double*, double*);

__global__ void adapt_row_max(int, int, double*, int*, int*, double*, int*, int*, double*, int*, int*, double*);

__global__ void add_const(int, int, int*);

__global__ void add_vecs(int, double*, double, double*);

#endif
