#pragma once

__global__ void lb_objective(int m, int n, double t, double* x, double* b, double* c, double* Q_x, double* A_x, double* out);
__global__ void lb_objective(int m, int n, double t, double* x, double* b, double* c, double* A_x, double* out);
__global__ void lb_gradient(int n, double t, double scale, double* b, double* c, double* Q_x, double* A_x, int* a_t_i, int* a_t_j, double* a_t_v, double* out);
__global__ void lb_gradient(int n, double t, double scale, double* b, double* c, double* A_x, int* a_t_i, int* a_t_j, double* a_t_v, double* out);
__global__ void lb_hessian(int m, double* b, double* A_x, double* out);