#pragma once

__device__ void cg_solve_kernel(int itmax, 
    double tol, 
    int n, 
    int* a_i, 
    int* a_j, 
    double* a_v, 
    int* a_t_i,
    int* a_t_j, 
    double* a_t_v, 
    double* b, 
    double* c,
    double* x, 
    double* r, 
    double* z, 
    double* p, 
    double* A_p);