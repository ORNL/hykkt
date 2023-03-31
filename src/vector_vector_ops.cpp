#include <cstdio>

#include "vector_vector_ops.hpp"
#include "matrix_vector_ops_cuda.hpp"
#include <cublas_v2.h>

#include "cuda_check_errors.hpp"


void sumVectors(cublasHandle_t& handle_cublas,
                int n,
                const double* x,
                double* y,
                const double* alpha)
{
  checkCudaErrors(cublasDaxpy(handle_cublas,
                              n,
                              alpha,
                              x, 1,
                              y, 1));
}

void dotProduct(cublasHandle_t& handle_cublas,
                int n,
                const double* x,
                const double* y,
                double* r)
{
  checkCudaErrors(cublasDdot(handle_cublas,
                             n,
                             x, 1,
                             y, 1,
                             r));
}

void scaleVector(cublasHandle_t& handle_cublas,
                 int n,
                 const double* alpha,
                 double* x)
{
  checkCudaErrors(cublasDscal(handle_cublas,
                              n,
                              alpha,
                              x, 1));
}
