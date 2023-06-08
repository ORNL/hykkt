#pragma once

__global__ void deviceDotProductKernel(int n, const double* x, const double* y, double* r);
__global__ void divide(const double* x, const double* y, double* z);
__global__ void mem_copy(const double* x, double* y);