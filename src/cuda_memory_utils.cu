#include <stdio.h>

#include <iostream>

#include "MMatrix.hpp"
#include "cuda_check_errors.hpp"

void deleteOnDevice(void* v)
{
  checkCudaErrors(cudaFree(v));
}

template <typename T>
void allocateValueOnDevice(T** v)
{
  checkCudaErrors(cudaMalloc((void**)v, sizeof(T)));
}
template void allocateValueOnDevice<double>(double**);

template <typename T1, typename T2>
void allocateVectorOnDevice(T1 n, T2** v)
{
  checkCudaErrors(cudaMalloc((void**)v, sizeof(T2) * n));
}
template void allocateVectorOnDevice<int, double>(int, double**);
template void allocateVectorOnDevice<int64_t, double>(int64_t, double**);

template <typename T>
void copyVectorToHost(int n, const T* src, T* dst)
{
  checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToHost));
}
template void copyVectorToHost<double>(int, const double*, double*);

template <typename T>
void copyDeviceVector(int n, const T* src, T* dst)
{
  checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToDevice));
}
template void copyDeviceVector<double>(int, const double*, double*);
template void copyDeviceVector<int>(int, const int*, int*);

template <typename T>
void copyVectorToDevice(int n, const T* src, T* dst)
{
  checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyHostToDevice));
}
template void copyVectorToDevice<double>(int, const double*, double*);


void allocateBufferOnDevice(void** b, size_t b_size)
{
  checkCudaErrors(cudaMalloc((void**)b, sizeof(char) * b_size));
}

void allocateMatrixOnDevice(int n, int nnz, int** a_i, int** a_j, double** a_v)
{
  allocateVectorOnDevice(n + 1, a_i);
  allocateVectorOnDevice(nnz, a_j);
  allocateVectorOnDevice(nnz, a_v);
}

void deleteMatrixOnDevice(int* a_i, int* a_j, double* a_v)
{
  deleteOnDevice(a_v);
  deleteOnDevice(a_i);
  deleteOnDevice(a_j);
}

void matrixDeviceToHostCopy(int n,
                            int nnz,
                            int* a_i,
                            int* a_j,
                            double* a_v,
                            int* b_i,
                            int* b_j,
                            double* b_v)
{
  copyVectorToHost(n + 1, a_i, b_i);
  copyVectorToHost(nnz, a_j, b_j);
  copyVectorToHost(nnz, a_v, b_v);
}

void matrixHostToDeviceCopy(int n,
    int nnz, 
    int* a_i, 
    int* a_j, 
    double* a_v, 
    int* b_i, 
    int* b_j, 
    double* b_v)
{
  copyVectorToDevice(n + 1, a_i, b_i);
  copyVectorToDevice(nnz, a_j, b_j);
  copyVectorToDevice(nnz, a_v, b_v);
}

void matrixDeviceToDeviceCopy(int n, 
    int nnz, 
    int* a_i, 
    int* a_j, 
    double* a_v, 
    int* b_i, 
    int* b_j, 
    double* b_v)
{
  copyDeviceVector(n + 1, a_i, b_i);
  copyDeviceVector(nnz, a_j, b_j);
  copyDeviceVector(nnz, a_v, b_v);
}

void copyMatrixToHost(const int* a_i,
    const int* a_j, 
    const double* a_v, 
    MMatrix& mat_a)
{
  copyVectorToHost(mat_a.n_ + 1, a_i, mat_a.csr_rows);
  copyVectorToHost(mat_a.nnz_, a_j, mat_a.csr_cols);
  copyVectorToHost(mat_a.nnz_, a_v, mat_a.csr_vals);
}

void copyMatrixToDevice(const MMatrix* mat_a, int* a_i, int* a_j, double* a_v)
{
  matrixHostToDeviceCopy(mat_a->n_, 
                         mat_a->nnz_,
                         mat_a->csr_rows,
                         mat_a->csr_cols,
                         mat_a->csr_vals,
                         a_i,
                         a_j,
                         a_v);
}

void copySymmetricMatrixToDevice(const MMatrix* mat_a, 
                                 int* a_i, 
                                 int* a_j, 
                                 double* a_v)
{
  matrixHostToDeviceCopy(mat_a->n_,
                         mat_a->nnz_,
                         mat_a->csr_rows,
                         mat_a->csr_cols,
                         mat_a->csr_vals,
                         a_i,
                         a_j,
                         a_v);
}


void cloneMatrixToDevice(const MMatrix* mat_a, int** a_i, int** a_j, double** a_v)
{
  allocateMatrixOnDevice(mat_a->n_, mat_a->nnz_, a_i, a_j, a_v);
  copyMatrixToDevice(mat_a, *a_i, *a_j, *a_v);
}

void cloneSymmetricMatrixToDevice(const MMatrix* mat_a,
                                  int** a_i, 
                                  int** a_j, 
                                  double** a_v)
{
  allocateMatrixOnDevice(mat_a->n_, mat_a->nnz_, a_i, a_j, a_v);
  copySymmetricMatrixToDevice(mat_a, *a_i, *a_j, *a_v);
}

void checkGpuMem()
{
  size_t avail;
  size_t total;
  cudaMemGetInfo(&avail, &total);
  size_t used = total - avail;
  printf("Available memory of a : %zu\n", avail);
  printf("Total memory of a : %zu\n", total);
  printf("Used memory of a : %zu\n", used);
}
