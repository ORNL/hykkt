#include <stdio.h>

#include <cusparse.h>
#include <iostream>

#include "matrix_vector_ops.hpp"
#include "matrix_vector_ops_cuda.hpp"
#include "MMatrix.hpp"
#include "cuda_memory_utils.hpp"
#include "cusparse_params.hpp"

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
  copyVectorToHost(mat_a.nnz_, a_j, mat_a.coo_cols);
  copyVectorToHost(mat_a.nnz_, a_v, mat_a.coo_vals);
}

void copyMatrixToDevice(const MMatrix& mat_a, int* a_i, int* a_j, double* a_v)
{
  matrixHostToDeviceCopy(mat_a.n_, 
                         mat_a.nnz_,
                         mat_a.csr_rows,
                         mat_a.coo_cols,
                         mat_a.coo_vals,
                         a_i,
                         a_j,
                         a_v);
}

void copySymmetricMatrixToDevice(const MMatrix& mat_a, 
                                 int* a_i, 
                                 int* a_j, 
                                 double* a_v)
{
  matrixHostToDeviceCopy(mat_a.n_,
                         mat_a.nnz_,
                         mat_a.csr_rows,
                         mat_a.csr_cols,
                         mat_a.csr_vals,
                         a_i,
                         a_j,
                         a_v);
}


void cloneMatrixToDevice(const MMatrix& mat_a, int** a_i, int** a_j, double** a_v)
{
  allocateMatrixOnDevice(mat_a.n_, mat_a.nnz_, a_i, a_j, a_v);
  copyMatrixToDevice(mat_a, *a_i, *a_j, *a_v);
}

void cloneSymmetricMatrixToDevice(const MMatrix& mat_a,
                                  int** a_i, 
                                  int** a_j, 
                                  double** a_v)
{
  allocateMatrixOnDevice(mat_a.n_, mat_a.nnz_, a_i, a_j, a_v);
  copySymmetricMatrixToDevice(mat_a, *a_i, *a_j, *a_v);
}

void transposeMatrixOnDevice(int n,
                             int m,
                             int nnz,
                             const int* a_i,
                             const int* a_j,
                             const double* a_v,
                             int* at_i,
                             int* at_j,
                             double* at_v)
{
  cusparseHandle_t handle = nullptr;
  checkCudaErrors(cusparseCreate(&handle));

  size_t buffersize;
  printf("Transpose A \n");
  checkCudaErrors(cusparseCsr2cscEx2_bufferSize(handle,
                                                n, 
                                                m, 
                                                nnz,
                                                a_v,  
                                                a_i,  
                                                a_j,
                                                at_v, 
                                                at_i, 
                                                at_j,
                                                COMPUTE_TYPE,
                                                CUSPARSE_ACTION_NUMERIC,
                                                INDEX_BASE,
                                                CUSPARSE_CSR2CSC_ALG1,
                                                &buffersize));

  // Create a buffer
  void *buffercsr = nullptr;
  allocateBufferOnDevice(&buffercsr,buffersize);
  printf("Buffer size is %d\n", buffersize);
  printf("A dimensions are %d by %d with %d nnz\n", n, m, nnz);

  checkCudaErrors(cusparseCsr2cscEx2(handle,
                                     n,
                                     m,
                                     nnz,
                                     a_v,
                                     a_i,
                                     a_j,
                                     at_v,
                                     at_i,
                                     at_j,
                                     COMPUTE_TYPE,
                                     CUSPARSE_ACTION_NUMERIC,
                                     INDEX_BASE,
                                     CUSPARSE_CSR2CSC_ALG1,
                                     buffercsr));

  deleteOnDevice(buffercsr);
  checkCudaErrors(cusparseDestroy(handle));
}

void createCsrMat(cusparseSpMatDescr_t* mat_desc, 
                  int n, 
                  int m, 
                  int nnz, 
                  int* a_i, 
                  int* a_j, 
                  double* a_v)
{

  checkCudaErrors(cusparseCreateCsr(mat_desc, 
                                    n, 
                                    m, 
                                    nnz,
                                    a_i, 
                                    a_j, 
                                    a_v, 
                                    INDEX_TYPE, 
                                    INDEX_TYPE, 
                                    INDEX_BASE, 
                                    COMPUTE_TYPE));
}

void createDnVec(cusparseDnVecDescr_t* vec_desc, int n, double* d_vec)
{
  checkCudaErrors(cusparseCreateDnVec(vec_desc, n, d_vec, COMPUTE_TYPE));
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

