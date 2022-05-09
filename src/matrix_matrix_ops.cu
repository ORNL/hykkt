#include "matrix_matrix_ops.hpp"
#include "matrix_vector_ops_cuda.hpp"
#include "matrix_vector_ops.hpp"
#include "cuda_memory_utils.hpp"
#include <assert.h>
#include <stdio.h>
#include "cuda_memory_utils.hpp"
#include "constants.hpp"

/*
@brief: wrapper for CUDA matrix-matrix product and sum

@inputs: matrix A, vectors b and c, scalars alpha and beta

@outputs: c = alpha*Ab+beta*c
*/


void allocate_for_product(cusparseHandle_t handle, cusparseOperation_t op,                 
      cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB,             
      cusparseSpMatDescr_t matC, cusparseSpGEMMDescr_t spgemmDesc)
{
    void*  dBuffer1    = NULL;
    void*  dBuffer2    = NULL;
    void*  dBuffer3    = NULL;
    void*  dBuffer4    = NULL;
    void*  dBuffer5    = NULL;
    size_t bufferSize1 = 0;
    size_t bufferSize2 = 0;
    size_t bufferSize3 = 0;
    size_t bufferSize4 = 0;
    size_t bufferSize5 = 0;
    checkCudaErrors(cusparseSpGEMMreuse_workEstimation(handle, op, op, matA,
        matB, matC, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL));
    allocateBufferOnDevice(&dBuffer1, bufferSize1); 
    checkCudaErrors(cusparseSpGEMMreuse_workEstimation(handle, op, op, matA,
        matB, matC, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1));
    checkCudaErrors(cusparseSpGEMMreuse_nnz(handle, op, op, matA, matB, matC, 
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, NULL, 
        &bufferSize3, NULL, &bufferSize4, NULL));  
    allocateBufferOnDevice(&dBuffer2, bufferSize2); 
    allocateBufferOnDevice(&dBuffer3, bufferSize3); 
    allocateBufferOnDevice(&dBuffer4, bufferSize4); 
    checkCudaErrors(cusparseSpGEMMreuse_nnz(handle, op, op, matA, matB, matC, 
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2, 
        &bufferSize3, dBuffer3, &bufferSize4, dBuffer4)); 
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    checkCudaErrors(cusparseSpMatGetSize(matC, 
        &C_num_rows1, &C_num_cols1, &C_nnz1));
    int* C_i;
    int* C_j;
    double* C_v;
    allocateMatrixOnDevice(C_num_rows1, C_nnz1, &C_i, &C_j, &C_v);
    checkCudaErrors(cusparseSpGEMMreuse_copy(handle, op, op, matA, matB, matC,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize5, NULL));
    allocateBufferOnDevice(&dBuffer5, bufferSize5); 
    checkCudaErrors(cusparseSpGEMMreuse_copy(handle, op, op, matA, matB, matC,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize5, dBuffer5));
    deleteOnDevice(dBuffer1);
    deleteOnDevice(dBuffer2);
    deleteOnDevice(dBuffer3);
    deleteOnDevice(dBuffer4);
    deleteOnDevice(dBuffer5);
}


void compute_product(cusparseHandle_t handle, cusparseOperation_t op, int alpha,
      cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, int beta,            
      cusparseSpMatDescr_t matC, cusparseSpGEMMDescr_t spgemmDesc)
{
    checkCudaErrors(cusparseSpGEMMreuse_compute(handle, op, op, &alpha, matA, matB, &beta,
        matC, COMPUTE_TYPE, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));
}

//C = A*B
void matrix_matrix_product(cusparseHandle_t handle, int alpha, 
    cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, int beta,
    cusparseSpMatDescr_t matC, cusparseSpGEMMDescr_t spgemmDesc)
{
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    allocate_for_product(handle, op, matA, matB, matC, spgemmDesc);
    compute_product(handle, op, alpha, matA, matB, beta, matC, spgemmDesc);
}

//C = A + B
void allocate_for_sum(cusparseHandle_t handle,
                      int* A_i, int* A_j, double* A_v,
                      int* B_i, int* B_j, double* B_v,
                      int* C_i, int* C_j, double* C_v,
                      int n, int nnzA, int nnzB, 
                      cusparseMatDescr_t &descrA, void* buffer_add, int* nnzTotal)
{
    double                one      = 1.0;
    size_t bufferSizeInBytes_add;
    cusparseDcsrgeam2_bufferSizeExt(handle, n, n, &one, descrA, H.nnz_, H_a, H_ia, H_ja, &one,
      descrA, JDtDxJD_nnz1, JDtDxJD_vals, JDtDxJD_rows, JDtDxJD_cols, descrA, Htil_vals, Htil_rows,
      Htil_cols, &bufferSizeInBytes_add);

  
}

void compute_sum(cusparseHandle_t handle,
                int* A_i, int* A_j, double* A_v,
                int* B_i, int* B_j, double* B_v,
                int* C_i, int* C_j, double* C_v,
                int n, int nnzA, int nnzB,
                cusparseMatDescr_t &descrA, void* buffer_add)
{
}

void matrix_sum(cusparseHandle_t handle,
               int* A_i, int* A_j, double* A_v,
               int* B_i, int* B_j, double* B_v, 
               int* C_i, int* C_j, double* C_v,
               int n, int nnzA, int nnzB,
               cusparseMatDescr_t &descrA, int* nnzTotal)
{
    void* buffer_add         = NULL;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

}

//C' = A*B + C
void spGEMM_product_sum(cusparseHandle_t handle,
                      int* A_i, int* A_j, double* A_v,
                      int* B_i, int* B_j, double* B_v, 
                      int* C_i, int* C_j, double* C_v,
                      int n, int nnzA, int nnzB, int& nnzC, cusparseMatDescr_t &descrA, void* buffer_add, int* nnzT)
{
}
