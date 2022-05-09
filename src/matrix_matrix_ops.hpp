#pragma once
#include <cusparse.h>
/*
@brief: wrapper for CUDA matrix-matrix product and sum

@inputs: matrix A, vectors b and c, scalars alpha and beta

@outputs: c = alpha*Ab+beta*c
*/

void allocate_for_sum(cusparseHandle_t handle,
                      int* A_i, int* A_j, double* A_v,
                      int* B_i, int* B_j, double* B_v,
                      int* C_i, int* C_j, double* C_v,
                      int n, int nnzA, int nnzB, int& nnzC,
                      cusparseMatDescr_t &descrA, void* buffer_add, int* nnzTotal);

  void allocate_for_product(cusparseHandle_t handle, cusparseOperation_t op,                 
      cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB,             
      cusparseSpMatDescr_t matC, cusparseSpGEMMDescr_t spgemmDesc);

void compute_product(cusparseHandle_t handle, cusparseOperation_t op, int alpha,
      cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, int beta,            
      cusparseSpMatDescr_t matC, cusparseSpGEMMDescr_t spgemmDesc);

void matrix_matrix_product(cusparseHandle_t handle, int alpha, 
    cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, int beta,
    cusparseSpMatDescr_t matC, cusparseSpGEMMDescr_t spgemmDesc);

void compute_sum(cusparseHandle_t handle,
                int* A_i, int* A_j, double* A_v,
                int* B_i, int* B_j, double* B_v,
                int* C_i, int* C_j, double* C_v,
                int n, int nnzA, int nnzB,
                cusparseMatDescr_t &descrA, void* buffer_add);

void matrix_sum(cusparseHandle_t handle,
               int* A_i, int* A_j, double* A_v,
               int* B_i, int* B_j, double* B_v, 
               int* C_i, int* C_j, double* C_v,
               int n, int nnzA, int nnzB,
               cusparseMatDescr_t &descrA, void* buffer_add);

void spGEMM_product_sum(cusparseHandle_t handle,
                      int* A_i, int* A_j, double* A_v,
                      int* B_i, int* B_j, double* B_v, 
                      int* C_i, int* C_j, double* C_v,
                      int n, int nnzA, int nnzB, int &nnzC,
                      cusparseMatDescr_t &descrA, void* buffer_add, int* nnzT);
