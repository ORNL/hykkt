#pragma once

#include <cusolver_common.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include "cusolverSp.h"
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverRf.h>

/*
 * @brief wrapper for matrix-vector product buffer size calculation wrapper 
 *        and matrix-vector product wrapper
 *
 * @param handle - handle for cuSPARSE library context
 * alpha - scalar for A*b
 * a_desc_sp - matrix A being multiplied
 * b_desc_dn - vector b being multiplied
 * beta - scalar for c
 * c_desc_dn - vector c where product A*b is stored
 *
 * @post c = alpha*Ab + beta*c stored in c_desc_dn
*/
void fun_SpMV_full(cusparseHandle_t handle, 
                   double alpha, 
                   cusparseSpMatDescr_t a_desc_sp, 
                   cusparseDnVecDescr_t b_desc_dn, 
                   double beta, 
                   cusparseDnVecDescr_t c_desc_dn);

/* 
 * @brief calculates the size of the workspace needed for fun_SpMV
 *
 * @param same for fun_SpMV_full
 * buffer_size - size of buffer to be calculated
 * 
 * @post buffer_size now has the size of the workspace needed
*/
void fun_SpMV_buffer(cusparseHandle_t handle, 
                     double alpha, 
                     cusparseSpMatDescr_t a_desc_sp, 
                     cusparseDnVecDescr_t b_desc_dn, 
                     double beta, 
                     cusparseDnVecDescr_t c_desc_dn, 
                     size_t* buffer_size);

/*
 * @brief wrapper for CUDA matrix-vector product
 *
 * @param same for fun_SpMV_full
 * buffer - used for memory on the device
 *
 * @post c = alpha*Ab + beta*c, stored in c_desc_dn
*/
void fun_SpMV_product(cusparseHandle_t handle, 
                      double alpha, 
                      cusparseSpMatDescr_t a_desc_sp, 
                      cusparseDnVecDescr_t b_desc_dn, 
                      double beta, 
                      cusparseDnVecDescr_t c_desc_dn, 
                      void* buffer);

/*
 * @brief diagonally scales a matrix from the left and right, and
 *        diagonally scales rhs (from the left)
 *
 * @param Size n of A, m - total rows in matrix (1), matrices A, B, B^T, in
 *        csr format (works also if the third matrix is not B^T provided 
 *        dimensions are correct), a scaling vector representing a 
 *        diagonal matrix, 
 *
 * @pre   scale is initialized to positive values 
 *        1/sqrt(the maximum entry magnitude of each row) of matrix (1)
 *
 * @post  The value arrays of the matrices (a_v, b_v, bt_v) are scaled along 
 *        with d_rhs the rhs vector. max_d is updated to include the 
 *        aggregate scaling
*/
void fun_adapt_diag_scale(int n, 
                          int m, 
                          double* a_v, 
                          int* a_i, 
                          int* a_j, 
                          double* b_v, 
                          int* b_i,
                          int* b_j, 
                          double* bt_v, 
                          int* bt_i, 
                          int* bt_j, 
                          double* scale, 
                          double* d_rhs1, 
                          double* d_rhs2, 
                          double* max_d);

/*
 * @brief Determines the correct scaling (corresponding to one iteration 
 *        of Ruiz scaling) to be used by the adapt_diag_scale kernel on 
 *        matrix (1) with form [A B^T; B 0]
 *
 * @param Size n of A, m - total rows in matrix (1), matrices A, B, B^T, in
 *        csr format (works also if the third matrix is not B^T provided 
 *        dimensions are correct), an empty scaling vector representing a 
 *        diagonal matrix
 * 
 * @post  The scaling vector scale is updated entry-wise with
 *        1/sqrt(the maximumentry magnitude of each row) of matrix (1)
*/
void fun_adapt_row_max(int n, 
                       int m, 
                       double* a_v, 
                       int* a_i, 
                       int* a_j, 
                       double* b_v, 
                       int* b_i,
                       int* b_j, 
                       double* bt_v, 
                       int* bt_i, 
                       int* bt_j, 
                       double* scale);

/*
 * @brief sets values of an array to a constant
 *
 * @param Length of array n, val - the value the array is set to,
 *        arr - a pointer to the array that is initialized
 * 
 * @post  arr with entries set to val
*/
void fun_set_const(int n, double val, double* arr);

/*
 * @brief adds a constant to an array
 * 
 * @param Length of array n, val - the value to be added,
 *        arr - a pointer to the array the constant is added to
 *
 * @post  arr with entries increased by val
*/
void fun_add_const(int n, int val, int* arr);

/*
 * @brief:  add arrays arr1, arr2 such that arr1 = arr1+ alp*arr2
 *
 * @params: Length of array n, arr1, arr2 - arrays to be added,
 *          alp - scaling constant
 *
 * @post:   arr1 += alp*arr2
 */
void fun_add_vecs(int n, double* arr1, double alp, double* arr2);

/*
 * @brief multiplies an array by a constant
 * 
 * @param Length of array n, val - the value to multiply,
 * arr - a pointer to the array the constant is added to
 *
 * @post Each entry in arr is scaled by val
*/
void fun_mult_const(int n, double val, double* arr);

/*
 * @brief adds a multiple of I to a CSR matrix A
 * 
 * @param Length of array n, val - the value to add,
 * and a_i, a_j, a_v - pointers for rows, columns and values
 * 
 * @post A[i][i]+=val for all i
*/
void fun_add_diag(int n, double val, int* a_i, int* a_j, double* a_v);

/*
 * @brief Applies the inverse of a diagonal matrix (stored as 
 *        a vector) on the left to a vector
 *
 * @param Size n of the matrix, d_rhs a (dense) vector, ds a dense 
 *        vector represting a diagonal matrix
 *
 * @post d_rhs=d_rhs./ds (elementwise)
*/
void fun_inv_vec_scale(int n, double* d_rhs, double* ds);

/*
 * @brief Applies a diagonal matrix (stored as a vector) on the 
 *         left to a vector
 * 
 * @param Size n of the matrix, d_rhs a (dense) vector, ds a dense 
 *        vector represting a diagonal matrix
 * 
 * @post d_rhs=ds.*d_rhs (elementwise)
 */
void fun_vec_scale(int n, double* d_rhs, double* ds);

/*
 * @brief concatenates 2 matrices into a third matrix (one under the other)
 * 
 * @param Row count n the matrix A and row count m of matrix B,
 *        number of non zeros (nnz) of matrices A and B matrices 
 *        A and B in CSR format, an empty matrix C to be overwritten
 * 
 * @post Matrix C in CSR format [A' B']'
 */
void fun_concatenate(int n, 
                     int m, 
                     int nnz_a, 
                     int nnz_b, 
                     double* a_v, 
                     int* a_i, 
                     int* a_j,
                     double* b_v, 
                     int* b_i, 
                     int* b_j, 
                     double* c_v, 
                     int* c_i, 
                     int* c_j);

/*
 * @brief Applies a diagonal matrix (stored as a vector) on the left to a 
 *        matrix and a vector and stores the result in separate arrays
 * 
 * @param Size n of the matrix, Matrix A in csr storage format, d_rhs a 
 *        (dense) vector, ds a dense vector represting a diagonal matrix
 * 
 * @post The value array of the matrix (A_v) is scaled along with d_rhs.
*/
void fun_row_scale(int n, 
                   double* a_v, 
                   int* a_i, 
                   int* a_j, 
                   double* a_vs, 
                   double* d_rhs,
                   double* d_rhs_s, 
                   double* ds);

/*
 * @brief diagonally scales a matrix from the left and right, and diagonally 
 *         scales rhs (from the left)
 *
 * @param Size n of the matrix, the lower half of the matrix stored in 
 *        csr format, the upper half of the matrix stored in csr format,
 *        a scaling vector representing a diagonal matrix, a rhs (vector) 
 *        of the equation, max_d a vector that aggregates scaling, a flag 
 *        to determine whether to scale the second matrix (not necessary 
 *        in last iteration)
 * 
 * @post The value arrays of the matrices (a_v, at_v) are scaled along with
 * d_rhs the rhs vector. max_d is updated to include the aggregate scaling
*/
void fun_diag_scale(int n, 
                    int m, 
                    double* a_v, 
                    int* a_i, 
                    int* a_J, 
                    double* at_v, 
                    int* at_i,
                    int* at_j, 
                    double* scale, 
                    double* d_rhs, 
                    double* max_d, 
                    int flag);

/*
 * @brief Determines the correct scaling (corresponding to one iteration 
 *        of Ruiz scaling) to be used by the diag_scale kernel
 * 
 * @param Size n of the matrix, the lower half of the matrix stored in 
 *        csr format, the upper half of the matrix stored in csr format,
 *        a scaling vector representing a diagonal matrix,
 * 
 * @post The scaling vector scale which is updated entry-wise with
 *       1/sqrt(the maximum of each row)
*/
void fun_row_max(int n, 
                 int m, 
                 double* a_v, 
                 int* a_i, 
                 int* a_j, 
                 double* at_v, 
                 int* at_i, 
                 int* at_j,
                 double* scale);
