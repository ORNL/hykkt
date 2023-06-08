#pragma once
#include <cusparse.h>

//*********************************SpGEMMM***********************************//

/*
 * @brief computes an upper bound of device memory for intermediate products
 *
 * @param handle - handle to cuSPARSE library context
 * a_desc - sparse matrix descriptor for first product matrix
 * b_desc - sparse matrix descriptor for second product matrix
 * c_desc - sparse matrix descriptor for result product matrix
 * spgemm_desc - spgemm descriptor necessary for computing
 * d_buffer - buffer for intermediate products
 *
 * @post d_buffer is now allocated on the device with an upper bound of
 *       memory need for the intermediate products
 */
void SpGEMM_workEstimation_reuse(cusparseHandle_t& handle,
    const cusparseSpMatDescr_t& a_desc,
    const cusparseSpMatDescr_t& b_desc,
    cusparseSpMatDescr_t& c_desc,
    cusparseSpGEMMDescr_t& spgemm_desc,
    void** d_buffer);

/*
 * @brief calculates the nonzero structure of the product result matrix c_desc
 *
 * @param handle - handle to the cuSPARSE library context
 * a_desc - sparse matrix descriptor for first product matrix
 * b_desc - sparse matrix descriptor for second product matrix
 * c_desc - sparse matrix descriptor for result product matrix
 * spgemm_desc - spgemm descriptor necessary for computing
 * d_buffer* - buffers needed for SpGEMM allocation
 *
 * @post nonzero structure of result matrix c_desc calculated and buffers 
 *       allocated properly for SpGEMM
 */
void SpGEMM_calculate_nnz_reuse(cusparseHandle_t& handle,
    const cusparseSpMatDescr_t& a_desc,
    const cusparseSpMatDescr_t& b_desc,
    cusparseSpMatDescr_t& c_desc,
    cusparseSpGEMMDescr_t& spgemm_desc,
    void** d_buffer1,
    void** d_buffer2,
    void** d_buffer3);

/*
 * @brief retrieves size of result product matrix, allocates the memory, 
 *        and sets up the descriptor c_desc for the result product matrix
 *
 * @param n - number of rows in c_desc
 * c_nnz - number of nonzeros to be retrieved from c_desc
 * c_i - row offsets pointer to be allocated in c_desc
 * c_j - column pointers to be allocated in c_desc
 * c_v - nonzero values pointer to be allocated in c_desc
 * c_desc - result product sparse matrix descriptor
 *
 * @post c_i, c_j, c_v are now pointers in c_desc and c_nnz is the number of
 *       nonzeros in the result product matrix C
 */
void SpGEMM_setup_product_descr(int n,
    int64_t& c_nnz,
    int** c_i,
    int** c_j,
    double** c_v,
    cusparseSpMatDescr_t& c_desc);

/*
 * @brief copies csr structure from the temporary buffers to the result matrix
 *
 * @param handle - handle to cuSPARSE library context
 * a_desc - sparse matrix descriptor for first product matrix
 * b_desc - sparse matrix descriptor for second product matrix
 * c_desc - sparse matrix descriptor for result product matrix
 * spgemm_desc - spgemm descriptor necessary for computing
 * d_buffer - buffer used in copying result to c_desc
 *
 * @post c_desc now contains the csr structure of the product result
 */
void SpGEMM_copy_result(cusparseHandle_t& handle,
    const cusparseSpMatDescr_t& a_desc,
    const cusparseSpMatDescr_t& b_desc,
    cusparseSpMatDescr_t& c_desc,
    cusparseSpGEMMDescr_t& spgemm_desc,
    void** d_buffer);

/*
 * @brief initializes descriptors and allocates the buffers needed for 
 *        repeated SpGEMM computations
 *
 * @param handle - handle to cuSPARSE library context
 * a_desc - sparse matrix descriptor for first product matrix
 * b_desc - sparse matrix descriptor for second product matrix
 * c_desc - sparse matrix descriptor for result product matrix
 * n - number of rows in matrix result
 * c_nnz - number of nonzeros in product result to be retrieved
 * c_i - row offsets of product result
 * c_j - column pointers of product result
 * c_v - nonzero values of product result
 * spgemm_desc - spgemm descriptor necessary for computing
 * d_buffer* - buffers needed for repeated SpGEMM computations
 *
 * @post CSR pointers c_i, c_j, c_v are properly set up in c_desc,
 *       spgemm_desc has necessary information and buffers are properly
 *       allocated for repeate SpGEMM computations
 */
void allocate_for_product(cusparseHandle_t& handle, 
    const cusparseSpMatDescr_t& a_desc, 
    const cusparseSpMatDescr_t& b_desc,             
    cusparseSpMatDescr_t& c_desc, 
    int n,
    int64_t& c_nnz,
    int** c_i,
    int** c_j,
    double** c_v,
    cusparseSpGEMMDescr_t& spgemm_desc,
    void** d_buffer1, 
    void** d_buffer2);

/*
 * @brief computes matrix matrix product (SpGEMM) C = alpha*A*B
 *
 * @param handle - handle to cuSPARSE library context
 * alpha - scalar for matrix matrix product
 * a_desc - sparse matrix descriptor for first product matrix
 * b_desc - sparse matrix descriptor for second product matrix
 * c_desc - sparse matrix descriptor for result product matrix
 * spgemm_desc - spgemm descriptor necessary for computing
 *
 * @post c_desc is now a descriptor for the matrix product alpha(A*B)
 */
void compute_product(cusparseHandle_t& handle, 
    double alpha,
    const cusparseSpMatDescr_t& a_desc, 
    const cusparseSpMatDescr_t& b_desc, 
    cusparseSpMatDescr_t& c_desc, 
    cusparseSpGEMMDescr_t& spgemm_desc);

//*******************************MATRIX***SUM********************************//

/*
 * @brief allocates buffer_add for matrix sum computation, 
 *        determines result row offsets and number of nonzeros
 *
 * @param handle - handle to cuSPARSE library context
 * a_i, a_j, a_v - CSR format for first sum matrix
 * alpha - scalar for first sum matrix
 * b_i, b_j, b_v - CSR format for second sum matrix
 * beta - scalar for second sum matrix
 * c_i, c_j, c_v - CSR format for result sum matrix
 * m - number of rows in matrices A, B, C
 * n - number of columns in matrices A, B, C
 * nnz_a - number of nonzeros in A
 * nnz_b - number of nonzeros in B
 * descr_a - descriptor of matrix A
 * buffer_add - buffer used  for marix sum computation
 * nnz_total_ptr - pointer to number of nonzeros in sum
 *
 * @post buffer_add is allocated on the device, CSR format of result matrix
 *       is set up, nnz_total_ptr points to the result matrix nnz
 */
void allocate_for_sum(cusparseHandle_t& handle,
    const int* a_i, 
    const int* a_j, 
    const double* a_v,
    double alpha,
    const int* b_i, 
    const int* b_j, 
    const double* b_v,
    double beta,
    int** c_i, 
    int** c_j, 
    double** c_v,
    int m,
    int n, 
    int nnz_a, 
    int nnz_b, 
    cusparseMatDescr_t& descr_a, 
    void** buffer_add, 
    int* nnz_total_ptr);

/*
 * @brief computes matrix sum C = alpha*A + beta*B
 *
 * @param handle - handle to cuSPARSE library context
 * a_i, a_j, a_v - CSR format for first sum matrix
 * alpha - scalar for first sum matrix
 * b_i, b_j, b_v - CSR format for second sum matrix
 * beta - scalar for second sum matrix
 * c_i, c_j, c_v - CSR format for result sum matrix
 * m - number of rows in matrices A, B, C
 * n - number of columns in matrices A, B, C
 * nnz_a - number of nonzeros in A
 * nnz_b - number of nonzeros in B
 * descr_a - descriptor of matrix A
 * buffer_add - buffer used  for marix sum computation
 *
 * @post CSR pointers c_i, c_j, c_v, now represent matrix sum C
 */
void compute_sum(cusparseHandle_t& handle,
    const int* a_i, 
    const int* a_j, 
    const double* a_v,
    double alpha,
    const int* b_i, 
    const int* b_j, 
    const double* b_v,
    double beta,
    int* c_i, 
    int* c_j, 
    double* c_v,
    int m, 
    int n,
    int nnz_a, 
    int nnz_b,
    cusparseMatDescr_t& descr_a,
    void** buffer_add);

/**
 * @brief Kernel wrapper for Q sparse product M = Sparse(Q + A'HA)
 * 
 * @param m, n, q_nnz, q_i, q_j, q_v, a_nnz, a_i, a_j, a_v - CSR information for matrix Q and A transpose
 * out - vector to store result
 * 
 * @pre Q is nxn SPD, A is mxn (m > n), H is nxn diagonal
 * 
 * @post stores sparse matrix product values in vector out corresponding to nonzero structure of Q
*/
void fun_q_sparse_product(int n, 
    int q_nnz, 
    int* q_i, 
    int* q_j, 
    double* q_v, 
    int a_nnz, 
    int* a_i, 
    int* a_j, 
    double* a_v, 
    double* h_v, 
    double* out);    

/**
 * @brief Kernel wrapper for diagonal product A'HA
 * 
 * @param n, a_i, a_j, a_v - CSR information for matrix A transpose
 * n - number of rows in A transpose
 * out - vector to store result
 * 
 * @pre A is mxn (m > n), H is nxn diagonal
 * 
 * @post stores diagonal of A'HA matrix product in out
*/
void fun_inv_diagonal_product(int n, 
    int* a_i, 
    int* a_j, 
    double* a_v, 
    double* h_v, 
    double* out);