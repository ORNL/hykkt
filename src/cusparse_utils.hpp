#pragma once

#include <string>

//***************************************************************************//
/*
 * @brief displays the nonzero values of a matrix from its sparse descriptor
 *
 * @param mat_desc - sparse matrix descriptor
 * @param start_i - first index of nonzero values to display
 * @param display_n - number of elements to display
 * @param label - name of values array
 *
 * @pre start_i<number of nonzeros and start_i+display_n-1<number of nonzeros
 * @post displays display_n number of nonzero values starting at start_i
*/
void displaySpMatValues(cusparseSpMatDescr_t mat_desc, 
    int start_i, 
    int display_n,
    std::string label = "Values");

/* 
 * @brief destroys cusparse descriptor
 *
 * @param desc - a cusparse descriptor
 *
 * @post desc is destroyed
 */
void deleteDescriptor(cusparseSpGEMMDescr_t& desc);

void deleteDescriptor(cusparseSpMatDescr_t& desc);

void deleteDescriptor(cusparseMatDescr_t& desc);

/* 
 * @brief creates the transpose of matrix A by converting from
 *        CSR format to CSC format
 *
 * @param  handle - handle to the cuSPARSE library context
 * n - number of rows in A
 * m - number of cols in A
 * nnz - number of nonzeros in A
 * a_i - row offsets for CSR format for A
 * a_j - column pointers for CSR format for A
 * a_v - nonzero values for CSR format for A
 * at_i - vector where transposed matrix row offsets are stored
 * at_j - vector where transposed matrix column pointers are stored
 * at_v - vector where transposed matrix nonzero values are stored
 * buffer - reusable buffer so transpose allocates only once
 * allocated - boolean for if buffer is allocated
 *
 * @post v at_ik at_j, at_v now represent the CSR format of the transpose of A
 */
void transposeMatrixOnDevice(cusparseHandle_t handle,
                             int n,
                             int m,
                             int nnz,
                             const int* a_i,
                             const int* a_j,
                             const double* a_v,
                             int* at_i,
                             int* at_j,
                             double* at_v,
                             void** buffer,
                             bool allocated);

/* 
 * @brief initializes mat_desc in CSR format of matrix A
 *
 * @param n - number of rows in A
 * @param m - number of cols in A
 * @param nnz - number of nonzeros in A
 * @param a_i - row offsets for CSR format for A
 * @param a_j - columns pointers for CSR format for A
 * @param a_v - nonzero values for CSR format for A
 *
 * @post mat_desc is now a sparse matrix descriptor in CSR format for A
 */
void createCsrMat(cusparseSpMatDescr_t* mat_desc,
                  int n,
                  int m,
                  int nnz,
                  int* a_i,
                  int* a_j,
                  double* a_v);

/*
 * @brief initializes dense vector descriptor
 *
 * @param vec_desc - dense vector descriptor on host
 * @param n - size of dense vector
 * @param d_vec - values of dense vector on device with size n
 * 
 * @post vec_desc is now initialized as a dense vector descriptor
*/
void createDnVec(cusparseDnVecDescr_t* vec_desc, int n, double* d_vec);

/**
 * @brief creates a SpGEMM cusparse descriptor
 *
 * @param spgemm_desc - descriptor to be initialized
 *
 * @post sgemm_desc is now a SpGEMM cusparse descriptor
 */
void createSpGEMMDescr(cusparseSpGEMMDescr_t* spgemm_desc);

/*
 * @brief creates a matrix descriptor
 *
 * @param descr - descriptor to be initialized
 *
 * @post descr is now a matrix descriptor with default setup
*/
void createSparseMatDescr(cusparseMatDescr_t& descr);

/*
 * @brief creates a handle for the cuSPARSE library context
 *
 * @param handle - handle to be initialized
 *
 * @post handle is now a handle for cuSPARSE
*/
void createSparseHandle(cusparseHandle_t& handle);

/*
 * @brief creates a handle for the cuBLAS library context
 *
 * @param handle - handle to be initialized
 *
 * @post handle is now a handle for cuBLAS
*/
void createCublasHandle(cublasHandle_t& handle);
