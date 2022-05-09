#pragma once

#include <stdio.h>
#include <cstring>
#include <iostream>

#include <cusparse.h>

#include "matrix_vector_ops.hpp"
#include "matrix_vector_ops_cuda.hpp"

// Forward declaration
struct MMatrix;

/* 
 * @brief deletes variable v from device
 *
 * @param v - a vector on the device
 *
 * @post v is freed from the device
 */
template <typename T>
void deleteOnDevice(T* v)
{
  checkCudaErrors(cudaFree(v));
}

/* 
 * @brief allocates vector v onto device
 *
 * @param n - size of the vector (int, size_t)
 * v - the vector to be allocated on the device
 *
 * @post v is now a vector with size n on the device
 */
template <typename T1, typename T2>
void allocateVectorOnDevice(T1 n, T2** v)
{
  checkCudaErrors(cudaMalloc((void**)v, sizeof(T2) * n));
}

/* 
 * @brief copies vector from device to host
 *
 * @param n - size of src vector
 * src - vector on device
 * dst - vector on host
 *
 * @pre src is a valid vector
 * @post src is copied onto dst
 */
template <typename T>
void copyVectorToHost(int n, const T* src, T* dst)
{
  checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToHost));
}

/* 
 * @brief copies a device vector onto another device vector
 *
 * @param n - size of src vector
 * src - vector on device to be copied
 * dst - vector on device to be copied onto
 *
 * @post src is copied onto dst
 */
template <typename T>
void copyDeviceVector(int n, const T* src, T* dst)
{
  checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToDevice));
}

/* 
 * @brief copies a vector from the host onto the device
 *
 * @param n - size of src vector
 * src - vector on host to be copied
 * dst - vector on device to be copied onto
 *
 * @post src is copied onto dst
 */
template <typename T>
void copyVectorToDevice(int n, const T* src, T* dst)
{
  checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyHostToDevice));
}

/* 
 * @brief copies a host vector onto a newly allocated vector on the device
 *
 * @param n - size of src vector
 * src - vector on host to be cloned
 * dst - vector on device on which src is cloned
 *
 * @post dst is a clone of src on the device
 */
template <typename T>
void cloneVectorToDevice(int n, T** src, T** dst)
{
  allocateVectorOnDevice(n, dst);
  copyVectorToDevice(n, *src, *dst);
}

/*
 * @brief prints vector from host
 *
 * @param v - vector on host
 * display_n - number of elements to print
 * label - name of vector
 *
 * @pre display_n <= number of elements in v
 * @post display_n elements of v printed
*/
template <typename T>
void displayHostVector(T* v, 
                       int display_n, 
                       std::string label = "Vector")
{
  std::cout<<"\n\n"<<label<<": {";
  for(int i = 0; i < display_n - 1; i++){
    std::cout<<v[i]<<", ";
  }
  std::cout<<v[display_n - 1]<<"}\n"<<std::endl; 
}

/*
 * @brief prints vector from device
 *
 * @param v - vector on host
 * display_n - number of elements to print
 * n - number of elements in v
 * label - name of vector
 *
 * @pre display_n <= n
 * @post display_n elements of v printed
*/
template <typename T>
void displayDeviceVector(T* v, 
                         int n, 
                         int display_n, 
                         std::string label = "Vector")
{
  T* h_v = new T[n];
  copyVectorToHost(n, v, h_v);
  displayHostVector(h_v, display_n, label);
}

/* 
 * @brief clones vector of size n from src to dst
 *
 * @param n - size of vector
 * src - vector to be cloned
 * dst - clone target
 *
 * @pre n contain an int length
 * src is a valid vector
 *
 * @post dst is a clone of src on device
 */
template <typename T>
void cloneDeviceVector(int n, T** src, T** dst)
{
  allocateVectorOnDevice(n, dst);
  copyDeviceVector(n, *src, *dst);
}

//***************************************************************************//
void allocateBufferOnDevice(void** b, size_t b_size);

/* 
 * @brief allocates a matrix on the device in CSR format
 *
 * @param n - size of matrix A
 * nnz - numer of nonzeros in matrix A
 * a_i - row offsets for CSR format for A
 * a_j - column pointers for CSR format for A
 * a_v - nonzero values for CSR format for A
 *
 * @post a_i, a_j, a_v allocated onto the device with size 
 *       n + 1, nnz, nnz to follow CSR format
 */
void allocateMatrixOnDevice(int n, int nnz, int** a_i, int** a_j, double** a_v);

/* 
 * @brief deletes a matrix in CSR format from the device
 *
 * @param a_i - row offsets for CSR format for A
 * a_j - column pointers for CSR format for A
 * a_v - nonzero values for CSR format for A
 *
 * @post a_i, a_j, a_v freed from the device
 */
void deleteMatrixOnDevice(int* a_i, int* a_j, double* a_v);

/* 
 * @brief copies device matrix A onto host matrix B
 *
 * @param n - dimensions of A
 * nnz - number of nonzeros in A
 * a_i - row offsets for CSR format for A
 * a_j - column pointers for CSR format for A
 * a_v - nonzero values for CSR format for A
 * b_i - row offsets for CSR format for B
 * b_j - column pointers for CSR format for B
 * b_v - nonzero values for CSR format for B
 *
 * @post a_i is copied onto b_i, a_j is copied onto b_j, a_v is
 *           copied onto b_v so that matrix A CSR format is copied
 *           onto the host 
 */
void matrixDeviceToHostCopy(int n,
                            int nnz,
                            int* a_i,
                            int* a_j,
                            double* a_v,
                            int* b_i,
                            int* b_j,
                            double* b_v);

/* 
 * @brief copies host matrix A onto device matrix B
 *
 * @param n - dimensions of A
 * nnz - number of nonzeros in A
 * a_i - row offsets for CSR format for A
 * a_j - column pointers for CSR format for A
 * a_v - nonzero values for CSR format for A
 * b_i - row offsets for CSR format for B
 * b_j - column pointers for CSR format for B
 * b_v - nonzero values for CSR format for B
 *
 * @post a_i is copied onto b_i, a_j is copied onto b_j, a_v is
 *           copied onto b_v so that matrix A CSR format is copied
 *           onto the device
 */
void matrixHostToDeviceCopy(int n,
                            int nnz,
                            int* a_i,
                            int* a_j,
                            double* a_v,
                            int* b_i,
                            int* b_j,
                            double* b_v);

/* 
 * @brief copies device matrix A to another device matrix B
 *
 * @param same for matrixHostToDeviceCopy
 *
 * @post a_i is copied onto b_i, a_j is copied onto b_j, a_v is
 *           copied onto b_v so that device matrix A CSR format 
 *           is recopied onto the device
 */
void matrixDeviceToDeviceCopy(int n,
                              int nnz,
                              int* a_i,
                              int* a_j, 
                              double* a_v, 
                              int* b_i, 
                              int* b_j, 
                              double* b_v);

/* 
 * @brief copies device matrix A onto the host in a MMatrix object
 *
 * @param a_i - row offsets for CSR format for A
 * a_j - column pointers for CSR format for A
 * a_v - nonzero values for CSR format for A
 * mat_a - MMatrix object on host to be copied onto
 *
 * @post a_i, a_j, a_v are copied from the device onto mat_a 
 *       member variables - csr_rows, coo_cols, coo_vals
 */
void copyMatrixToHost(const int* a_i,
                      const int* a_j, 
                      const double* a_v, 
                      MMatrix& mat_a);

/* 
 * @brief copies host MMatrix object mat_a onto the device in CSR format
 *
 * @param mat_a - MMatrix object on host to be copied
 * a_i - row offsets to be copied onto
 * a_j - column pointers to be copied onto
 * a_v - nonzero values to be copied onto
 *
 * @post mat_a member variables - csr_ rows, coo_cols, coo_vals - 
 *       are copied onto device vectors a_i, a_j, a_v
 */
void copyMatrixToDevice(const MMatrix& mat_a, int* a_i, int* a_j, double* a_v);

/* 
 * @brief copies host symmetric MMatrix object mat_a onto the device in CSR format
 *
 * @param same as for copyMatrixToDevice
 *
 * @post mat_a member variables - csr_ rows, csr_cols, csr_vals - 
 *       are copied onto device vectors a_i, a_j, a_v
 */
void copySymmetricMatrixToDevice(const MMatrix& mat_a, 
                                 int* a_i, 
                                 int* a_j, 
                                 double* a_v);

/* 
 * @brief copies CSR format of MMatrix object mat_a onto newly 
 *        allocated vectors a_i, a_j, a_v
 *
 * @param mat_a - MMatrix object on host to be cloned to device
 * a_i - row offsets to be copied onto
 * a_j - column pointers to be copied onto
 * a_v - nonzero values to be copied onto
 *
 * @post mat_a member variables - csr_rows, coo_cols, coo_vals -
 *       are copied onto a_i, a_j, a_v which are first allocated
 *       onto the device
 */
void cloneMatrixToDevice(const MMatrix& mat_a, int** a_i, int** a_j, double** a_v);

/* 
 * @brief copies CSR format of symmetric MMatrix object mat_a onto 
 *        newly allocated vectors a_i, a_j, a_v
 *
 * @param same as for cloneMatrixToDevice
 *
 * @post mat_a member variables - csr_rows, csr_cols, csr_vals -
 *       are copied onto a_i, a_j, a_v which are first allocated
 *       onto the device
 */
void cloneSymmetricMatrixToDevice(const MMatrix& mat_a,
                                  int** a_i, 
                                  int** a_j, 
                                  double** a_v);

/* 
 * @brief creates the transpose of matrix A by converting from
 *        CSR format to CSC format
 *
 * @param n - number of rows in A
 * m - number of cols in A
 * nnz - number of nonzeros in A
 * a_i - row offsets for CSR format for A
 * a_j - column pointers for CSR format for A
 * a_v - nonzero values for CSR format for A
 * at_i - vector where transformed matrix row offsets are stored
 * at_j - vector where transformed matrix column pointers are stored
 * at_v - vector where transformed matrix nonzero values are stored
 *
 * @post v at_ik at_j, at_v now represent the CSR format of the transform of A
 */
void transposeMatrixOnDevice(int n,
                             int m,
                             int nnz,
                             const int* a_i,
                             const int* a_j,
                             const double* a_v,
                             int* at_i,
                             int* at_j,
                             double* at_v);

/* 
 * @brief initializes mat_desc in CSR format of matrix A
 *
 * @param n - number of rows in A
 * m - number of cols in A
 * nnz - number of nonzeros in A
 * a_i - row offsets for CSR format for A
 * a_j - columns pointers for CSR format for A
 * a_v - nonzero values for CSR format for A
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
 * n - size of dense vector
 * d_vec - values of dense vector on device with size n
 * 
 * @pre
 * @post vec_desc is now initialized as a dense vector descriptor
*/
void createDnVec(cusparseDnVecDescr_t* vec_desc, int n, double* d_vec);
