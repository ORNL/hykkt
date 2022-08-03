#pragma once

#include <algorithm>
#include "matrix_vector_ops.hpp"
#include "cuda_memory_utils.hpp"
#include "matrix_vector_ops_cuda.hpp"
#include "cusparse_params.hpp"

/*
 * Class for symbolic and numerical factorization of A,
 * triangular solve of Ax = b
 */
class CholeskyClass
{
public:
  // constructor
  CholeskyClass(int n, int nnz, double* a_v, int* a_i, int* a_j)
    : n_(n),
      nnz_(nnz),
      a_v_(a_v),
      a_i_(a_i),
      a_j_(a_j)
  {
  }

  // destructor
  ~CholeskyClass()
  {
    deleteOnDevice(buffer_gpu_);
    checkCudaErrors(cusolverSpDestroy(handle_cusolver_));
    checkCudaErrors(cusparseDestroyMatDescr(descr_a_));
    checkCudaErrors(cusolverSpDestroyCsrcholInfo(info_));
  }

/* 
 * @brief Symbolic analysis, memory allocation for Cholesky factorization
 *
 * @pre Member variables n_, nnz_, a_v_, a_i_, a_j_ have been initialized
 * to the dimension of matrix A, the number of nonzeros it has, and its values,
 * row offsets, and column arrays.
 *
 * @post buffer_gpu_ is allocated to the size needed for the numerical 
 * Cholesky factorization.
 * info_ contains the symbolic factorization structure of A
 */

  void Analysis()
  {
    checkCudaErrors(cusparseCreateMatDescr(&descr_a_));
    checkCudaErrors(cusparseSetMatType(descr_a_, 
                                       CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(descr_a_, INDEX_BASE));
    checkCudaErrors(cusolverSpCreate(&handle_cusolver_));
    checkCudaErrors(cusolverSpCreateCsrcholInfo(&info_));
    
    checkCudaErrors(cusolverSpXcsrcholAnalysis(handle_cusolver_,
                                               n_, 
                                               nnz_, 
                                               descr_a_, 
                                               a_i_, 
                                               a_j_, 
                                               info_));
    size_t internalDataInBytes = 0; 
    size_t workspaceInBytes = 0;
    checkCudaErrors(cusolverSpDcsrcholBufferInfo(handle_cusolver_, 
                                                 n_, 
                                                 nnz_, 
                                                 descr_a_, 
                                                 a_v_, 
                                                 a_i_,  
                                                 a_j_,
                                                 info_, 
                                                 &internalDataInBytes, 
                                                 &workspaceInBytes));
    allocateBufferOnDevice(&buffer_gpu_, workspaceInBytes);
  }
  
/* 
 * @brief Set pivot tolerance for numeric factorization
 *
 * @param tol - factorization tolerance 
 *
 * @pre tol is initialized to a double smaller than 1
 *
 * @post tol_ is set to tol
 */

  void set_pivot_tolerance(const double tol)
  {
    tol_ = tol;
  }

/* 
 * @brief Set matrix values (for optimization solver iterations after the first
 * there is no need to reconstruct the class, only the values should change).
 * This is for a new matrix with same nonzero structure.
 *
 * @param tol - double a_v
 *
 * @pre tol is initialized to a double smaller than 1
 *
 * @post tol_ is set to tol
 */

  void set_matrix_values(double* a_v) 
  {
    a_v_ = a_v;
  }

/* 
 * @brief numerical Cholesky factorization
 *
 * @param <NAME> <DESCRIPTION>
 *
 * @pre Member variables n_, nnz_, a_v_, a_i_, a_j_ have been initialized
 * to the dimension of matrix A, the number of nonzeros it has, and its values,
 * row offsets, and column arrays.
 * buffer_gpu_ is allocated to the size needed 
 * for the numerical Cholesky factorization.
 * info_ contains the symbolic factorization structure of A
 *
 * @post info contains the Cholesky factors of A
 */

  void Factorization()
  {
    int singularity = 0;
    checkCudaErrors(cusolverSpDcsrcholFactor(handle_cusolver_,
                                             n_,
                                             nnz_,
                                             descr_a_,
                                             a_v_, 
                                             a_i_, 
                                             a_j_, 
                                             info_, 
                                             buffer_gpu_));
    checkCudaErrors(cusolverSpDcsrcholZeroPivot(handle_cusolver_,
                                                info_, 
                                                tol_, 
                                                &singularity));
    if (singularity >= 0) {
      fprintf(stderr, "Error: H not invertible, singularity=%d\n", singularity);
    }
  }

/* 
 * @brief solves Ax = b for x via Cholesky factors of A
 *
 * @param x - empty vector for solution to Ax = b
 * b - right hand side
 *
 * @pre variable of size n_ is initialized to the right hand side
 * buffer_gpu_ is allocated to the size needed 
 * info_ contains the symbolic factorization structure of A
 *
 * @post x - contains solution to Ax = b 
 */

  void Solve(double* x, double* b)
  {
    checkCudaErrors(cusolverSpDcsrcholSolve(handle_cusolver_, 
                                            n_, 
                                            b, 
                                            x, 
                                            info_, 
                                            buffer_gpu_));
  }


private:

  // member variables
  int n_; // dimension of A 
  int nnz_; // nonzeros of A
  double tol_ = 1e-12; // pivot tolerance for Cholesky

  int*    a_i_; // row offsets of csr storage of A
  int*    a_j_; // column pointers of csr storage of A
  double* a_v_; // value pointers of csr storage of A

  cusolverSpHandle_t handle_cusolver_ = NULL; 
  cusparseMatDescr_t descr_a_ = NULL; 
  csrcholInfo_t info_ = NULL; // stores Cholesky factorization
  void* buffer_gpu_ = nullptr; // buffer for Cholesky factorization

};

