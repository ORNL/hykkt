#pragma once

#include <cusparse.h>
#include "cusolverSp.h"
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

/*
 * Class for symbolic and numerical factorization of A,
 * triangular solve of Ax = b
 */
class CholeskyClass
{

public:
  
  // constructor
  CholeskyClass(int n, int nnz, double* a_v, int* a_i, int* a_j);

  // destructor
  ~CholeskyClass();

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
  void symbolic_analysis();

/* 
 * @brief Set pivot tolerance for numeric factorization
 *
 * @param[in] tol - factorization tolerance to set in the class
 *
 * @pre tol is initialized to a double smaller than 1
 *
 * @post tol_ is set to tol
 */
  void set_pivot_tolerance(const double tol);

/* 
 * @brief Set matrix values (for optimization solver iterations after the first
 * there is no need to reconstruct the class, only the values should change).
 * This is for a new matrix with same nonzero structure.
 *
 * @param[in] tol - double a_v to set in the class
 *
 * @pre tol is initialized to a double smaller than 1
 *
 * @post tol_ is set to tol
 */
  void set_matrix_values(double* a_v); 

/* 
 * @brief numerical Cholesky factorization
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
  void numerical_factorization();

/* 
 * @brief solves Ax = b for x via Cholesky factors of A
 *
 * @param[in, out] x - empty vector for solution to Ax = b, set to solution
 * @param[in] b - right hand side
 *
 * @pre variable of size n_ is initialized to the right hand side
 * buffer_gpu_ is allocated to the size needed 
 * info_ contains the symbolic factorization structure of A
 *
 * @post x - contains solution to Ax = b 
 */
  void solve(double* x, double* b);

private:

  // member variables
  int n_; // dimension of A 
  int nnz_; // nonzeros of A
  double tol_ = 1e-12; // pivot tolerance for Cholesky

  int*    a_i_; // row offsets of csr storage of A
  int*    a_j_; // column pointers of csr storage of A
  double* a_v_; // value pointers of csr storage of A

  //handle to the cuSPARSE library context
  cusolverSpHandle_t handle_cusolver_ = NULL;
  cusparseMatDescr_t descr_a_ = NULL;//descriptor for matrix A 
  csrcholInfo_t info_ = NULL; // stores Cholesky factorization
  void* buffer_gpu_ = nullptr; // buffer for Cholesky factorization

};

