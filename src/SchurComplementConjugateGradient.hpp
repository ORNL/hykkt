#pragma once

#include "matrix_vector_ops.hpp"
#include "vector_vector_ops.hpp"
#include "CholeskyClass.hpp"

class SchurComplementConjugateGradient
{

public:
  // default constructor
  SchurComplementConjugateGradient();

  // parametrized constructor
  SchurComplementConjugateGradient(cusparseSpMatDescr_t jc_desc, 
                                   cusparseSpMatDescr_t jct_desc, 
                                   double* x0, 
                                   double* b, 
                                   int n, 
                                   int m, 
                                   CholeskyClass* cc,
                                   cusparseHandle_t handle, 
                                   cublasHandle_t handle_cublas);

  // destructor
  ~SchurComplementConjugateGradient();
 
/*
 * @brief allocate and create dense vector descriptors
 *
 * @pre Member variables m_ and n_ are initialized to the row and
 *      column number of the JC
 * @post y_, z_, r_, w_, p_, s_ are all allocated on the device and vecx_,
 *       vecb_, vecy_, vecz_, vecr_, vecw_, vecp_, vecs_ are the respective
 *       dense vector descriptors; y_ and z_ have size m_, the other vectors
 *       have size n_, and ycp_ is now a vector of size m_ with values of 0
*/
  void allocate();

/*
 * @brief copy ycp_ onto the device and copy device vectors
 *
 * @pre Member variables m_, n_, y_, b_, r_, w_ are initialized to the
 *      number of rows in JC, number of columns in JC, and the vectors
 *      used in solving the Schur Complement
 * @post ycp_ is copied onto the device on y_, y_ is copied onto z_,
 *       b_ onto r_, b_ onto w_, r_ onto p_, and w_ onto s_
*/
  void setup();

/*
 * @brief solve Schur Complement using conjugate gradient 
 *
 * @pre Member variables handle_, handle_cublas_,
 *      all dense vector descriptors, all vectors on device, tol_m_,
 *      itmax_, gam_i_, gam_i1_, alpha_, beta_, minalpha_, are
 *      initialized to a cuSPARSE handler, a cuBLAS handler, initialized
 *      to their respective device vectors, allocated on the device,
 *      initialized to the max tolerance of the conjugate gradient, the
 *      maximum number of iterations for conjugate gradient, and the
 *      respective scalar values
 * @post iterative solver on the Schur complement using Cholesky solver
*/
  int solve();

/*
 * @brief set solver tolerance for conjugate gradient
 *
 * @param tol - the new tolerance level
 *
 * @pre
 * @post tol_ is now set to tol
*/
  void set_solver_tolerance(double tol);

/*
 * @brief set the maximum number of iterations for conjugate gradient
 *
 * @param itmax - the new max iterations
 *
 * @pre
 * @post itmax_ is now set to itmax
*/
  void set_solver_itmax(int itmax);

private:

  // member variables
  int n_; // dimension of outer system
  int m_; // dimension of inner system
  int itmax_  = 100; // maximum iterations for conjugate gradient
  double tol_ = 1e-12; // solver tolerance for Schur

  cusparseSpMatDescr_t jc_desc_; // sparse matrix descriptor for JC
  cusparseSpMatDescr_t jct_desc_; // sparse matrix descriptor for JC transform

  cusparseHandle_t handle_; // handle to the cuSPARSE library context
  cublasHandle_t handle_cublas_; //handle to the cuBLAS library context

  // scalars used for conjugate gradient
  double gam_i_;
  double beta_ = 0;
  double delta_;
  double alpha_;
  double minalpha_;
  double gam_i1_;

  double* x0_; // lhs of entire system
  double* b_; // rhs of entire system
  double* ycp_; // used to copy to y_
  double* y_; // internal rhs of system
  double* z_; // internal lhs of system
  double* r_; // residual
  
  // vectors used for conjugate gradient
  double* p_;
  double* s_;
  double* w_;
  
  // resepctive dense vector descriptors on host
  cusparseDnVecDescr_t vecx_ = NULL;
  cusparseDnVecDescr_t vecb_ = NULL;
  cusparseDnVecDescr_t vecy_ = NULL;
  cusparseDnVecDescr_t vecz_ = NULL;
  cusparseDnVecDescr_t vecr_ = NULL;
  cusparseDnVecDescr_t vecw_ = NULL;
  cusparseDnVecDescr_t vecp_ = NULL;
  cusparseDnVecDescr_t vecs_ = NULL;
  
  CholeskyClass* cc_; // cholesky factorization on 1,1 block
 
  // timers used for testing
  double timeIO_ = 0.0;
  struct timeval t1_;
  struct timeval t2_;

};

