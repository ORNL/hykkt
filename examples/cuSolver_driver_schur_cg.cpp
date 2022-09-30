#include <unistd.h>
#include <cstdlib>
#include <assert.h>
#include <ctype.h>
#include <string.h>
#include <iostream>
#include <memory>
#include <string>
#include "input_functions.hpp"
#include "SchurComplementConjugateGradient.hpp"
#include "matrix_vector_ops_cuda.hpp"
#include "cuda_memory_utils.hpp"
#include "MMatrix.hpp"
#include "cusparse_utils.hpp"
#include "CholeskyClass.hpp"


#include "cuda_check_errors.hpp"


/**
  *@brief Driver file demonstrates use of both Cholesky 
          decomposition and Schur complement using Cholesky
          and SchurComplementConjugateGradient classes
  * 
  *@pre Only NORMAL mtx matrices are read; don't have to be sorted
 **/

int main(int argc, char *argv[]){

  const double tol = 1e-12;
  const double abs_tol = 1e-6;
  /*** cuda stuff ***/

  cusparseHandle_t handle = 0;
  checkCudaErrors(cusparseCreate(&handle));
  cublasHandle_t handle_cublas;
  checkCudaErrors(cublasCreate(&handle_cublas));

  char const* const jc_file_name = argv[1];
  char const* const h_file_name = argv[2];
  char const* const rhs_file_name = argv[3];

  MMatrix mat_h  = MMatrix();
  MMatrix mat_jc = MMatrix();
  
  read_mm_file_into_coo(jc_file_name, mat_jc, 3);
  printf("\n/******* Matrix size: %d x %d nnz: %d *******/\n\n", 
         mat_jc.n_, 
         mat_jc.m_, 
         mat_jc.nnz_); 
  coo_to_csr(mat_jc);

  read_mm_file_into_coo(h_file_name, mat_h, 3);
  printf("\n/******* Matrix size: %d x %d nnz: %d *******/\n\n", 
         mat_h.n_, 
         mat_h.m_, 
         mat_h.nnz_);  
  coo_to_csr(mat_h);
  
  double* h_rhs = new double[mat_h.n_];
  
  read_rhs(rhs_file_name, h_rhs);
  printf("RHS reading completed ..............................\n"); 

  /*** now copy data to GPU and format convert ***/
  double* d_rhs = nullptr;
  int*    h_i   = nullptr; //columns and rows of H
  int*    h_j   = nullptr;
  double* h_v   = nullptr;
  int*    jc_i  = nullptr;//columns and rows of JC
  int*    jc_j  = nullptr;
  double* jc_v  = nullptr;
 
  cloneVectorToDevice(mat_jc.n_, &h_rhs, &d_rhs);
  cloneMatrixToDevice(&mat_jc, &jc_i, &jc_j, &jc_v);
  cloneMatrixToDevice(&mat_h, &h_i, &h_j, &h_v);
  
  /*** Transpose JC */
  double* jc_t_v = nullptr;
  int*    jc_t_j = nullptr;
  int*    jc_t_i = nullptr;
  
  allocateMatrixOnDevice(mat_jc.m_, mat_jc.nnz_, &jc_t_i, &jc_t_j, &jc_t_v);
  
  void* buffer;
  transposeMatrixOnDevice(handle,
      mat_jc.n_,
      mat_jc.m_, 
      mat_jc.nnz_, 
      jc_i, 
      jc_j, 
      jc_v, 
      jc_t_i, 
      jc_t_j, 
      jc_t_v,
      &buffer,
      false);
  deleteOnDevice(buffer);
  /*** matrix structures */
  cusparseSpMatDescr_t jc_desc;
  createCsrMat(&jc_desc,
      mat_jc.n_,
      mat_jc.m_,
      mat_jc.nnz_,
      jc_i,
      jc_j,
      jc_v);

  cusparseSpMatDescr_t jc_t_desc;
  createCsrMat(&jc_t_desc,
      mat_jc.m_,
      mat_jc.n_,
      mat_jc.nnz_,
      jc_t_i,
      jc_t_j,
      jc_t_v);
 
  /*** malloc x */
  double* h_x = new double[mat_jc.n_]{0.0};
  double* d_x = nullptr;
  
  cloneVectorToDevice(mat_h.n_, &h_x, &d_x);
  
  //factorize H
  CholeskyClass* cc = new CholeskyClass(mat_h.n_, mat_h.nnz_, h_v, h_i, h_j);
  cc->symbolic_analysis();
  cc->set_pivot_tolerance(tol);
  cc->numerical_factorization();


//class implementation
  SchurComplementConjugateGradient* sccg = 
    new SchurComplementConjugateGradient(jc_desc, 
        jc_t_desc, 
        d_x, 
        d_rhs, 
        mat_jc.n_, 
        mat_jc.m_, 
        cc,
        handle, 
        handle_cublas);
  
  sccg->setup();
  sccg->solve();
 

  int fails = 0;
  copyVectorToHost(mat_jc.n_, d_x, h_x);
  if (fabs(h_x[0] - 22.171865776354700) > abs_tol){
    fails ++;
    printf("x[0] incorrect, x[0] = %32.32g\n", h_x[0]);
  }
  if (fabs(h_x[6] + 4.446628667344612e+03) > abs_tol){
    fails ++;
    printf("x[6] incorrect, x[6] = %32.32g\n", h_x[6]);
  }
  
  delete sccg;
  delete cc;
  
  deleteOnDevice(d_rhs);
  deleteOnDevice(d_x);
  
  deleteMatrixOnDevice(h_i, h_j, h_v);
  deleteMatrixOnDevice(jc_i, jc_j, jc_v);
  deleteMatrixOnDevice(jc_t_i, jc_t_j, jc_t_v);
  
  delete [] h_rhs; 
  delete [] h_x;
  
  checkCudaErrors(cusparseDestroy(handle));
  checkCudaErrors(cublasDestroy(handle_cublas));
  
  return fails;
}
