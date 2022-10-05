#include <iostream>
#include "input_functions.hpp"
#include "CholeskyClass.hpp"
#include "cuda_memory_utils.hpp"
#include "MMatrix.hpp"

/**
  *@brief Driver file demonstrates use of Cholesky class Analysis,
          Factorization, and Solve
  *
  *@pre Only NORMAL mtx matrices are read; don't have to be sorted
**/

int main(int argc, char *argv[]){

  const double tol = 1e-12;
  const double abs_tol = 3e-2;
	/*** cuda stuff ***/

	char const* const h_file_name = argv[1];
	char const* const rhs_file_name = argv[2];
  
  MMatrix mat_h = MMatrix();

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
  double* h_v = nullptr;
	int* h_j = nullptr;
  int* h_i = nullptr; //columns and rows of H
 
  cloneVectorToDevice(mat_h.n_, &h_rhs, &d_rhs);
  cloneMatrixToDevice(&mat_h, &h_i, &h_j, &h_v);

  /*** malloc x */
  double* h_x = new double[mat_h.n_]{0.0};
	double* d_x;

  cloneVectorToDevice(mat_h.n_, &h_x, &d_x);
  //factorize mat_h

  CholeskyClass* cc = new CholeskyClass(mat_h.n_, mat_h.nnz_, h_v, h_i, h_j);
  cc->symbolic_analysis();
  cc->set_pivot_tolerance(tol);
  // cc.set_matrix_values(H_a); // needed only at subsequent iterations
  cc->numerical_factorization();
  cc->solve(d_x, d_rhs);
  
  int fails = 0;
  copyVectorToHost(mat_h.n_, d_x, h_x);
  if (fabs(h_x[0] + 97.663191790329750) > abs_tol){
    fails ++;
    printf("x[0] incorrect, x[0] = %32.32g\n", h_x[0]);
  }
  if (fabs(h_x[6] + 1.389053208413921) > abs_tol){
    fails ++;
    printf("x[6] incorrect, x[6] = %32.32g\n", h_x[6]);
  }
  delete cc;
  
  deleteOnDevice(d_rhs);
  deleteOnDevice(d_x);
  deleteMatrixOnDevice(h_i, h_j, h_v);
  
  delete [] h_rhs;
	delete [] h_x;
	
  return fails;
}
