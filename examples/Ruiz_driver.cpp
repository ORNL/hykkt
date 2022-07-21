#include <stdio.h>
#include <assert.h>

#include <matrix_vector_ops.hpp>
#include <RuizClass.hpp>
#include <MMatrix.hpp>
#include <cuda_memory_utils.hpp>

/*
 * @brief Initializes matrices with testing values to demonstrate Ruiz scaling 
 *
 * @param mat_a - Matrix to be initialized
 * mat_h - Matrix to be initialized
 * h_rhs - RHS to be initialized
 *
 * @pre h_rhs has size equal to mat_a.n_ + mat_h.n_
 *
 * @post mat_a, mat_h, h_rhs are initialized with test values
*/

void initializeTestMatrices(MMatrix& mat_a, MMatrix& mat_h, double* h_rhs)
{
  assert(mat_a.n_ == mat_h.n_);
  int i;
  int n = mat_a.n_;
  int totn = mat_a.n_ + mat_h.n_;
  //initialize the matrix and the RHS
  mat_a.csr_rows[0] = 0;
  for(i = 0; i < (mat_a.n_); i++){
    if(i){
      mat_a.coo_vals[i * 2 - 1] = i + 1;
      mat_a.coo_cols[i * 2 - 1] = i - 1;
      mat_a.csr_rows[i] = i * 2 - 1;
    }
    mat_a.coo_vals[i * 2] = 0;
    mat_a.coo_cols[i * 2] = i;
  }
  mat_a.csr_rows[i] = mat_a.nnz_;
  for(i = 0; i < (mat_h.n_); i++){
    mat_h.coo_vals[i] = sqrt(n);
    mat_h.csr_rows[i] = i;
    mat_h.coo_cols[i] = i;
  }
  mat_h.csr_rows[i] = mat_h.nnz_;
  for(i = 0; i < totn; i++){
    h_rhs[i] = 1;
  }
}

/*
  * @brief Driver demonstrates the use of RuizClass for Ruiz scaling
*/

int main(int argc, char *argv[])
{
  const double tol = 1e-8;
  // Size of matrix block  
  int n = 1024;
  // Create (1,2) block A
  MMatrix mat_a(n, n, 2 * n - 1);
  // Create (1,1) block H
  MMatrix mat_h(n, n, n);
  // Size of rhs vector
  int totn = mat_h.n_ + mat_a.n_;
  // Create rhs vector
  double* h_rhs = new double[totn]{0.0};

  // Initialize matrices to testing values
  initializeTestMatrices(mat_a, mat_h, h_rhs);

  // Create copy of (1,2) block on the device
  int* a_i;
  int* a_j;
  double* a_v;
  cloneMatrixToDevice(&mat_a, &a_i, &a_j, &a_v);  

  // Create copy of (1,2) block on the device
  int* h_i;
  int* h_j;
  double *h_v;
  cloneMatrixToDevice(&mat_h, &h_i, &h_j, &h_v);  

  // Create copy of the rhs vector on the device
  double* d_rhs = nullptr;
  allocateVectorOnDevice(totn, &d_rhs);
  copyVectorToDevice(totn, h_rhs, d_rhs);

  // Test adding to diagonal
  fun_add_diag(mat_a.n_, 1.0, a_i, a_j, a_v);

  // Transpose A to have its upper triangular part
  // Allocate matrix mat_at to store the transpose
  double* at_v;
  int* at_i;
  int* at_j;
  allocateMatrixOnDevice(mat_a.m_, mat_a.nnz_, &at_i, &at_j, &at_v);

  // Transpose A and store it in mat_at
  void* buffer;
  cusparseHandle_t handle;
  createSparseHandle(handle);
  transposeMatrixOnDevice(handle,
                          mat_a.n_,
                          mat_a.m_,
                          mat_a.nnz_,
                          a_i,
                          a_j,
                          a_v,
                          at_i,
                          at_j,
                          at_v,
                          &buffer,
                          false);

  deleteOnDevice(buffer);
  // Copy data to host
  copyMatrixToHost(a_i, a_j, a_v, mat_a);
  copyMatrixToHost(h_i, h_j, h_v, mat_h);
  MMatrix mat_at(mat_a.m_, mat_a.n_, mat_a.nnz_);
  copyMatrixToHost(at_i, at_j, at_v, mat_at);

  
  //Ruiz scaling
  const int ruiz_its = 2;
  double* max_h = new double[totn]{0.0};
  double* max_d = nullptr;
  allocateVectorOnDevice(totn, &max_d);
  
  RuizClass* rz = new RuizClass(ruiz_its, n, totn);
  rz->add_block11(h_v, h_i, h_j);
  rz->add_block12(at_v, at_i, at_j);
  rz->add_block21(a_v, a_i, a_j);
  rz->add_rhs1(d_rhs);
  rz->add_rhs2(&d_rhs[n]);
  rz->ruiz_scale();
  max_d = rz->get_max_d();

// Copy data back to the host
  copyMatrixToHost(a_i, a_j, a_v, mat_a);
  copyMatrixToHost(h_i, h_j, h_v, mat_h);
  copyMatrixToHost(at_i, at_j, at_v, mat_at);
  copyVectorToHost(totn, d_rhs, h_rhs);
  copyVectorToHost(totn, max_d, max_h);

  // Test to compare with MATLAB
  int fails = 0;
  if (fabs(mat_h.coo_vals[n / 2 - 1] - 0.062378167641326) > tol){
    fails++;
    printf("H not scaled correctly H[n/2-1][n/2-1] = %32.32g\n",
           mat_h.coo_vals[(mat_h.n_) / 2 - 1]);
  }
  if (fabs(mat_a.coo_vals[(mat_a.nnz_) - 1] - 0.005524271728020) > tol){
    fails++;
    printf("A not scaled correctly A[n-1][n-1] = %32.32g\n",
           mat_a.coo_vals[(mat_a.nnz_) - 1]);
  }
  if (fabs(mat_at.coo_vals[1] - 0.5) > tol){
    fails++;
    printf("mat_at not scaled correctly mat_at[0][1] = %32.32g \n",
           mat_at.coo_vals[1]);
  }
  if (fabs(h_rhs[n / 2 - 1] - 0.044151078568835) > tol){
    fails++;
    printf("rhs not scaled correctly h_rhs[n/2-1]= %32.32g\n", 
           h_rhs[n / 2 - 1]);
  }
  if (fabs(h_rhs[3 * n / 2 - 1] - 0.044194173824159) > tol){
    fails++;
    printf("rhs not scaled correctly h_rhs[3*n/2-1]= %32.32g\n", 
           h_rhs[3 * n / 2 - 1]);
  }
  if (fabs(max_h[32] - 0.171498585142) > tol){
    fails++;
    printf("max_d not scaled correctly max_d[32]= %32.32g\n", 
           max_h[32]);
  }

  if (fails==0){
    printf("All tests passed\n");
  } else{
    printf("%d tests failed\n",fails);
  }
  
  delete rz;
  
  deleteMatrixOnDevice(a_i, a_j, a_v); 
  deleteMatrixOnDevice(at_i, at_j, at_v); 
  deleteMatrixOnDevice(h_i, h_j, h_v);
  deleteOnDevice(d_rhs);
  delete [] h_rhs;
  delete [] max_h;

  return fails;
}
