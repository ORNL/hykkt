#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <ctype.h>
#include <string.h>
#include <cusolver_common.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include <algorithm>
#include "cusolverSp.h"
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverRf.h>
#include <iostream>
#include <memory>
#include <string>
#include <tgmath.h>
#include <math.h>
#include <matrix_vector_ops.hpp>

#define ruiz_its 2
#define tol 1e-8

typedef struct {
  int * coo_rows;
  int * coo_cols;
  double * coo_vals;

  int * csr_ia;

  int n;
  int m;
  int nnz;
} mmatrix;

int main(int argc, char *argv[]){

  cusparseStatus_t status;
  cusparseHandle_t handle=NULL;
  status= cusparseCreate(&handle);
  cusolverSpHandle_t handle_cusolver = NULL;
  cusolverSpCreate(&handle_cusolver);
  cusparseMatDescr_t descrA; 
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA,  CUSPARSE_INDEX_BASE_ZERO);
  cublasHandle_t handle_cublas;
  cublasCreate(&handle_cublas);
  
  int n=1024, *A_i, *A_j;
  double *A_v, *H_rhs, *D_rhs;
  int *H_i, *H_j;
  double *H_v;
  mmatrix *A = (mmatrix *) calloc(1, sizeof(mmatrix));
  mmatrix *H = (mmatrix *) calloc(1, sizeof(mmatrix));
  A->n=n;
  A->m=n;
  A->nnz=2*n-1;
  A->coo_vals =(double*)  calloc(A->nnz, sizeof(double));
  A->csr_ia =(int*)  calloc((A->n)+1, sizeof(int));
  A->coo_cols = (int *)  calloc(A->nnz, sizeof(int));
  H->n=n;
  H->m=n;
  H->nnz=n;
  int totn = (H->n)+(A->n);
  H->coo_vals =(double*)  calloc(H->nnz, sizeof(double));
  H->csr_ia =(int*)  calloc((H->n)+1, sizeof(int));
  H->coo_cols = (int *)  calloc(H->nnz, sizeof(int));
  H_rhs =(double*)  calloc(totn, sizeof(double));
  
  int i;
  //initialize the matrix and the RHS
  A->csr_ia[0]=0;
  for(i=0;i<(A->n);i++){
    if (i){
      A->coo_vals[i*2-1]=i+1;
      A->coo_cols[i*2-1]=i-1;
      A->csr_ia[i]=i*2-1;
    }
    A->coo_vals[i*2]=0;
    A->coo_cols[i*2]=i;
  }
  A->csr_ia[i]=A->nnz;
  for(i=0;i<(H->n);i++){
    H->coo_vals[i]=sqrt(n);
    H->csr_ia[i]=i;
    H->coo_cols[i]=i;
  }
  H->csr_ia[i]=H->nnz;
  for(i=0;i<totn;i++){
    H_rhs[i]=1;
  }

  cudaMalloc((void**)&A_v, (A->nnz)*sizeof(double));
  cudaMalloc((void**)&A_j, (A->nnz)*sizeof(int));
  cudaMalloc((void**)&A_i, ((A->n)+1)*sizeof(int));
  cudaMalloc((void**)&H_v, (H->nnz)*sizeof(double));
  cudaMalloc((void**)&H_j, (H->nnz)*sizeof(int));
  cudaMalloc((void**)&H_i, ((H->n)+1)*sizeof(int));
  cudaMalloc((void**)&D_rhs, totn*sizeof(double));

  cudaMemcpy(D_rhs, H_rhs, sizeof(double)*totn, cudaMemcpyHostToDevice);
  cudaMemcpy(A_v, A->coo_vals, sizeof(double)*A->nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(A_j, A->coo_cols, sizeof(int)*A->nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(A_i, A->csr_ia, sizeof(int)*((A->n)+1), cudaMemcpyHostToDevice);
  cudaMemcpy(H_v, H->coo_vals, sizeof(double)*H->nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(H_j, H->coo_cols, sizeof(int)*H->nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(H_i, H->csr_ia, sizeof(int)*((H->n)+1), cudaMemcpyHostToDevice);

// Test adding to diagonal
  fun_add_diag(A->n, 1.0, A_i, A_j, A_v);

//Transpose A to have its upper triangular part
  double* At_v;
  int *At_i, *At_j;
  cudaMalloc(&At_v, (A->nnz)*sizeof(double));
  cudaMalloc(&At_j, (A->nnz)*sizeof(int));
  cudaMalloc(&At_i, ((A->m)+1)*sizeof(int));
  void *buffercsr=NULL;
  size_t buffersize;
  printf("Transpose A \n");
  status = cusparseCsr2cscEx2_bufferSize(handle, A->n,A->m, A->nnz,A_v,A_i,A_j,
      At_v,At_i,At_j,CUDA_R_64F,CUSPARSE_ACTION_NUMERIC,CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG1,&buffersize);
  printf("Buffer allocation status %d\n",status);
  cudaMalloc(&buffercsr, sizeof(char)*buffersize);
  printf("Buffer size is %d\n",buffersize);
  printf("A dimensions are %d by %d with %d nnz\n", A->n,A->m,A->nnz);
  cusparseCsr2cscEx2(handle,A->n,A->m, A->nnz,A_v,A_i,A_j,At_v,At_i,At_j,
      CUDA_R_64F,
      CUSPARSE_ACTION_NUMERIC,CUSPARSE_INDEX_BASE_ZERO,CUSPARSE_CSR2CSC_ALG1,buffercsr);
  printf("tanspose status %d\n",status);

  cudaMemcpy(A->coo_vals, A_v, sizeof(double)*A->nnz, cudaMemcpyDeviceToHost);
  cudaMemcpy(A->coo_cols, A_j, sizeof(int)*A->nnz, cudaMemcpyDeviceToHost);
  cudaMemcpy(A->csr_ia, A_i, sizeof(int)*((A->n)+1), cudaMemcpyDeviceToHost);
  cudaMemcpy(H->coo_vals, H_v, sizeof(double)*H->nnz, cudaMemcpyDeviceToHost);
  cudaMemcpy(H->coo_cols, H_j, sizeof(int)*H->nnz, cudaMemcpyDeviceToHost);
  cudaMemcpy(H->csr_ia, H_i, sizeof(int)*((H->n)+1), cudaMemcpyDeviceToHost);
  double* Ah_v =(double *) calloc((A->nnz), sizeof(double));
  int* Ah_j =(int *) calloc((A->nnz), sizeof(int));
  int* Ah_i =(int *) calloc((A->m+1), sizeof(int));
  cudaMemcpy(Ah_v, At_v, sizeof(double)*A->nnz, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ah_j, At_j, sizeof(int)*A->nnz, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ah_i, At_i, sizeof(int)*((A->m)+1), cudaMemcpyDeviceToHost);
#if 0
  printf("printing A\n");
  for(i=0; i<n; i++)
  {
    printf("Row %d\n",i);
    for (int j=A->csr_ia[i]; j<A->csr_ia[i+1]; j++)
    {
      printf("Column %d, value %f\n", A->coo_cols[j], A->coo_vals[j]);
    }
  }
#endif
#if 0
  printf("printing H\n");
  for(i=0; i<n; i++)
  {
    printf("Row %d\n",i);
    for (int j=H->csr_ia[i]; j<H->csr_ia[i+1]; j++)
    {
      printf("Column %d, value %f\n", H->coo_cols[j], H->coo_vals[j]);
    }
  }
#endif
#if 0
  printf("printing A transpose\n");
  for(i=0; i<2; i++)
  {
    printf("Row %d\n",i);
    for (int j=Ah_i[i]; j<Ah_i[i+1]; j++)
    {
      printf("Column %d, value %f\n", Ah_j[j], Ah_v[j]);
    }
  }
#endif
double *max_d,*scale;
cudaMalloc(&max_d, totn*sizeof(double));
cudaMalloc(&scale, totn*sizeof(double));
double* max_h = (double *) calloc(totn, sizeof(double));
for(i=0;i<totn;i++){
  max_h[i]=1; 
}
cudaMemcpy(max_d, max_h, sizeof(double)*totn, cudaMemcpyHostToDevice);
/*
     This is where the Ruiz magic happens
     */
for(i=0;i<ruiz_its;i++){
  fun_adapt_row_max(n, totn, H_v, H_i, H_j, A_v, A_i, A_j,At_v,At_i, At_j, scale);
  //if(i==ruiz_its-1) flag=0;
  fun_adapt_diag_scale(n, totn, H_v, H_i, H_j, A_v, A_i, A_j, At_v, At_i, At_j,
      scale, D_rhs, &D_rhs[n], max_d);
}
  cudaMemcpy(A->coo_vals, A_v, sizeof(double)*A->nnz, cudaMemcpyDeviceToHost);
  cudaMemcpy(A->coo_cols, A_j, sizeof(int)*A->nnz, cudaMemcpyDeviceToHost);
  cudaMemcpy(A->csr_ia, A_i, sizeof(int)*((A->n)+1), cudaMemcpyDeviceToHost);
  cudaMemcpy(H->coo_vals, H_v, sizeof(double)*H->nnz, cudaMemcpyDeviceToHost);
  cudaMemcpy(H->coo_cols, H_j, sizeof(int)*H->nnz, cudaMemcpyDeviceToHost);
  cudaMemcpy(H->csr_ia, H_i, sizeof(int)*((H->n)+1), cudaMemcpyDeviceToHost);
  cudaMemcpy(Ah_v, At_v, sizeof(double)*A->nnz, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ah_j, At_j, sizeof(int)*A->nnz, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ah_i, At_i, sizeof(int)*((A->m)+1), cudaMemcpyDeviceToHost);
  cudaMemcpy(max_h, max_d, sizeof(double)*totn, cudaMemcpyDeviceToHost);
  cudaMemcpy(H_rhs, D_rhs, sizeof(double)*totn, cudaMemcpyDeviceToHost);
#if 0
  printf("max_d\n");
  for(i=0; i<2; i++)
  {
    printf("max_d at row %d = %32.32g\n",i,max_h[i]);
  }
#endif
#if 0
  printf("printing A\n");
  for(i=n-2; i<n; i++)
  {
    printf("Row %d\n",i);
    for (int j=A->csr_ia[i]; j<A->csr_ia[i+1]; j++)
    {
      printf("Column %d, value %f\n", A->coo_cols[j], A->coo_vals[j]);
    }
  }
#endif
#if 0 
  printf("printing H\n");
  for(i=0; i<2; i++)
  {
    printf("Row %d\n",i);
    for (int j=H->csr_ia[i]; j<H->csr_ia[i+1]; j++)
    {
      printf("Column %d, value %f\n", H->coo_cols[j], H->coo_vals[j]);
    }
  }
#endif
#if 0
  printf("printing A transpose\n");
  for(i=n-2; i<n; i++)
  {
    printf("Row %d\n",i);
    for (int j=Ah_i[i]; j<Ah_i[i+1]; j++)
    {
      printf("Column %d, value %f\n", Ah_j[j], Ah_v[j]);
    }
  }
#endif
// Test to compare with MATLAB
  int fails=0;
  if (fabs(H->coo_vals[n/2-1]-0.062378167641326)>tol){
    fails++;
    printf("H not scaled correctly H[n/2-1][n/2-1] = %32.32g\n",H->coo_vals[(H->n)/2-1]);
  }
  if (fabs(A->coo_vals[(A->nnz)-1]-0.005524271728020)>tol){
    fails++;
    printf("A not scaled correctly A[n-1][n-1] = %32.32g\n",A->coo_vals[(A->nnz)-1]);
  }
  if (fabs(Ah_v[1]-0.5)>tol){
    fails++;
    printf("At not scaled correctly At[0][1] = %32.32g \n",Ah_v[1]);
  }
  if (fabs(H_rhs[n/2-1]-0.044151078568835)>tol){
    fails++;
    printf("rhs not scaled correctly H_rhs[n/2-1]= %32.32g\n", H_rhs[n/2-1]);
  }
  if (fabs(max_h[n/2-1]-0.044151078568835)>tol){
    fails++;
    printf("Incorrect scaling factor max_h[n/2-1] = %32.32g\n", max_h[n/2-1]);
  }
  if (fabs(H_rhs[3*n/2-1]-0.044194173824159)>tol){
    fails++;
    printf("rhs not scaled correctly H_rhs[3*n/2-1]= %32.32g\n", H_rhs[3*n/2-1]);
  }
  if (fabs(max_h[3*n/2-1]-0.044194173824159)>tol){
    fails++;
    printf("Incorrect scaling factor max_h[3*n/2-1] = %32.32g\n", max_h[3*n/2-1]);
  }
  if (fails==0) printf("All tests passed\n");
  else 
  {
    printf("%d tests failed\n",fails);
    return 1;
  }
#if 0 //printing
  printf("printing A\n");
  for(i=0; i<n; i++)
  {
    printf("Row %d\n",i);
    for (int j=A->csr_ia[i]; j<A->csr_ia[i+1]; j++)
    {
      printf("Column %d, value %f\n", A->coo_cols[j], A->coo_vals[j]);
    }
  }
#endif
#if 0
  printf("printing A transpose\n");
  for(i=0; i<n; i++)
  {
    printf("Row %d\n",i);
    for (int j=Ah_i[i]; j<Ah_i[i+1]; j++)
    {
      printf("Column %d, value %f\n", Ah_j[j], Ah_v[j]);
    }
  }
#endif
#if 0 //We're no longer concatenating, so no test for this
  //now that matrices are different, test concatenation
  double* C_v;
  int *C_i, *C_j;
  cudaMalloc(&C_v, (A->nnz)*2*sizeof(double));
  cudaMalloc(&C_j, (A->nnz)*2*sizeof(int));
  cudaMalloc(&C_i, ((A->m)*2+1)*sizeof(int));
  blockSize = 32;
  numBlocks = (n + A->m + 1 + blockSize - 1) / blockSize;
  concatenate<<<blockSize,numBlocks>>>(n, n, A->nnz, A->nnz, A_v, A_i, A_j,
   At_v, At_i, At_j, C_v, C_i, C_j);
// Copy back
  double* Ch_v = (double *) calloc((A->nnz)*2, sizeof(double));
  int* Ch_j = (int *) calloc((A->nnz)*2, sizeof(int));
  int* Ch_i = (int *) calloc((A->n)*2+1, sizeof(int));
  cudaMemcpy(Ch_v, C_v, sizeof(double)*(A->nnz)*2, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ch_j, C_j, sizeof(int)*(A->nnz)*2, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ch_i, C_i, sizeof(int)*((A->n)*2+1), cudaMemcpyDeviceToHost);
  for(i=0; i<n*2; i++)
  {
    printf("Row %d\n",i);
    for (int j=Ch_i[i]; j<Ch_i[i+1]; j++)
    {
      printf("Column %d, value %f\n", Ch_j[j], Ch_v[j]);
    }
  }
#endif
  return 0;
}
