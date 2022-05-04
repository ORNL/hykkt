#include <stdio.h>
#include <stdlib.h>
#include <cusolver_common.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include <algorithm>
#include "cusolverSp.h"
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverRf.h>
#include <schur_complement_cg.hpp>

/** Brief: Solves the equation JC H^{-1} JC^T x = b
  via Chronopoulous Gear conjugate gradient 
  Input: JC and JCt in csr format, factorized H,
  initial guss x0, rhs b, max iterations,convergence tolerance, 
  dimensions and nonzeros of JC, matrix description,
  handles for cusparse, cusolver, cublas
  Output: x0 is changed to the solution to JH^{-1}J^Tx=b
  Transpose Jc as we will need it repeatedly
*/
void schur_cg(cusparseSpMatDescr_t matJC, cusparseSpMatDescr_t matJCt,
    csrcholInfo_t dH, double* x0, double* b, const int itmax, const double tol,
    int n, int m, int nnz, void* buffer_gpu, cusparseHandle_t handle, 
    cusolverSpHandle_t handle_cusolver, cublasHandle_t handle_cublas)
{
  // Start of block - CG setup
  // create constants for multiplication and allocation - happens once
  double               one      = 1.0;
  double               zero     = 0.0;
  double               minusone = -1.0;
  cusparseDnVecDescr_t vecx     = NULL;
  cusparseCreateDnVec(&vecx, n, x0, CUDA_R_64F);
  cusparseDnVecDescr_t vecb = NULL;
  cusparseCreateDnVec(&vecb, n, b, CUDA_R_64F);
  // create vectors necessary
  double* ycp = (double*)calloc(m, sizeof(double));
  double *y, *z, *r, *p, *s, *w;
  for(int i = 0; i < m; i++)
  {
    ycp[i] = 0;
  }
  cudaMalloc((void**)&y, m * sizeof(double));
  cudaMalloc((void**)&z, m * sizeof(double));
  cudaMalloc((void**)&r, n * sizeof(double));
  cudaMalloc((void**)&w, n * sizeof(double));
  cudaMalloc((void**)&p, n * sizeof(double));
  cudaMalloc((void**)&s, n * sizeof(double));
  //  Initializing values - happens every iteration
  cudaMemcpy(y, ycp, sizeof(double) * (m), cudaMemcpyHostToDevice);
  cudaMemcpy(z, y, sizeof(double) * (m), cudaMemcpyDeviceToDevice);
  cudaMemcpy(r, b, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
  cudaMemcpy(w, b, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
  cudaMemcpy(p, r, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
  cudaMemcpy(s, w, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
  double timeIO = 0.0;

  //  Allocation - happens once
  cusparseDnVecDescr_t vecy = NULL;
  cusparseCreateDnVec(&vecy, m, y, CUDA_R_64F);
  cusparseDnVecDescr_t vecz = NULL;
  cusparseCreateDnVec(&vecz, m, z, CUDA_R_64F);
  cusparseDnVecDescr_t vecr = NULL;
  cusparseCreateDnVec(&vecr, n, r, CUDA_R_64F);
  cusparseDnVecDescr_t vecw = NULL;
  cusparseCreateDnVec(&vecw, n, w, CUDA_R_64F);
  cusparseDnVecDescr_t vecp = NULL;
  cusparseCreateDnVec(&vecp, n, p, CUDA_R_64F);
  cusparseDnVecDescr_t vecs = NULL;
  cusparseCreateDnVec(&vecs, n, s, CUDA_R_64F);
  struct timeval t1, t2;
  gettimeofday(&t1, 0);
  /* This is zero anyways
  size_t         bufferSizet = 0, bufferSize = 0;
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matJCt, vecx, &zero, vecy,
    CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSizet);
  void* buffert = NULL;
  cudaMalloc(&buffert, bufferSizet);
  */
  // Start of block - 0 iteration opertaions for Chronopoulos Gear CG (every iteration)
  fun_SpMV(one, matJCt, vecx, zero, vecy);
  cusolverSpDcsrcholSolve(handle_cusolver, m, y, z, dH, buffer_gpu);
  /* This is zero anyways
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusone, matJC, vecz, &one,
    vecr, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
  void* buffer = NULL;
  cudaMalloc(&buffer, bufferSize);
  */
  fun_SpMV(minusone, matJC, vecz, one, vecr);
  double gam_i;
#if 0
	double *h_r =(double*)  calloc(n, sizeof(double));
  cudaMemcpy(h_r, r, sizeof(double)*(n), cudaMemcpyDeviceToHost);
  for(int i=n-10; i<n; i++){
    printf("h_r[%d] = %f\n", i, h_r[i]);
  }
#endif
  cublasDdot(handle_cublas, n, r, 1, r, 1, &gam_i);
  // printf("Iteration 0 gamma = %f\n", gam_i);
  // product with w=Ar starts here
  fun_SpMV(one, matJCt, vecr, zero, vecy);
  cusolverSpDcsrcholSolve(handle_cusolver, m, y, z, dH, buffer_gpu);
  fun_SpMV(one, matJC, vecz, zero, vecw);
  double beta = 0, delta, alpha, gam_i1;
  cublasDdot(handle_cublas, n, w, 1, r, 1, &delta);
  alpha           = gam_i / delta;
  double minalpha = -alpha;
  //  printf("Iteration 0 delta = %f, gamma = %f, alpha = %f\n", delta, gam_i, alpha);
  // Start of block - Main CG iteration
  for(int i = 0; i < itmax; i++)
  {
    cublasDscal(handle_cublas, n, &beta, p, 1);
    cublasDaxpy(handle_cublas, n, &one, r, 1, p, 1);

    cublasDscal(handle_cublas, n, &beta, s, 1);
    cublasDaxpy(handle_cublas, n, &one, w, 1, s, 1);

    cublasDaxpy(handle_cublas, n, &alpha, p, 1, x0, 1);
    minalpha = -alpha;
    cublasDaxpy(handle_cublas, n, &minalpha, s, 1, r, 1);

    cublasDdot(handle_cublas, n, r, 1, r, 1, &gam_i1);
    if(sqrt(gam_i1) < tol)
    {
      printf("Convergence occured at iteration %d\n", i);
      break;
    }
    // product with w=Ar starts here
    fun_SpMV(one, matJCt, vecr, zero, vecy);
    cusolverSpDcsrcholSolve(handle_cusolver, m, y, z, dH, buffer_gpu);
    fun_SpMV(one, matJC, vecz, zero, vecw);

    cublasDdot(handle_cublas, n, w, 1, r, 1, &delta);
    beta  = gam_i1 / gam_i;
    gam_i = gam_i1;
    alpha = gam_i / (delta - beta * gam_i / alpha);
  }
  gettimeofday(&t2, 0);
  timeIO = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  printf("time for CG ev(ms). : %16.16f\n", timeIO);
  printf("Error is %32.32g \n", sqrt(gam_i1));
  free(ycp);
  cudaFree(y);
  cudaFree(z);
  cudaFree(r);
  cudaFree(w);
  cudaFree(p);
  cudaFree(s);
  // cudaFree(buffert);
  // cudaFree(buffer);
}
