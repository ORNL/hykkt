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

#include "SchurComplementConjugateGradient.hpp"
  // parametrized constructor
  SchurComplementConjugateGradient::SchurComplementConjugateGradient(
      cusparseSpMatDescr_t matJC, cusparseSpMatDescr_t matJCt, csrcholInfo_t dH,
      double* x0, double* b, int n, int m, int nnz, void* buffer_gpu) :
      matJC_(matJC), matJCt_(matJCt), dH_(dH), x0_(x0), b_(b), n_(n), m_(m),
      nnz_(nnz), buffer_gpu_(buffer_gpu){}

  // destructor
  SchurComplementConjugateGradient::~SchurComplementConjugateGradient(){
  free(ycp);
  cudaFree(y);
  cudaFree(z);
  cudaFree(r);
  cudaFree(w);
  cudaFree(p);
  cudaFree(s);
  };

  // solver API
  void SchurComplementConjugateGradient::allocate(){
  cusparseCreate(&handle);  
  cusolverSpCreate(&handle_cusolver);
  cublasCreate(&handle_cublas);
  ycp = (double*)calloc(m_, sizeof(double));
  for(int i = 0; i < m_; i++)
  {
    ycp[i] = 0;
  }
  cudaMalloc((void**)&y, m_ * sizeof(double));
  cudaMalloc((void**)&z, m_ * sizeof(double));
  cudaMalloc((void**)&r, n_ * sizeof(double));
  cudaMalloc((void**)&w, n_ * sizeof(double));
  cudaMalloc((void**)&p, n_ * sizeof(double));
  cudaMalloc((void**)&s, n_ * sizeof(double));

  //  Allocation - happens once
  cusparseCreateDnVec(&vecx, n_, x0_, CUDA_R_64F);
  cusparseCreateDnVec(&vecb, n_, b_, CUDA_R_64F);
  cusparseCreateDnVec(&vecy, m_, y, CUDA_R_64F);
  cusparseCreateDnVec(&vecz, m_, z, CUDA_R_64F);
  cusparseCreateDnVec(&vecr, n_, r, CUDA_R_64F);
  cusparseCreateDnVec(&vecw, n_, w, CUDA_R_64F);
  cusparseCreateDnVec(&vecp, n_, p, CUDA_R_64F);
  cusparseCreateDnVec(&vecs, n_, s, CUDA_R_64F);

  }
  void SchurComplementConjugateGradient::setup(){
  cudaMemcpy(y, ycp, sizeof(double) * (m_), cudaMemcpyHostToDevice);
  cudaMemcpy(z, y, sizeof(double) * (m_), cudaMemcpyDeviceToDevice);
  cudaMemcpy(r, b_, sizeof(double) * (n_), cudaMemcpyDeviceToDevice);
  cudaMemcpy(w, b_, sizeof(double) * (n_), cudaMemcpyDeviceToDevice);
  cudaMemcpy(p, r, sizeof(double) * (n_), cudaMemcpyDeviceToDevice);
  cudaMemcpy(s, w, sizeof(double) * (n_), cudaMemcpyDeviceToDevice);
  
  }
  int SchurComplementConjugateGradient::solve(){
  gettimeofday(&t1, 0);
  fun_SpMV(one, matJCt_, vecx, zero, vecy);
  cusolverSpDcsrcholSolve(handle_cusolver, m_, y, z, dH_, buffer_gpu_);
  fun_SpMV(minusone, matJC_, vecz, one, vecr);
  double gam_i;
  cublasDdot(handle_cublas, n_, r, 1, r, 1, &gam_i);
  fun_SpMV(one, matJCt_, vecr, zero, vecy);
  cusolverSpDcsrcholSolve(handle_cusolver, m_, y, z, dH_, buffer_gpu_);
  fun_SpMV(one, matJC_, vecz, zero, vecw);
  double beta = 0, delta, alpha, gam_i1;
  cublasDdot(handle_cublas, n_, w, 1, r, 1, &delta);
  alpha           = gam_i / delta;
  double minalpha = -alpha;
  int i;
  for(i = 0; i < itmax_; i++)
  {
    cublasDscal(handle_cublas, n_, &beta, p, 1);
    cublasDaxpy(handle_cublas, n_, &one, r, 1, p, 1);

    cublasDscal(handle_cublas, n_, &beta, s, 1);
    cublasDaxpy(handle_cublas, n_, &one, w, 1, s, 1);

    cublasDaxpy(handle_cublas, n_, &alpha, p, 1, x0_, 1);
    minalpha = -alpha;
    cublasDaxpy(handle_cublas, n_, &minalpha, s, 1, r, 1);

    cublasDdot(handle_cublas, n_, r, 1, r, 1, &gam_i1);
    if(sqrt(gam_i1) < tol_)
    {
      gettimeofday(&t2, 0);
      timeIO = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
      printf("time for CG ev(ms). : %16.16f\n", timeIO);
      printf("Convergence occured at iteration %d\n", i);
      break;
    }
    // product with w=Ar starts here
    fun_SpMV(one, matJCt_, vecr, zero, vecy);
    cusolverSpDcsrcholSolve(handle_cusolver, m_, y, z, dH_, buffer_gpu_);
    fun_SpMV(one, matJC_, vecz, zero, vecw);

    cublasDdot(handle_cublas, n_, w, 1, r, 1, &delta);
    beta  = gam_i1 / gam_i;
    gam_i = gam_i1;
    alpha = gam_i / (delta - beta * gam_i / alpha);
  }
  printf("Error is %32.32g \n", sqrt(gam_i1));
  if (i==itmax_){
    gettimeofday(&t2, 0);
    timeIO = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("time for CG ev(ms). : %16.16f\n", timeIO);
    printf("No CG convergence in %d iterations\n", itmax_);
    return 1;
  }
    return 0;
  }
  void SchurComplementConjugateGradient::set_solver_tolerance(double tol)
  {
    tol_ = tol;
  }
  void SchurComplementConjugateGradient::set_solver_itmax(int itmax)
  {
    itmax_ = itmax;
  }

