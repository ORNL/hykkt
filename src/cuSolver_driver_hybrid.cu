/* written by SR based on a code by KS
         How to compile:
         nvcc -lcusparse -lcusolver -lcublas cuSolver_driver_chol.cu
 */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
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
#include "matrix_vector_ops.hpp"
#include "permcheck.hpp"
#include "input_functions.hpp"
#include "schur_complement_cg.hpp"
#include "SchurComplementConjugateGradient.hpp"
#include <RuizClass.hpp>
#define tol 1e-12
#define norm_tol 1e-2
#define ruiz_its 2
// this version reads NORMAL mtx matrices; dont have to be sorted.
int main(int argc, char* argv[])
{
  // Start of block: reading matrices from files and allocating structures for
  // them, to be replaced by HiOp structures
  struct timeval t1, t2;
  double         timeIO = 0.0f, timeM = 0.0f;
  /*** cuda stuff ***/
  cusparseStatus_t status;
  cusparseHandle_t handle            = NULL;
  status                             = cusparseCreate(&handle);
  cusolverSpHandle_t handle_cusolver = NULL;
  cusolverSpCreate(&handle_cusolver);
  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cublasHandle_t handle_cublas;
  cublasCreate(&handle_cublas);
  void * dBuffer3 = NULL, *dBuffer4 = NULL;
  size_t bufferSize3 = 0, bufferSize4 = 0;

  // Get matrix block files
  char const* const HFileName  = argv[1];
  char const* const DsFileName = argv[2];
  char const* const JCFileName = argv[3];
  char const* const JDFileName = argv[4];

  // Get rhs block files
  char const* const rxFileName  = argv[5];
  char const* const rsFileName  = argv[6];
  char const* const ryFileName  = argv[7];
  char const* const rydFileName = argv[8];
 // char const* const permFileName = argv[11];
  int skip_lines = atoi(argv[9]);
  double gamma = atof(argv[10]);
  // Matix structure allocations
  // Start block - allocating memory for matrices and vectors
  mmatrix* H  = (mmatrix*)calloc(1, sizeof(mmatrix));
  mmatrix* Ds = (mmatrix*)calloc(1, sizeof(mmatrix));
  mmatrix* JC = (mmatrix*)calloc(1, sizeof(mmatrix));
  mmatrix* JD = (mmatrix*)calloc(1, sizeof(mmatrix));
  // Vector allocations
  double *rx, *rs, *ry, *ryd;

  // read matrices

  read_mm_file_into_coo(HFileName, H, skip_lines);
  sym_coo_to_csr(H);

  read_mm_file_into_coo(DsFileName, Ds, skip_lines);
  coo_to_csr(Ds);

  read_mm_file_into_coo(JCFileName, JC, skip_lines);
  coo_to_csr(JC);

  read_mm_file_into_coo(JDFileName, JD, skip_lines);
  coo_to_csr(JD);
  int jd_flag = (JD->nnz > 0);
  // read right hand side
  rx = (double*)calloc(H->n, sizeof(double));
  read_rhs(rxFileName, rx);
  rs = (double*)calloc(Ds->n, sizeof(double));
  read_rhs(rsFileName, rs);
  ry = (double*)calloc(JC->n, sizeof(double));
  read_rhs(ryFileName, ry);
  ryd = (double*)calloc(JD->n, sizeof(double));
  read_rhs(rydFileName, ryd);
  // now copy data to GPU and format convert
  double *d_rx, *d_rs, *d_ry, *d_ry_c, *d_ryd, *d_ryd_s;
  double *H_a, *Ds_a, *JC_a;
  int *   H_ja, *H_ia;     // columns and rows of H
  int *   JC_ja, *JC_ia;   // columns and rows of JC
  int *   JD_ja, *JD_ia;   // columns and rows of JD
  double *JD_a, *JD_as;
  // allocate space for rhs and copy it to device

  cudaMalloc((void**)&d_rx, H->n * sizeof(double));
  cudaMalloc((void**)&d_rs, Ds->n * sizeof(double));
  cudaMalloc((void**)&d_ry, JC->n * sizeof(double));
  cudaMalloc((void**)&d_ry_c, JC->n * sizeof(double));
  cudaMalloc((void**)&d_ryd, JD->n * sizeof(double));
  cudaMalloc((void**)&d_ryd_s, JD->n * sizeof(double));

  cudaMemcpy(d_rx, rx, sizeof(double) * H->n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_rs, rs, sizeof(double) * Ds->n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ry, ry, sizeof(double) * JC->n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ry_c, d_ry, sizeof(double) * JC->n, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_ryd, ryd, sizeof(double) * JD->n, cudaMemcpyHostToDevice);

  // allocate space for matrix and copy it to device
  cudaMalloc((void**)&H_a, (H->nnz) * sizeof(double));
  cudaMalloc((void**)&H_ja, (H->nnz) * sizeof(int));
  cudaMalloc((void**)&H_ia, (H->n + 1) * sizeof(int));

  cudaMemcpy(H_a, H->csr_vals, sizeof(double) * H->nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(H_ia, H->csr_ia, sizeof(int) * (H->n + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(H_ja, H->csr_ja, sizeof(int) * H->nnz, cudaMemcpyHostToDevice);
#if 0 
  printf("CSR H\n");
  for(int i=0; i<10; i++)
  {
    printf("%d\n",i);
    for (int j=H->csr_ia[i]; j<H->csr_ia[i+1]; j++)
    {
      printf("Column %d, value %f\n", H->coo_cols[j], H->coo_vals[j]);
    }
  }
#endif
  cusparseSpMatDescr_t matH;
  cusparseCreateCsr(&matH, H->n, H->m, H->nnz, H_ia, H_ja, H_a, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

  cudaMalloc((void**)&Ds_a, (Ds->nnz) * sizeof(double));
  cudaMemcpy(Ds_a, Ds->coo_vals, sizeof(double) * Ds->nnz, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&JC_a, (JC->nnz) * sizeof(double));
  cudaMalloc((void**)&JC_ja, (JC->nnz) * sizeof(int));
  cudaMalloc((void**)&JC_ia, (JC->n + 1) * sizeof(int));

  cudaMemcpy(JC_a, JC->coo_vals, sizeof(double) * JC->nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(JC_ia, JC->csr_ia, sizeof(int) * (JC->n + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(JC_ja, JC->coo_cols, sizeof(int) * JC->nnz, cudaMemcpyHostToDevice);

  if(jd_flag)
  {
    cudaMalloc((void**)&JD_a, (JD->nnz) * sizeof(double));
    cudaMalloc((void**)&JD_as, (JD->nnz) * sizeof(double));
    cudaMalloc((void**)&JD_ja, (JD->nnz) * sizeof(int));
    cudaMalloc((void**)&JD_ia, (JD->n + 1) * sizeof(int));

    cudaMemcpy(JD_a, JD->coo_vals, sizeof(double) * JD->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(JD_ia, JD->csr_ia, sizeof(int) * (JD->n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(JD_ja, JD->coo_cols, sizeof(int) * JD->nnz, cudaMemcpyHostToDevice);
  }
  // malloc initial guess (potentially supplied by HiOp)
  // could change at each iteration, but might only happen once

  double* h_x  = (double*)calloc(H->m, sizeof(double));
  double* h_s  = (double*)calloc(Ds->m, sizeof(double));
  double* h_y  = (double*)calloc(JC->n, sizeof(double));
  double* h_yd = (double*)calloc(JD->n, sizeof(double));

  double *d_x, *d_s, *d_y, *d_yd;

  for(int i = 0; i < H->m; i++)
  {
    h_x[i] = 0;
  }

  for(int i = 0; i < Ds->m; i++)
  {
    h_s[i] = 0;
  }

  for(int i = 0; i < JC->n; i++)
  {
    h_y[i] = 0;
  }

  for(int i = 0; i < JD->n; i++)
  {
    h_yd[i] = 0;
  }

  cudaMalloc((void**)&d_x, H->m * sizeof(double));
  cudaMemcpy(d_x, h_x, sizeof(double) * (H->m), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_s, Ds->m * sizeof(double));
  cudaMemcpy(d_s, h_s, sizeof(double) * (Ds->m), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_y, JC->n * sizeof(double));
  cudaMemcpy(d_y, h_y, sizeof(double) * (JC->n), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_yd, JD->n * sizeof(double));
  cudaMemcpy(d_yd, h_yd, sizeof(double) * (JD->n), cudaMemcpyHostToDevice);

  cusparseSpMatDescr_t matJC;
  cusparseCreateCsr(&matJC, JC->n, JC->m, JC->nnz, JC_ia, JC_ja, JC_a, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  // set up vectors to store products

  double *d_rx_til, *d_rs_til;

  cudaMalloc((void**)&d_rx_til, H->n * sizeof(double));
  cudaMalloc((void**)&d_rs_til, Ds->n * sizeof(double));
  cudaMalloc((void**)&d_ryd_s, JD->n * sizeof(double));
  gettimeofday(&t1, 0);
  cudaMemcpy(d_rx_til, d_rx, sizeof(double) * H->n, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_rs_til, d_rs, sizeof(double) * Ds->n, cudaMemcpyDeviceToDevice);
  gettimeofday(&t2, 0);
  timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  cusparseDnVecDescr_t vec_d_ryd = NULL;
  cusparseCreateDnVec(&vec_d_ryd, JD->n, d_ryd, CUDA_R_64F);
  cusparseDnVecDescr_t vec_d_rs_til = NULL;
  cusparseCreateDnVec(&vec_d_rs_til, Ds->n, d_rs_til, CUDA_R_64F);

  cusparseDnVecDescr_t vec_d_ryd_s = NULL;
  cusparseCreateDnVec(&vec_d_ryd_s, JD->n, d_ryd_s, CUDA_R_64F);
  cusparseDnVecDescr_t vec_d_rx_til = NULL;
  cusparseCreateDnVec(&vec_d_rx_til, H->n, d_rx_til, CUDA_R_64F);
  // Start of block: Setting up eq (4) from the paper
  // start products
  double                one      = 1.0;
  double                zero     = 0.0;
  double                minusone = -1.0;
  int                   nnzHtil;
  double*               Htil_vals = NULL;
  int *                 Htil_cols = NULL, *Htil_rows = NULL;
  cusparseSpGEMMDescr_t spgemmDesc;
  cusparseSpGEMM_createDescr(&spgemmDesc);
  cusparseSpMatDescr_t matJD = NULL;   // create once and overwrite at each iteration
  cusparseCreateCsr(&matJD, JD->n, JD->m, JD->nnz, JD_ia, JD_ja, JD_a, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

  cusparseSpMatDescr_t matJDt = NULL;
  double* JDt_a;
  int *   JDt_ja, *JDt_ia;
  if(jd_flag)   // if JD is not all zeros (otherwise computation is saved)
  {
    // Creating a CSR matrix and buffer for transposing - done only once
    cudaMalloc(&JDt_a, (JD->nnz) * sizeof(double));
    cudaMalloc(&JDt_ja, (JD->nnz) * sizeof(int));
    cudaMalloc(&JDt_ia, ((JD->m) + 1) * sizeof(int));
    void*  buffercsr = NULL;
    size_t buffersize;
    cusparseCsr2cscEx2_bufferSize(handle, JD->n, JD->m, JD->nnz, JD_a, JD_ia, JD_ja, JDt_a, JDt_ia,
      JDt_ja, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
      &buffersize);
    cudaMalloc(&buffercsr, sizeof(char) * buffersize);
    // Applying the transpose to the matrix - done every iteration
    gettimeofday(&t1, 0);
    cusparseCsr2cscEx2(handle, JD->n, JD->m, JD->nnz, JD_a, JD_ia, JD_ja, JDt_a, JDt_ia, JDt_ja,
      CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
      buffercsr);
    gettimeofday(&t2, 0);
    timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    cusparseCreateCsr(&matJDt, JD->m, JD->n, JD->nnz, JDt_ia, JDt_ja, JDt_a, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    // math ops for eq (4) done at every iteration
    gettimeofday(&t1, 0);
    fun_row_scale(JD->n, JD_a, JD_ia, JD_ja, JD_as, d_ryd, d_ryd_s, Ds_a);
    gettimeofday(&t2, 0);
    timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    cusparseSpMatDescr_t matJDs = NULL;   //(except this part)
    cusparseCreateCsr(&matJDs, JD->n, JD->m, JD->nnz, JD_ia, JD_ja, JD_as, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    gettimeofday(&t1, 0);
    fun_add_vecs(JD->n, d_ryd_s, one, d_rs);
    // create buffer for matvec - done once
    /*
      size_t bufferSize_rx = 0;
      cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE, &one, matJD, vec_d_ryd_s, &one,
        vec_d_rx_til, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize_rx);
      void* buffer_rx = NULL;
      cudaMalloc(&buffer_rx, bufferSize_rx);
      printf("bufferSize_rx is %d", bufferSize_rx); //this is 0
      */
    // matvec done every iteration
    fun_SpMV(one, matJDt, vec_d_ryd_s, one, vec_d_rx_til);
    gettimeofday(&t2, 0);
    timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    // Compute H_til= H+J_d^T * D_s * J_d
    // Allocating for SPGEMM - done once
    cusparseSpMatDescr_t matJDtDxJD = NULL;
    cusparseCreateCsr(&matJDtDxJD, JD->m, JD->m, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    // ask bufferSize3 bytes for external memory
    cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matJDt, matJDs, &zero, matJDtDxJD, CUDA_R_64F,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize3, NULL);
    cudaMalloc((void**)&dBuffer3, bufferSize3);
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matJDt, matJDs, &zero, matJDtDxJD, CUDA_R_64F,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize3, dBuffer3);
    cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matJDt, matJDs, &zero, matJDtDxJD, CUDA_R_64F,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize4, NULL);
    cudaMalloc((void**)&dBuffer4, bufferSize4);
    // SPGEMM - done every iteration
    gettimeofday(&t1, 0);
    cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matJDt, matJDs, &zero, matJDtDxJD, CUDA_R_64F,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize4, dBuffer4);
    gettimeofday(&t2, 0);
    timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;

    // compute the intermediate product of A * B - happens once
    int64_t JDtDxJD_num_rows1, JDtDxJD_num_cols1, JDtDxJD_nnz1;
    cusparseSpMatGetSize(matJDtDxJD, &JDtDxJD_num_rows1, &JDtDxJD_num_cols1, &JDtDxJD_nnz1);
    int *   JDtDxJD_rows, *JDtDxJD_cols;
    double* JDtDxJD_vals;
    cudaMalloc((void**)&JDtDxJD_rows, (JDtDxJD_num_rows1 + 1) * sizeof(int));
    cudaMalloc((void**)&JDtDxJD_cols, JDtDxJD_nnz1 * sizeof(int));
    cudaMalloc((void**)&JDtDxJD_vals, JDtDxJD_nnz1 * sizeof(double));
    // SPGEMM - happens every iteration
    cusparseCsrSetPointers(matJDtDxJD, JDtDxJD_rows, JDtDxJD_cols, JDtDxJD_vals);
  gettimeofday(&t1, 0);
    cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, matJDt, matJDs, &zero, matJDtDxJD, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);
  gettimeofday(&t2, 0);
  timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    /* It's time for the sum Htilde= H + (J_d^TD_xJ_d)
     nnzTotalDevHostPtr points to host memory
     Allocation for matrix addition - happens once
    */
    size_t bufferSizeInBytes_add;
    void*  buffer_add         = NULL;
    int*   nnzTotalDevHostPtr = &nnzHtil;
    cudaMalloc((void**)&Htil_rows, sizeof(int) * ((H->n) + 1));
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    cusparseDcsrgeam2_bufferSizeExt(handle, H->n, H->n, &one, descrA, H->nnz, H_a, H_ia, H_ja, &one,
      descrA, JDtDxJD_nnz1, JDtDxJD_vals, JDtDxJD_rows, JDtDxJD_cols, descrA, Htil_vals, Htil_rows,
      Htil_cols, &bufferSizeInBytes_add);
    cudaMalloc((void**)&buffer_add, sizeof(char) * bufferSizeInBytes_add);
    cusparseXcsrgeam2Nnz(handle, H->n, H->n, descrA, H->nnz, H_ia, H_ja, descrA, JDtDxJD_nnz1,
      JDtDxJD_rows, JDtDxJD_cols, descrA, Htil_rows, nnzTotalDevHostPtr, buffer_add);
    nnzHtil = *nnzTotalDevHostPtr;
    cudaMalloc((void**)&Htil_cols, sizeof(int) * (nnzHtil));
    cudaMalloc((void**)&Htil_vals, sizeof(double) * (nnzHtil));
    // Matrix addition, happens every iteration
    gettimeofday(&t1, 0);
    cusparseDcsrgeam2(handle, H->n, H->n, &one, descrA, H->nnz, H_a, H_ia, H_ja, &one, descrA,
      JDtDxJD_nnz1, JDtDxJD_vals, JDtDxJD_rows, JDtDxJD_cols, descrA, Htil_vals, Htil_rows,
      Htil_cols, buffer_add);
    gettimeofday(&t2, 0);
    timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    // Free only happens at last iteration
    cudaFree(JDtDxJD_rows);
    cudaFree(JDtDxJD_cols);
    cudaFree(JDtDxJD_vals);
    cudaFree(buffercsr);
    cudaFree(buffer_add);
  }   // This closes the if J_d!=0 statement
  else
  {   // overwite H with Htil if JD==0
    cudaMalloc((void**)&Htil_rows, sizeof(int) * ((H->n) + 1));
    cudaMalloc((void**)&Htil_cols, sizeof(int) * (H->nnz));
    cudaMalloc((void**)&Htil_vals, sizeof(double) * (H->nnz));
    gettimeofday(&t1, 0);
    cudaMemcpy(Htil_vals, H_a, sizeof(double) * (H->nnz), cudaMemcpyDeviceToDevice);
    gettimeofday(&t2, 0);
    timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    cudaMemcpy(Htil_rows, H_ia, sizeof(int) * (H->n + 1), cudaMemcpyDeviceToDevice);
    cudaMemcpy(Htil_cols, H_ja, sizeof(int) * (H->nnz), cudaMemcpyDeviceToDevice);
    nnzHtil = H->nnz;
  }
  // Start of block: Ruiz scaling
  // Allocation - happens once
  int     nHJ = (H->n) + (JC->n);
  double* JCt_a;
  int *   JCt_ja, *JCt_ia;
  cudaMalloc(&JCt_a, (JC->nnz) * sizeof(double));
  cudaMalloc(&JCt_ja, (JC->nnz) * sizeof(int));
  cudaMalloc(&JCt_ia, ((JC->m) + 1) * sizeof(int));
  void*  buffercsr3 = NULL;
  size_t buffersize3;
  cusparseCsr2cscEx2_bufferSize(handle, JC->n, JC->m, JC->nnz, JC_a, JC_ia, JC_ja, JCt_a, JCt_ia,
    JCt_ja, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
    &buffersize3);
  cudaMalloc(&buffercsr3, sizeof(char) * buffersize3);
  // Transpose JC - happens every iteration
    gettimeofday(&t1, 0);
  cusparseCsr2cscEx2(handle, JC->n, JC->m, JC->nnz, JC_a, JC_ia, JC_ja, JCt_a, JCt_ia, JCt_ja,
    CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
    buffercsr3);
    gettimeofday(&t2, 0);
    timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  cusparseSpMatDescr_t matJCt = NULL;
  cusparseCreateCsr(&matJCt, JC->m, JC->n, JC->nnz, JCt_ia, JCt_ja, JCt_a, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
#if 1 //this block is only activated to check solution (requires more copying)
  // saves the original JC and JCt  
  double*               JC_a_c = NULL;
  int *                 JC_ia_c = NULL, *JC_ja_c = NULL;
  cudaMalloc((void**)&JC_ia_c, sizeof(int) * ((JC->n) + 1));
  cudaMalloc((void**)&JC_ja_c, sizeof(int) * (JC->nnz));
  cudaMalloc((void**)&JC_a_c, sizeof(double) * (JC->nnz));
  cudaMemcpy(JC_a_c, JC_a, sizeof(double) * (JC->nnz), cudaMemcpyDeviceToDevice);
  cudaMemcpy(JC_ia_c, JC_ia, sizeof(int) * (JC->n + 1), cudaMemcpyDeviceToDevice);
  cudaMemcpy(JC_ja_c, JC_ja, sizeof(int) * (JC->nnz), cudaMemcpyDeviceToDevice);
  cusparseSpMatDescr_t matJC_c = NULL;
  cusparseCreateCsr(&matJC_c, JC->n, JC->m, JC->nnz, JC_ia_c, JC_ja_c, JC_a_c, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  double*               JCt_a_c = NULL;
  int *                 JCt_ia_c = NULL, *JCt_ja_c = NULL;
  cudaMalloc((void**)&JCt_ia_c, sizeof(int) * ((JC->m) + 1));
  cudaMalloc((void**)&JCt_ja_c, sizeof(int) * (JC->nnz));
  cudaMalloc((void**)&JCt_a_c, sizeof(double) * (JC->nnz));
  cudaMemcpy(JCt_a_c, JCt_a, sizeof(double) * (JC->nnz), cudaMemcpyDeviceToDevice);
  cudaMemcpy(JCt_ia_c, JCt_ia, sizeof(int) * ((JC->m) + 1), cudaMemcpyDeviceToDevice);
  cudaMemcpy(JCt_ja_c, JCt_ja, sizeof(int) * (JC->nnz), cudaMemcpyDeviceToDevice);
  cusparseSpMatDescr_t matJCt_c = NULL;
  cusparseCreateCsr(&matJCt_c, JC->m, JC->n, JC->nnz, JCt_ia_c, JCt_ja_c, JCt_a_c, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
#endif
#if 0 
  double *JC_a_h;
  int *JC_ia_h, *JC_ja_h;
  JC_a_h = (double*)calloc(JC->nnz, sizeof(double));
  JC_ja_h = (int*)calloc(JC->nnz, sizeof(int));
  JC_ia_h = (int*)calloc((JC->n)+1, sizeof(int));
  cudaMemcpy(JC_a_h, JC_a_c, sizeof(double)*(JC->nnz), cudaMemcpyDeviceToHost);
  cudaMemcpy(JC_ja_h, JC_ja_c, sizeof(int)*(JC->nnz), cudaMemcpyDeviceToHost);
  cudaMemcpy(JC_ia_h, JC_ia_c, sizeof(int)*((JC->n)+1), cudaMemcpyDeviceToHost);
  printf("CSR J_c\n");
  for(int i=(JC->n)-2; i<(JC->n); i++)
  {
    printf("%d\n",i);
    for (int j=JC_ia_h[i]; j<JC_ia_h[i+1]; j++)
    {
      printf("Column %d, value %f\n", JC_ja_h[j], JC_a_h[j]);
    }
  }
  free(JC_a_h);
  free(JC_ia_h);
  free(JC_ja_h);
#endif
  // setup vectors for scaling
#if 0 //class implementation
RuizClass hjr(H->n, nHJ, Htil_vals, Htil_rows, Htil_cols, JC_a,
    JC_ia, JC_ja, JCt_a, JCt_ia, JCt_ja, d_rx_til, d_ry);
hjr.setup();
hjr.init_max_d();
for(int i=0;i<ruiz_its;i++){
  hjr.row_max();
  hjr.diag_scale();
}
double* max_d;
max_d = hjr.get_max_d();
#endif
#if 1 //function implemention
  // Allocation - happens once
  double *max_d, *scale;
  cudaMalloc(&max_d, nHJ * sizeof(double));
  cudaMalloc(&scale, nHJ * sizeof(double));
  double* max_h = (double*)calloc(nHJ, sizeof(double));
  // Initialization and actual scaling - happen every iteration
  for(int i = 0; i < nHJ; i++)
  {
    max_h[i] = 1;
  }
  gettimeofday(&t1, 0);
  cudaMemcpy(max_d, max_h, sizeof(double) * nHJ, cudaMemcpyHostToDevice);
  for(int i = 0; i < ruiz_its; i++)
  {
    fun_adapt_row_max(H->n, nHJ, Htil_vals, Htil_rows, Htil_cols, JC_a, JC_ia,
        JC_ja, JCt_a, JCt_ia, JCt_ja, scale);
    fun_adapt_diag_scale(H->n, nHJ, Htil_vals, Htil_rows, Htil_cols, JC_a,
      JC_ia, JC_ja, JCt_a, JCt_ia, JCt_ja, scale, d_rx_til, d_ry, max_d);
  }
  gettimeofday(&t2, 0);
  timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
#endif
#if 0 
  double *Ht_a_h;
  int *Ht_ia_h, *Ht_ja_h;
  Ht_a_h = (double*)calloc(nnzHtil, sizeof(double));
  Ht_ja_h = (int*)calloc(nnzHtil, sizeof(int));
  Ht_ia_h = (int*)calloc((H->n)+1, sizeof(int));
  cudaMemcpy(Ht_a_h, Htil_vals, sizeof(double)*(nnzHtil), cudaMemcpyDeviceToHost);
  cudaMemcpy(Ht_ja_h, Htil_cols, sizeof(int)*(nnzHtil), cudaMemcpyDeviceToHost);
  cudaMemcpy(Ht_ia_h, Htil_rows, sizeof(int)*((H->n)+1), cudaMemcpyDeviceToHost);
  printf("CSR H\n");
  for(int i=(H->n)-2; i<(H->n); i++)
  {
    printf("%d\n",i);
    for (int j=Ht_ia_h[i]; j<Ht_ia_h[i+1]; j++)
    {
      printf("Column %d, value %f\n", Ht_ja_h[j], Ht_a_h[j]);
    }
  }
  free(Ht_a_h);
  free(Ht_ia_h);
  free(Ht_ja_h);
#endif
#if 0
  cudaMemcpy(max_h,max_d, sizeof(double)*nHJ, cudaMemcpyDeviceToHost);
  for (int i=0;i<10;i++)
  printf("D[%d] = %f\n", i, max_h[i]);
#endif
  // Start of block, setting up eq (5)
  // Allocation for SPGEMM - happens once
  cusparseSpMatDescr_t matJCtJC = NULL;
  cusparseCreateCsr(&matJCtJC, JC->m, JC->m, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  void*  bufferJC = NULL;
  size_t buffersizeJC;
  // ask bufferSize1 bytes for external memory
  cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &gamma, matJCt, matJC, &zero, matJCtJC, CUDA_R_64F,
    CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &buffersizeJC, NULL);
  cudaMalloc((void**)&bufferJC, buffersizeJC);
  // inspect the matrices A and B to understand the memory requirement for
  // the next step
  cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &gamma, matJCt, matJC, &zero, matJCtJC, CUDA_R_64F,
    CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &buffersizeJC, bufferJC);
  void*  bufferJC2 = NULL;
  size_t buffersizeJC2;
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &gamma, matJCt, matJC, &zero, matJCtJC, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
    &buffersizeJC2, NULL);
  cudaMalloc((void**)&bufferJC2, buffersizeJC2);
  // compute the intermediate product of A * B
  // Compute SPGEMM - done every iteration
  gettimeofday(&t1, 0);
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &gamma, matJCt, matJC, &zero, matJCtJC, CUDA_R_64F,
    CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &buffersizeJC2, bufferJC2);
  gettimeofday(&t2, 0);
  timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  // Allocation - happens once
  int64_t JCtJC_num_rows1, JCtJC_num_cols1, JCtJC_nnz1;
  cusparseSpMatGetSize(matJCtJC, &JCtJC_num_rows1, &JCtJC_num_cols1, &JCtJC_nnz1);
  int *   JCtJC_rows, *JCtJC_cols;
  double* JCtJC_vals;
  cudaMalloc((void**)&JCtJC_rows, (JC->m + 1) * sizeof(int));
  cudaMalloc((void**)&JCtJC_cols, JCtJC_nnz1 * sizeof(int));
  cudaMalloc((void**)&JCtJC_vals, JCtJC_nnz1 * sizeof(double));
  // SPGEMM - happens very iterations
  cusparseCsrSetPointers(matJCtJC, JCtJC_rows, JCtJC_cols, JCtJC_vals);
  gettimeofday(&t1, 0);
  cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &gamma, matJCt, matJC, &zero, matJCtJC, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);
  gettimeofday(&t2, 0);
  timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
#if 0
  int *JCtJC_i, *JCtJC_j;
  double *JCtJC_v;
  JCtJC_i=(int*) malloc((H->n + 1)*sizeof(int));
  JCtJC_j=(int*) malloc((JCtJC_nnz1)*sizeof(int));
  JCtJC_v=(double*) malloc((JCtJC_nnz1)*sizeof(double));
    
  cudaMemcpy(JCtJC_v, JCtJC_vals, sizeof(double)*JCtJC_nnz1, cudaMemcpyDeviceToHost);
  cudaMemcpy(JCtJC_i, JCtJC_rows, sizeof(int)*(H->n+1), cudaMemcpyDeviceToHost);  
  cudaMemcpy(JCtJC_j, JCtJC_cols, sizeof(int)*JCtJC_nnz1, cudaMemcpyDeviceToHost);
  printf("gamma*J_c^TJ_c num rows = %d, nnz = %d\n",H->n, JCtJC_nnz1);
  for(int i=3000; i<3001; i++)
  {
    printf("Row %d starts at place %d\n",i,JCtJC_i[i]);
    for (int j=JCtJC_i[i]; j<JCtJC_i[i+1]; j++)
    {
      printf("Column %d value %f\n", JCtJC_j[j], JCtJC_v[j]);
    }
  }
  free(JCtJC_i);
  free(JCtJC_j);
  free(JCtJC_v);
#endif
  /* It's time for the sum Hgamma= Htilde + gamma(J_c^TJ_c)
   nnzTotalDevHostPtr2 points to host memory
   Allocation for matrix addition - happens once*/
  size_t  bufferSizeInBytes_add2;
  char*   buffer_add2 = NULL;
  int     nnzHgam;
  int*    nnzTotalDevHostPtr2 = &nnzHgam;
  double* Hgam_vals           = NULL;
  int *   Hgam_cols = NULL, *Hgam_rows = NULL;
  cudaMalloc((void**)&Hgam_rows, sizeof(int) * ((H->n) + 1));
  cusparseDcsrgeam2_bufferSizeExt(handle, H->n, H->n, &one, descrA, nnzHtil, Htil_vals, Htil_rows,
    Htil_cols, &one, descrA, JCtJC_nnz1, JCtJC_vals, JCtJC_rows, JCtJC_cols, descrA, Hgam_vals,
    Hgam_rows, Hgam_cols, &bufferSizeInBytes_add2);
  cudaMalloc((void**)&buffer_add2, sizeof(char) * bufferSizeInBytes_add2);
  cusparseXcsrgeam2Nnz(handle, H->n, H->n, descrA, nnzHtil, Htil_rows, Htil_cols, descrA,
    JCtJC_nnz1, JCtJC_rows, JCtJC_cols, descrA, Hgam_rows, nnzTotalDevHostPtr2, buffer_add2);
  nnzHgam = *nnzTotalDevHostPtr2;
  printf("nnzHgam = %d\n", nnzHgam);
  cudaMalloc((void**)&Hgam_cols, sizeof(int) * (nnzHgam));
  cudaMalloc((void**)&Hgam_vals, sizeof(double) * (nnzHgam));
  // Matrix addition - happens every iteration
  gettimeofday(&t1, 0);
  cusparseDcsrgeam2(handle, H->n, H->n, &one, descrA, nnzHtil, Htil_vals, Htil_rows, Htil_cols,
    &one, descrA, JCtJC_nnz1, JCtJC_vals, JCtJC_rows, JCtJC_cols, descrA, Hgam_vals, Hgam_rows,
    Hgam_cols, buffer_add2);
  gettimeofday(&t2, 0);
  timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  double* d_rx_hat;
  cudaMalloc((void**)&d_rx_hat, H->n * sizeof(double));
  gettimeofday(&t1, 0);
  cudaMemcpy(d_rx_hat, d_rx_til, sizeof(double) * H->n, cudaMemcpyDeviceToDevice);
  gettimeofday(&t2, 0);
  timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  cusparseDnVecDescr_t vec_d_rx_hat = NULL;
  cusparseCreateDnVec(&vec_d_rx_hat, H->n, d_rx_hat, CUDA_R_64F);
  cusparseDnVecDescr_t vec_d_ry = NULL;
  cusparseCreateDnVec(&vec_d_ry, JC->n, d_ry, CUDA_R_64F);
  /* this size is 0 anyways
  size_t  bufferSize_rx_hat = 0;
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &gamma, matJCt, vec_d_ry, &one,
    vec_d_rx_hat, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize_rx_hat);
  void* buffer_rx_hat = NULL;
  cudaMalloc(&buffer_rx_hat, bufferSize_rx_hat);
  */
  gettimeofday(&t1, 0);
  fun_SpMV(gamma, matJCt, vec_d_ry, one, vec_d_rx_hat);
  gettimeofday(&t2, 0);
  timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  // Start of block: permutation calculation (happens once)
  int *Hgam_h_rows, *Hgam_h_cols;
  Hgam_h_rows = (int*)malloc((H->n + 1) * sizeof(int));
  Hgam_h_cols = (int*)malloc((nnzHgam) * sizeof(int));
  cudaMemcpy(Hgam_h_rows, Hgam_rows, sizeof(int) * (H->n + 1), cudaMemcpyDeviceToHost);
  cudaMemcpy(Hgam_h_cols, Hgam_cols, sizeof(int) * nnzHgam, cudaMemcpyDeviceToHost);
#if 0
  double *Hgam_h_vals;
  Hgam_h_vals=(double*) malloc((nnzHgam)*sizeof(double));
    
  cudaMemcpy(Hgam_h_vals, Hgam_vals, sizeof(double)*nnzHgam, cudaMemcpyDeviceToHost);
  printf("Hgam num rows = %d, nnz = %d\n",H->n, nnzHgam);
  for(int i=500; i<502; i++)
  {
    printf("Row %d\n",i);
    for (int j=Hgam_h_rows[i]; j<Hgam_h_rows[i+1]; j++)
    {
      printf("Column %d, value %f\n", Hgam_h_cols[j], Hgam_h_vals[j]);
    }
  }
  free(Hgam_h_vals);
#endif
  int* perm       = NULL;
  int* rev_perm   = NULL;
  int* perm_mapH  = NULL;
  int* perm_mapJ  = NULL;
  int* perm_mapJt = NULL;
  perm            = (int*)calloc(H->n, sizeof(int));
  perm_mapH       = (int*)calloc(nnzHgam, sizeof(int));
  perm_mapJ       = (int*)calloc(JC->nnz, sizeof(int));
  perm_mapJt      = (int*)calloc(JC->nnz, sizeof(int));
  rev_perm        = (int*)calloc(H->n, sizeof(int));
  cusolverSpXcsrsymamdHost(handle_cusolver, H->n, nnzHgam, descrA, Hgam_h_rows, Hgam_h_cols,
    perm);   // overwriting perm in next line for test
#if 0
  printf("Overwriting permutation \n");
  int *MLperm=(int*)  calloc(H->n, sizeof(int));
  read_1idx_perm(permFileName, MLperm);
  perm=MLperm;
#endif
  int *Hgam_p_rows, *Hgam_p_cols;
  Hgam_p_rows = (int*)malloc((H->n + 1) * sizeof(int));
  Hgam_p_cols = (int*)malloc((nnzHgam) * sizeof(int));
  reverse_perm(H->n, perm, rev_perm);
  make_vec_map_rc(
    H->n, Hgam_h_rows, Hgam_h_cols, perm, rev_perm, Hgam_p_rows, Hgam_p_cols, perm_mapH);

  int* Jc_p_cols;
  Jc_p_cols = (int*)malloc((JC->nnz) * sizeof(int));
  make_vec_map_c(JC->n, JC->csr_ia, JC->coo_cols, rev_perm, Jc_p_cols, perm_mapJ);

  int* Jct_p_cols;
  int* Jct_p_rows;
  Jct_p_cols = (int*)malloc((JC->nnz) * sizeof(int));
  Jct_p_rows = (int*)malloc((JC->m + 1) * sizeof(int));
  int* Jct_cols;
  int* Jct_rows;
  Jct_cols = (int*)malloc((JC->nnz) * sizeof(int));
  Jct_rows = (int*)malloc((JC->m + 1) * sizeof(int));
  cudaMemcpy(Jct_rows, JCt_ia, sizeof(int) * (JC->m + 1), cudaMemcpyDeviceToHost);
  cudaMemcpy(Jct_cols, JCt_ja, sizeof(int) * (JC->nnz), cudaMemcpyDeviceToHost);
  make_vec_map_r(JC->m, Jct_rows, Jct_cols, perm, Jct_p_rows, Jct_p_cols, perm_mapJt);

  cudaMemcpy(Hgam_rows, Hgam_p_rows, sizeof(int) * (H->n + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(Hgam_cols, Hgam_p_cols, sizeof(int) * nnzHgam, cudaMemcpyHostToDevice);
  cudaMemcpy(JCt_ja, Jct_p_cols, sizeof(int) * (JC->nnz), cudaMemcpyHostToDevice);
  cudaMemcpy(JCt_ia, Jct_p_rows, sizeof(int) * (JC->m + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(JC_ja, Jc_p_cols, sizeof(int) * (JC->nnz), cudaMemcpyHostToDevice);

  int *drev_perm, *d_perm, *d_perm_mapH, *d_perm_mapJ, *d_perm_mapJt;
  cudaMalloc(&drev_perm, (H->n) * sizeof(int));
  cudaMalloc(&d_perm, (H->n) * sizeof(int));
  cudaMalloc(&d_perm_mapH, (nnzHgam) * sizeof(int));
  cudaMalloc(&d_perm_mapJ, (JC->nnz) * sizeof(int));
  cudaMalloc(&d_perm_mapJt, (JC->nnz) * sizeof(int));
  cudaMemcpy(drev_perm, rev_perm, sizeof(int) * (H->n), cudaMemcpyHostToDevice);
  cudaMemcpy(d_perm, perm, sizeof(int) * (H->n), cudaMemcpyHostToDevice);
  cudaMemcpy(d_perm_mapH, perm_mapH, sizeof(int) * nnzHgam, cudaMemcpyHostToDevice);
  cudaMemcpy(d_perm_mapJ, perm_mapJ, sizeof(int) * (JC->nnz), cudaMemcpyHostToDevice);
  cudaMemcpy(d_perm_mapJt, perm_mapJt, sizeof(int) * (JC->nnz), cudaMemcpyHostToDevice);

  double *Hgamp_val, *Jcp_val, *Jctp_val;
  cudaMalloc(&Hgamp_val, (nnzHgam) * sizeof(double));
  cudaMalloc(&Jcp_val, (JC->nnz) * sizeof(double));
  cudaMalloc(&Jctp_val, (JC->nnz) * sizeof(double));
  // Start of block: permutation application - happens every iteration
  gettimeofday(&t1, 0);
  fun_map_idx(nnzHgam, d_perm_mapH, Hgam_vals, Hgamp_val);
  fun_map_idx(JC->nnz, d_perm_mapJ, JC_a, Jcp_val);
  fun_map_idx(JC->nnz, d_perm_mapJt, JCt_a, Jctp_val);
  gettimeofday(&t2, 0);
  timeM += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  printf("time for forming Hgamma ev(ms). : %16.16f\n", timeM);
  cusparseSpMatDescr_t matJCp = NULL;
  cusparseCreateCsr(&matJCp, JC->n, JC->m, JC->nnz, JC_ia, JC_ja, Jcp_val, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
#if 1
  gettimeofday(&t1, 0);
  fun_add_diag(H->n, zero, Hgam_rows, Hgam_cols, Hgamp_val);
  gettimeofday(&t2, 0);
  timeIO = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  printf("time for forming Hdelta ev(ms). : %16.16f\n", timeIO);
#endif
#if 0
  cudaMemcpy(JC->coo_vals, Jcp_val, sizeof(double)*(JC->nnz), cudaMemcpyDeviceToHost);
  cudaMemcpy(JC->csr_ia, JC_ia, sizeof(int)*(JC->n+1), cudaMemcpyDeviceToHost);  
  cudaMemcpy(JC->coo_cols, JC_ja, sizeof(int)*(JC->nnz), cudaMemcpyDeviceToHost);
  printf("JC num rows = %d, nnz = %d\n",JC->n, JC->nnz);
  for(int i=1099; i<1100; i++)
  {
    printf("Row %d\n",i);
    for (int j=JC->csr_ia[i]; j<JC->csr_ia[i+1]; j++)
    {
      printf("Column %d, value %f\n", JC->coo_cols[j], JC->coo_vals[j]);
    }
  }
#endif
#if 0
  double *Hgamp_h_vals;
  Hgamp_h_vals=(double*) malloc((nnzHgam)*sizeof(double));
  cudaMemcpy(Hgamp_h_vals, Hgamp_val, sizeof(double)*nnzHgam, cudaMemcpyDeviceToHost);
  printf("Hgamp num rows = %d, nnz = %d\n",H->n, nnzHgam);
  for(int i=1099; i<1100; i++)
  {
    printf("Row %d starts at place %d\n",i,Hgam_p_rows[i]);
    for (int j=Hgam_p_rows[i]; j<Hgam_p_rows[i+1]; j++)
    {
      printf("Column %d, value %f\n", Hgam_p_cols[j], Hgamp_h_vals[j]);
    }
  }
  free(Hgamp_h_vals);
#endif
  cusparseSpMatDescr_t matJCtp = NULL;
  cusparseCreateCsr(&matJCtp, JC->m, JC->n, JC->nnz, JCt_ia, JCt_ja, Jctp_val, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
#if 0
  double *JCth_v;
  int *JCth_i, *JCth_j;
  JCth_v=(double*) malloc((JC->nnz)*sizeof(double));
  JCth_j=(int*) malloc((JC->nnz)*sizeof(int));
  JCth_i=(int*) malloc((JC->m+1)*sizeof(int));
  cudaMemcpy(JCth_v, Jctp_val, sizeof(double)*(JC->nnz), cudaMemcpyDeviceToHost);
  cudaMemcpy(JCth_j, JCt_ja, sizeof(int)*(JC->nnz), cudaMemcpyDeviceToHost);
  cudaMemcpy(JCth_i, JCt_ia, sizeof(int)*(JC->m +1), cudaMemcpyDeviceToHost);
  printf("JCt num rows = %d, nnz = %d\n",JC->m, JC->nnz);
  for(int i=1099; i<1100; i++)
  {
    printf("Row %d starts at place %d\n",i,JCth_i[i]);
    for (int j=JCth_i[i]; j<JCth_i[i+1]; j++)
    {
      printf("Column %d, value %f\n", JCth_j[j], JCth_v[j]);
    }
  }
  free(JCth_v);
  free(JCth_i);
  free(JCth_j);
#endif

  double* d_rxp;
  cudaMalloc((void**)&d_rxp, H->n * sizeof(double));
  fun_map_idx(H->n, d_perm, d_rx_hat, d_rxp);
  //  Start of block: Factorization of Hgamma
  //  Symbolic analysis: Happens once
  csrcholInfo_t info = NULL;
  cusolverSpCreateCsrcholInfo(&info);
  gettimeofday(&t1, 0);
  cusolverSpXcsrcholAnalysis(handle_cusolver, H->n, nnzHgam, descrA, Hgam_rows, Hgam_cols, info);
  size_t internalDataInBytes, workspaceInBytes;
  cusolverSpDcsrcholBufferInfo(handle_cusolver, H->n, nnzHgam, descrA, Hgamp_val, Hgam_rows,
    Hgam_cols, info, &internalDataInBytes, &workspaceInBytes);
  gettimeofday(&t2, 0);
  timeIO = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  void* buffer_gpu = NULL;
  cudaMalloc(&buffer_gpu, sizeof(char) * workspaceInBytes);
  printf("time for symbolic analysis ev(ms). : %16.16f\n", timeIO);
  int singularity = 0;
  gettimeofday(&t1, 0);
  // Numerical factorization - happens every iteration
  cusolverSpDcsrcholFactor(
    handle_cusolver, H->n, nnzHgam, descrA, Hgamp_val, Hgam_rows, Hgam_cols, info, buffer_gpu);
  gettimeofday(&t2, 0);
  timeIO = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  cusolverSpDcsrcholZeroPivot(handle_cusolver, info, tol, &singularity);
  printf("time for factorization analysis ev(ms). : %16.16f\n", timeIO);
  if(singularity >= 0)
  {
    fprintf(stderr, "Error: H is not invertible, singularity=%d\n", singularity);
    return 1;
  }
  else
    printf("matrix nonsingular, proceed\n");
  //  Start of block : setting up the right hand side for equation 7
  //  Allocation - happens once
  double* d_Hrxp;
  cudaMalloc((void**)&d_Hrxp, H->n * sizeof(double));
  double* d_schur;
  cudaMalloc((void**)&d_schur, JC->n * sizeof(double));
  //  Solve and copy - happen every iteration
  cusolverSpDcsrcholSolve(handle_cusolver, H->n, d_rxp, d_Hrxp, info, buffer_gpu);
  cudaMemcpy(d_schur, d_ry, sizeof(double) * JC->n, cudaMemcpyDeviceToDevice);
#if 0
  printf("printing ry\n");
  cudaMemcpy(ry, d_schur, sizeof(double)*(JC->n), cudaMemcpyDeviceToHost);
  for (int i=(JC->n)-10; i<JC->n; i++){
    printf("ry[%d] = %f\n", i, ry[i]);
  }
#endif
  // Allocation - happens once
  cusparseDnVecDescr_t vec_d_schur = NULL;
  cusparseCreateDnVec(&vec_d_schur, JC->n, d_schur, CUDA_R_64F);
  cusparseDnVecDescr_t vec_d_Hrxp = NULL;
  cusparseCreateDnVec(&vec_d_Hrxp, H->n, d_Hrxp, CUDA_R_64F);
  /* this is just zero
  size_t               bufferSize_schur = 0;
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matJCp, vec_d_Hrxp,
    &minusone, vec_d_schur, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize_schur);
  void* buffer_schur = NULL;
  cudaMalloc(&buffer_schur, bufferSize_schur);
  */
  //  Matrix vector multiply - happens every iteration
  fun_SpMV(one, matJCp, vec_d_Hrxp, minusone, vec_d_schur);
#if 0 
  double *h_schur;
  h_schur=(double*) malloc((JC->n)*sizeof(double));
  cudaMemcpy(h_schur, d_schur, sizeof(double)*(JC->n), cudaMemcpyDeviceToHost);
  for (int i=15; i<485; i++){
     printf("schur[%d] = %f\n", i, h_schur[i]);
  }
  free(h_schur);
#endif
  // Start of block - conjugate gradient on eq (7)
  // Solving eq (7) via CG - happens every iteration
  //function implementation
#if 0
  int itmax = (JC->n) / 10;
  schur_cg(matJCp, matJCtp, info, d_y, d_schur, itmax, tol, JC->n, JC->m, JC->nnz,
    buffer_gpu, handle, handle_cusolver, handle_cublas);
#endif
  // class implementation
#if 1
  SchurComplementConjugateGradient sccg(
      matJCp, matJCtp, info, d_y, d_schur, JC->n, JC->m, JC->nnz,buffer_gpu);
  sccg.allocate();
  sccg.setup();
  sccg.solve();
#endif
#if 0
  cudaMemcpy(h_y, d_y, sizeof(double)*(JC->n), cudaMemcpyDeviceToHost);
  for (int i=(JC->n)-10; i<JC->n; i++){
     printf("y[%d] = %f\n", i, h_y[i]);
  }
#endif
  // Start of block - recovering the solution to the original system by parts
  // this part is to recover delta_x
  // Allocation - happens once
  cusparseDnVecDescr_t vec_d_y = NULL;
  cusparseCreateDnVec(&vec_d_y, JC->n, d_y, CUDA_R_64F);
  cusparseDnVecDescr_t vec_d_rxp = NULL;
  cusparseCreateDnVec(&vec_d_rxp, H->n, d_rxp, CUDA_R_64F);
  /* this is zero anyways
  size_t               bufferSize_d_z = 0;
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusone, matJCtp, vec_d_y,
    &one, vec_d_rxp, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize_d_z);
  void* buffer_d_z = NULL;
  cudaMalloc(&buffer_d_z, bufferSize_d_z);
  */
  // Matrix-vector product - happens every iteration
  gettimeofday(&t1, 0);
  fun_SpMV(minusone, matJCtp, vec_d_y, one, vec_d_rxp);
  //  Allocation - happens once
  double* d_z;
  cudaMalloc((void**)&d_z, H->n * sizeof(double));
  //  Solve - happens every iteration
  cusolverSpDcsrcholSolve(handle_cusolver, H->n, d_rxp, d_z, info, buffer_gpu);
  fun_map_idx(H->n, drev_perm, d_z, d_x);
#if 0
  double *h_rx_hat;
  printf("delta_x\n");
  h_rx_hat=(double*) malloc((H->n)*sizeof(double));
  cudaMemcpy(h_rx_hat, d_x, sizeof(double)*(H->n), cudaMemcpyDeviceToHost);
  for (int i=(H->n)-10; i<H->n; i++){
    printf("delta_x[%d] = %f\n", i, h_rx_hat[i]);
  }
  free(h_rx_hat);
#endif
  // scale back delta_y and delta_x (every iteration)
  fun_vec_scale(H->n, d_x, max_d);
  fun_vec_scale(JC->n, d_y, &max_d[H->n]);
#if 0 
  cudaMemcpy(h_x, d_x, sizeof(double)*(H->n), cudaMemcpyDeviceToHost);
  for (int i=(H->n)-10; i<H->n; i++){
     printf("x[%d] = %f\n", i, h_x[i]);
  }
#endif

#if 0   // check max_d
  cudaMemcpy(max_h, max_d, sizeof(double)*(nHJ), cudaMemcpyDeviceToHost);
  for (int i=0; i<10; i++){
     printf("max[%d] = %f\n", i, max_h[i]);
  }
  for (int i=H->n; i<H->n+10; i++){
     printf("max[%d] = %f\n", i, max_h[i]);
  }
#endif
  // now recover delta_s and delta_yd
  //  Allocation - happens once
  cusparseDnVecDescr_t vec_d_x = NULL;
  cusparseCreateDnVec(&vec_d_x, H->n, d_x, CUDA_R_64F);
  cusparseDnVecDescr_t vec_d_s = NULL;
  cusparseCreateDnVec(&vec_d_s, Ds->n, d_s, CUDA_R_64F);
  cudaMemcpy(d_s, d_ryd, sizeof(double) * (Ds->m), cudaMemcpyDeviceToDevice);
  if(jd_flag)
  {
    /* This is zero anyways
    size_t               bufferSize_dx = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matJD, vec_d_x,
      &minusone, vec_d_ryd, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize_dx);
    void* buffer_dx = NULL;
    cudaMalloc(&buffer_dx, bufferSize_dx);
    cudaFree(buffer_dx);
    */
    //  Matrix-vector product - happens every iteration
    fun_SpMV(one, matJD, vec_d_x, minusone, vec_d_s);
  }
  else
  {   //  Math operations - happens every iteration
    fun_mult_const(Ds->n, minusone, d_s);
  }
  //  Math operations - happens every iteration
  cudaMemcpy(d_yd, d_s, sizeof(double) * (Ds->m), cudaMemcpyDeviceToDevice);
  fun_vec_scale(Ds->n, d_yd, Ds_a);
  fun_add_vecs(Ds->n, d_yd, minusone, d_rs);
  gettimeofday(&t2, 0);
  timeIO = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  printf("time for recovering solution ev(ms). : %16.16f\n", timeIO);
#if 0 
  cudaMemcpy(h_yd, d_yd, sizeof(double)*(JD->n), cudaMemcpyDeviceToHost);
  for (int i=(JD->n)-10; i<JD->n; i++){
     printf("yd[%d] = %f\n", i, h_yd[i]);
  }
  cudaMemcpy(h_s, d_s, sizeof(double)*(Ds->n), cudaMemcpyDeviceToHost);
  for (int i=(Ds->n)-10; i<Ds->n; i++){
     printf("s[%d] = %f\n", i, h_s[i]);
  }
#endif
  //  Start of block, calculate error of Ax-b 
  //  Calculate error in rx
  gettimeofday(&t1, 0);
  double norm_rx_sq=0, norm_rs_sq=0, norm_ry_sq=0, norm_ryd_sq=0;
  double norm_resx_sq=0, norm_resy_sq=0; 
  // This will aggregate the squared norms of the residual and rhs
  // Note that by construction the residuals of rs and ryd are 0
  cublasDdot(handle_cublas, H->n, d_rx, 1, d_rx, 1, &norm_rx_sq);
  cublasDdot(handle_cublas, Ds->n, d_rs, 1, d_rs, 1, &norm_rs_sq);
  cublasDdot(handle_cublas, JC->n, d_ry_c, 1, d_ry_c, 1, &norm_ry_sq);
  cublasDdot(handle_cublas, JD->n, d_ryd, 1, d_ryd, 1, &norm_ryd_sq);
  norm_rx_sq+= norm_rs_sq + norm_ry_sq + norm_ryd_sq;
  cusparseDnVecDescr_t vec_d_rx = NULL;
  cusparseCreateDnVec(&vec_d_rx, H->n, d_rx, CUDA_R_64F);
  cusparseDnVecDescr_t vec_d_yd = NULL;
  cusparseCreateDnVec(&vec_d_yd, JD->n, d_yd, CUDA_R_64F);
  fun_SpMV(minusone, matH, vec_d_x, one, vec_d_rx);
  cublasDdot(handle_cublas, H->n, d_rx, 1, d_rx, 1, &norm_resx_sq);
  if (jd_flag){
    fun_SpMV(minusone, matJDt, vec_d_yd, one, vec_d_rx);
    cublasDdot(handle_cublas, H->n, d_rx, 1, d_rx, 1, &norm_resx_sq);
  }
#if 0 
  double *JCt_a_h;
  int *JCt_ia_h, *JCt_ja_h;
  JCt_a_h = (double*)calloc(JC->nnz, sizeof(double));
  JCt_ja_h = (int*)calloc(JC->nnz, sizeof(int));
  JCt_ia_h = (int*)calloc((JC->m)+1, sizeof(int));
  cudaMemcpy(JCt_a_h, JCt_a_c, sizeof(double)*(JC->nnz), cudaMemcpyDeviceToHost);
  cudaMemcpy(JCt_ja_h, JCt_ja_c, sizeof(int)*(JC->nnz), cudaMemcpyDeviceToHost);
  cudaMemcpy(JCt_ia_h, JCt_ia_c, sizeof(int)*((JC->m)+1), cudaMemcpyDeviceToHost);
  printf("CSR J_c\n");
  for(int i=1500; i<1502; i++)
  {
    printf("%d\n",i);
    for (int j=JCt_ia_h[i]; j<JCt_ia_h[i+1]; j++)
    {
     printf("Column %d, value %f\n", JCt_ja_h[j], JCt_a_h[j]);
    }
  }
  free(JCt_a_h);
  free(JCt_ia_h);
  free(JCt_ja_h);
#endif
#if 0
  cudaMemcpy(h_y, d_y, sizeof(double)*(JC->n), cudaMemcpyDeviceToHost);
  for (int i=(JC->n)-10; i<JC->n; i++){
     printf("y[%d] = %f\n", i, h_y[i]);
  }
#endif
  fun_SpMV(minusone, matJCt_c, vec_d_y, one, vec_d_rx);
  cublasDdot(handle_cublas, H->n, d_rx, 1, d_rx, 1, &norm_resx_sq);
  //  Calculate error in ry
  cusparseDnVecDescr_t vec_d_ry_c = NULL;
  cusparseCreateDnVec(&vec_d_ry_c, JC->n, d_ry_c, CUDA_R_64F);
  fun_SpMV(minusone, matJC_c, vec_d_x, one, vec_d_ry_c);
  cublasDdot(handle_cublas, JC->n, d_ry_c, 1, d_ry_c, 1, &norm_resy_sq);
  // Calculate final relative norm
  norm_resx_sq+=norm_resy_sq;
  double norm_res = sqrt(norm_resx_sq)/sqrt(norm_rx_sq);
  printf("||Ax-b||/||b|| = %32.32g\n", norm_res);
  //  Start of block - free memory
  free(rx);
  free(rs);
  free(ry);
  free(ryd);
  cudaFree(d_x);
  cudaFree(d_s);
  cudaFree(d_y);
  cudaFree(d_yd);
  cudaFree(d_z);
  cudaFree(d_rx);
  cudaFree(d_rxp);
  cudaFree(d_Hrxp);
  cudaFree(d_schur);
  cudaFree(d_rs);
  cudaFree(d_ry);
  cudaFree(d_ry_c);
  cudaFree(d_ryd);
  cudaFree(d_ryd_s);
  cudaFree(d_rx_til);
  cudaFree(d_rx_hat);
  cudaFree(d_rs_til);
  cudaFree(H_a);
  cudaFree(H_ja);
  cudaFree(H_ia);
  cudaFree(Ds_a);
  cudaFree(JC_a);
  cudaFree(JC_ja);
  cudaFree(JC_ia);
  cudaFree(JCt_a);
  cudaFree(JCt_ja);
  cudaFree(JCt_ia);
  cudaFree(JC_a_c);
  cudaFree(JC_ja_c);
  cudaFree(JC_ia_c);
  cudaFree(JCt_a_c);
  cudaFree(JCt_ja_c);
  cudaFree(JCt_ia_c);
  cudaFree(JD_a);
  cudaFree(JD_as);
  cudaFree(JD_ja);
  cudaFree(JD_ia);
  cudaFree(JDt_a);
  cudaFree(JDt_ja);
  cudaFree(JDt_ia);
  free(h_x);
  free(h_s);
  free(h_y);
  free(h_yd);
  free(H->csr_ia);
  free(H->csr_ja);
  free(H->csr_vals);
  free(H->coo_cols);
  free(H->coo_rows);
  free(H->coo_vals);
  free(H);
  free(Ds->coo_cols);
  free(Ds->csr_ia);
  free(Ds->coo_rows);
  free(Ds->coo_vals);
  free(Ds);
  free(JC->coo_cols);
  free(JC->csr_ia);
  free(JC->coo_rows);
  free(JC->coo_vals);
  free(JC);
  free(JD->coo_cols);
  free(JD->csr_ia);
  free(JD->coo_rows);
  free(JD->coo_vals);
  free(JD);
  cudaFree(buffercsr3);
  cudaFree(buffer_gpu);
  // cudaFree(buffer_schur);
  // cudaFree(buffer_d_z);
  cudaFree(dBuffer3);
  cudaFree(dBuffer4);
  cudaFree(buffer_add2);
  // cudaFree(buffer_rx_hat);
  cudaFree(bufferJC);
  cudaFree(bufferJC2);
  cudaFree(max_d);
#if 0
  free(max_h);
  cudaFree(scale);
#endif
  cudaFree(d_perm);
  cudaFree(drev_perm);
  cudaFree(d_perm_mapH);
  cudaFree(d_perm_mapJ);
  cudaFree(d_perm_mapJt);
  cudaFree(Htil_rows);
  cudaFree(Htil_cols);
  cudaFree(Htil_vals);
  cudaFree(JCtJC_rows);
  cudaFree(JCtJC_cols);
  cudaFree(JCtJC_vals);
  cudaFree(Hgam_rows);
  cudaFree(Hgam_cols);
  cudaFree(Hgam_vals);
  cudaFree(Hgamp_val);
  cudaFree(Jcp_val);
  cudaFree(Jctp_val);
  free(Hgam_h_rows);
  free(Hgam_h_cols);
  free(Hgam_p_rows);
  free(Hgam_p_cols);
  free(Jc_p_cols);
  free(Jct_p_cols);
  free(Jct_rows);
  free(Jct_cols);
  free(Jct_p_rows);
  gettimeofday(&t2, 0);
  timeIO = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  printf("time for IO+API+error ev(ms). : %16.16f\n", timeIO);
  if (norm_res<norm_tol){
    printf("Residual test passed ");
  }
  else{
    printf("Residual test failed ");
    return 1;
  }
  return 0;
}
