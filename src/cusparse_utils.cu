#include <cusparse.h>
#include <cublas.h>

#include <cusparse_utils.hpp>
#include "cuda_memory_utils.hpp"
#include "cusparse_params.hpp"
#include "matrix_vector_ops.hpp"

#include "cuda_check_errors.hpp"

void displaySpMatValues(cusparseSpMatDescr_t mat_desc, 
    int start_i, 
    int display_n,
    std::string label)
{
  int64_t rows;
  int64_t cols;
  int64_t nnz;
  checkCudaErrors(cusparseSpMatGetSize(mat_desc,
                  &rows,
                  &cols,
                  &nnz));

  double* mat_v;
  allocateVectorOnDevice(nnz, &mat_v);
  cusparseSpMatGetValues(mat_desc, (void**)(&mat_v));
  displayDeviceVector(mat_v, 
                      nnz, 
                      start_i,
                      display_n,
                      label);
}

void deleteDescriptor(cusparseSpGEMMDescr_t& desc)
{
  checkCudaErrors(cusparseSpGEMM_destroyDescr(desc));
}

void deleteDescriptor(cusparseSpMatDescr_t& desc)
{
  checkCudaErrors(cusparseDestroySpMat(desc));
}

void deleteDescriptor(cusparseMatDescr_t& desc)
{
  checkCudaErrors(cusparseDestroyMatDescr(desc));
}

void transposeMatrixOnDevice(cusparseHandle_t handle,
                             int n,
                             int m,
                             int nnz,
                             const int* a_i,
                             const int* a_j,
                             const double* a_v,
                             int* at_i,
                             int* at_j,
                             double* at_v,
                             void** buffer,
                             bool allocated)
{
  if(!allocated){
    size_t buffersize;
    checkCudaErrors(cusparseCsr2cscEx2_bufferSize(handle,
                                                  n, 
                                                  m, 
                                                  nnz,
                                                  a_v,  
                                                  a_i,  
                                                  a_j,
                                                  at_v, 
                                                  at_i, 
                                                  at_j,
                                                  COMPUTE_TYPE,
                                                  CUSPARSE_ACTION_NUMERIC,
                                                  INDEX_BASE,
                                                  CUSPARSE_CSR2CSC_ALG1,
                                                  &buffersize));

    // Create a buffer
    allocateBufferOnDevice(buffer,buffersize);
  }
  checkCudaErrors(cusparseCsr2cscEx2(handle,
                                     n,
                                     m,
                                     nnz,
                                     a_v,
                                     a_i,
                                     a_j,
                                     at_v,
                                     at_i,
                                     at_j,
                                     COMPUTE_TYPE,
                                     CUSPARSE_ACTION_NUMERIC,
                                     INDEX_BASE,
                                     CUSPARSE_CSR2CSC_ALG1,
                                     *buffer));
}

void createCsrMat(cusparseSpMatDescr_t* mat_desc, 
                  int n, 
                  int m, 
                  int nnz, 
                  int* a_i, 
                  int* a_j, 
                  double* a_v)
{

  checkCudaErrors(cusparseCreateCsr(mat_desc, 
                                    n, 
                                    m, 
                                    nnz,
                                    a_i, 
                                    a_j, 
                                    a_v, 
                                    INDEX_TYPE, 
                                    INDEX_TYPE, 
                                    INDEX_BASE, 
                                    COMPUTE_TYPE));
}

void createDnVec(cusparseDnVecDescr_t* vec_desc, int n, double* d_vec)
{
  checkCudaErrors(cusparseCreateDnVec(vec_desc, n, d_vec, COMPUTE_TYPE));
}

void createSpGEMMDescr(cusparseSpGEMMDescr_t* spgemm_desc)
{
  checkCudaErrors(cusparseSpGEMM_createDescr(spgemm_desc));
}

void createSparseMatDescr(cusparseMatDescr_t& descr)
{ 
  checkCudaErrors(cusparseCreateMatDescr(&descr));
  checkCudaErrors(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  checkCudaErrors(cusparseSetMatIndexBase(descr, INDEX_BASE));
}

void createSparseHandle(cusparseHandle_t& handle)
{
  checkCudaErrors(cusparseCreate(&handle));  
}

void createCublasHandle(cublasHandle_t& handle)
{
  checkCudaErrors(cublasCreate(&handle));
}
