#pragma once

#include <algorithm>
#include "matrix_vector_ops.hpp"
#include "cuda_memory_utils.hpp"
#include "constants.hpp"
#include "cusparse_params.hpp"

//C = A*B
//E = D + C

class SpgemmClass
{
public:
  // constructor
  SpgemmClass(int n, 
              cusparseHandle_t handle, 
              double alpha_p, 
              double alpha_s, 
              double beta_s) 
    : n_(n),
      handle_(handle),
      alpha_p_(alpha_p),
      alpha_s_(alpha_s),
      beta_s_(beta_s)
  {
    allocate_workspace();
  }


  // destructor
  ~SpgemmClass()
  {
    checkCudaErrors(cusparseSpGEMM_destroyDescr(spgemm_desc_));
    checkCudaErrors(cusparseDestroySpMat(c_desc_));
    
    deleteMatrixOnDevice(c_i_, c_j_, c_v_);
    deleteOnDevice(buffer_add_);
  }

  void load_product_matrices(int m_c,
                             cusparseSpMatDescr_t a_desc, 
                             cusparseSpMatDescr_t b_desc)
  {
    a_desc_ = a_desc; 
    b_desc_ = b_desc;
  
    c_desc_ = NULL;
    createCsrMat(&c_desc_, m_c, m_c, 0, NULL, NULL, NULL);
  }

  void allocate_product()
  {
    void*  d_buffer1    = NULL;
    void*  d_buffer2    = NULL;
    void*  d_buffer3    = NULL;
    void*  d_buffer4    = NULL;
    void*  d_buffer5    = NULL;
    
    size_t buffer_size1 = 0;
    size_t buffer_size2 = 0;
    size_t buffer_size3 = 0;
    size_t buffer_size4 = 0;
    size_t buffer_size5 = 0;
    
    //------------------------------------------------------------------------
    
    checkCudaErrors(cusparseSpGEMMreuse_workEstimation(handle_,
          CUSPARSE_OPERATION,
          CUSPARSE_OPERATION,
          a_desc_,
          b_desc_,
          c_desc_,
          CUSPARSE_ALGORITHM,
          spgemm_desc_,
          &buffer_size1,
          NULL));
    
    allocateBufferOnDevice(&d_buffer1, buffer_size1);
 
    checkCudaErrors(cusparseSpGEMMreuse_workEstimation(handle_,
          CUSPARSE_OPERATION,
          CUSPARSE_OPERATION,
          a_desc_,
          b_desc_,
          c_desc_,
          CUSPARSE_ALGORITHM,
          spgemm_desc_,
          &buffer_size1,
          d_buffer1));

    //-----------------------------------------------------------------------

    checkCudaErrors(cusparseSpGEMMreuse_nnz(handle_,
          CUSPARSE_OPERATION,
          CUSPARSE_OPERATION,
          a_desc_,
          b_desc_,
          c_desc_,
          CUSPARSE_ALGORITHM,
          spgemm_desc_,
          &buffer_size2,
          NULL,
          &buffer_size3,
          NULL,
          &buffer_size4,
          NULL));
    
    allocateBufferOnDevice(&d_buffer2, buffer_size2);
    allocateBufferOnDevice(&d_buffer3, buffer_size3);
    allocateBufferOnDevice(&d_buffer4, buffer_size4);

    checkCudaErrors(cusparseSpGEMMreuse_nnz(handle_,
          CUSPARSE_OPERATION,
          CUSPARSE_OPERATION,
          a_desc_,
          b_desc_,
          c_desc_,
          CUSPARSE_ALGORITHM,
          spgemm_desc_,
          &buffer_size2,
          d_buffer2,
          &buffer_size3,
          d_buffer3,
          &buffer_size4,
          d_buffer4));

    //-----------------------------------------------------------------------
  
    int64_t c_n; //rows
    int64_t c_m; //cols

    checkCudaErrors(cusparseSpMatGetSize(c_desc_, &c_n, &c_m, &nnz_c_));
    allocateMatrixOnDevice(n_, nnz_c_, &c_i_, &c_j_, &c_v_);
    checkCudaErrors(cusparseCsrSetPointers(c_desc_,
          c_i_,
          c_j_,
          c_v_));
    
    printf("\n\nNNZ_C: %d\n\n", nnz_c_);
    printf("\n\nBuffers: %d, %d, %d, %d, %d\n\n",
        buffer_size1,
        buffer_size2,
        buffer_size3,
        buffer_size4,
        buffer_size5);

    //------------------------------------------------------------
   
    checkCudaErrors(cusparseSpGEMMreuse_copy(handle_,
          CUSPARSE_OPERATION,
          CUSPARSE_OPERATION,
          a_desc_,
          b_desc_,
          c_desc_,
          CUSPARSE_ALGORITHM,
          spgemm_desc_,
          &buffer_size5,
          NULL));

    allocateBufferOnDevice(&d_buffer5, buffer_size5);

    checkCudaErrors(cusparseSpGEMMreuse_copy(handle_,
          CUSPARSE_OPERATION,
          CUSPARSE_OPERATION,
          a_desc_,
          b_desc_,
          c_desc_,
          CUSPARSE_ALGORITHM,
          spgemm_desc_,
          &buffer_size5,
          d_buffer5));
 
    //------------------------------------------------------------

    deleteOnDevice(d_buffer1);
    deleteOnDevice(d_buffer2);
    deleteOnDevice(d_buffer3);
    deleteOnDevice(d_buffer4);
    deleteOnDevice(d_buffer5);
  }

  void compute_product()
  { 
    checkCudaErrors(cusparseSpGEMMreuse_compute(handle_,
          CUSPARSE_OPERATION,
          CUSPARSE_OPERATION,
          &alpha_p_,
          a_desc_,
          b_desc_,
          &beta_p_,
          c_desc_,
          COMPUTE_TYPE,
          CUSPARSE_ALGORITHM,
          spgemm_desc_));
    
    return; 
    displayDeviceVector(c_v_, nnz_c_, 10, "Product");
  }

  void load_sum_matrices(int* d_i, 
                         int* d_j, 
                         double* d_v,
                         int nnz_d)
  {
    d_i_ = d_i;
    d_j_ = d_j;
    d_v_ = d_v;
    
    nnz_d_ = nnz_d; 
  }

  void allocate_sum()
  {
    size_t buffer_add_size;
    allocateVectorOnDevice(n_ + 1, &e_i_);
    checkCudaErrors(cusparseSetPointerMode(handle_, 
                                           CUSPARSE_POINTER_MODE_HOST));
    checkCudaErrors(cusparseDcsrgeam2_bufferSizeExt(handle_, 
                                    n_, 
                                    n_, 
                                    &alpha_s_, 
                                    descr_a_, 
                                    nnz_d_ , 
                                    d_v_, 
                                    d_i_, 
                                    d_j_, 
                                    &beta_s_,
                                    descr_a_, 
                                    nnz_c_, 
                                    c_v_, 
                                    c_i_, 
                                    c_j_, 
                                    descr_a_, 
                                    e_v_, 
                                    e_i_,
                                    e_j_, 
                                    &buffer_add_size));
    allocateBufferOnDevice(&buffer_add_, buffer_add_size);
    checkCudaErrors(cusparseXcsrgeam2Nnz(handle_, 
                         n_, 
                         n_, 
                         descr_a_, 
                         nnz_d_, 
                         d_i_, 
                         d_j_, 
                         descr_a_, 
                         nnz_c_,
                         c_i_, 
                         c_j_, 
                         descr_a_, 
                         e_i_, 
                         nnz_e_pointer_, 
                         buffer_add_));
    
    nnz_e_ = *nnz_e_pointer_;
    
    allocateVectorOnDevice(nnz_e_, &e_j_);
    allocateVectorOnDevice(nnz_e_, &e_v_);
  }

  void compute_sum()
  {
    checkCudaErrors(cusparseDcsrgeam2(handle_, 
                      n_, 
                      n_, 
                      &alpha_s_, 
                      descr_a_, 
                      nnz_d_, 
                      d_v_, 
                      d_i_, 
                      d_j_, 
                      &beta_s_, 
                      descr_a_,
                      nnz_c_, 
                      c_v_, 
                      c_i_, 
                      c_j_, 
                      descr_a_, 
                      e_v_, 
                      e_i_,
                      e_j_, 
                      buffer_add_));
  }

  int getResultMatrix(int** e_i, int** e_j, double** e_v)
  {
    cloneDeviceVector(n_ + 1, &e_i_, e_i);
    cloneDeviceVector(nnz_e_, &e_j_, e_j);
    cloneDeviceVector(nnz_e_, &e_v_, e_v);
    
    return nnz_e_;
  }

private:

  void allocate_workspace()
  {
    checkCudaErrors(cusparseSpGEMM_createDescr(&spgemm_desc_));
    checkCudaErrors(cusparseCreateMatDescr(&descr_a_));
    checkCudaErrors(cusparseSetMatType(descr_a_, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(descr_a_, INDEX_BASE));
    nnz_e_pointer_ = &nnz_e_;
  }
  
  // member variables
  cusparseHandle_t handle_;
  cusparseSpGEMMDescr_t spgemm_desc_;
  cusparseMatDescr_t descr_a_;

  int n_;

  double alpha_p_;
  double beta_p_ = ZERO;
  double alpha_s_;
  double beta_s_;
 
  cusparseSpMatDescr_t a_desc_;
  cusparseSpMatDescr_t b_desc_;
  cusparseSpMatDescr_t c_desc_;
  
  void* buffer_add_; 

  int* c_i_;
  int* c_j_;
  double* c_v_;
  
  int* d_i_;
  int* d_j_;
  double* d_v_;

  int* e_i_;
  int* e_j_;
  double* e_v_;

  int64_t nnz_c_;
  int nnz_d_;
  int* nnz_e_pointer_;
  int nnz_e_;
};
