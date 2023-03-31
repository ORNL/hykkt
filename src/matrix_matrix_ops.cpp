#include "matrix_matrix_ops.hpp"
#include "matrix_vector_ops.hpp"
#include "cuda_memory_utils.hpp"
#include <assert.h>
#include <stdio.h>
#include "cuda_memory_utils.hpp"
#include "cusparse_params.hpp"
#include "constants.hpp"

#include "cuda_check_errors.hpp"


void SpGEMM_workEstimation(cusparseHandle_t& handle,
    const cusparseSpMatDescr_t& a_desc,
    const cusparseSpMatDescr_t& b_desc,
    cusparseSpMatDescr_t& c_desc,
    cusparseSpGEMMDescr_t& spgemm_desc,
    void** d_buffer)
{
    size_t buffer_size = 0;
    
      checkCudaErrors(cusparseSpGEMMreuse_workEstimation(handle,
          CUSPARSE_OPERATION,
          CUSPARSE_OPERATION,
          a_desc,
          b_desc,
          c_desc,
          CUSPARSE_ALGORITHM,
          spgemm_desc,
          &buffer_size,
          NULL)); 

    allocateBufferOnDevice(d_buffer, buffer_size);

    checkCudaErrors(cusparseSpGEMMreuse_workEstimation(handle,
          CUSPARSE_OPERATION,
          CUSPARSE_OPERATION,
          a_desc,
          b_desc,
          c_desc,
          CUSPARSE_ALGORITHM,
          spgemm_desc,
          &buffer_size,
          *d_buffer));
}

void SpGEMM_calculate_nnz_reuse(cusparseHandle_t& handle,
    const cusparseSpMatDescr_t& a_desc,
    const cusparseSpMatDescr_t& b_desc,
    cusparseSpMatDescr_t& c_desc,
    cusparseSpGEMMDescr_t& spgemm_desc,
    void** d_buffer1,
    void** d_buffer2,
    void** d_buffer3)
{
    size_t buffer_size1 = 0;
    size_t buffer_size2 = 0;
    size_t buffer_size3 = 0;
    
    checkCudaErrors(cusparseSpGEMMreuse_nnz(handle,
          CUSPARSE_OPERATION,
          CUSPARSE_OPERATION,
          a_desc,
          b_desc,
          c_desc,
          CUSPARSE_ALGORITHM,
          spgemm_desc,
          &buffer_size1,
          NULL,
          &buffer_size2,
          NULL,
          &buffer_size3,
          NULL));
    
    allocateBufferOnDevice(d_buffer1, buffer_size1);
    allocateBufferOnDevice(d_buffer2, buffer_size2);
    allocateBufferOnDevice(d_buffer3, buffer_size3);

    checkCudaErrors(cusparseSpGEMMreuse_nnz(handle,
          CUSPARSE_OPERATION,
          CUSPARSE_OPERATION,
          a_desc,
          b_desc,
          c_desc,
          CUSPARSE_ALGORITHM,
          spgemm_desc,
          &buffer_size1,
          *d_buffer1,
          &buffer_size2,
          *d_buffer2,
          &buffer_size3,
          *d_buffer3));
}

void SpGEMM_setup_product_descr(int n,
                                int64_t& c_nnz,
                                int** c_i,
                                int** c_j,
                                double** c_v,
                                cusparseSpMatDescr_t& c_desc)
{
    int64_t c_num_rows;
    int64_t c_num_cols;
    checkCudaErrors(cusparseSpMatGetSize(c_desc, 
                                         &c_num_rows,
                                         &c_num_cols,
                                         &c_nnz));

    allocateMatrixOnDevice(n, static_cast<int>(c_nnz), c_i, c_j, c_v);

    checkCudaErrors(cusparseCsrSetPointers(c_desc, *c_i, *c_j, *c_v));
}

void SpGEMM_copy_result(cusparseHandle_t& handle,
    const cusparseSpMatDescr_t& a_desc,
    const cusparseSpMatDescr_t& b_desc,
    cusparseSpMatDescr_t& c_desc,
    cusparseSpGEMMDescr_t& spgemm_desc,
    void** d_buffer)
{
    size_t buffer_size = 0;

    checkCudaErrors(cusparseSpGEMMreuse_copy(handle,
          CUSPARSE_OPERATION,
          CUSPARSE_OPERATION,
          a_desc,
          b_desc,
          c_desc,
          CUSPARSE_ALGORITHM,
          spgemm_desc,
          &buffer_size,
          NULL));

    allocateBufferOnDevice(d_buffer, buffer_size);

    checkCudaErrors(cusparseSpGEMMreuse_copy(handle,
          CUSPARSE_OPERATION,
          CUSPARSE_OPERATION,
          a_desc,
          b_desc,
          c_desc,
          CUSPARSE_ALGORITHM,
          spgemm_desc,
          &buffer_size,
          *d_buffer));
}

void allocate_for_product(cusparseHandle_t& handle, 
    const cusparseSpMatDescr_t& a_desc, 
    const cusparseSpMatDescr_t& b_desc,             
    cusparseSpMatDescr_t& c_desc, 
    int n,
    int64_t& c_nnz,
    int** c_i,
    int** c_j,
    double** c_v,
    cusparseSpGEMMDescr_t& spgemm_desc,
    void** d_buffer4, 
    void** d_buffer5)
{
    
    void* d_buffer1;
    void* d_buffer2;
    void* d_buffer3;

    SpGEMM_workEstimation(handle,
        a_desc,
        b_desc,
        c_desc,
        spgemm_desc,
        &d_buffer1);
    
    SpGEMM_calculate_nnz_reuse(handle,
        a_desc,
        b_desc,
        c_desc,
        spgemm_desc,
        &d_buffer2,
        &d_buffer3,
        d_buffer4); 

    deleteOnDevice(d_buffer1);
    deleteOnDevice(d_buffer2);
    
    SpGEMM_setup_product_descr(n, c_nnz, c_i, c_j, c_v, c_desc);

    SpGEMM_copy_result(handle,
        a_desc,
        b_desc,
        c_desc,
        spgemm_desc,
        d_buffer5);

    deleteOnDevice(d_buffer3);
}

void compute_product(cusparseHandle_t& handle, 
    double alpha,
    const cusparseSpMatDescr_t& a_desc, 
    const cusparseSpMatDescr_t& b_desc, 
    cusparseSpMatDescr_t& c_desc, 
    cusparseSpGEMMDescr_t& spgemm_desc)
{
    checkCudaErrors(cusparseSpGEMMreuse_compute(handle,
          CUSPARSE_OPERATION,
          CUSPARSE_OPERATION,
          &alpha,
          a_desc,
          b_desc,
          &ZERO,
          c_desc,
          COMPUTE_TYPE,
          CUSPARSE_ALGORITHM,
          spgemm_desc));
}

//***************************************************************************//

void allocate_for_sum(cusparseHandle_t& handle,
    const int* a_i, 
    const int* a_j, 
    const double* a_v,
    double alpha,
    const int* b_i, 
    const int* b_j, 
    const double* b_v,
    double beta,
    int** c_i, 
    int** c_j, 
    double** c_v,
    int m, 
    int n,
    int nnz_a, 
    int nnz_b, 
    cusparseMatDescr_t& descr_a, 
    void** buffer_add, 
    int* nnz_total_ptr)
{
    size_t buffer_byte_size_add;
    
    allocateVectorOnDevice(n + 1, c_i);
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  
    //calculates sum buffer
    cusparseDcsrgeam2_bufferSizeExt(handle, 
                                    m, 
                                    n, 
                                    &alpha, 
                                    descr_a, 
                                    nnz_a, 
                                    a_v, 
                                    a_i, 
                                    a_j, 
                                    &beta,
                                    descr_a, 
                                    nnz_b, 
                                    b_v, 
                                    b_i, 
                                    b_j, 
                                    descr_a, 
                                    *c_v, 
                                    *c_i,
                                    *c_j, 
                                    &buffer_byte_size_add);
    
    allocateBufferOnDevice(buffer_add, buffer_byte_size_add);
  
    //determines sum row offsets and total number of nonzeros
    cusparseXcsrgeam2Nnz(handle, 
                         m, 
                         n, 
                         descr_a, 
                         nnz_a, 
                         a_i, 
                         a_j, 
                         descr_a, 
                         nnz_b,
                         b_i, 
                         b_j, 
                         descr_a, 
                         *c_i,
                         nnz_total_ptr, 
                         *buffer_add);
    
    allocateVectorOnDevice(*nnz_total_ptr, c_j);
    allocateVectorOnDevice(*nnz_total_ptr, c_v);
}

void compute_sum(cusparseHandle_t& handle,
    const int* a_i, 
    const int* a_j, 
    const double* a_v,
    double alpha,
    const int* b_i, 
    const int* b_j, 
    const double* b_v,
    double beta,
    int* c_i, 
    int* c_j, 
    double* c_v,
    int m,
    int n, 
    int nnz_a, 
    int nnz_b,
    cusparseMatDescr_t& descr_a, 
    void** buffer_add)
{
    cusparseDcsrgeam2(handle, 
        m, 
        n, 
        &alpha, 
        descr_a, 
        nnz_a, 
        a_v, 
        a_i, 
        a_j, 
        &beta, 
        descr_a,
        nnz_b, 
        b_v, 
        b_i, 
        b_j, 
        descr_a, 
        c_v, 
        c_i,
        c_j, 
        *buffer_add);
}
