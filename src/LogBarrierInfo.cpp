#include "LogBarrierInfo.hpp"
#include "cusparse_utils.hpp"
#include "cuda_memory_utils.hpp"

/**
 * @brief constructor used in linear problem where Q is 0
*/
LogBarrierInfo::LogBarrierInfo(int m, 
    int n, 
    int a_nnz, 
    int* a_i, 
    int* a_j, 
    double* a_v,
    double* b,
    double* c,
    cusparseHandle_t& cusparse_handle)
    : m_{m},
    n_{n},
    a_nnz_{a_nnz},
    a_i_{a_i},
    a_j_{a_j},
    a_v_{a_v},
    b_{b},
    c_{c},
    cusparse_handle_{cusparse_handle},
    quadratic_{false}
{ 
    allocateLinearComponents();
}

/**
 * @brief constructor used in linear problem where Q is nonzero
*/
LogBarrierInfo::LogBarrierInfo(int m, 
    int n, 
    int q_nnz,
    int* q_i,
    int* q_j,
    double* q_v,
    int a_nnz, 
    int* a_i, 
    int* a_j, 
    double* a_v, 
    double* b,
    double* c,
    cusparseHandle_t& cusparse_handle)
    : m_{m},
    n_{n},
    q_nnz_{q_nnz},
    q_i_{q_i},
    q_j_{q_j},
    q_v_{q_v},
    a_nnz_{a_nnz},
    a_i_{a_i},
    a_j_{a_j},
    a_v_{a_v},
    b_{b},
    c_{c},
    cusparse_handle_{cusparse_handle},
    quadratic_{true}
{   
    allocateLinearComponents();
    allocateQuadraticComponents();
}

/**
 * @brief destructor
*/
LogBarrierInfo::~LogBarrierInfo()
{
    deleteDescriptor(a_desc_);
    deleteDescriptor(a_t_desc_);
    if (quadratic_)
    {
        deleteDescriptor(q_desc_);
    }
    deleteOnDevice(buffer0_);
}

/**
 * @brief allocates A and A transpose descriptors
*/
void LogBarrierInfo::allocateLinearComponents()
{
    allocateMatrixOnDevice(n_, a_nnz_, &a_t_i_, &a_t_j_, &a_t_v_);
    transposeMatrixOnDevice(cusparse_handle_, m_, n_, a_nnz_, a_i_, a_j_, a_v_, a_t_i_, a_t_j_, a_t_v_, &buffer0_, false);
    createCsrMat(&a_desc_, m_, n_, a_nnz_, a_i_, a_j_, a_v_);
    createCsrMat(&a_t_desc_, n_, m_, a_nnz_, a_t_i_, a_t_j_, a_t_v_);
}

/**
 * @brief allocates Q descriptor
*/
void LogBarrierInfo::allocateQuadraticComponents()
{
    createCsrMat(&q_desc_, n_, n_, q_nnz_, q_i_, q_j_, q_v_);
}