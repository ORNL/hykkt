#pragma once

#include <cusparse.h>
#include <cublas.h>

//TODO: change to using get methods rather than . operator to access variables

class LogBarrierInfo
{
public:
    LogBarrierInfo(int m, 
        int n, 
        int a_nnz, 
        int* a_i, 
        int* a_j, 
        double* a_v, 
        double* b,
        double* c,
        cusparseHandle_t& cusparse_handle);
    
    LogBarrierInfo(int m, 
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
        cusparseHandle_t& cusparse_handle);

    ~LogBarrierInfo();

    //member variables
    cusparseHandle_t cusparse_handle_;

    cusparseSpMatDescr_t a_desc_;
    cusparseSpMatDescr_t a_t_desc_;
    cusparseSpMatDescr_t q_desc_;

    int m_; //m dimension of mxn A matrix
    int n_; //n dimension of mxn A matrix, nxn Q matrix, nxn H matrix

    int q_nnz_; //number of nonzeros in Q
    int* q_i_ = nullptr; //csr rows of Q
    int* q_j_ = nullptr; //csr cols of Q
    double* q_v_ = nullptr; //nonzero values of Q

    int a_nnz_; //number of nonzeros in A
    int* a_i_ = nullptr; //csr rows of A
    int* a_j_ = nullptr; //csr cols of A
    double* a_v_ = nullptr; //nonzero values of A

    int* a_t_i_ = nullptr; //csr rows of A transpose
    int* a_t_j_ = nullptr; //csr cols of A transpose
    double* a_t_v_ = nullptr; //nonzero values of A transpose

    double* c_ = nullptr; //c vector in problem formulation
    double* b_ = nullptr; //barrier vector in problem formulation
    bool quadratic_; //whether the problem is quadratic or not

private:
    void allocateLinearComponents();
    void allocateQuadraticComponents();

    void* buffer0_ = nullptr; //buffer for taking transpose of A
};