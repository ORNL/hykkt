#pragma once

#include <cusparse.h>
#include <cublas.h>
#include "cusparse_utils.hpp"
#include "constants.hpp"
#include "CholeskyClass.hpp"
#include "PCGOperator.hpp"
#include "LogBarrierInfo.hpp"

class LQOperator : public PCGOperator {
public:
    LQOperator(LogBarrierInfo& info, double* h_v);
    ~LQOperator();

    void load_H_matrix(double* h_v);
    void set_Q_scalar(double t);
    int get_operator_size() const;
    void apply(double* v, double* out);
    void extract_sparse_structure(double* out);
    void extract_inv_linear_diagonal(double* out);
    void preconditioner_solve(double* b, double* out);
private:
    void updateParams(bool params_updated);
    void factorize();
    
    LogBarrierInfo& info_; //log barrier problem formulation
    cudaEvent_t event_;
    
    //vector structures for intermediateds in matrix multiplication
    cusparseDnVecDescr_t w_desc_;
    cusparseDnVecDescr_t r_desc_;
    cusparseDnVecDescr_t v_desc_;
    cusparseDnVecDescr_t y_desc_;

    CholeskyClass* cholesky_ = nullptr; //pointer to cholesky class for factorization of sparsified operator
    
    bool params_updated_; //boolean for if H or t parameters have been updated so S values need to be updated
    bool factorized_ = false; //boolean for if the operator is factorized through taking the inverse of the diagonal already
    bool linear_allocated_ = false; //boolean for if linear system buffers allocated
    bool quadratic_allocated_ = false; //boolean for if quadradic system buffers allocated

    void* buffer1_; //buffer used in Av product
    void* buffer2_; //buffer used in AtHAv and Qv product

    double t_; //scalar multiplying matrix Q
    double tol_ = 1e-12;

    double* h_v_; //diagonal values of H
    double* s_v_; //values of sparsified operator S = Sparse(Q + A'HA), size n if Q = 0, size q_nnz if Q != 0

    double* w_; //intermediate vector used in matrix multiplaction, size m_
    double* r_; //intermediate vector used in matrix multiplaction, size n_
};