#pragma once

#include "LQOperator.hpp"

class PreconditionedCG {
public:
    PreconditionedCG();
    ~PreconditionedCG();
    int solve(int itmax, double tol, PCGOperator& p_operator, double* b, double* x);
private:
    void allocate_workspace(int n); 
    void clean_workspace();

    bool workspace_allocated_ = false; //whether or not operator has been loaded

    int n_; //number of rows in operator

    double* r_; //used in solver
    double* z_; //used in solver
    double* p_; //used in solver
    double* A_p_; //used in solver

    volatile double* h_r_norm2_;
    volatile double* d_r_norm2_;
    volatile double* h_b_norm2_;
    volatile double* d_b_norm2_;
    double* d_r_dot_z_;
    double* d_p_Ap_;
    double* d_r_dot_z_prev_;
};