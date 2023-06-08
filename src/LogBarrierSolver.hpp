#pragma once

#include "PreconditionedCG.hpp"
#include "LogBarrierInfo.hpp"

class LogBarrierSolver
{
public:
    LogBarrierSolver();
    ~LogBarrierSolver();
    int solve(LogBarrierInfo& info, double* out, bool verbose = false);
    void set_barrier_tol(double);
    void set_newton_tol(double);
    void set_cg_tol(double);
    void set_mu(double);
    void set_barrier_itmax(int);
    void set_newton_itmax(int);
    void set_cg_itmax(int);
private:
    cudaEvent_t event_;

    double barrier_tol_ = 1e-5; //barrier method tolerance
    double newton_tol_ = 1e-8; //newton method tolerance
    double cg_tol_ = 1e-12; //conjugate gradient tolerance
    double mu_ = 1.2; //how fast t grows
    int barrier_itmax_ = 1; //maximum number of barrier iterations
    int newton_itmax_ = 1; //maximum number of newton iterations
    int cg_itmax_ = 1000; //maximum number of cg iterations per newton iteration

    volatile double* h_lam_sq_;
    volatile double* d_lam_sq_;
    volatile double* h_obj_val_;
    volatile double* d_obj_val_;
};