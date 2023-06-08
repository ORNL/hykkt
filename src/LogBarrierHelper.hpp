#pragma once

#include "LogBarrierInfo.hpp"
#include "LBObjectiveInfo.hpp"
#include <cusparse.h>
#include <cublas.h>

//TODO: optimize to reduce recalculating Ax, 1.0 ./ (b - Ax), etc
class LogBarrierHelper
{
public:
    LogBarrierHelper(LogBarrierInfo& info);
    ~LogBarrierHelper();
    double line_search(double t, cusparseDnVecDescr_t& x_desc, double* x, double* dx, double min_alpha = 1e-15);
    void minus_gradient(double t, cusparseDnVecDescr_t& x_desc, double* out);
    void gradient(double t, cusparseDnVecDescr_t& x_desc, double* out, double scale = 1);
    double update_get_objective(double t, double* x, cusparseDnVecDescr_t& x_desc);
    void hessian(cusparseDnVecDescr_t& x_desc, double* out);
    void set_line_search_params(double c1, double c2);
private:
    void update_Qx(cusparseDnVecDescr_t& x_desc);
    void update_Ax(cusparseDnVecDescr_t& x_desc);
    void update_objective(double t, double* x, cusparseDnVecDescr_t& x_desc);

    LogBarrierInfo& problem_info_; //log barrier problem formulation

    cusparseDnVecDescr_t A_x_desc_;
    cusparseDnVecDescr_t Q_x_desc_;
    cusparseDnVecDescr_t x_cache_desc_;

    cudaEvent_t event_;

    double c1_ = 1e-4; //line search parameter
    double c2_ = 0.5; //line search parameter
    
    volatile double* h_obj_value_; //objective value for use by objective function
    volatile double* d_obj_value_; //objective value on device for use by objective function

    volatile double* h_grad_dot_dx_; //gradient dot dx for use by line search
    volatile double* d_grad_dot_dx_; //gradient dot dx on device for use by line search

    double* A_x_; //matrix vector mult Ax
    double* Q_x_; //matrix vector mult Qx
    double* gradient_; //gradient of objective
    double* x_cache_; //cache of x for use in line search

    void* buffer1_; //buffer used for A_x_
    void* buffer2_; //buffer used for Q_x_

    bool buffer1_allocated_ = false; //whether buffer1_ has been allocated
    bool buffer2_allocated_ = false; //whether buffer2_ has been allocated
};