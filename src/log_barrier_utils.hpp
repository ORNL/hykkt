#pragma once

#include "LBObjectiveInfo.hpp"
#include "LogBarrierInfo.hpp"

void fun_lb_objective(LogBarrierInfo& problem_info, double* x, double* Q_x, double* A_x, double t, double* out); //quadratic
void fun_lb_objective(LogBarrierInfo& problem_info, double* x, double* A_x, double t, double* out); //linear
void fun_lb_gradient(LogBarrierInfo& info, double* Q_x, double* A_x, double* out, double t, double scale); //quadratic 
void fun_lb_gradient(LogBarrierInfo& info, double* A_x, double* out, double t, double scale); //linear
void fun_lb_hessian(LogBarrierInfo& info, double* A_x, double* out);