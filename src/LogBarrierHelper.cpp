#include "LogBarrierHelper.hpp"
#include "log_barrier_utils.hpp"
#include "cuda_memory_utils.hpp"
#include "cusparse_utils.hpp"
#include "constants.hpp"
#include "matrix_vector_ops.hpp"
#include "vector_vector_ops.hpp"
#include "cuda_check_errors.hpp"
#include "chrono"

/**
 * @brief constructor
*/
LogBarrierHelper::LogBarrierHelper(LogBarrierInfo& problem_info)
    : problem_info_{problem_info}
{
    cudaSetDeviceFlags(cudaDeviceMapHost);
    allocateVectorOnDevice(problem_info_.m_, &A_x_);
    allocateVectorOnDevice(problem_info_.n_, &Q_x_);
    allocateVectorOnDevice(problem_info_.n_, &gradient_);
    allocateVectorOnDevice(problem_info_.n_, &x_cache_);
    
    createDnVec(&A_x_desc_, problem_info_.m_, A_x_);
    createDnVec(&Q_x_desc_, problem_info_.n_, Q_x_);
    createDnVec(&x_cache_desc_, problem_info_.n_, x_cache_);

    cudaHostAlloc((void**)&h_grad_dot_dx_, sizeof(double), cudaHostAllocMapped);
    cudaHostAlloc((void**)&h_obj_value_, sizeof(double), cudaHostAllocMapped);
    cudaHostGetDevicePointer((double **)&d_grad_dot_dx_, (double *)h_grad_dot_dx_, 0);
    cudaHostGetDevicePointer((double **)&d_obj_value_, (double *)h_obj_value_, 0);
    cudaEventCreate(&event_);
}

/**
 * @brief destructor
*/
LogBarrierHelper::~LogBarrierHelper()
{
    deleteOnDevice(A_x_);
    deleteOnDevice(Q_x_);
    deleteOnDevice(gradient_);
    deleteOnDevice(x_cache_);

    deleteDescriptor(A_x_desc_);
    deleteDescriptor(Q_x_desc_);
    deleteDescriptor(x_cache_desc_);
    cudaEventDestroy(event_);
}

/**
 * @brief Find ideal step size
 * 
 * @param x - current x
 * dx - current dx
 * 
 * @return returns ideal step size, alpha
*/
double LogBarrierHelper::line_search(double t, cusparseDnVecDescr_t& x_desc, double* x, double* dx, double min_alpha)
{
    update_objective(t, x, x_desc); //objective at x
    gradient(t, x_desc, gradient_); //gradient at xs
    deviceDotProduct(problem_info_.n_, gradient_, dx, (double*)d_grad_dot_dx_);
    cudaEventRecord(event_);
    cudaEventSynchronize(event_);
    double alpha = 1.0;
    double f_0 = *h_obj_value_;
    double obj_value;
    while (alpha > min_alpha)
    {
        fun_add_vecs(problem_info_.n_, ONE, alpha, x, dx, x_cache_); //x_cache_ = x + alpha*dx
        update_objective(t, x_cache_, x_cache_desc_); //objective at x + alpha*dx
        cudaEventRecord(event_);
        cudaEventSynchronize(event_);
        obj_value = *h_obj_value_;
        if (std::isnan(obj_value) || obj_value > f_0 + c1_ * alpha * (*h_grad_dot_dx_)) {
            alpha *= c2_;
        }
        else {
            break;
        }
    }

    return alpha;
}

/**
 * @brief compute objective value at x
*/
void LogBarrierHelper::update_objective(double t, double* x, cusparseDnVecDescr_t& x_desc)
{
    update_Ax(x_desc);
    if (problem_info_.quadratic_) {
        update_Qx(x_desc);
        fun_lb_objective(problem_info_, x, Q_x_, A_x_, t, (double*)d_obj_value_);
    } else{
        fun_lb_objective(problem_info_, x, A_x_, t, (double *)d_obj_value_);
    }
}

/**
 * @brief compute objective value at x and return it
*/
double LogBarrierHelper::update_get_objective(double t, double* x, cusparseDnVecDescr_t& x_desc)
{
    update_objective(t, x, x_desc);
    cudaEventRecord(event_);
    cudaEventSynchronize(event_);
    return *h_obj_value_;
}

/**
 * @brief compute minus gradient at x
*/
void LogBarrierHelper::minus_gradient(double t, cusparseDnVecDescr_t& x_desc, double* out)
{
    gradient(t, x_desc, out, MINUS_ONE);
}

/**
 * @brief compute minus gradient at x
*/
void LogBarrierHelper::gradient(double t, cusparseDnVecDescr_t& x_desc, double* out, double scale)
{
    update_Ax(x_desc);
    if (problem_info_.quadratic_) {
        update_Qx(x_desc);
        fun_lb_gradient(problem_info_, Q_x_, A_x_, out, t, scale);
    }
    else {
        fun_lb_gradient(problem_info_, A_x_, out, t, scale);
    }
}

/**
 * @brief compute inverse hessian at x
*/
void LogBarrierHelper::hessian(cusparseDnVecDescr_t& x_desc, double* out)
{
    update_Ax(x_desc);
    fun_lb_hessian(problem_info_, A_x_, out);
}

/**
 * @brief update Qx
*/
void LogBarrierHelper::update_Qx(cusparseDnVecDescr_t& x_desc)
{
    SpMV_product_reuse(problem_info_.cusparse_handle_, ONE, problem_info_.q_desc_, x_desc, ZERO, Q_x_desc_, &buffer2_, buffer2_allocated_);
    buffer2_allocated_ = true;
}

/**
 * @brief update Ax
*/
void LogBarrierHelper::update_Ax(cusparseDnVecDescr_t& x_desc)
{
    SpMV_product_reuse(problem_info_.cusparse_handle_, ONE, problem_info_.a_desc_, x_desc, ZERO, A_x_desc_, &buffer1_, buffer1_allocated_);
    buffer1_allocated_ = true;
}

/**
 * @brief Set the line search parameters
*/
void LogBarrierHelper::set_line_search_params(double c1, double c2)
{
    c1_ = c1;
    c2_ = c2;
}