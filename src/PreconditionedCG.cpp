#include "PreconditionedCG.hpp"

#include "cuda_memory_utils.hpp"
#include "cuda_check_errors.hpp"
#include "matrix_vector_ops.hpp"
#include "vector_vector_ops.hpp"
#include "constants.hpp"
#include "chrono"
#include "sys/time.h"

void displayTime(std::chrono::steady_clock::time_point& start, std::string name)
{
    auto end = std::chrono::steady_clock::now();
    printf("%s: %f\n", name.c_str(), std::chrono::duration<double>(end - start).count());
    start = end;
}

PreconditionedCG::PreconditionedCG()
{
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc((void**)&h_r_norm2_, sizeof(double), cudaHostAllocMapped);
    cudaHostAlloc((void**)&h_b_norm2_, sizeof(double), cudaHostAllocMapped);
    cudaHostGetDevicePointer((double **)&d_r_norm2_, (double *)h_r_norm2_, 0);
    cudaHostGetDevicePointer((double **)&d_b_norm2_, (double *)h_b_norm2_, 0);
    *h_r_norm2_ = 1;
    *h_b_norm2_ = 1;
    allocateValueOnDevice(&d_p_Ap_);
    allocateValueOnDevice(&d_r_dot_z_);
    allocateValueOnDevice(&d_r_dot_z_prev_);
}

PreconditionedCG::~PreconditionedCG()
{
    clean_workspace();
    deleteOnDevice(d_r_dot_z_);
    deleteOnDevice(d_r_dot_z_prev_);
    deleteOnDevice(d_p_Ap_);
}   

/**
 * @brief Allocate workspace
 * 
 * @param n - size of nxn operator
 * 
 * @post allocate r_, z_, p_, A_p_ - vectors of length n for use in conjugate gradient solver
*/
void PreconditionedCG::allocate_workspace(int n)
{
    clean_workspace(); //makes sure there are no memory leaks on gpu
    n_ = n;
    allocateVectorOnDevice(n_, &r_);
    allocateVectorOnDevice(n_, &z_);
    allocateVectorOnDevice(n_, &p_);
    allocateVectorOnDevice(n_, &A_p_);
    workspace_allocated_ = true;
 }

/**
 * @brief Deallocate workspace
 * 
 * @post if workspace has been allocated, deletes r_, z_, p_, A_p_ from device
*/
void PreconditionedCG::clean_workspace()
{
    if (workspace_allocated_){
        deleteOnDevice(r_);
        deleteOnDevice(z_);
        deleteOnDevice(p_);
        deleteOnDevice(A_p_);
    }
}

/**
 * @brief Setup r_0, z_0, p_0 vectors
 * 
 * @param solver - solver for operator of size nxn
 * b, x - vectors in system Ax = b
 * itmax - max number of iterations
 * tol - min value for convergence
 * 
 * @pre A is size nxn, b is size n, x is size n, all on the devec
 * 
 * @post returns num iterations if solved system under itmax iterations, else returns -1
 * stores resulting solution vector on device x
 * 
*/
int PreconditionedCG::solve(int itmax, double tol, PCGOperator& p_operator, double* b, double* x)
{
    int n = p_operator.get_operator_size();
    if (!workspace_allocated_ || n != n_) {
        allocate_workspace(n);
    }
    p_operator.apply(x, r_); //r_0 = A x_0

    fun_add_vecs_scaled(n_, MINUS_ONE, ONE, r_, b); //r_0 = b - A x_0
    p_operator.preconditioner_solve(r_, z_); //z_0 = M^-1 r_0
    copyDeviceVector(n_, z_, p_); //p_0 = z_0
    deviceDotProduct(n_, r_, z_, d_r_dot_z_); //<r_, z_>
    deviceDotProduct(n_, b, b, (double *) d_b_norm2_); //<b, b>

    cudaEvent_t event;
    cudaEventCreate(&event);
    int k;
    for (k = 0; k < itmax; k++) {
        p_operator.apply(p_, A_p_); //A_p = A * p 

        deviceDotProduct(n_, p_, A_p_, d_p_Ap_); //<p_, A_p_>

        fun_cg_helper1(n, d_r_dot_z_, d_p_Ap_, x, r_, p_, A_p_); //out = out + alpha * p_, r_ = r_ - alpha * A_p_

        deviceDotProduct(n_, r_, r_, (double *) d_r_norm2_);
        cudaEventRecord(event);
        cudaEventSynchronize(event);

        if (*h_r_norm2_ / *h_b_norm2_ < tol * tol) {
            return k;
        }
        else if(k == itmax - 1) {
            break;
        }

        p_operator.preconditioner_solve(r_, z_); //z_ = M^-1 r_

        fun_mem_copy(d_r_dot_z_, d_r_dot_z_prev_); //r_dot_z_prev_ = r_dot_z_

        deviceDotProduct(n_, r_, z_, d_r_dot_z_); //<r_, z_>

        fun_cg_helper2(n, d_r_dot_z_, d_r_dot_z_prev_, p_, z_); //p_ = z_ + <r_, z_> / <r_prev_, z_prev_> * p_
    }
    return -1;
}
