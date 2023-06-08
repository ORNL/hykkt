#include "LogBarrierSolver.hpp"

#include "cuda_memory_utils.hpp"
#include "cuda_check_errors.hpp"
#include "matrix_vector_ops.hpp"
#include "vector_vector_ops.hpp"
#include "constants.hpp"
#include "LogBarrierHelper.hpp"

#include <chrono>

void printTime(std::chrono::steady_clock::time_point& start, std::string name)
{
    auto end = std::chrono::steady_clock::now();
    printf("%s: %f\n", name.c_str(), std::chrono::duration<double>(end - start).count());
    start = end;
}

//TODO: find clean way of displaying info with functions & use spd::log rather than printf
//TODO: remove dot product host registering and use memcpy instead

LogBarrierSolver::LogBarrierSolver()
{
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc((void**)&h_lam_sq_, sizeof(double), cudaHostAllocMapped);
    cudaHostAlloc((void**)&h_obj_val_, sizeof(double), cudaHostAllocMapped);
    checkCudaErrors(cudaHostGetDevicePointer((double**)&d_lam_sq_, (double*)h_lam_sq_, 0));
    checkCudaErrors(cudaHostGetDevicePointer((double**)&d_obj_val_, (double*)h_obj_val_, 0));
    cudaEventCreate(&event_);
}

LogBarrierSolver::~LogBarrierSolver()
{
    cudaEventDestroy(event_);
}

/**
 * @brief Solve the system using the log barrier method
 * 
 * @param info - info containing problem data for barrier method
 * out - output vector
 * verbose - whether or not to print out information
 * 
 * @return returns number of iterations converged or -1 if failed
*/
int LogBarrierSolver::solve(LogBarrierInfo& info, double* out, bool verbose)
{    
    int n = info.n_;
    int m = info.m_;

    double t = n * barrier_tol_; //initial t
    double alpha; //step size for newton method

    double* d_hessian_diag;
    allocateVectorOnDevice(m, &d_hessian_diag);

    LogBarrierHelper helper(info);
    LQOperator lqop(info, d_hessian_diag);
    PreconditionedCG cg_solver;

    double* x_cg;
    double* minus_gradient;
    allocateVectorOnDevice(n, &x_cg);
    allocateVectorOnDevice(n, &minus_gradient);

    cusparseDnVecDescr_t out_desc;
    createDnVec(&out_desc, n, out);
    
    int barrier_it;
    int newton_it;
    int total_cg_its;
    int cg_its;
    for (barrier_it = 0; barrier_it < barrier_itmax_; barrier_it++) {
        printf("t: %.12f\n", t);
        if (verbose) {
            printf("Barrier it: %d\n", barrier_it + 1);
        }
        for (newton_it = 0; newton_it < newton_itmax_; newton_it++) {
            auto n_start = std::chrono::steady_clock::now();
            if (verbose) {
                printf("    Newton it: %d\n", newton_it + 1);
            }
            //compute gradient
            helper.minus_gradient(t, out_desc, minus_gradient);
            //compute hessian
            helper.hessian(out_desc, d_hessian_diag);
            lqop.load_H_matrix(d_hessian_diag); //update with hessian matrix (actually is inverse of hessian)
            //solve system
            auto start = std::chrono::steady_clock::now();
            cg_its = cg_solver.solve(cg_itmax_, cg_tol_, lqop, minus_gradient, x_cg);
            if (verbose) {
                printf("    Total cg time: %f\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
            }
            //compute lambda squared
            deviceDotProduct(n, minus_gradient, x_cg, (double*)d_lam_sq_);
            //update alpha
            alpha = helper.line_search(t, out_desc, out, x_cg);
            //update x
            fun_add_vecs(n, out, alpha, x_cg);
            cudaEventRecord(event_);
            cudaEventSynchronize(event_);
            if (verbose) {
                printf("    Total newton time: %f\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - n_start).count());
            }
            //check convergence
            if (cg_its != -1) { //converged
                total_cg_its += cg_its;
                if (verbose) {
                    printf("        CG its: %d\n", cg_its);
                }
            }
            else
            {
                if (verbose) {
                    printf("        CG failed to converge in %d iterations\n", cg_itmax_);
                }
                //break;
            }
            if (verbose) {
                printf("        lambda sqaured: %.12f\n", *h_lam_sq_);
            }
            if (*h_lam_sq_ / 2 <= newton_tol_) //converged
            {
                //break;
            }
            else if (verbose && newton_it == newton_itmax_ - 1) //failed to converge
            {
                printf("        Newton failed to converge in %d iterations\n", newton_itmax_);
            }
            printf("\n");
        }
        deviceDotProduct(n, info.c_, out, (double*)d_obj_val_);
        cudaEventRecord(event_);
        cudaEventSynchronize(event_);
        //check convergence
        if (verbose) {
            printf("Objective value is: %.12f\n", *h_obj_val_);
        }
        if (n / t <= barrier_tol_) //converged
        {
            break;
        }
        t *= mu_; //increase t
    }

    deleteDescriptor(out_desc);
    deleteOnDevice(d_hessian_diag);
    deleteOnDevice(x_cg);
    deleteOnDevice(minus_gradient);

    return 0;
}


/**
 * @brief Set the barrier tolerance
*/
void LogBarrierSolver::set_barrier_tol(double barrier_tol)
{
    barrier_tol_ = barrier_tol;
}

/**
 * @brief Set the newton tolerance
*/
void LogBarrierSolver::set_newton_tol(double newton_tol)
{
    newton_tol_ = newton_tol;
}

/**
 * @brief Set the cg tolerance
*/
void LogBarrierSolver::set_cg_tol(double cg_tol)
{
    cg_tol_ = cg_tol;
}

/**
 * @brief Set mu, how fast t grows
*/
void LogBarrierSolver::set_mu(double mu)
{
    mu_ = mu;
}

/**
 * @brief Set the maximum number of newton iterations
*/
void LogBarrierSolver::set_newton_itmax(int newton_itmax)
{
    newton_itmax_ = newton_itmax;
}

/**
 * @brief Set the maximum number of barrier iterations
*/
void LogBarrierSolver::set_barrier_itmax(int barrier_itmax)
{
    barrier_itmax_ = barrier_itmax;
}

/**
 * @brief Set the maximum number of cg iterations
*/
void LogBarrierSolver::set_cg_itmax(int cg_itmax)
{
    cg_itmax_ = cg_itmax;
}

