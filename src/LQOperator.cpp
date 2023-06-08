#include "LQOperator.hpp"
#include "matrix_vector_ops.hpp"
#include "cuda_memory_utils.hpp"
#include "matrix_matrix_ops.hpp"
#include "chrono"
#include "sparse_mat_mul.hpp"

void updateTime(std::chrono::steady_clock::time_point& start, std::string name)
{
    auto end = std::chrono::steady_clock::now();
    printf("%s: %f\n", name.c_str(), std::chrono::duration<double>(end - start).count());
    start = end;
}

/**
 * @brief Constructor
 * 
 * @param info - info containing problem data for barrier method
 * h_v - diagonal values of H matrix
 * handle - cusparse handle
 * 
 * @pre info is loaded with problem data, h_v is on the device with size m
 * 
 * @post matrices are loaded
*/
LQOperator::LQOperator(LogBarrierInfo& info, double* h_v)
    : info_{info}, h_v_{h_v}
{
    set_Q_scalar(ONE);
    load_H_matrix(h_v);
    cudaEventCreate(&event_);
}

/**
 * @brief Destructor
*/
LQOperator::~LQOperator() 
{
    if (linear_allocated_) {
        //deleteOnDevice(buffer1_);
        //deleteOnDevice(buffer2_);
        deleteOnDevice(w_);
        deleteDescriptor(w_desc_);
        deleteDescriptor(v_desc_);
        deleteDescriptor(y_desc_);
    }
    if (quadratic_allocated_) {
        deleteOnDevice(r_);
        deleteDescriptor(r_desc_);
    }
    
    if (factorized_) {
        if (info_.quadratic_) {
            delete cholesky_;
        }
        deleteOnDevice(s_v_);
    }
}

/**
 * @brief applies the operatotr (tQ + A'HA) to v where Q is nxn, 
 *        A is mxn (m > n), H is diagonal mxm
 *
 * @param v - the vector to which the loaded operator is applied to
 * y - the result of applying the operator is copied into this vector
 * 
 * @pre v and out have size n
 * 
 * @post y holds the result of applying the operator to v
 */
void LQOperator::apply(double* v, double* y)
{   
    if (!linear_allocated_) {
        allocateVectorOnDevice(info_.m_, &w_);
        createDnVec(&w_desc_, info_.m_, w_);
        createDnVec(&v_desc_, info_.n_, v);
        createDnVec(&y_desc_, info_.n_, y);
    }
    else {
        cusparseDnVecSetValues(v_desc_, v);
        cusparseDnVecSetValues(y_desc_, y);
    }
    //fun_csr_spmv_kernel(info_.m_, info_.n_, info_.a_j_, info_.a_i_, info_.a_v_, v, w_);
    SpMV_product_reuse(info_.cusparse_handle_, ONE, info_.a_desc_, v_desc_, ZERO, w_desc_, &buffer1_, linear_allocated_); //Av
    fun_vec_scale(info_.m_, w_, h_v_); //HAv 

    //fun_csr_spmv_kernel(info_.n_, info_.m_, info_.a_t_j_, info_.a_t_i_, info_.a_t_v_, w_, y); //A'HAv
    SpMV_product_reuse(info_.cusparse_handle_, ONE, info_.a_t_desc_, w_desc_, ZERO, y_desc_, &buffer2_, linear_allocated_); //AtHAv

    linear_allocated_ = true;

    if (info_.quadratic_) {
        if (!quadratic_allocated_) {
            allocateVectorOnDevice(info_.n_, &r_);
            createDnVec(&r_desc_, info_.n_, r_);
            quadratic_allocated_ = true;
        }
        //fun_csr_spmv_kernel(info_.n_, info_.m_, info_.q_j_, info_.q_i_, info_.q_v_, v, r_); //Qv
        SpMV_product_reuse(info_.cusparse_handle_, ONE, info_.q_desc_, v_desc_, ZERO, r_desc_, &buffer2_, linear_allocated_); //Qv
        fun_add_vecs(info_.n_, y, t_, r_); //tQv + A'HAv
    }
}

/**
 * @brief extracts the sparse structure of the operator Q + A'HA corresponding to Q nonzero sparsity
 * 
 * @param out - the resulting nonzero values corresponding to CSR of sparse structure of Q + A'HA
 * 
 * @pre out has size q_nnz
 * 
 * @post out is filled with values of Q + A'HA corresponding to nonzero Q entries
*/
void LQOperator::extract_sparse_structure(double* out) {
    fun_q_sparse_product(info_.n_, info_.q_nnz_, info_.q_i_, info_.q_j_, info_.q_v_, info_.a_nnz_, info_.a_t_i_, info_.a_t_j_, info_.a_t_v_, h_v_, out);
}

/**
 * @brief extracts the inverse of the diagonal of the operator A'HA
 * 
 * @param out - the vector where the resulting extracted inverse diagonal is stored
 * 
 * @pre out has size n
 * 
 * @post out is filled with values corresponding to the inverse of the diagonal elements of A'HA
*/
void LQOperator::extract_inv_linear_diagonal(double* out) {
    fun_inv_diagonal_product(info_.n_, info_.a_t_i_, info_.a_t_j_, info_.a_t_v_, h_v_, out);
}

/**
 * @brief factorizes the sparsified operator S
 * 
 * @pre all matrices are loaded
 * 
 * @post cholesky_ object has factorization of S if quadratic, else S is inverse diagonal of A'HA
*/
void LQOperator::factorize() {
    if (!factorized_) {
        if (info_.quadratic_) {
            allocateVectorOnDevice(info_.q_nnz_, &s_v_);
            cholesky_ = new CholeskyClass(info_.n_, info_.q_nnz_, s_v_, info_.q_i_, info_.q_j_);
            cholesky_->set_pivot_tolerance(tol_);
            cholesky_->symbolic_analysis();
        }
        else {
            allocateVectorOnDevice(info_.n_, &s_v_);
        }
    }
    if (params_updated_) {
        if (info_.quadratic_) {
            extract_sparse_structure(s_v_);
            cholesky_->set_matrix_values(s_v_);
            cholesky_->numerical_factorization();
        }
        else {
            extract_inv_linear_diagonal(s_v_);
        }
        updateParams(false);
    }
    factorized_ = true;
}

/**
 * @brief Solves Sx = b and stores x in out
 * 
 * @param b - the input array to which the operator is applied
 * out - the output array to which x is stored in
 * 
 * @pre b and out have size n, on the device
 * 
 * @post x is solved using preconditioned conjugate gradient and stored in out
*/
void LQOperator::preconditioner_solve(double* b, double* out)
{
    factorize();

    if (info_.quadratic_) {
        cholesky_->solve(out, b);
    }
    else {
        fun_vec_scale(info_.n_, b, s_v_, out);
    }
}

/**
 * @brief loads CSR information for H matrix which is a diagonal matrix
 *
 * @param h_v - vector representing the diagonal of H
 * 
 * @pre h_v has size m and is on the device
 * 
 * @post Diagonal for H matrix is loaded
 */
void LQOperator::load_H_matrix(double* h_v)
{
    h_v_ = h_v;
    updateParams(true);
}

/**
 * @brief sets scalar that multiplies Q
 * 
 * @param t - scalar
 * 
 * @pre t is some double value
 * 
 * @post t_ value is set to t
*/
void LQOperator::set_Q_scalar(double t) {
    t_ = t;
    if (info_.quadratic_) {
        updateParams(true);
    }
}

/**
 * @brief Updates the value of params_updated_.
 *
 * @param params_updated the new value of params_updated_
 *
 * @post params_updated_ is updated
 */
void LQOperator::updateParams(bool params_updated) {
    params_updated_ = params_updated;
}

/**
 * @brief returns the number of rows of the operator
*/
int LQOperator::get_operator_size() const
{
    return info_.n_;
}