#pragma once

#include <algorithm>
#include "matrix_vector_ops.hpp"
#include "matrix_vector_ops_cuda.hpp"
#include "cuda_memory_utils.hpp"
#include "permcheck_cuda.hpp"
#include "permcheck.hpp"
#include "cusparse_params.hpp"

enum Permutation_Type { perm_v, rev_perm_v, perm_h_v, perm_j_v, perm_jt_v }; 

class PermClass
{
public:
  // constructor
  PermClass(int n_h, int nnz_h, int nnz_j) 
    : n_h_(n_h), 
      nnz_h_(nnz_h), 
      nnz_j_(nnz_j)
  {
    allocate_workspace();
  }

  // destructor
  ~PermClass()
  {
    deleteOnDevice(d_perm_);
    deleteOnDevice(d_rev_perm_);
    deleteOnDevice(d_perm_map_h_);
    deleteOnDevice(d_perm_map_j_);
    deleteOnDevice(d_perm_map_jt_);
    
    if(perm_is_default_){
      delete [] perm_;
    }
    delete [] rev_perm_;
    delete [] perm_map_h_;
    delete [] perm_map_j_;
    delete [] perm_map_jt_;
  }

  /*
   * @brief loads CSR structure for matrix H
   *
   * @param h_i - Row offsets for H
   * h_j - Column indeces for H
   *
   * @pre
   * @post h_i_ set to h_i, h_j_ set to h_j
  */
  void add_H_info(int* h_i, int* h_j)
  {
    h_i_ = h_i;
    h_j_ = h_j;
  }

  /*
   * @brief loads CSR structure for matrix J
   *
   * @param j_i - Row offsets for J
   * j_j - Column indeces for j
   * n_j, m_j - dimensions of J
   *
   * @pre
   * @post j_i_ set to j_i, j_j_ set to j_j, n_j_ set to n_j, m_j_ set to m_j
  */
  void add_J_info(int* j_i, int* j_j, int n_j, int m_j)
  {
    j_i_ = j_i;
    j_j_ = j_j;
    n_j_ = n_j;
    m_j_ = m_j;
  }

  /*
   * @brief loads CSR structure for matrix Jt
   *
   * @param jt_i - Row offsets for Jt
   * jt_j - Column indeces for Jt
   *
   * @pre
   * @post jt_i_ set to jt_i, jt_j_ set to jt_j
  */
  void add_Jt_info(int* jt_i, int* jt_j)
  {
    jt_i_ = jt_i;
    jt_j_ = jt_j;
  }

  /*
   * @brief Implements Symmetric Approximate Minimum Degree 
   *        to reduce zero-fill in Cholesky Factorization
   *
   * @pre Member variables n_h_, nnz_h_, h_i_, h_j_ have been 
   *      initialized to the dimensions of matrix H, the number 
   *      of nonzeros it has, its row offsets, and column arrays
   *
   * @post perm is the perumation vector that implements symamd
   *       on the 2x2 system
  */
  void symamd()
  {
    cusolverSpHandle_t handle_cusolver = NULL;
    cusparseMatDescr_t descr_a = NULL;
    checkCudaErrors(cusparseCreateMatDescr(&descr_a));
    checkCudaErrors(cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(descr_a, INDEX_BASE));
    checkCudaErrors(cusolverSpCreate(&handle_cusolver));
    checkCudaErrors(cusolverSpXcsrsymamdHost(handle_cusolver, n_h_, nnz_h_, 
           descr_a, h_i_, h_j_, perm_));
    checkCudaErrors(cusolverSpDestroy(handle_cusolver));
    checkCudaErrors(cusparseDestroyMatDescr(descr_a));
    
    cloneVectorToDevice(n_h_, &perm_, &d_perm_); 
  }

  /*
   * @brief sets custom permutation of matrix
   *
   * @param custom_perm - custom permutation vector
   *
   * @pre Member variable n_h_ initialized to dimension of matrix
   *
   * @post perm points to custom_perm out of scope so perm_is_default
   *       set to false so that custom_perm not deleted twice in destructors,
   *       permutation vector copied onto device d_perm
  */
  void add_perm(int* custom_perm)
  {
    perm_is_default_ = 0;
    perm_ = custom_perm;
    cloneVectorToDevice(n_h_, &perm_, &d_perm_); 
  }

  /*
   * @brief copies reverse permutation of perm onto device
   *
   * @pre Member variables n_h_, perm intialized to dimension of matrix
   *      and to a permutation vector
   * 
   * @post rev_perm is now the reverse permuation of perm and copied onto
   *       the device d_perm
  */
  void invert_perm()
  {
    reverse_perm(n_h_, perm_, rev_perm_);
    cloneVectorToDevice(n_h_, &rev_perm_, &d_rev_perm_); 
  }

  /*
   * @brief copies permutation of rows and columns of matrix onto device
   *
   * @param b_i - row offsets of permutation
   * b_j - column indeces of permutation
   *
   * @pre Member variables n_h_, nnz_h_, h_i_, h_j_, perm, rev_perm
   *      initialized to the dimension of matrix H, number of nonzeros
   *      in H, row offsets for H, column indeces for H, permutation
   *      and reverse permutation of H
   * 
   * @post perm_map_h is now permuted rows/columns of H and copied onto
   *       the device d_perm_map_h
  */
  void vec_map_rc(int* b_i, int* b_j)
  {
    make_vec_map_rc(n_h_, h_i_, h_j_, perm_, rev_perm_, b_i, b_j, perm_map_h_);
    cloneVectorToDevice(nnz_h_, &perm_map_h_, &d_perm_map_h_); 
  }

  /*
   * @brief copies the permutation of the columns of matrix J onto device
   *
   * @param b_j - column indeces of permutation
   *
   * @pre Member variables n_j_, nnz_j_, j_i_, j_j_, rev_perm initialized
   *      to the dimension of matrix J, number of nonzeros in J, row
   *      offsets for J, column indeces for J, and reverse permutation
   * 
   * @post perm_map_j is now the column permutation and is copied onto
   *       the device d_perm_map_j
  */
  void vec_map_c(int* b_j)
  {
    make_vec_map_c(n_j_, j_i_, j_j_, rev_perm_, b_j, perm_map_j_);
    cloneVectorToDevice(nnz_j_, &perm_map_j_, &d_perm_map_j_); 
  }

  /*
   * @brief copies the permutation of the rows of matrix Jt onto device
   *
   * @param b_i - row offsets of permutation
   * b_j - column indeces of permutation
   *
   * @pre Member variables m_j_, nnz_j_, jt_i_, jt_j_, initialized to
   *      the dimension of matrix J, the number of nonzeros in J, the
   *      row offsets for J transform, the column indeces for J transform
   * 
   * @post perm_map_jt is now the permuations of the rows of J transform
   *       and is copied onto the device d_perm_map_jt
  */
  void vec_map_r(int* b_i, int* b_j)
  {
    make_vec_map_r(m_j_, jt_i_, jt_j_, perm_, b_i, b_j, perm_map_jt_);
    cloneVectorToDevice(nnz_j_, &perm_map_jt_, &d_perm_map_jt_); 
  }

  /*
   * @brief maps the permutated values of old_val to new_val
   *
   * @param permutation - the type of permutation of the 2x2 system
   * old_val - the old values in the matrix
   * new_val - the permuted values
   *
   * @pre Member variables n_h_, nnz_h_, nnz_j_, d_perm, d_rev_perm,
   *      d_perm_map_h, d_perm_map_j, d_perm_map_jt initialized to
   *      the dimension of matrix H, number of nonzeros in H, number
   *      of nonzeros in matrix J, the device permutation and reverse
   *      permutation vectors, the device permutation mappings for 
   *      H, J, and J transform
   * 
   * @post new_val contains the permuted old_val
  */
  void map_index(Permutation_Type permutation, double* old_val, double* new_val)
  {
    switch(permutation)
    {
      case perm_v: 
        fun_map_idx(n_h_, d_perm_, old_val, new_val);
        break;
      case rev_perm_v: 
        fun_map_idx(n_h_, d_rev_perm_, old_val, new_val);
        break;
      case perm_h_v: 
        fun_map_idx(nnz_h_, d_perm_map_h_, old_val, new_val);
        break;
      case perm_j_v: 
        fun_map_idx(nnz_j_, d_perm_map_j_, old_val, new_val);
        break;
      case perm_jt_v: 
        fun_map_idx(nnz_j_, d_perm_map_jt_, old_val, new_val);
        break;
      default:
        printf("Valid arguments are perm_v, rev_perm_v, perm_h_v, perm_j_v, perm_jt_v\n");
    }
  }

private:

/*
 * @brief allocates memory on host for permutation vectors
 *
 * @pre Member variables n_h_, nnz_h_, nnz_j_ are initialized to the
 *      dimension of matrix H, number of nonzeros in H, and number of
 *      nonzeros in matrix J
 * @post perm_ and rev_perm_ are now vectors with size n_h_, perm_map_h
 *       is now a vector with size nnz_h_, perm_map_j and perm_map_jt
 *       are now vectors with size nnz_j_
*/
  void allocate_workspace()
  {
    perm_ = new int[n_h_];
    rev_perm_ = new int[n_h_];
    perm_map_h_ = new int[nnz_h_];
    perm_map_j_ = new int[nnz_j_];
    perm_map_jt_ = new int[nnz_j_];
  }

  // member variables
  int perm_is_default_ = 0; // boolean if perm set custom
  
  int n_h_; // dimension of H
  int nnz_h_; // nonzeros of H

  int n_j_; // dimensions of J
  int m_j_;
  int nnz_j_; // nonzeros of J

  int* perm_; // permutation of 2x2 system
  int* rev_perm_; // reverse of permutation
  int* perm_map_h_; // mapping of permuted H
  int* perm_map_j_; // mapping of permuted J
  int* perm_map_jt_; // mapping of permuted Jt
  
  //device permutations
  int* d_perm_;
  int* d_rev_perm_;
  int* d_perm_map_h_;
  int* d_perm_map_j_;
  int* d_perm_map_jt_;
  
  int* h_i_; // row offsets of csr storage of H
  int* h_j_; // column pointers of csr storage of H

  int* j_i_; // row offsets of csr storage of J
  int* j_j_; // column pointers of csr storage of J

  int* jt_i_; // row offsets of csr storage of J transform
  int* jt_j_; // column pointers of csr storage of J transform 

  // right hand side of 2x2 system
  double* rhs1_; // first block in vector
  double* rhs2_; // second block in vector
};

