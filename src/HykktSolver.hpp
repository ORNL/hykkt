#pragma once

#include <algorithm>
#include <unistd.h>
#include <cstdlib>
#include <assert.h>
#include <ctype.h>
#include <string.h>
#include <iostream>
#include <memory>
#include <string>
#include "permcheck.hpp"
#include "input_functions.hpp"
#include "SchurComplementConjugateGradient.hpp"
#include "RuizClass.hpp"
#include "matrix_vector_ops_cuda.hpp"
#include "vector_vector_ops.hpp"
#include "SpgemmClass.hpp"
#include "MMatrix.hpp"
#include "CholeskyClass.hpp"
#include "PermClass.hpp"
#include "SpgemmClass.hpp"
#include "constants.hpp"
#include "cuda_memory_utils.hpp"

class HykktSolver
{
public:
  
  //constructor
  HykktSolver(double gamma);

  //destructor  
  ~HykktSolver();
  
  //methods
  
  /*
   * @brief loads KKT system into solver
   *
   * @param file names for different components of KKT system
   *        with same nonzero structure
   *
   * @post mat_h_, mat_ds_, mat_jc_, mat_jd_, rx_, rs_, ry_,
   *       ryd_ have new values for the system in a following
   *       solver iteration with same nonzero structure as
   *       previous iterations
  */
  void read_matrix_files(char const* const h_file,
      char const* const ds_file,
      char const* const jc_file,
      char const* const jd_file,
      char const* const rx_file,
      char const* const rs_file,
      char const* const ry_file,
      char const* const ryd_file,
      int skip_lines);

  /*
   * @brief sets gamma value for hykkt solver
   * 
   * @param gamma - new value for gamma_
   *
   * @post gamma_ is now equal to gamma
  */
  void set_gamma(double gamma);

  /*
   * @brief uses Hykkt algorithm to solve KKT system
   *
   * @pre matrix files have been loaded into the solver
   *
   * @post solution to given KKT system is computed using Hykkt
  */
  int execute();

private:
 
  /*
   * @brief creates handles and sets up matrices
   *
   * @post handle_cusparse_ is now a handle for the cuSPARSE
   *       library context and handle_cublas_ is now a handle
   *       for the cuBLAS library context
  */
  void allocate_workspace();
 
  /*
   * @brief computes Spgemm to calculate Htilda matrix
   *
   * @pre matrices and sc_til_ properly allocated using setup
   *      method for spgemm_htil
   *
   * @post Htilda calculated using JD matrix if JD nnz > 0 and
   *       is set to H if JD nnz == 0
  */
  void compute_spgemm_htil();
  
  /*
   * @brief computes ruiz scaling so we can judge the size of
   *        gamma and delta min relative to H gammma system
   *
   * @pre matrices and rz_ properly allocated using setup method
   *      for ruiz_scaling 
   *
   * @post max_d_ now contains the aggregated Ruiz scaling
  */
  void compute_ruiz_scaling();
  
  /*
   * @brief computes Spgemm to calculate Hgamma matrix
   *
   * @pre matrices and sc_gamma_ properly allocated using setup
   *      method for spgemm_hgamma
   *
   * @post Hgamma CSR now represent jc_t_desc_*jc_desc_ + htil
  */
  void compute_spgemm_hgamma();
  
  /*
   * @brief computes permutations for H, J, J transpose matrices
   *
   * @pre permutation matrix computed using setup method for
   *      permutation
   *
   * @post hgam_v_p_, jc_v_p_, jct_v_p_, d_rxp_ are now permuted
   *       values of hgam_v_, jc-v_, jc_t_v_, d_rx_hat_
  */
  void apply_permutation();
  
  /*
   * @brief sparse Cholesky factorization on permuted (1,1) block   *        so that LDLt does not have to be used
   *
   * @pre symbolic analysis already computed using setup method
   *      for hgamma_factorization
   *
   * @post hgamma inner factorization is computed, thus updating
   *       hgam_v_p_
  */
  void compute_hgamma_factorization();
  
  /*
   * @brief iterative solver on the Schur complement
   *
   * @pre matrices and sccg_ properly allocated using setup
   *      method for conjugate_gradient
   *
   * @post converged to approximate solution of block system
  */
  void compute_conjugate_gradient();
  
  /*
   * @brief recovers solution from hykkt solver
   *
   * @pre execute functions setup and computed correctly
   *
   * @post d_rx_, d_rs_, d_ryc_, and d_ryd_ contain the solution
           on the device
  */
  void recover_solution();
  
  /*
   * @brief calculates the error of Ax - b
   *
   * @pre solution properly recovered using recover_solution()
   * @return int - 0 if error small enough, 1 if hykkt failed
   *
   * @post solver status = success if error is smaller than
           optimization solver error
  */
  int check_error();
  
  /*
   * @brief allocates and initiates variables for KKT system
   *
   * @pre jd_flag_ determines if variables used for Spgemm Htil
          should be initiated
   *
   * @post all variables used for hykkt are allocated for; jd
           related variables are not initiated if JD nnz == 0
  */
  void setup_parameters();
  void setup_spgemm_htil();
  void setup_solution_check();
  void setup_ruiz_scaling();
  void setup_spgemm_hgamma();
  void setup_permutation();
  void setup_hgamma_factorization();
  void setup_conjugate_gradient();

  //constants
  const int ruiz_its_ = 2;
  const double norm_tol_ = 1e-2;
  const double tol_ = 1e-12;
  
  //gamma value used in HYKKT
  double gamma_;
  
  //booleans for reuse functionality
  bool allocated_ = false;
  bool jd_flag_ = false;
  bool jd_buffers_set_ = false;
  
  //status of if solver is correctly used with matrices of same
  //nonzero structure
  bool status = true;

  cusparseHandle_t handle_cusparse_; //handle to cuSPARSE library
  cublasHandle_t handle_cublas_; //handle to cuBLAS library
    
  //KKT system/solution matrix and vector descriptors
  cusparseSpMatDescr_t h_desc_;
  cusparseSpMatDescr_t jc_desc_;
  cusparseSpMatDescr_t jd_desc_;
  cusparseSpMatDescr_t jd_t_desc_;
  cusparseSpMatDescr_t jd_s_desc_;
  cusparseSpMatDescr_t jc_t_desc_;
  cusparseSpMatDescr_t jc_c_desc_;
  cusparseSpMatDescr_t jc_t_desc_c_;
  cusparseSpMatDescr_t jc_p_desc_;
  cusparseSpMatDescr_t jc_t_descp_;

  cusparseDnVecDescr_t vec_d_ryd_;
  cusparseDnVecDescr_t vec_d_rs_til_;
  cusparseDnVecDescr_t vec_d_ryd_s_;
  cusparseDnVecDescr_t vec_d_rx_til_;
  cusparseDnVecDescr_t vec_d_rx_hat_;
  cusparseDnVecDescr_t vec_d_ry_;
  cusparseDnVecDescr_t vec_d_schur_;
  cusparseDnVecDescr_t vec_d_hrxp_;
  cusparseDnVecDescr_t vec_d_y_;
  cusparseDnVecDescr_t vec_d_rxp_;
  cusparseDnVecDescr_t vec_d_x_;
  cusparseDnVecDescr_t vec_d_s_;
  cusparseDnVecDescr_t vec_d_rx_;
  cusparseDnVecDescr_t vec_d_yd_;
  cusparseDnVecDescr_t vec_d_ryc_;
  
  //2x2 system of matrices
  MMatrix mat_h_;
  MMatrix mat_ds_;
  MMatrix mat_jc_;
  MMatrix mat_jd_;

  //helper objects used for steps of algorithm
  RuizClass* rz_;
  SpgemmClass* sc_til_;
  SpgemmClass* sc_gamma_;
  PermClass* pc_;
  CholeskyClass* cc_;
  SchurComplementConjugateGradient* sccg_;

  //buffers used for matrix vector products
  void* buffer_htil_;
  void* buffer_hgam_;
  void* buffer_schur_;
  void* buffer_solve1_;
  void* buffer_solve2_;
  
  //buffers used for calculating error of solution
  void* buffer_error1_;
  void* buffer_error2_;
  void* buffer_error3_;
  void* buffer_error4_;

  //bufers used for transposing onto device
  void* buffer_trans1_;
  void* buffer_trans2_;
  
  double* max_d_; // ruiz scaling
 
  //host vectors 
  double* rx_;
  double* rs_;
  double* ry_;
  double* ryd_;

  //device vectors
  double* d_rx_;
  double* d_rs_;
  double* d_ry_;
  double* d_ryc_;
  double* d_ryd_;
  double* d_ryd_s_;
  double* h_v_;
  double* ds_v_;
  double* jc_v_;
  double* jd_v_;
  double* jd_vs_;

  //device vectors
  double* d_x_;
  double* d_s_;
  double* d_y_;
  double* d_yd_;
  double* d_rx_til_;
  double* d_rs_til_;
  double* d_rxp_;
  double* d_hrxp_;
  double* d_schur_;
  double* d_rx_hat_;
  double* d_z_;
  
  //CSR FORMAT POINTERS
  int* h_j_;
  int* h_i_;
  double* h_x_;
  double* h_s_;
  double* h_y_;
  double* h_yd_;

  int* jc_j_;
  int* jc_i_;
  int* jc_j_p_;
  int* jc_i_p_;
  double* jc_t_v_;
  int* jc_t_j_;
  int* jc_t_i_;
  int* jct_j_p_;
  int* jct_i_p_;

  double* htil_v_;
  int*    htil_j_;
  int*    htil_i_;
  int     nnz_htil_;

  int* jd_j_;
  int* jd_i_;
  double* jd_t_v_;
  int*    jd_t_j_;
  int*    jd_t_i_;

  // saves the original JC
  double* jc_v_c_;
  int* jc_i_c_;
  int* jc_j_c_;
  
  //saves the original JCt
  double* jct_v_c_;
  int*    jct_i_c_;
  int*    jct_j_c_;

  double* hgam_v_;
  int*    hgam_j_;
  int*    hgam_i_;
  int     nnz_hgam_;

  int* hgam_h_i_;
  int* hgam_h_j_;
  int* hgam_p_i_;
  int* hgam_p_j_;
  int* jc_p_j_;
  int* jct_p_j_;
  int* jct_p_i_;
  int* jct_j_;
  int* jct_i_;

  double* hgam_v_p_;
  int* hgam_i_p_;
  int* hgam_j_p_;
  double* jc_v_p_;
  double* jct_v_p_;
};
