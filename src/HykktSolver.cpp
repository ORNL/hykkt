#include <cassert>

#include <cusparse.h>
#include <cublas.h>

#include "HykktSolver.hpp"

#include "input_functions.hpp"
#include "SchurComplementConjugateGradient.hpp"
#include "RuizClass.hpp"
#include "vector_vector_ops.hpp"
#include "SpgemmClass.hpp"
#include "CholeskyClass.hpp"
#include "PermClass.hpp"
#include "SpgemmClass.hpp"
#include "matrix_vector_ops.hpp"

  HykktSolver::HykktSolver(double gamma)
    : gamma_(gamma),
      rx_(nullptr),
      rs_(nullptr),
      ry_(nullptr),
      ryd_(nullptr),
      d_x_(nullptr),
      d_s_(nullptr),
      d_y_(nullptr),
      d_yd_(nullptr),
      h_x_(nullptr),
      h_s_(nullptr),
      h_y_(nullptr),
      h_yd_(nullptr)
  {
    allocate_workspace();
  }
  
  HykktSolver::~HykktSolver()
  {
    deleteMatrixOnDevice(htil_i_, htil_j_, htil_v_);
    deleteMatrixOnDevice(hgam_i_, hgam_j_, hgam_v_);
    deleteMatrixOnDevice(h_i_, h_j_, h_v_);
    deleteMatrixOnDevice(jc_i_, jc_j_, jc_v_);
    deleteMatrixOnDevice(jc_t_i_, jc_t_j_, jc_t_v_);
    deleteMatrixOnDevice(jc_i_c_, jc_j_c_, jc_v_c_);
    deleteMatrixOnDevice(jct_i_c_, jct_j_c_, jct_v_c_);
    deleteMatrixOnDevice(jc_i_p_, jc_j_p_, jc_v_p_);
    deleteMatrixOnDevice(jct_i_p_, jct_j_p_, jct_v_p_);
    deleteMatrixOnDevice(jd_i_, jd_j_, jd_v_);
    deleteMatrixOnDevice(jd_t_i_, jd_t_j_, jd_t_v_);
    deleteMatrixOnDevice(hgam_i_p_, hgam_j_p_, hgam_v_p_);

    deleteOnDevice(d_x_);
    deleteOnDevice(d_s_);
    deleteOnDevice(d_y_);
    deleteOnDevice(d_yd_);
    deleteOnDevice(d_z_);
    deleteOnDevice(d_rx_);
    deleteOnDevice(d_rxp_);
    deleteOnDevice(d_hrxp_);
    deleteOnDevice(d_schur_);
    deleteOnDevice(d_rs_);
    deleteOnDevice(d_ry_);
    deleteOnDevice(d_ryc_);
    deleteOnDevice(d_ryd_);
    deleteOnDevice(d_ryd_s_);
    deleteOnDevice(d_rx_til_);
    deleteOnDevice(d_rx_hat_);
    deleteOnDevice(d_rs_til_);
    deleteOnDevice(ds_v_);
    deleteOnDevice(jd_vs_);

    deleteOnDevice(buffer_htil_);
    deleteOnDevice(buffer_hgam_);
    deleteOnDevice(buffer_schur_);
    deleteOnDevice(buffer_solve1_);
    deleteOnDevice(buffer_solve2_);
    deleteOnDevice(buffer_trans1_);
    deleteOnDevice(buffer_trans2_);

    // Uncommenting lines below causes double free segfault
    //delete [] rx_;
    //delete [] rs_;
    //delete [] ry_;
    // delete [] ryd_;

    // delete [] h_x_;
    // delete [] h_s_;
    // delete [] h_y_;
    // delete [] h_yd_;
    delete [] hgam_h_i_;
    delete [] hgam_h_j_;
    delete [] hgam_p_i_;
    delete [] hgam_p_j_;
    delete [] jc_p_j_;
    delete [] jct_p_j_;
    delete [] jct_i_;
    delete [] jct_j_;
    delete [] jct_p_i_;
    
    delete rz_;
    delete sc_til_;
    delete sc_gamma_;
    delete pc_;
    delete cc_;
    delete sccg_; 
  }
  
  void HykktSolver::allocate_workspace()
  {
    createSparseHandle(handle_cusparse_);
    createCublasHandle(handle_cublas_);
  }
  
  void HykktSolver::read_matrix_files(char const* const h_file,
      char const* const ds_file,
      char const* const jc_file,
      char const* const jd_file,
      char const* const rx_file,
      char const* const rs_file,
      char const* const ry_file,
      char const* const ryd_file,
      int skip_lines)
  {
    // read matrices
    read_mm_file_into_coo(h_file, mat_h_, skip_lines);
    sym_coo_to_csr(mat_h_);
	mat_h_.owns_csr_data(true);

    read_mm_file_into_coo(ds_file, mat_ds_, skip_lines);
    coo_to_csr(mat_ds_);
    mat_ds_.owns_csr_data(true);

    read_mm_file_into_coo(jc_file, mat_jc_, skip_lines);
    coo_to_csr(mat_jc_);
    mat_jc_.owns_csr_data(true);

    read_mm_file_into_coo(jd_file, mat_jd_, skip_lines);
    coo_to_csr(mat_jd_);
    mat_jd_.owns_csr_data(true);
    
    bool jd_flag = mat_jd_.nnz_ > 0;
    
    status_ = (jd_flag_ == jd_flag); //if new nonzero structure then broken
      
    jd_flag_ = jd_flag;
    // read right hand side
    if(!allocated_){
      rx_ = new double[mat_h_.n_];
      rs_ = new double[mat_ds_.n_];
      ry_ = new double[mat_jc_.n_];
      ryd_ = new double[mat_jd_.n_];
      h_x_ = new double[mat_h_.n_]{0.0};
      h_s_ = new double[mat_ds_.n_]{0.0};
      h_y_ = new double[mat_jc_.n_]{0.0};
      h_yd_ = new double[mat_jd_.n_]{0.0};
    }
    
    read_rhs(rx_file, rx_);
    read_rhs(rs_file, rs_);
    read_rhs(ry_file, ry_);
    read_rhs(ryd_file, ryd_);
  }

  void HykktSolver::set_H(int* rowptrs, int* colidx, double* vals, int n, int m, int nnz)
  {
    assert(m == n);
	mat_h_.owns_csr_data(false);
    set_matrix_block(mat_h_, rowptrs, colidx, vals, n, m, nnz);
  }

  void HykktSolver::set_Ds(int* rowptrs, int* colidx, double* vals, int n, int m, int nnz)
  {
    assert(n == m);
	mat_ds_.owns_csr_data(false);
    set_matrix_block(mat_ds_, rowptrs, colidx, vals, n, m, nnz);
  }

  void HykktSolver::set_Jc(int* rowptrs, int* colidx, double* vals, int n, int m, int nnz)
  {
    mat_jc_.owns_csr_data(false);
    set_matrix_block(mat_jc_, rowptrs, colidx, vals, n, m, nnz);
  }

  void HykktSolver::set_Jd(int* rowptrs, int* colidx, double* vals, int n, int m, int nnz)
  {
    mat_jd_.owns_csr_data(false);
    set_matrix_block(mat_jd_, rowptrs, colidx, vals, n, m, nnz);

    bool jd_flag = mat_jd_.nnz_ > 0;
    // status_ = (jd_flag_ == jd_flag);
    status_ = true; // when using API, we can't check if sparsity pattern changed
    jd_flag_ = jd_flag;
  }

  void HykktSolver::set_rx(double*  vals, int n)
  {
    set_vector_block(&rx_, vals, n);
  }

  void HykktSolver::set_rs(double*  vals, int n)
  {
    set_vector_block(&rs_, vals, n);
  }

  void HykktSolver::set_ry(double*  vals, int n)
  {
    set_vector_block(&ry_, vals, n);
  }

  void HykktSolver::set_ryd(double*  vals, int n)
  {
    set_vector_block(&ryd_, vals, n);
  }

  void HykktSolver::set_x_host(double*  vals, int n)
  {
    set_vector_block(&h_x_, vals, n);
  }

  void HykktSolver::set_s_host(double*  vals, int n)
  {
    set_vector_block(&h_s_, vals, n);
  }

  void HykktSolver::set_y_host(double*  vals, int n)
  {
    set_vector_block(&h_y_, vals, n);
  }

  void HykktSolver::set_yd_host(double*  vals, int n)
  {
    set_vector_block(&h_yd_, vals, n);
  }


  void HykktSolver::set_gamma(double gamma)
  {
    gamma_ = gamma;
  }

  int HykktSolver::execute()
  {
    // TODO: Review sparsity pattern checking in HyKKT
    if(!status_ && allocated_){
      printf("\n\nERROR: USING HYKKT WITH NEW NONZERO STRUCTURE\n\n");
      std::cout << "status = "      << status_ 
                << ", allocated = " << allocated_
                << ", Jd flag = "   << jd_flag_
                << "\n";
      return 1;
    }

    setup_parameters();
    
    if(!allocated_){
      setup_spgemm_htil();
    }
    compute_spgemm_htil();
    
    setup_solution_check();

    if(!allocated_){
      setup_ruiz_scaling();
    }
    compute_ruiz_scaling();

    if(!allocated_){
      setup_spgemm_hgamma();
    }
    compute_spgemm_hgamma();

    if(!allocated_){
      setup_permutation();
    }
    apply_permutation();

    if(!allocated_){
      setup_hgamma_factorization();
    }
    compute_hgamma_factorization();
  
    setup_conjugate_gradient();
    compute_conjugate_gradient();

    recover_solution();
    return check_error();
  }

  // Computes $H_\tilde$ in eq (4)
  void HykktSolver::compute_spgemm_htil()
  {
    if(jd_flag_){
      transposeMatrixOnDevice(handle_cusparse_,
          mat_jd_.n_,
          mat_jd_.m_,
          mat_jd_.nnz_,
          jd_i_,
          jd_j_,
          jd_v_,
          jd_t_i_,
          jd_t_j_,
          jd_t_v_,
          &buffer_trans1_,
          jd_buffers_set_);  
      
      fun_row_scale(mat_jd_.n_,
          jd_v_,
          jd_i_,
          jd_j_,
          jd_vs_,
          d_ryd_,
          d_ryd_s_,
          ds_v_);
      
      fun_add_vecs(mat_jd_.n_, d_ryd_s_, ONE, d_rs_);
      SpMV_product_reuse(handle_cusparse_,
          ONE,
          jd_t_desc_,
          vec_d_ryd_s_,
          ONE,
          vec_d_rx_til_,
          &buffer_htil_,
          jd_buffers_set_);
      
      sc_til_->spGEMM_reuse();
    } else{
      matrixDeviceToDeviceCopy(mat_h_.n_,
          mat_h_.nnz_,
          h_i_,
          h_j_,
          h_v_,
          htil_i_,
          htil_j_,
          htil_v_);

      nnz_htil_ = mat_h_.nnz_;
    }
  }
 
  // Ruiz scaling of eq (4)
  void HykktSolver::compute_ruiz_scaling()
  {
    rz_->add_block11(htil_v_, htil_i_, htil_j_);
    rz_->add_block12(jc_t_v_, jc_t_i_, jc_t_j_);
    rz_->add_block21(jc_v_, jc_i_, jc_j_);
    rz_->add_rhs1(d_rx_til_);
    rz_->add_rhs2(d_ry_);
    rz_->ruiz_scale();
    max_d_ = rz_->get_max_d();
  }

  // Sets up $H_\gamma$ of eq (6)
  void HykktSolver::setup_spgemm_hgamma()
  {
    sc_gamma_ = new SpgemmClass(mat_h_.n_,
        mat_h_.n_,
        handle_cusparse_,
        gamma_,
        ONE,
        ONE);

    sc_gamma_->load_product_matrices(jc_t_desc_, jc_desc_);
    sc_gamma_->load_sum_matrices(htil_i_,
        htil_j_,
        htil_v_,
        nnz_htil_);
    sc_gamma_->load_result_matrix(&hgam_i_, &hgam_j_, &hgam_v_, &nnz_hgam_); 
  }

  // Computes $H_\gamma$ of eq (6)
  void HykktSolver::compute_spgemm_hgamma()
  {
    sc_gamma_->spGEMM_reuse();
    
    copyDeviceVector(mat_h_.n_, d_rx_til_, d_rx_hat_);
    SpMV_product_reuse(handle_cusparse_,
        gamma_,
        jc_t_desc_,
        vec_d_ry_,
        ONE,
        vec_d_rx_hat_,
        &buffer_hgam_,
        allocated_);
  }
  
  // Applies permutation to $H_\gamma$ of eq (6)
  void HykktSolver::apply_permutation()
  {
    pc_->map_index(perm_h_v, hgam_v_, hgam_v_p_);
    pc_->map_index(perm_j_v, jc_v_, jc_v_p_);
    pc_->map_index(perm_jt_v, jc_t_v_, jct_v_p_);

    fun_add_diag(mat_h_.n_, ZERO, hgam_i_p_, hgam_j_p_, hgam_v_p_);
   
    pc_->map_index(perm_v, d_rx_hat_, d_rxp_);
  }
  
  // Factorization of $H_\gamma$ of eq (6)
  void HykktSolver::compute_hgamma_factorization()
  {
    cc_->numerical_factorization();
  }
  
  // Computes conjugate gradient using $S$ of eq (7)
  void HykktSolver::compute_conjugate_gradient()
  {
    sccg_->solve();
  }
  
  // Using the solution to eq (7) recover the solution to (3)
  void HykktSolver::recover_solution()
  {
    // block-recovering the solution to the original system by parts
    // this part is to recover delta_x
    SpMV_product_reuse(handle_cusparse_,
        MINUS_ONE,
        jc_t_descp_,
        vec_d_y_,
        ONE,
        vec_d_rxp_,
        &buffer_solve1_,
        allocated_);
    
    cc_->solve(d_z_, d_rxp_);
    pc_->map_index(rev_perm_v, d_z_, d_x_);
    // scale back delta_y and delta_x (every iteration)
    fun_vec_scale(mat_h_.n_, d_x_, max_d_);
    fun_vec_scale(mat_jc_.n_, d_y_, &max_d_[mat_h_.n_]);
   
    copyDeviceVector(mat_ds_.m_, d_ryd_, d_s_);
    if(jd_flag_)
    {
      SpMV_product_reuse(handle_cusparse_,
                         ONE,
                         jd_desc_,
                         vec_d_x_,
                         MINUS_ONE,
                         vec_d_s_,
                         &buffer_solve2_,
                         jd_buffers_set_);
    }
    else
    { //Math ops - happens every iteration
      fun_mult_const(mat_ds_.n_, MINUS_ONE, d_s_);
    }
   
    copyDeviceVector(mat_ds_.m_, d_s_, d_yd_);
    fun_vec_scale(mat_ds_.n_, d_yd_, ds_v_);
    fun_add_vecs(mat_ds_.n_, d_yd_, MINUS_ONE, d_rs_);

    // Copy solution back to host
#if 0
    copyVectorToHost(mat_h_.n_,  d_x_,  h_x_);
    copyVectorToHost(mat_ds_.n_, d_s_,  h_s_);
    copyVectorToHost(mat_jc_.n_, d_y_,  h_y_);
    copyVectorToHost(mat_jd_.n_, d_yd_, h_yd_);
#endif
  }

  void HykktSolver::setup_parameters()
  {
    std::cout << "Matrix nnz = " << mat_ds_.nnz_ << "\n";
    if(!allocated_){
      // h_x_ = new double[mat_h_.m_]{0.0};
      // h_s_ = new double[mat_ds_.m_]{0.0};
      // h_y_ = new double[mat_jc_.n_]{0.0};
      // h_yd_ = new double[mat_jd_.n_]{0.0};
      hgam_h_i_ = new int[mat_h_.n_ + 1];
      hgam_p_i_ = new int[mat_h_.n_ + 1];
      jc_p_j_ = new int[mat_jc_.nnz_];
      jct_p_j_ = new int[mat_jc_.nnz_];
      jct_p_i_ = new int[mat_jc_.m_ + 1];
      jct_j_ = new int [mat_jc_.nnz_];
      jct_i_ = new int [mat_jc_.m_ + 1];
      
      allocateVectorOnDevice(mat_h_.n_, &d_rx_);
      allocateVectorOnDevice(mat_ds_.n_, &d_rs_);
      allocateVectorOnDevice(mat_jc_.n_, &d_ry_);
      allocateVectorOnDevice(mat_jd_.n_, &d_ryd_);
      allocateVectorOnDevice(mat_jd_.n_, &d_ryd_s_);
      allocateVectorOnDevice(mat_jc_.n_, &d_ryc_);
      allocateVectorOnDevice(mat_ds_.nnz_, &ds_v_);
      allocateVectorOnDevice(mat_jd_.nnz_, &jd_vs_);
      allocateVectorOnDevice(mat_h_.m_, &d_x_);
      allocateVectorOnDevice(mat_ds_.m_, &d_s_);
      allocateVectorOnDevice(mat_jc_.n_, &d_y_);
      allocateVectorOnDevice(mat_jd_.n_, &d_yd_);
      allocateVectorOnDevice(mat_h_.n_, &d_rx_til_);
      allocateVectorOnDevice(mat_ds_.n_, &d_rs_til_);
      allocateVectorOnDevice(mat_jd_.n_, &d_ryd_s_);
      allocateVectorOnDevice(mat_h_.n_, &d_rx_hat_);
      allocateVectorOnDevice(mat_h_.n_, &d_rxp_);
      allocateVectorOnDevice(mat_h_.n_, &d_hrxp_);
      allocateVectorOnDevice(mat_jc_.n_, &d_schur_);
      allocateVectorOnDevice(mat_h_.n_, &d_z_);
      
      allocateMatrixOnDevice(mat_h_.n_,
          mat_h_.nnz_,
          &h_i_,
          &h_j_,
          &h_v_);
    
      allocateMatrixOnDevice(mat_jd_.n_,
          mat_jd_.nnz_,
          &jd_i_,
          &jd_j_,
          &jd_v_);
      allocateMatrixOnDevice(mat_jc_.n_,
          mat_jc_.nnz_,
          &jc_i_,
          &jc_j_,
          &jc_v_);
    
      allocateMatrixOnDevice(mat_jc_.m_,
          mat_jc_.nnz_,
          &jc_t_i_,
          &jc_t_j_,
          &jc_t_v_);
      
      allocateMatrixOnDevice(mat_jd_.m_,
          mat_jd_.nnz_,
          &jd_t_i_,
          &jd_t_j_,
          &jd_t_v_);
      
      allocateMatrixOnDevice(mat_jc_.n_,
          mat_jc_.nnz_,
          &jc_i_p_,
          &jc_j_p_,
          &jc_v_p_);
    
      allocateMatrixOnDevice(mat_jc_.m_,
          mat_jc_.nnz_,
          &jct_i_p_,
          &jct_j_p_,
          &jct_v_p_);
      
      allocateMatrixOnDevice(mat_h_.n_,
        mat_h_.nnz_,
        &htil_i_,
        &htil_j_,
        &htil_v_);

      createDnVec(&vec_d_ryd_, mat_jd_.n_, d_ryd_);
      createDnVec(&vec_d_rs_til_, mat_ds_.n_, d_rs_til_);
      createDnVec(&vec_d_ryd_s_, mat_jd_.n_, d_ryd_s_);
      createDnVec(&vec_d_rx_til_, mat_h_.n_, d_rx_til_);
      createDnVec(&vec_d_rx_hat_, mat_h_.n_, d_rx_hat_);
      createDnVec(&vec_d_ry_, mat_jc_.n_, d_ry_);
      createDnVec(&vec_d_schur_, mat_jc_.n_, d_schur_);
      createDnVec(&vec_d_hrxp_, mat_h_.n_, d_hrxp_);
      createDnVec(&vec_d_y_, mat_jc_.n_, d_y_);
      createDnVec(&vec_d_rxp_, mat_h_.n_, d_rxp_);
      createDnVec(&vec_d_x_, mat_h_.n_, d_x_);
      createDnVec(&vec_d_s_, mat_ds_.n_, d_s_);
      createDnVec(&vec_d_rx_, mat_h_.n_, d_rx_);
      createDnVec(&vec_d_yd_, mat_jd_.n_, d_yd_);
      createDnVec(&vec_d_ryc_, mat_jc_.n_, d_ryc_);
      
      createCsrMat(&jc_t_desc_,
          mat_jc_.m_,
          mat_jc_.n_,
          mat_jc_.nnz_,
          jc_t_i_,
          jc_t_j_,
          jc_t_v_);
    
      createCsrMat(&h_desc_,
          mat_h_.n_,
          mat_h_.m_,
          mat_h_.nnz_,
          h_i_,
          h_j_,
          h_v_);
    
      createCsrMat(&jc_desc_,
          mat_jc_.n_,
          mat_jc_.m_,
          mat_jc_.nnz_,
          jc_i_,
          jc_j_,
          jc_v_);
      
      createCsrMat(&jd_desc_,
          mat_jd_.n_,
          mat_jd_.m_,
          mat_jd_.nnz_,
          jd_i_,
          jd_j_,
          jd_v_);

      createCsrMat(&jd_t_desc_,
          mat_jd_.m_,
          mat_jd_.n_,
          mat_jd_.nnz_,
          jd_t_i_,
          jd_t_j_,
          jd_t_v_);
      
      createCsrMat(&jd_s_desc_,
          mat_jd_.n_,
          mat_jd_.m_,
          mat_jd_.nnz_,
          jd_i_,
          jd_j_,
          jd_vs_);
    
      createCsrMat(&jc_p_desc_,
          mat_jc_.n_,
          mat_jc_.m_,
          mat_jc_.nnz_,
          jc_i_p_,
          jc_j_p_,
          jc_v_p_);
    
      createCsrMat(&jc_t_descp_,
          mat_jc_.m_,
          mat_jc_.n_,
          mat_jc_.nnz_,
          jct_i_p_,
          jct_j_p_,
          jct_v_p_);
      std::cout << "H size: " << mat_h_.n_ << " "<< mat_h_.m_ << " "<< mat_h_.nnz_ << " \n";
      copySymmetricMatrixToDevice(&mat_h_, 
          h_i_, 
          h_j_, 
          h_v_);
      std::cout << "Jc size: " << mat_jc_.n_ << "  " << mat_jc_.m_ << "  "<< mat_jc_.nnz_ << " \n";
      // mat_jc_.print_csr();
      copyMatrixToDevice(&mat_jc_,
          jc_i_,
          jc_j_,
          jc_v_);
    } else{
    //nonzero structure stays same so only update values each it
      copyVectorToDevice(mat_h_.nnz_, mat_h_.csr_vals, h_v_);
      copyVectorToDevice(mat_jc_.nnz_, mat_jc_.csr_vals, jc_v_);
    }
    
    if(jd_flag_){
      if(!jd_buffers_set_){
      std::cout << "Jd size: " << mat_jd_.n_ << "  " << mat_jd_.m_ << "  "<< mat_jd_.nnz_ << " \n";
      copyMatrixToDevice(&mat_jd_,
          jd_i_,
          jd_j_,
          jd_v_);

      }
      else
      {
        //nonzero structure stays same so only update values each it
        copyVectorToDevice(mat_jd_.nnz_, mat_jd_.csr_vals, jd_v_);
      }
    }
    // for(int i=0; i<mat_h_.n_; ++i) std::cout << rx_[i] << " ";
    // std::cout << "rx size: " << mat_ds_.n_ << " " << rx_[2] << "\n";
    copyVectorToDevice(mat_h_.n_, rx_, d_rx_);
    copyVectorToDevice(mat_ds_.n_, rs_, d_rs_);
    copyVectorToDevice(mat_jc_.n_, ry_, d_ry_);
    copyVectorToDevice(mat_jd_.n_, ryd_, d_ryd_);
    copyVectorToDevice(mat_ds_.nnz_, mat_ds_.csr_vals, ds_v_);
    copyVectorToDevice(mat_h_.m_, h_x_, d_x_);
    copyVectorToDevice(mat_ds_.m_, h_s_, d_s_);
    copyVectorToDevice(mat_jc_.n_, h_y_, d_y_);
    copyVectorToDevice(mat_jd_.n_, h_yd_, d_yd_);
    copyVectorToDevice(mat_h_.n_, d_rx_, d_rx_til_);
    copyVectorToDevice(mat_ds_.n_, d_rs_, d_rs_til_);
    
    copyDeviceVector(mat_jc_.n_, d_ry_, d_ryc_);
    
    transposeMatrixOnDevice(handle_cusparse_,
        mat_jc_.n_,
        mat_jc_.m_,
        mat_jc_.nnz_,
        jc_i_,
        jc_j_,
        jc_v_,
        jc_t_i_,
        jc_t_j_,
        jc_t_v_,
        &buffer_trans2_,
        allocated_);
  }

  // Sets up matrix products used in eqs (4) and (6)
  void HykktSolver::setup_spgemm_htil()
  {
    sc_til_ = new SpgemmClass(mat_h_.n_,
        mat_h_.n_,
        handle_cusparse_,
        ONE,
        ONE,
        ONE);
    
    sc_til_->load_product_matrices(jd_t_desc_, jd_s_desc_);
    sc_til_->load_sum_matrices(h_i_, h_j_, h_v_, mat_h_.nnz_);
    sc_til_->load_result_matrix(&htil_i_, &htil_j_, &htil_v_, &nnz_htil_);  
  }

  // Set up to check the residual of eq (3) to make sure it's small
  void HykktSolver::setup_solution_check()
  {
    if(!allocated_){
      allocateMatrixOnDevice(mat_jc_.n_,
          mat_jc_.nnz_,
          &jc_i_c_,
          &jc_j_c_,
          &jc_v_c_);
    
      allocateMatrixOnDevice(mat_jc_.m_,
          mat_jc_.nnz_,
          &jct_i_c_,
          &jct_j_c_,
          &jct_v_c_);
    
      createCsrMat(&jc_c_desc_,
          mat_jc_.n_,
          mat_jc_.m_,
          mat_jc_.nnz_,
          jc_i_c_,
          jc_j_c_,
          jc_v_c_);
    
      createCsrMat(&jc_t_desc_c_,
          mat_jc_.m_,
          mat_jc_.n_,
          mat_jc_.nnz_,
          jct_i_c_,
          jct_j_c_,
          jct_v_c_);
    }

    matrixDeviceToDeviceCopy(mat_jc_.n_,
        mat_jc_.nnz_,
        jc_i_,
        jc_j_,
        jc_v_,
        jc_i_c_,
        jc_j_c_,
        jc_v_c_);

    matrixDeviceToDeviceCopy(mat_jc_.m_,
        mat_jc_.nnz_,
        jc_t_i_,
        jc_t_j_,
        jc_t_v_,
        jct_i_c_,
        jct_j_c_,
        jct_v_c_);
  }

  //Sets up the Ruiz scaling of (4)
  void HykktSolver::setup_ruiz_scaling()
  {
    int n_hj = mat_h_.n_ + mat_jc_.n_;
    allocateVectorOnDevice(n_hj, &max_d_);
    rz_ = new RuizClass(ruiz_its_, mat_h_.n_, n_hj); 
  }

  // Sets up permutation of $H_\gamma$ of eq (6)
  void HykktSolver::setup_permutation()
  {
    hgam_h_j_ = new int[nnz_hgam_];
    hgam_p_j_ = new int[nnz_hgam_];
    
    allocateVectorOnDevice(nnz_hgam_, &hgam_v_p_);
    allocateMatrixOnDevice(mat_h_.n_,
        nnz_hgam_,
        &hgam_i_p_,
        &hgam_j_p_,
        &hgam_v_p_);

    copyVectorToHost(mat_h_.n_ + 1, hgam_i_, hgam_h_i_);
    copyVectorToHost(nnz_hgam_, hgam_j_, hgam_h_j_);
    pc_ = new PermClass(mat_h_.n_, nnz_hgam_, mat_jc_.nnz_);
    pc_->add_h_info(hgam_h_i_, hgam_h_j_);
    pc_->add_j_info(mat_jc_.csr_rows, 
        mat_jc_.csr_cols, 
        mat_jc_.n_, 
        mat_jc_.m_); 
      
    pc_->add_jt_info(jct_i_, jct_j_);  
    pc_->symamd(); 
    pc_->invert_perm();
    pc_->vec_map_rc(hgam_p_i_, hgam_p_j_);
    pc_->vec_map_c(jc_p_j_);
   
    copyVectorToHost(mat_jc_.m_ + 1, jc_t_i_, jct_i_);
    copyVectorToHost(mat_jc_.nnz_, jc_t_j_, jct_j_);
    pc_->vec_map_r(jct_p_i_, jct_p_j_);

    copyVectorToDevice(mat_h_.n_ + 1, hgam_p_i_, hgam_i_p_);
    copyVectorToDevice(nnz_hgam_, hgam_p_j_, hgam_j_p_);
    
    copyVectorToDevice(mat_jc_.nnz_, jct_p_j_, jct_j_p_);
    copyVectorToDevice(mat_jc_.m_ + 1, jct_p_i_, jct_i_p_);
    copyVectorToDevice(mat_jc_.nnz_, jc_p_j_, jc_j_p_);
    copyDeviceVector(mat_jc_.n_ + 1, jc_i_, jc_i_p_);
  }

// Sets up factorization of $H_\gamma$ of eq (6)
  void HykktSolver::setup_hgamma_factorization()
  {
    cc_ = new CholeskyClass(mat_h_.n_,
        nnz_hgam_,
        hgam_v_p_,
        hgam_i_p_,
        hgam_j_p_);

    cc_->symbolic_analysis();
    cc_->set_pivot_tolerance(tol_);
  }

// Sets up conjugate gradient solution of eq (7)
  void HykktSolver::setup_conjugate_gradient()
  {
    //start of block: setting up the right hand side for equation 7
    cc_->solve(d_hrxp_, d_rxp_);
    copyDeviceVector(mat_jc_.n_, d_ry_, d_schur_);

    SpMV_product_reuse(handle_cusparse_,
        ONE,
        jc_p_desc_,
        vec_d_hrxp_,
        MINUS_ONE,
        vec_d_schur_,
        &buffer_schur_,
        allocated_);
    
    if(!allocated_){
      sccg_ = new SchurComplementConjugateGradient(jc_p_desc_,
          jc_t_descp_,
          d_y_,
          d_schur_,
          mat_jc_.n_,
          mat_jc_.m_,
          cc_,
          handle_cusparse_,
          handle_cublas_);
    } else {
      sccg_->update(d_y_, d_schur_, cc_, jc_p_desc_, jc_t_descp_); 
    }
    sccg_->setup();
  }

// Checks the residual of eq (3) to make sure it's small
  int HykktSolver::check_error()
  {
    //  Start of block, calculate error of Ax-b 
    //  Calculate error in rx
    double norm_rx_sq   = 0;
    double norm_rs_sq   = 0;
    double norm_ry_sq   = 0;
    double norm_ryd_sq  = 0;
    double norm_resx_sq = 0;
    double norm_resy_sq = 0; 
    
    // This will aggregate the squared norms of the residual and rhs
    // Note that by construction the residuals of rs and ryd are 0
    dotProduct(handle_cublas_, mat_h_.n_, d_rx_, d_rx_, &norm_rx_sq);
    dotProduct(handle_cublas_, mat_ds_.n_, d_rs_, d_rs_, &norm_rs_sq);
    dotProduct(handle_cublas_, mat_jc_.n_, d_ryc_, d_ryc_, &norm_ry_sq);
    dotProduct(handle_cublas_, mat_jd_.n_, d_ryd_, d_ryd_, &norm_ryd_sq);

    norm_rx_sq += norm_rs_sq + norm_ry_sq + norm_ryd_sq;
    
    SpMV_product_reuse(handle_cusparse_,
        MINUS_ONE,
        h_desc_,
        vec_d_x_,
        ONE,
        vec_d_rx_,
        &buffer_error1_,
        allocated_);
    
    dotProduct(handle_cublas_, mat_h_.n_, d_rx_, d_rx_, &norm_resx_sq);
    if (jd_flag_){
      SpMV_product_reuse(handle_cusparse_,
          MINUS_ONE,
          jd_t_desc_,
          vec_d_yd_,
          ONE,
          vec_d_rx_,
          &buffer_error2_,
          jd_buffers_set_);
    
      dotProduct(handle_cublas_, mat_h_.n_, d_rx_, d_rx_, &norm_resx_sq);   
    }

    SpMV_product_reuse(handle_cusparse_,
        MINUS_ONE,
        jc_t_desc_c_,
        vec_d_y_,
        ONE,
        vec_d_rx_,
        &buffer_error3_,
        allocated_);
    
    dotProduct(handle_cublas_, mat_h_.n_, d_rx_, d_rx_, &norm_resx_sq);

    //  Calculate error in ry
    SpMV_product_reuse(handle_cusparse_,
        MINUS_ONE,
        jc_c_desc_,
        vec_d_x_,
        ONE,
        vec_d_ryc_,
        &buffer_error4_,
        allocated_);
    
    dotProduct(handle_cublas_, mat_jc_.n_, d_ryc_, d_ryc_, &norm_resy_sq);
    
    // Calculate final relative norm
    norm_resx_sq += norm_resy_sq;
    double norm_res = sqrt(norm_resx_sq)/sqrt(norm_rx_sq);
    printf("||Ax-b||/||b|| = %32.32g\n\n", norm_res);
    
    if(jd_flag_){
      jd_buffers_set_ = true;
    }
    
    allocated_ = true;
  
    if (norm_res<norm_tol_){
      printf("Residual test passed\n");
      return 0;
    }else{
      printf("Residual test failed\n");
      return 1;
    }
  }

  void HykktSolver::set_matrix_block(MMatrix& mat,
                                     int* rowptrs,
                                     int* colidx,
                                     double* vals,
                                     int n,
                                     int m,
                                     int nnz)
  {
    assert(mat.owns_csr_data() == false);

    // Use external data
    mat.csr_rows = rowptrs;
    mat.csr_cols = colidx;
    mat.csr_vals = vals;
    mat.n_ = n;
    mat.m_ = m;
    mat.nnz_ = nnz;
  }

  void HykktSolver::set_vector_block(double** vec, double* vals, int n)
  {
    assert(*vec == nullptr);

    // Set to external vector data
    *vec = vals;

    (void) n; // surpress warning that n is unused
  }
