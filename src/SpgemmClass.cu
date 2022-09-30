#include "SpgemmClass.hpp"
#include "matrix_matrix_ops.hpp"
#include "matrix_vector_ops.hpp"
#include "cuda_memory_utils.hpp"
#include "constants.hpp"
#include "cusparse_utils.hpp"

  SpgemmClass::SpgemmClass(int n,
      int m,
      cusparseHandle_t handle,
      double alpha_p,
      double alpha_s,
      double beta_s)
  :n_(n),
  m_(m),
  handle_(handle),
  alpha_p_(alpha_p),
  alpha_s_(alpha_s),
  beta_s_(beta_s)
  {
    allocate_workspace();
  }

  SpgemmClass::~SpgemmClass()
  {
    deleteMatrixOnDevice(c_i_, c_j_, c_v_);
    
    deleteDescriptor(spgemm_desc_);
    deleteDescriptor(c_desc_);

    deleteOnDevice(d_buffer1_);
    deleteOnDevice(d_buffer2_);
    deleteOnDevice(buffer_add_);
  }
  
  void SpgemmClass::load_product_matrices(cusparseSpMatDescr_t a_desc, 
      cusparseSpMatDescr_t b_desc)
  {
    a_desc_ = a_desc; 
    b_desc_ = b_desc;
  }
  
  void SpgemmClass::load_sum_matrices(int* d_i, 
      int* d_j, 
      double* d_v,
      int nnz_d)
  {
    d_i_ = d_i;
    d_j_ = d_j;
    d_v_ = d_v;
    
    nnz_d_ = nnz_d; 
  }

  void SpgemmClass::load_result_matrix(int** e_i, 
      int** e_j, 
      double** e_v, 
      int* nnz_e)
  {
    e_i_ = e_i;
    e_j_ = e_j;
    e_v_ = e_v;

    nnz_e_ = nnz_e;
  }
  
  void SpgemmClass::spGEMM_reuse()
  {
    if(!spgemm_allocated_){
      allocate_spGEMM_product();
    }

    compute_spGEMM_product();

    if(!spgemm_allocated_){
      allocate_spGEMM_sum();
    }

    compute_spGEMM_sum();
  
    spgemm_allocated_ = true;
  }

  void SpgemmClass::allocate_workspace()
  {
    createSpGEMMDescr(&spgemm_desc_);
    createSparseMatDescr(descr_d_);
    
    c_desc_ = NULL;
    createCsrMat(&c_desc_, n_, n_, 0, NULL, NULL, NULL);
  }
  
  void SpgemmClass::allocate_spGEMM_product()
  {
    allocate_for_product(handle_,
        a_desc_,
        b_desc_,
        c_desc_,
        n_,
        nnz_c_,
        &c_i_,
        &c_j_,
        &c_v_,
        spgemm_desc_,
        &d_buffer1_,
        &d_buffer2_);
  }
  
  void SpgemmClass::compute_spGEMM_product()
  {
    compute_product(handle_,
        alpha_p_,
        a_desc_,
        b_desc_,
        c_desc_,
        spgemm_desc_);
  }

  void SpgemmClass::allocate_spGEMM_sum()
  {
    allocate_for_sum(handle_,
        d_i_,
        d_j_,
        d_v_,
        alpha_s_,
        c_i_,
        c_j_,
        c_v_,
        beta_s_,
        e_i_,
        e_j_,
        e_v_,
        m_,
        n_,
        nnz_d_,
        nnz_c_,
        descr_d_,
        &buffer_add_,
        nnz_e_);
  }
  
  void SpgemmClass::compute_spGEMM_sum()
  {
    compute_sum(handle_,
        d_i_,
        d_j_,
        d_v_,
        alpha_s_,
        c_i_,
        c_j_,
        c_v_,
        beta_s_,
        *e_i_,
        *e_j_,
        *e_v_,
        m_,
        n_,
        nnz_d_,
        nnz_c_,
        descr_d_,
        &buffer_add_);
  }
