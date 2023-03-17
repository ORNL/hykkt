#include "PermClass.hpp"
#include "matrix_vector_ops.hpp"
#include "cuda_memory_utils.hpp"
#include "permcheck.hpp"
#include <cusparse_utils.hpp>

#include "cuda_check_errors.hpp"

#include <iostream>
#include <fstream>

#include "amd.h"


// Creates a class for the permutation of $H_\gamma$ in (6)
PermClass::PermClass(int n_h, int nnz_h, int nnz_j) 
  : n_h_(n_h),
    nnz_h_(nnz_h),
    nnz_j_(nnz_j)
  {
    allocate_workspace();
  }

  PermClass::~PermClass()
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

  void PermClass::add_h_info(int* h_i, int* h_j)
  {
    h_i_ = h_i;
    h_j_ = h_j;
  }
  
  void PermClass::add_j_info(int* j_i, int* j_j, int n_j, int m_j)
  {
    j_i_ = j_i;
    j_j_ = j_j;
    n_j_ = n_j;
    m_j_ = m_j;
  }
  
  void PermClass::add_jt_info(int* jt_i, int* jt_j)
  {
    jt_i_ = jt_i;
    jt_j_ = jt_j;
  }
  
  void PermClass::add_perm(int* custom_perm)
  {
    perm_is_default_ = false;
    perm_ = custom_perm;
    cloneVectorToDevice(n_h_, &perm_, &d_perm_);
  }
 
// Symamd permutation of $H_\gamma$ in (6)
  void PermClass::symamd()
  {
    std::cout << "Testing SYMAMD" << std::endl;
    std::cout << n_h_ << "\t" << nnz_h_ << std::endl;
    
    std::ofstream zOut1("h_i.txt", std::ios::out | std::ios::binary);
    zOut1.write(reinterpret_cast<char*>(h_i_), sizeof(int)*(n_h_+1));
    zOut1.close();

    std::ofstream zOut2("h_j.txt", std::ios::out | std::ios::binary);
    zOut2.write(reinterpret_cast<char*>(h_j_), sizeof(int)*(nnz_h_));
    zOut2.close();

    for (int i = 0; i < 10; i++)
    {
        std::cout << h_i_[i] << "\t" << h_j_[i] << std::endl;
    }


#if 1

    std::cout << "Using - AMD" << std::endl;
    double Control[AMD_CONTROL], Info[AMD_INFO];
	
	amd_defaults(Control);
	amd_control(Control);
	
	int result = amd_order(n_h_, h_i_, h_j_, perm_, Control, Info);
	
	if (result != AMD_OK)
	{
		printf("AMD failed\n");
		exit(1);
	}

#else
    cusolverSpHandle_t handle_cusolver = NULL;
    cusparseMatDescr_t descr_a = NULL;
    createSparseMatDescr(descr_a);
    checkCudaErrors(cusolverSpCreate(&handle_cusolver));
    checkCudaErrors(cusolverSpXcsrsymamdHost(handle_cusolver, n_h_, nnz_h_, 
           descr_a, h_i_, h_j_, perm_));
    checkCudaErrors(cusolverSpDestroy(handle_cusolver));
    deleteDescriptor(descr_a);
	
    for (int i = 0; i < 10; i++)
    {
        std::cout << perm_[i] << std::endl;
    }
#endif
    
    cloneVectorToDevice(n_h_, &perm_, &d_perm_); 
  }
  
  void PermClass::invert_perm()
  {
    reverse_perm(n_h_, perm_, rev_perm_);
    cloneVectorToDevice(n_h_, &rev_perm_, &d_rev_perm_); 
  }

  void PermClass::vec_map_rc(int* b_i, int* b_j)
  {
    make_vec_map_rc(n_h_, h_i_, h_j_, perm_, rev_perm_, b_i, b_j, perm_map_h_);
    cloneVectorToDevice(nnz_h_, &perm_map_h_, &d_perm_map_h_);
  }

  void PermClass::vec_map_c(int* b_j)
  {
    make_vec_map_c(n_j_, j_i_, j_j_, rev_perm_, b_j, perm_map_j_);
    cloneVectorToDevice(nnz_j_, &perm_map_j_, &d_perm_map_j_); 
  }

  void PermClass::vec_map_r(int* b_i, int* b_j)
  {
    make_vec_map_r(m_j_, jt_i_, jt_j_, perm_, b_i, b_j, perm_map_jt_);
    cloneVectorToDevice(nnz_j_, &perm_map_jt_, &d_perm_map_jt_); 
  }
  
  void PermClass::map_index(Permutation_Type permutation,
      double* old_val,
      double* new_val)
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

  void PermClass::display_perm() const
  {
    displayDeviceVector(d_perm_,
        n_h_,
        0,
        10,
        "PERMUTATION"); 

  }
  
  void PermClass::allocate_workspace()
  {
    perm_ = new int[n_h_];
    rev_perm_ = new int[n_h_];
    perm_map_h_ = new int[nnz_h_];
    perm_map_j_ = new int[nnz_j_];
    perm_map_jt_ = new int[nnz_j_];
  }
