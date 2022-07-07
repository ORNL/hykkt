#include "RuizClass.hpp"
#include <algorithm>
#include "matrix_vector_ops.hpp"
#include "cuda_memory_utils.hpp"
#include "constants.hpp"

  RuizClass::RuizClass(int ruiz_its, int n, int totn) 
: ruiz_its_(ruiz_its),
  n_(n),
  totn_(totn),
  scale_(nullptr),
  max_d_(nullptr)
  {
    allocate_workspace();
  }

  RuizClass::~RuizClass()
  {
    deleteOnDevice(scale_);
    deleteOnDevice(max_d_);
  }
  
  void RuizClass::add_block11(double* h_v, int* h_i, int* h_j)
  {
    h_v_ = h_v;
    h_i_ = h_i;
    h_j_ = h_j;
  }
  
  void RuizClass::add_block12(double* jt_v, int* jt_i, int* jt_j)
  {
    jt_v_ = jt_v;
    jt_i_ = jt_i;
    jt_j_ = jt_j;
  }
  
  void RuizClass::add_block21(double* j_v, int* j_i, int* j_j)
  {
    j_v_ = j_v;
    j_i_ = j_i;
    j_j_ = j_j;
  }

  void RuizClass::add_rhs1(double* rhs1)
  {
    rhs1_ = rhs1;
  }

  void RuizClass::add_rhs2(double* rhs2)
  {
    rhs2_ = rhs2;
  }
  
  void RuizClass::ruiz_scale()
  {
    for(int i = 0; i < ruiz_its_; i++) {
      fun_adapt_row_max(n_,
          totn_,
          h_v_,
          h_i_,
          h_j_,
          j_v_,
          j_i_,
          j_j_,
          jt_v_,
          jt_i_,
          jt_j_,
          scale_);
      
      fun_adapt_diag_scale(n_,
          totn_,
          h_v_,
          h_i_,
          h_j_,
          j_v_,
          j_i_,
          j_j_,
          jt_v_,
          jt_i_,
          jt_j_,
          scale_,
          rhs1_,
          rhs2_,
          max_d_);
    }
  }

  double* RuizClass::get_max_d() const
  {
    return max_d_;
  }

  void RuizClass::allocate_workspace()
  {
    allocateVectorOnDevice(totn_, &scale_);
    allocateVectorOnDevice(totn_, &max_d_);
    fun_set_const(totn_, ONE, max_d_);
  }
