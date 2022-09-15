#include <stdio.h>
#include "SchurComplementConjugateGradient.hpp"
#include "matrix_vector_ops_cuda.hpp"
#include "vector_vector_ops.hpp"
#include "cuda_memory_utils.hpp"
#include "constants.hpp"

  // parametrized constructor
  SchurComplementConjugateGradient::SchurComplementConjugateGradient(
      cusparseSpMatDescr_t jc_desc, 
      cusparseSpMatDescr_t jct_desc,
      double* x0, 
      double* b, 
      int n, 
      int m, 
      CholeskyClass* cc, 
      cusparseHandle_t handle, 
      cublasHandle_t handle_cublas) 
    :jc_desc_(jc_desc), 
     jct_desc_(jct_desc), 
     x0_(x0), 
     b_(b), 
     n_(n), 
     m_(m),
     cc_(cc), 
     handle_(handle), 
     handle_cublas_(handle_cublas)
  {
    allocate_workspace();
  }

  // destructor
  SchurComplementConjugateGradient::~SchurComplementConjugateGradient()
  {
    deleteOnDevice(y_);
    deleteOnDevice(z_);
    deleteOnDevice(r_);
    deleteOnDevice(w_);
    deleteOnDevice(p_);
    deleteOnDevice(s_);
    
    deleteOnDevice(buffer1_);
    deleteOnDevice(buffer2_);
    deleteOnDevice(buffer3_);
    deleteOnDevice(buffer4_);
    
    delete [] ycp_;
  };
  
  // solver API
  void SchurComplementConjugateGradient::allocate_workspace()
  {
    ycp_ = new double[m_]{0.0};
    
    allocateVectorOnDevice(m_, &y_);
    allocateVectorOnDevice(m_, &z_);
    allocateVectorOnDevice(n_, &r_);
    allocateVectorOnDevice(n_, &w_);
    allocateVectorOnDevice(n_, &p_);
    allocateVectorOnDevice(n_, &s_);

    //  Allocation - happens once
    createDnVec(&vecx_, n_, x0_);
    createDnVec(&vecb_, n_, b_);
    createDnVec(&vecy_, m_, y_);
    createDnVec(&vecz_, m_, z_);
    createDnVec(&vecr_, n_, r_);
    createDnVec(&vecw_, n_, w_);
    createDnVec(&vecp_, n_, p_);
    createDnVec(&vecs_, n_, s_);
  }

  void SchurComplementConjugateGradient::setup()
  {
    copyVectorToDevice(m_, ycp_, y_);

    copyDeviceVector(m_, y_, z_);
    copyDeviceVector(n_, b_, r_);
    copyDeviceVector(n_, b_, w_);
    copyDeviceVector(n_, r_, p_);
    copyDeviceVector(n_, w_, s_);
    
    beta_ = 0;
  }
 
  int SchurComplementConjugateGradient::solve()
  {
    SpMV_product_reuse(handle_,
        ONE,
        jct_desc_,
        vecx_,
        ZERO,
        vecy_,
        &buffer1_,
        allocated_);
    cc_->solve(z_, y_);
  
    SpMV_product_reuse(handle_,
        MINUS_ONE,
        jc_desc_,
        vecz_,
        ONE,
        vecr_,
        &buffer2_,
        allocated_);
    
    dotProduct(handle_cublas_, n_, r_, r_, &gam_i_);
    SpMV_product_reuse(handle_,
        ONE,
        jct_desc_,
        vecr_,
        ZERO,
        vecy_,
        &buffer3_,
        allocated_);
    cc_->solve(z_, y_);
  
    SpMV_product_reuse(handle_,
        ONE,
        jc_desc_,
        vecz_,
        ZERO,
        vecw_,
        &buffer4_,
        allocated_);
    dotProduct(handle_cublas_, n_, w_, r_, &delta_);
    alpha_    = gam_i_ / delta_;
    minalpha_ = -alpha_;
    int i;
  
    for(i = 0; i < itmax_; i++){
      scaleVector(handle_cublas_, n_, &beta_, p_);
      sumVectors(handle_cublas_, n_, r_, p_);
      scaleVector(handle_cublas_, n_, &beta_, s_);
      sumVectors(handle_cublas_, n_, w_, s_);
      sumVectors(handle_cublas_, n_, p_, x0_, &alpha_);
      minalpha_ = -alpha_;
      sumVectors(handle_cublas_, n_, s_, r_, &minalpha_);
      dotProduct(handle_cublas_, n_, r_, r_, &gam_i1_);
      if(sqrt(gam_i1_) < tol_){
        printf("Convergence occured at iteration %d\n", i);
        break;
      }
      // product with w=Ar starts here
      SpMV_product_reuse(handle_,
          ONE,
          jct_desc_,
          vecr_,
          ZERO,
          vecy_,
          &buffer3_,
          allocated_);
    
      cc_->solve(z_, y_);
      SpMV_product_reuse(handle_,
          ONE,
          jc_desc_,
          vecz_,
          ZERO,
          vecw_,
          &buffer4_,
          allocated_);

      dotProduct(handle_cublas_, n_, w_, r_, &delta_);
      beta_  = gam_i1_ / gam_i_;
      gam_i_ = gam_i1_;
      alpha_ = gam_i_ / (delta_ - beta_ * gam_i_ / alpha_);
    }
    
    allocated_ = true;

    printf("Error is %32.32g \n", sqrt(gam_i1_));
    if (i == itmax_){
      printf("No CG convergence in %d iterations\n", itmax_);
      return 1;
    }
    return 0;
  }

  void SchurComplementConjugateGradient::set_solver_tolerance(double tol)
  {
    tol_ = tol;
  }

  void SchurComplementConjugateGradient::set_solver_itmax(int itmax)
  {
    itmax_ = itmax;
  }
