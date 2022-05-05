#include <stdio.h>
#include <stdlib.h>
#include <cusolver_common.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include <algorithm>
#include "cusolverSp.h"
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverRf.h>

#include "RuizClass.hpp"
  // parametrized constructor
  RuizClass::RuizClass(int n, int m, double* H_v, int* H_i, int* H_j,
      double* J_v, int* J_i, int* J_j, double* Jt_v, int* Jt_i, int* Jt_j,
      double* rhs_1, double* rhs_2) :
      n_(n), m_(m), H_v_(H_v), H_i_(H_i), H_j_(H_j), 
      J_v_(J_v), J_i_(J_i), J_j_(J_j), Jt_v_(J_v), Jt_i_(J_i), Jt_j_(J_j),
      rhs_1_(rhs_1), rhs_2_(rhs_2){}

  // destructor
  RuizClass::~RuizClass(){
    free(max_h);
    cudaFree(scale);
  };

  //  Initialization
  void RuizClass::setup(){
    cudaMalloc(&max_d, m_*sizeof(double));
    cudaMalloc(&scale, m_*sizeof(double));
    max_h = (double *) calloc(m_, sizeof(double));
    for(int i=0;i<m_;i++){
      max_h[i]=1; 
    }
  }

  void RuizClass::init_max_d(){
    cudaMemcpy(max_d, max_h, sizeof(double)*m_, cudaMemcpyHostToDevice);
  }
  
  void RuizClass::row_max(){
    fun_adapt_row_max(n_, m_, H_v_, H_i_, H_j_, J_v_, J_i_, J_j_, 
        Jt_v_, Jt_i_, Jt_j_, scale);
  }
  
  void RuizClass::diag_scale(){
    fun_adapt_diag_scale(n_, m_, H_v_, H_i_, H_j_, J_v_, J_i_, J_j_, 
        Jt_v_, Jt_i_, Jt_j_, scale, rhs_1_, rhs_2_, max_d);
  }

  double* RuizClass::get_max_d(){
    return max_d;
  }
