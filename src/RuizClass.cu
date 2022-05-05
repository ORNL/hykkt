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

#include "RuizScale.hpp"
  // parametrized constructor
  RuizClass::RuizClass(int n, int m, double* H_v, int* H_i, int* H_j,
      double* J_v, int* J_i, int* J_j, double* Jt_v, int* Jt_i, int* Jt_j,
      double* rhs_1, double* rhs_2) :
      n_(n), m_(m), H_v_(H_v), H_i_(H_i), H_j_(H_j), 
      J_v_(J_v), J_i_(J_i), J_j_(J_j), Jt_v_(J_v), Jt_i_(J_i), Jt_j_(J_j),
      rhs_1_(rhs_1), rhs_2_(rhs_2){}

  // destructor
  RuizScale::~RuizScale(){
  cudaFree(scale);
  };

  //  Initialization
  void RuizScale::row_max(){

  }
  void RuizScale::row_max(){

  }
