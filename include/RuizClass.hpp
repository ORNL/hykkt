#ifndef CGCL__H__
#define CGCL__H__

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
#include "matrix_vector_ops.hpp"
class SchurComplementConjugateGradient
{
public:
  // default constructor
  RuizClass();

  // parametrized constructor
  RuizClass(int n, int m, double* H_v, int* H_i, int* H_j,
      double* J_v, int* J_i, int* J_j, double* Jt_v, int* Jt_i, int* Jt_j,
      double* rhs_1, double* rhs_2);

  // destructor
  ~RuizClass();

  // Ruiz functions
  void row_max();
  void diag_scale();

  // Return scaling pointer
  double* get_max_d();

private:
  // member variables
  int n_, m_;
  double *H_v_;
  int *H_i_, *H_j_;
  double *J_v_;
  int *J_i_, *J_j_;
  double *Jt_v_;
  int *Jt_i_, *Jt_j_;
  double *scale, *max_d;
};

#endif
