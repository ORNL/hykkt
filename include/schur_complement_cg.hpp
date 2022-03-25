#ifndef CG__H__
#define CG__H__

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

void schur_cg(cusparseSpMatDescr_t, cusparseSpMatDescr_t, csrcholInfo_t,
    double*, double*, const int, const double, int, int, int, void*,
    cusparseHandle_t, cusolverSpHandle_t, cublasHandle_t);
#endif
