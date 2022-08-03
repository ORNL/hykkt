#pragma once
#include <cusparse.h>

const cudaDataType COMPUTE_TYPE = CUDA_R_64F; 
const cusparseIndexType_t INDEX_TYPE = CUSPARSE_INDEX_32I;
const cusparseIndexBase_t INDEX_BASE = CUSPARSE_INDEX_BASE_ZERO;
const cusparseOperation_t CUSPARSE_OPERATION = CUSPARSE_OPERATION_NON_TRANSPOSE;
const cusparseSpGEMMAlg_t CUSPARSE_ALGORITHM = CUSPARSE_SPGEMM_DEFAULT;
