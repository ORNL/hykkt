#pragma once
#include <cusparse.h>

constexpr cudaDataType COMPUTE_TYPE = CUDA_R_64F; 
constexpr cusparseIndexType_t INDEX_TYPE = CUSPARSE_INDEX_32I;
constexpr cusparseIndexBase_t INDEX_BASE = CUSPARSE_INDEX_BASE_ZERO;
constexpr cusparseOperation_t CUSPARSE_OPERATION = CUSPARSE_OPERATION_NON_TRANSPOSE;
constexpr cusparseSpGEMMAlg_t CUSPARSE_ALGORITHM = CUSPARSE_SPGEMM_DEFAULT;

