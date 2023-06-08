#pragma once

void fun_csr_spmv_kernel(
    int n_rows,
    int n_cols,
    int* col_ids,
    int* row_ptr,
    double* data,
    double* x,
    double* y);