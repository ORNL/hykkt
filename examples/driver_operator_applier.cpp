#include <iostream>
#include <iomanip>
#include "input_functions.hpp"
#include "LQOperator.hpp"
#include "cuda_memory_utils.hpp"
#include "matrix_matrix_ops.hpp"
#include "MMatrix.hpp"
#include "LogBarrierInfo.hpp"

int compareVectors(double* v1, double*v2, int n) {
  int fails = 0;
  double val1;
  double val2;
  double diff;
  for (int i = 0; i < n; i++) {
      val1 = v1[i];
      val2 = v2[i];
      diff = val1 - val2;
      if (abs(diff) > 1e-12) {
        printf("Error at index %.12d\nRESULT: %.12f EXPECTED: %.12f DIFFERENCE: %.12f\n", i, val1, val2, val1 - val2);
        fails++;
      }
  }
  return fails;
}

/**
  * @brief Driver file demonstrates use of Operator Applier S = tQ + A'HA
  *
  * @pre Q is nxn SPD, A is mxn (m > n), H is nxn diagonal
  * 
  */
int main(int argc, char *argv[]) 
{  
    if(argc != 9)
    {
        printf("Incorrect number of inputs. Exiting ...\n");
        return -1;
    }
    
    int n;
    int m;
    int q_nnz;
    int a_nnz;

    int* q_i = nullptr;
    int* q_j = nullptr;
    double* q_v = nullptr;

    int* a_i = nullptr;
    int* a_j = nullptr;
    double* a_v = nullptr;

    double* h_v = nullptr;
    double* d_h_v = nullptr;

    double* v_in = nullptr;
    double* d_v_in = nullptr;
    double* result_apply_vec = nullptr;
    double* result_apply_d_vec = nullptr;
    double* result_extract_vec = nullptr;
    double* result_extract_d_vec = nullptr;
    double* result_diag_vec = nullptr;
    double* result_diag_d_vec = nullptr;
    double* expect_apply_vec = nullptr;
    double* expect_extract_vec = nullptr;
    double* expect_diag_vec = nullptr;
    //***************************FILE READING**************************//
    char const* const q_file_name = argv[1];
    char const* const a_file_name = argv[2];
    char const* const h_file_name = argv[3];
    char const* const v_file_name = argv[4];
    char const* const expect_apply_file_name = argv[5];
    char const* const expect_extract_file_name = argv[6];
    char const* const expect_diag_file_name = argv[7];
    double t = std::stod(argv[8]);

    MMatrix mat_q = MMatrix();
    MMatrix mat_a = MMatrix();

    read_mm_file_into_coo(q_file_name, mat_q, 2);
    read_mm_file_into_coo(a_file_name, mat_a, 2);

    sym_coo_to_csr(mat_q);
    coo_to_csr(mat_a);

    m = mat_a.n_; //rows
    n = mat_a.m_; //cols
    q_nnz = mat_q.nnz_;
    a_nnz = mat_a.nnz_;

    h_v = new double[m];
    v_in = new double[n];
    expect_apply_vec = new double[n];
    expect_extract_vec = new double[q_nnz];
    expect_diag_vec = new double[n];
    read_rhs(h_file_name, h_v);
    read_rhs(v_file_name, v_in);
    read_rhs(expect_apply_file_name, expect_apply_vec);
    read_rhs(expect_extract_file_name, expect_extract_vec);
    read_rhs(expect_diag_file_name, expect_diag_vec);
    
    if (mat_q.n_ != mat_a.m_ || mat_q.n_ != mat_q.m_) {
      printf("Invalid matrix dimensions. Exiting ...\n");
      return -1;
    }
    printf("File reading completed ..........................\n");
    //**************************MEMORY COPYING*************************//
    cusparseHandle_t cusparse_handle = NULL;
    createSparseHandle(cusparse_handle);
    cloneMatrixToDevice(&mat_q, &q_i, &q_j, &q_v);
    cloneMatrixToDevice(&mat_a, &a_i, &a_j, &a_v);
    cloneVectorToDevice(m, &h_v, &d_h_v);
    cloneVectorToDevice(n, &v_in, &d_v_in);
    allocateVectorOnDevice(n, &result_apply_d_vec);
    allocateVectorOnDevice(q_nnz, &result_extract_d_vec);
    allocateVectorOnDevice(n, &result_diag_d_vec);
    //************************APPLYING OPERATORS***********************//
    LogBarrierInfo info(m, n, q_nnz, q_i, q_j, q_v, a_nnz, a_i, a_j, a_v, nullptr, nullptr, cusparse_handle);
    LQOperator applier(info, d_h_v);
    applier.set_Q_scalar(t);
    applier.apply(d_v_in, result_apply_d_vec);
    printf("SECOND APPLY\n");
    applier.apply(d_v_in, result_apply_d_vec);
    applier.extract_sparse_structure(result_extract_d_vec);
    applier.extract_inv_linear_diagonal(result_diag_d_vec);
    //*************************TESTING OPERATOR************************//

    result_apply_vec = new double[n];
    result_extract_vec = new double[q_nnz];
    result_diag_vec = new double[n];
   
    copyVectorToHost(n, result_apply_d_vec, result_apply_vec);
    copyVectorToHost(q_nnz, result_extract_d_vec, result_extract_vec);
    copyVectorToHost(n, result_diag_d_vec, result_diag_vec);
    int fails1 = compareVectors(result_apply_vec, expect_apply_vec, n);
    if (fails1 == 0) {
      printf("Operator Applier test passed!\n");
    }
    else {
      printf("Operator Applier test failed!\n");
    }
    int fails2 = compareVectors(result_extract_vec, expect_extract_vec, q_nnz);
    if (fails2 == 0) {
      printf("Sparse extraction test passed!\n");
    }
    else {
      printf("Sparse extraction test failed!\n");
    }
    int fails3 = compareVectors(result_diag_vec, expect_diag_vec, n);
    if (fails3 == 0) {
     printf("Diagonal extraction test passed!\n");
    }
    else {
      printf("Diagonal extraction test failed!\n");
    }

    int fails = fails1 + fails2 + fails3;
    //**************************FREEING MEMORY*************************//

    deleteHandle(cusparse_handle);
    
    deleteOnDevice(d_h_v);
    deleteOnDevice(d_v_in);
    deleteOnDevice(result_apply_d_vec);
    deleteOnDevice(result_extract_d_vec);
    deleteOnDevice(result_diag_d_vec);
    deleteMatrixOnDevice(q_i, q_j, q_v);
    deleteMatrixOnDevice(a_i, a_j, a_v);

    delete[] h_v;
    delete[] v_in;
    delete[] result_apply_vec;
    delete[] result_extract_vec;
    delete[] result_diag_vec;
    delete[] expect_apply_vec;
    delete[] expect_extract_vec;
    delete[] expect_diag_vec;
    
    return fails;
}