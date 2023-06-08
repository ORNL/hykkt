#include <string>
#include "stdio.h"
#include "MMatrix.hpp"
#include "input_functions.hpp"
#include "cuda_memory_utils.hpp"
#include "log_barrier_utils.hpp"
#include "cusparse_utils.hpp"
#include "LogBarrierHelper.hpp"

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
        printf("Error at index %d\nRESULT: %f EXPECTED: %f DIFFERENCE: %f\n", i, val1, val2, val1 - val2);
        fails++;
      }
  }
  return fails;
}

/**
 * @brief Driver file tests function helpers: objective, gradient, hessian for Log Barrier method
*/

int main(int argc, char* argv[])
{
    if(argc != 12) {
        printf("Incorrect number of inputs. Exiting ...\n");
        return -1;
    }

    double t;

    int m, n;

    double q_nnz;
    int* q_i = nullptr;
    int* q_j = nullptr;
    double* q_v = nullptr;

    double a_nnz;
    int* a_i = nullptr;
    int* a_j = nullptr;
    double* a_v = nullptr;

    double* d_b = nullptr;
    double* h_b = nullptr;
    double* d_c = nullptr;
    double* h_c = nullptr;
    double* d_x = nullptr;
    double* h_x = nullptr;
    cusparseDnVecDescr_t x_desc;

    double qp_expected_objective;
    double lp_expected_objective;
    double* qp_grad_expected = nullptr;
    double* lp_grad_expected = nullptr;
    double* hess_expected = nullptr;

    double qp_test_objective;
    double lp_test_objective;
    double* qp_d_grad_test = nullptr;
    double* qp_d_hess_test = nullptr;
    double* lp_d_grad_test = nullptr;
    double* lp_d_hess_test = nullptr;

    double* qp_h_grad_test = nullptr;
    double* qp_h_hess_test = nullptr;
    double* lp_h_grad_test = nullptr;
    double* lp_h_hess_test = nullptr;

    char* q_file_name = nullptr;
    char* a_file_name = nullptr;
    char* b_file_name = nullptr;
    char* c_file_name = nullptr;
    char* x_file_name = nullptr;
    char* qp_gradient_file_name = nullptr;
    char* lp_gradient_file_name = nullptr;
    char* hessian_file_name = nullptr;
    //***************************FILE READING**************************//
    q_file_name = argv[1];
    a_file_name = argv[2];
    b_file_name = argv[3];
    c_file_name = argv[4];
    x_file_name = argv[5];
    qp_expected_objective = std::stod(argv[6]);
    lp_expected_objective = std::stod(argv[7]);
    qp_gradient_file_name = argv[8];
    lp_gradient_file_name = argv[9];
    hessian_file_name = argv[10];
    t = std::stod(argv[11]);

    MMatrix mat_q = MMatrix();
    MMatrix mat_a = MMatrix();

    read_mm_file_into_coo(q_file_name, mat_q, 2);
    sym_coo_to_csr(mat_q);
    q_nnz = mat_q.nnz_;
    
    read_mm_file_into_coo(a_file_name, mat_a, 2);
    coo_to_csr(mat_a);
    a_nnz = mat_a.nnz_;

    m = mat_a.n_; //rows
    n = mat_a.m_; //cols

    h_b = new double[m];
    h_c = new double[n];
    h_x = new double[n];
    qp_grad_expected = new double[n];
    lp_grad_expected = new double[n];
    hess_expected = new double[m];
    read_rhs(b_file_name, h_b);
    read_rhs(c_file_name, h_c);
    read_rhs(x_file_name, h_x);
    read_rhs(qp_gradient_file_name, qp_grad_expected);
    read_rhs(lp_gradient_file_name, lp_grad_expected);
    read_rhs(hessian_file_name, hess_expected);
    printf("File reading completed ..........................\n");
    //**************************MEMORY COPYING*************************//
    cusparseHandle_t cusparse_handle = NULL;
    createSparseHandle(cusparse_handle);
    cloneMatrixToDevice(&mat_q, &q_i, &q_j, &q_v);
    cloneMatrixToDevice(&mat_a, &a_i, &a_j, &a_v);
    cloneVectorToDevice(m, &h_b, &d_b);
    cloneVectorToDevice(n, &h_c, &d_c);
    cloneVectorToDevice(n, &h_x, &d_x);
    createDnVec(&x_desc, n, d_x);

    allocateVectorOnDevice(n, &qp_d_grad_test);
    allocateVectorOnDevice(m, &qp_d_hess_test);
    allocateVectorOnDevice(n, &lp_d_grad_test);
    allocateVectorOnDevice(m, &lp_d_hess_test);
    qp_h_grad_test = new double[n];
    qp_h_hess_test = new double[m];
    lp_h_grad_test = new double[n];
    lp_h_hess_test = new double[m];
    //*************************TESTING FUNCTIONS**************************//
    int fails = 0;
    LogBarrierInfo qp_info(m, n, q_nnz, q_i, q_j, q_v, a_nnz, a_i, a_j, a_v, d_b, d_c, cusparse_handle);
    LogBarrierHelper qp_helper(qp_info);
    qp_helper.gradient(t, x_desc, qp_d_grad_test);
    qp_helper.hessian(x_desc, qp_d_hess_test);
    qp_test_objective = qp_helper.update_get_objective(t, d_x, x_desc);

    LogBarrierInfo lp_info(m, n, a_nnz, a_i, a_j, a_v, d_b, d_c, cusparse_handle);
    LogBarrierHelper lp_helper(lp_info);
    lp_helper.gradient(t, x_desc, lp_d_grad_test);
    lp_helper.hessian(x_desc, lp_d_hess_test);
    lp_test_objective = lp_helper.update_get_objective(t, d_x, x_desc);

    copyVectorToHost(n, qp_d_grad_test, qp_h_grad_test);
    copyVectorToHost(m, qp_d_hess_test, qp_h_hess_test);
    copyVectorToHost(n, lp_d_grad_test, lp_h_grad_test);
    copyVectorToHost(m, lp_d_hess_test, lp_h_hess_test);
    int fails1 = compareVectors(qp_h_grad_test, qp_grad_expected, n);
    if (fails1 > 0) {
      printf("QP Gradient test failed\n");
      fails += fails1;
    }
    else {
      printf("QP Gradient test passed!\n");
    }

    int fails2 = compareVectors(lp_h_grad_test, lp_grad_expected, n);
        if (fails2 > 0) {
      printf("LP Gradient test failed\n");
      fails += fails2;
    }
    else {
      printf("LP Gradient test passed!\n");
    }
    
    int fails3 = compareVectors(qp_h_hess_test, hess_expected, m);
    if (fails3 > 0) {
      printf("QP Hessian test failed\n");
      fails += fails2;
    }
    else {
      printf("QP Hessian test passed!\n");
    }

    //Note: should be same as LP
    int fails4 = compareVectors(lp_h_hess_test, hess_expected, m);
    if (fails4 > 0) {
      printf("LP Inverse Hessian test failed\n");
      fails += fails4;
    }
    else {
      printf("LP Inverse Hessian test passed!\n");
    }

    int fails5 = abs(qp_test_objective - qp_expected_objective) > 1e-12;
    if (fails5 > 0) {
      printf("QP Objective test failed. Expected: %.12f Result: %.12f\n", qp_expected_objective, qp_test_objective);
      fails += fails5;
    }
    else {
      printf("QP Objective test passed!\n");
    }

    int fails6 = abs(lp_test_objective - lp_expected_objective) > 1e-12;
    if (fails6 > 0) {
      printf("LP Objective test failed. Expected: %.12f Result: %.12f\n", lp_expected_objective, lp_test_objective);
      fails += fails6;
    }
    else {
      printf("LP Objective test passed!\n");
    }
    //**************************FREEING MEMORY*************************//
    deleteHandle(cusparse_handle);
    deleteMatrixOnDevice(q_i, q_j, q_v);
    deleteMatrixOnDevice(a_i, a_j, a_v);
    deleteOnDevice(d_b);
    deleteOnDevice(d_c);
    deleteOnDevice(d_x);
    deleteOnDevice(qp_d_grad_test);
    deleteOnDevice(qp_d_hess_test);
    deleteOnDevice(lp_d_grad_test);
    deleteOnDevice(lp_d_hess_test);
    deleteDescriptor(x_desc);

    delete[] h_b;
    delete[] h_c;
    delete[] h_x;
    delete[] qp_grad_expected;
    delete[] lp_grad_expected;
    delete[] hess_expected;
    delete[] qp_h_grad_test;
    delete[] qp_h_hess_test;
    delete[] lp_h_grad_test;
    delete[] lp_h_hess_test;

    return fails;
}