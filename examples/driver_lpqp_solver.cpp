#include "input_functions.hpp"
#include "PreconditionedCG.hpp"
#include "LQOperator.hpp"
#include "cuda_memory_utils.hpp"
#include "matrix_matrix_ops.hpp"
#include "MMatrix.hpp"
#include "chrono"

int compareVectorsDifMag(double* v1, double*v2, int n, double tol)
{
  int fails = 0;
  double total = 0;
  double val1;
  double val2;
  double diff;
  for (int i = 0; i < n; i++) {
      val1 = v1[i];
      val2 = v2[i];
      diff = val1 - val2;
      total += diff * diff;
  }
  double norm_err = sqrt(total);
  printf("Norm Err: %.20f\n", norm_err);
  return norm_err > tol * 10; //1 if greater than tol, else 0 and passes
}

int testConvergence(LQOperator& lqop, double* d_test_v, double* h_expected, double tol, int iterations, int itmax)
{
  int n = lqop.get_operator_size();
  double* d_test_res;
  allocateVectorOnDevice(n, &d_test_res);
  
  lqop.apply(d_test_v, d_test_res);

  double* h_test_res = new double[n];
  copyVectorToHost(n, d_test_res, h_test_res);
  
  if (iterations == -1) {
    printf("Solver failed to converge (Iterations limit: %d)\n", itmax);
  }
  else {
    printf("Solver converged in %d iterations.\n", iterations);
  }

  int result = compareVectorsDifMag(h_test_res, h_expected, n, tol);
  delete[] h_test_res;
  deleteOnDevice(d_test_res);
  return result;
}

/**
  * @brief Driver file demonstrates use of Preconditioned Conjugate Gradient for LQOperator
  */
int main(int argc, char *argv[]) 
{  
    bool quadratic;

    if(argc == 6) {
        quadratic = false;
    }
    else if(argc == 8) {
        quadratic = true;
    }
    else {
        printf("Incorrect number of inputs. Exiting ...\n");
        return -1;
    }

    const int itmax = 200;
    const double tol = 1e-12;

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

    double* h1_v = nullptr;
    double* d_h1_v = nullptr;
    double* h2_v = nullptr;
    double* d_h2_v = nullptr;

    double* res1 = nullptr;
    double* d_res1 = nullptr;
    double* res2 = nullptr;
    double* d_res2 = nullptr;
    double* test_v = nullptr;
    double* d_test_v = nullptr;

    char* q_file_name = nullptr;
    char* a_file_name = nullptr;
    char* h1_file_name = nullptr;
    char* h2_file_name = nullptr;
    char* res1_file_name = nullptr;
    char* res2_file_name = nullptr;
    double t;
    //***************************FILE READING**************************//
    if (quadratic) {
        q_file_name = argv[1];
        a_file_name = argv[2];
        h1_file_name = argv[3];
        h2_file_name = argv[4];
        res1_file_name = argv[5];
        res2_file_name = argv[6];
        double t = std::stod(argv[7]);
    }
    else {
        a_file_name = argv[1];
        h1_file_name = argv[2];
        h2_file_name = argv[3];
        res1_file_name = argv[4];
        res2_file_name = argv[5];
    }

    MMatrix mat_q = MMatrix();
    MMatrix mat_a = MMatrix();

    if (quadratic) {
        read_mm_file_into_coo(q_file_name, mat_q, 2);
        sym_coo_to_csr(mat_q);
        q_nnz = mat_q.nnz_;
    }

    read_mm_file_into_coo(a_file_name, mat_a, 2);
    coo_to_csr(mat_a);

    m = mat_a.n_; //rows
    n = mat_a.m_; //cols
    a_nnz = mat_a.nnz_;

    h1_v = new double[m];
    h2_v = new double[m];
    res1 = new double[n];
    res2 = new double[n];
    read_rhs(h1_file_name, h1_v);
    read_rhs(h2_file_name, h2_v);
    read_rhs(res1_file_name, res1);
    read_rhs(res2_file_name, res2);

    if (quadratic && (mat_q.n_ != mat_a.m_ || mat_q.n_ != mat_q.m_)) {
      printf("Invalid matrix dimensions. Exiting ...\n");
      return -1;
    }
    printf("File reading completed ..........................\n");
    //**************************MEMORY COPYING*************************//
    cusparseHandle_t cusparse_handle = NULL; //each handle takes about 4s to be created
    createSparseHandle(cusparse_handle);
    if (quadratic) {
        cloneMatrixToDevice(&mat_q, &q_i, &q_j, &q_v);
    }
    cloneMatrixToDevice(&mat_a, &a_i, &a_j, &a_v);
    cloneVectorToDevice(m, &h1_v, &d_h1_v);
    cloneVectorToDevice(m, &h2_v, &d_h2_v);
    cloneVectorToDevice(n, &res1, &d_res1);
    cloneVectorToDevice(n, &res2, &d_res2);
    test_v = new double[n]{0.0};
    //*************************TESTING SOLVER*************************//
    int fails = 0; //used for testing
    int iterations;
    if (quadratic) { //demonstration of using quadratic solve of S = tQ + A'HA
        cloneVectorToDevice(n, &test_v, &d_test_v); //initial x0 = 0
        LogBarrierInfo info(m, n, q_nnz, q_i, q_j, q_v, a_nnz, a_i, a_j, a_v, nullptr, nullptr, cusparse_handle);
        LQOperator lqop(info, d_h1_v);
        lqop.set_Q_scalar(t);
        PreconditionedCG cg_solver;
        iterations = cg_solver.solve(itmax, tol, lqop, d_res1, d_test_v);
        fails += testConvergence(lqop, d_test_v, res1, tol, iterations, itmax);
        //********************UPDATING H MATRIX*************************//
        cloneVectorToDevice(n, &test_v, &d_test_v); //initial x0 = 0
        lqop.load_H_matrix(d_h2_v);
        iterations = cg_solver.solve(itmax, tol, lqop, d_res2, d_test_v);
        fails += testConvergence(lqop, d_test_v, res2, tol, iterations, itmax);
    }
    else { //demonstration of using linear solve of S = A'HA
        cloneVectorToDevice(n, &test_v, &d_test_v); //initial x0 = 0
        LogBarrierInfo info(m, n, a_nnz, a_i, a_j, a_v, nullptr, nullptr, cusparse_handle);
        LQOperator lqop(info, d_h1_v);
        PreconditionedCG cg_solver;
        iterations = cg_solver.solve(itmax, tol, lqop, d_res1, d_test_v);
        fails += testConvergence(lqop, d_test_v, res1, tol, iterations, itmax);
        //********************UPDATING H MATRIX*************************//
        cloneVectorToDevice(n, &test_v, &d_test_v); //initial x0 = 0
        lqop.load_H_matrix(d_h2_v);
        iterations = cg_solver.solve(itmax, tol, lqop, d_res2, d_test_v);
        fails += testConvergence(lqop, d_test_v, res2, tol, iterations, itmax);
    }
    //**************************FREEING MEMORY*************************//
    deleteHandle(cusparse_handle);
    deleteOnDevice(d_h1_v);
    deleteOnDevice(d_h2_v);
    deleteOnDevice(d_res1);
    deleteOnDevice(d_res2);
    deleteOnDevice(d_test_v);
    deleteMatrixOnDevice(q_i, q_j, q_v);
    deleteMatrixOnDevice(a_i, a_j, a_v);

    delete[] h1_v;
    delete[] h2_v;
    delete[] res1;
    delete[] res2;
    delete[] test_v;

    return fails;
}