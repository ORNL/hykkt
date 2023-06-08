#include <stdio.h>
#include "LogBarrierSolver.hpp"
#include "input_functions.hpp"
#include "cuda_memory_utils.hpp"
#include "cuda_check_errors.hpp"
#include "vector_vector_ops.hpp"
#include "log_barrier_utils.hpp"
#include "chrono"

/**
 * @brief Driver file demonstrates Log Barrier method
*/
int main(int argc, char* argv[])
{
    if(argc != 4) {
        printf("Incorrect number of inputs. Exiting ...\n");
        return -1;
    }
    
    //int q_nnz
    int a_nnz, m, n;

    double* d_x_test;
    double* h_x_test;

    /*int* q_i = nullptr;
    int* q_j = nullptr;
    double* q_v = nullptr;*/

    int* a_i = nullptr;
    int* a_j = nullptr;
    double* a_v = nullptr;

    double* d_b = nullptr;
    double* h_b = nullptr;
    double* d_c = nullptr;
    double* h_c = nullptr;

    //char* q_file_name = nullptr;
    char* a_file_name = nullptr;
    char* b_file_name = nullptr;
    char* c_file_name = nullptr;
    //***************************FILE READING**************************//
    //q_file_name = argv[1];
    a_file_name = argv[1];
    b_file_name = argv[2];
    c_file_name = argv[3];

    //MMatrix mat_q = MMatrix();
    MMatrix mat_a = MMatrix();

    //read_mm_file_into_coo(q_file_name, mat_q, 2);
    //sym_coo_to_csr(mat_q);
    //q_nnz = mat_q.nnz_;
    
    read_mm_file_into_coo(a_file_name, mat_a, 2);
    coo_to_csr(mat_a);
    a_nnz = mat_a.nnz_;

    m = mat_a.n_; //rows
    n = mat_a.m_; //cols

    h_b = new double[m];
    h_c = new double[n];
    read_rhs(b_file_name, h_b);
    read_rhs(c_file_name, h_c);

    printf("File reading completed ..........................\n");
    //**************************MEMORY COPYING*************************//
    cusparseHandle_t cusparse_handle = NULL;
    createSparseHandle(cusparse_handle);
    //cloneMatrixToDevice(&mat_q, &q_i, &q_j, &q_v);
    cloneMatrixToDevice(&mat_a, &a_i, &a_j, &a_v);
    cloneVectorToDevice(m, &h_b, &d_b);
    cloneVectorToDevice(n, &h_c, &d_c);
    h_x_test = new double[n]{0.0};
    cloneVectorToDevice(n, &h_x_test, &d_x_test);
    //*************************RUNNING SOLVER**************************//
    int fails = 0;
    bool display_info = true;
    //LogBarrierInfo info(m, n, q_nnz, q_i, q_j, q_v, a_nnz, a_i, a_j, a_v, d_b, d_c, cusparse_handle);
    LogBarrierInfo info(m, n, a_nnz, a_i, a_j, a_v, d_b, d_c, cusparse_handle);
    LogBarrierSolver solver;
    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(info, d_x_test, display_info);
    printf("Time to solve: %f\n", std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count());
    //*************************TESTING SOLVER**************************//
    copyVectorToHost(n, d_x_test, h_x_test);
    //**************************FREEING MEMORY*************************//
    deleteHandle(cusparse_handle);
    //deleteMatrixOnDevice(q_i, q_j, q_v);
    deleteMatrixOnDevice(a_i, a_j, a_v);
    deleteOnDevice(d_b);
    deleteOnDevice(d_c);
    deleteOnDevice(d_x_test);
    
    delete[] h_b;
    delete[] h_c;
    delete[] h_x_test;

    return fails;
}