#include "stdio.h"
#include <string>
#include "input_functions.hpp"
#include "vector_vector_ops.hpp"
#include "cuda_check_errors.hpp"
#include "cuda_memory_utils.hpp"
#include "cublas.h"
#include "cusparse.h"
#include "cusparse_utils.hpp"
#include "chrono"
#include "sys/time.h"
#include "matrix_vector_ops.hpp"
#include "sparse_mat_mul.hpp"
#include "cusparse_params.hpp"
#include <benchmark/benchmark.h>
#include <unistd.h>

#if 1
void spmv(cusparseHandle_t& handle, cusparseSpMatDescr_t& a_desc, cusparseDnVecDescr_t& x_desc, cusparseDnVecDescr_t& test_desc, void* buffer, cudaEvent_t& event) {
    checkCudaErrors(cusparseSpMV(handle, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        &ONE, 
                        a_desc, 
                        x_desc,
                        &ZERO, 
                        test_desc, 
                        COMPUTE_TYPE, 
                        CUSPARSE_MV_ALG_DEFAULT, 
                        buffer));
    cudaEventRecord(event);
    cudaEventSynchronize(event);
}

static void BM_spmv(benchmark::State& state) {    
    //*****************************FILE READING********************************//
    std::string a_file_name = "../../src/mats/sandbox/a.mtx";
    std::string x_file_name = "../../src/mats/sandbox/x.mtx";
    std::string y_file_name = "../../src/mats/sandbox/y.mtx";
    char* a_char, *x_char, *y_char;
    strcpy(a_char, a_file_name.c_str());
    strcpy(x_char, x_file_name.c_str());
    strcpy(y_char, y_file_name.c_str());

    int* a_i, *a_j;
    double* a_v, *h_x, *d_x, *h_y, *h_test, *d_test;
    MMatrix mat_a;
    read_mm_file_into_coo(a_char, mat_a, 2);
    coo_to_csr(mat_a);
    double a_nnz = mat_a.nnz_;
    int m = mat_a.n_; //rows
    int n = mat_a.m_; //cols
    h_x = new double[n];
    h_y = new double[m];
    h_test = new double[m];
    d_test = new double[m];
    read_rhs(x_char, h_x);
    read_rhs(y_char, h_y);
    cloneVectorToDevice(n, &h_x, &d_x);
    allocateVectorOnDevice(m, &d_test);
    cloneMatrixToDevice(&mat_a, &a_i, &a_j, &a_v);

    cusparseHandle_t handle;
    createSparseHandle(handle);
    cusparseSpMatDescr_t a_desc;
    cusparseDnVecDescr_t x_desc, test_desc;
    createDnVec(&x_desc, n, d_x);
    createDnVec(&test_desc, m, d_test);
    createCsrMat(&a_desc, m, n, a_nnz, a_i, a_j, a_v);
    cudaEvent_t event;
    cudaEventCreate(&event);
    struct timeval t1, t2;
    double time = 0.0;
//****************************BUFFER ALLOCATION****************************//
    void* buffer;
    bool allocated = false;
    size_t buffer_size = 0;
    checkCudaErrors(cusparseSpMV_bufferSize(handle, 
                                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                        &ONE, 
                                        a_desc, 
                                        x_desc,
                                        &ZERO, 
                                        test_desc, 
                                        COMPUTE_TYPE, 
                                        CUSPARSE_MV_ALG_DEFAULT,
                                        &buffer_size));
    allocateBufferOnDevice(&buffer, buffer_size);

    //for (int i = 0; i < 10; i++) {
        //auto start = std::chrono::high_resolution_clock::now();
        //spmv(handle, a_desc, x_desc, test_desc, buffer, event);
        //auto end = std::chrono::high_resolution_clock::now();
        //printf("Time: %.12f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0);
    //}

    fun_csr_spmv_kernel(m, n, a_j, a_i, a_v, d_x, d_test);

    copyVectorToHost(m, d_test, h_test);
    int fails1 = 0;
    for (int i = 0; i < m; i++) {
        if (abs(h_test[i] - h_y[i]) > 1e-12) {
            printf("i: %d h_test: %.20f h_y: %.20f diff: %.20f\n", i, h_test[i], h_y[i], h_test[i] - h_y[i]);
            fails1++;
        }
    }


    //code below gets timed
    for (auto _ : state) {
        spmv(handle, a_desc, x_desc, test_desc, buffer, event);
    }

    delete[] h_x;
    delete[] h_y;
    delete[] h_test;
    deleteOnDevice(d_x);
    deleteOnDevice(d_test);
    deleteOnDevice(a_i);
    deleteOnDevice(a_j);
    deleteOnDevice(a_v);
    deleteOnDevice(buffer);
    cudaEventDestroy(event);
    deleteHandle(handle);
    deleteDescriptor(x_desc);
    deleteDescriptor(test_desc);
    deleteDescriptor(a_desc);
}
BENCHMARK(BM_spmv);

BENCHMARK_MAIN();
#else
int main(int argc, char* argv[]) {
    if (argc != 8) {
        printf("%d arguments provided\n", argc);
        printf("Incorrect number of inputs. Exiting...\n");
        return -1;
    }

    char* v1_file_name = argv[1];
    char* v2_file_name = argv[2];
    int n = std::stoi(argv[3]);
    double expected = std::stod(argv[4]);

    int k = 1000;
    double* h_w = new double[k]{1.0};
    double* d_w;
    cloneVectorToDevice(k, &h_w, &d_w);

    double* h_v1 = new double[n];
    double* h_v2 = new double[n];
    double* d_v1;
    double* d_v2;

    read_rhs(v1_file_name, h_v1);
    read_rhs(v2_file_name, h_v2);
    cloneVectorToDevice(n, &h_v1, &d_v1);
    cloneVectorToDevice(n, &h_v2, &d_v2);

    //double res;
    //double* h_res;
    //double* d_res;
    //allocateValueOnDevice(&d_res);

    double* d_test;
    allocateValueOnDevice(&d_test);

    struct timeval t1, t2;
    double time = 0.0;

    double* div;
    allocateValueOnDevice(&div);
    double res1;
    
    cudaSetDeviceFlags(cudaDeviceMapHost);
    
    volatile double* d_res;
    volatile double* h_res;
    cudaHostAlloc((void**)&h_res, sizeof(double), cudaHostAllocMapped);
    cudaHostGetDevicePointer((double **)&d_res, (double *)h_res, 0);

    cudaEvent_t event;
    cudaEventCreate(&event);
        deviceDotProduct(n, d_v1, d_v2, (double *)d_res);

    /*for (int i = 0; i < 1; i++) {
        gettimeofday(&t1, NULL);
        deviceDotProduct(n, d_v1, d_v2, (double *)d_res);
        cudaEventRecord(event);
        cudaEventSynchronize(event);
        gettimeofday(&t2, NULL);
        time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("\nTime: %.12f\n", time);
        //cudaDeviceSynchronize();
        //copyVectorToHost(1, d_res, &res1);
        //copyDeviceVector(1, d_res, d_test);


        printf("%.12f\n", *h_res);

    }*/
    deviceDotProduct(k, d_w, d_w, (double*)d_test);
    cudaEventRecord(event);
    cudaEventSynchronize(event);
    gettimeofday(&t1, NULL);
    deviceDotProduct(k, d_w, d_w, (double*)d_test);
    cudaEventRecord(event);
    cudaEventSynchronize(event);
    gettimeofday(&t2, NULL);
    time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("\nTime: %.12f\n", time);

    //copyVectorToHost(1, d_res, &res1);

    //fun_divide(d_test, (double *) d_res, div);

    //cudaDeviceSynchronize();

    //displayDeviceVector(div, 1, 0, 1, "div");

    double res2;
    cublasHandle_t handle;
    createCublasHandle(handle);
    dotProduct(handle, n, d_v1, d_v2, &res2);

    gettimeofday(&t1, NULL);
    dotProduct(handle, k, d_w, d_w, &res2);
    gettimeofday(&t2, NULL);
    time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("cublasDotProduct: %.12f\n", time);

    int fails = 0;

    if(abs(*h_res - expected) > 1e-12) {
        printf("RESULT1: %f EXPECTED: %f DIFFERENCE: %f\n", res1, expected, res1 - expected);
        fails++;
    }
/*
    if(abs(res2 - expected) > 1e-12) {
        printf("RESULT2: %f EXPECTED: %f DIFFERENCE: %f\n", res2, expected, res2 - expected);
        fails++;
    }

    delete[] h_v1;
    delete[] h_v2;
    deleteOnDevice(d_v1);
    deleteOnDevice(d_v2);
    deleteHandle(handle);
    checkCudaErrors(cudaHostUnregister(h_res));
*/
    return fails;
}
#endif