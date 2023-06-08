#include <benchmark/benchmark.h>
#include "input_functions.hpp"
#include "cuda_memory_utils.hpp"
#include "Models/BiELLMat.hpp"

void func()
{

}

static void BM_func(benchmark::State& state) 
{
    const int i = state.range(0);
    std::string a_file_name = "../../src/mats/BiELL/a" + std::to_string(i) + ".mtx";
    std::string x_file_name = "../../src/mats/BiELL/x" + std::to_string(i) + ".mtx";
    const char* a_char = a_file_name.c_str();
    const char* x_char = x_file_name.c_str();

    int* a_i;
    int* a_j;
    double* a_v;
    double* h_x;
    double* d_x;

    MMatrix mat_a;
    read_mm_file_into_coo(a_char, mat_a, 2, false);
    coo_to_csr(mat_a);
    int a_nnz = mat_a.nnz_;
    int m = mat_a.n_; //rows
    int n = mat_a.m_; //cols
    h_x = new double[n];
    read_rhs(x_char, h_x);
    cloneVectorToDevice(n, &h_x, &d_x);
    cloneMatrixToDevice(&mat_a, &a_i, &a_j, &a_v);

    BiELLMat<double> biell_a(a_i, a_j, a_v);

    for (auto _ : state) {
        func();
    }

    delete[] h_x;
    deleteOnDevice(d_x);
    deleteMatrixOnDevice(a_i, a_j, a_v);
}

BENCHMARK(BM_func)->Arg(1);

BENCHMARK_MAIN();