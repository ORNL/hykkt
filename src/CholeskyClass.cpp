#include "CholeskyClass.hpp"
#include "matrix_vector_ops.hpp"
#include "cuda_memory_utils.hpp"
#include "cusparse_utils.hpp"

#include "cuda_check_errors.hpp"


CholeskyClass::CholeskyClass(int n, 
                             int nnz, 
                             double* a_v, 
                             int* a_i, 
                             int* a_j)
  : n_(n),
    nnz_(nnz),
    a_i_(a_i),
    a_j_(a_j),
    a_v_(a_v)
{
}

  CholeskyClass::~CholeskyClass()
  {
    deleteOnDevice(buffer_gpu_);
    checkCudaErrors(cusolverSpDestroy(handle_cusolver_));
    deleteDescriptor(descr_a_);
    checkCudaErrors(cusolverSpDestroyCsrcholInfo(info_));
  }

// Symbolic factorization of $H_\gamma$ in (6)
  void CholeskyClass::symbolic_analysis()
  {
    createSparseMatDescr(descr_a_);
    
    checkCudaErrors(cusolverSpCreate(&handle_cusolver_));
    checkCudaErrors(cusolverSpCreateCsrcholInfo(&info_));
    
    checkCudaErrors(cusolverSpXcsrcholAnalysis(handle_cusolver_,
          n_, 
          nnz_, 
          descr_a_, 
          a_i_, 
          a_j_, 
          info_));
    size_t internal_data_in_bytes = 0; 
    size_t worksepace_in_bytes = 0;
    checkCudaErrors(cusolverSpDcsrcholBufferInfo(handle_cusolver_, 
          n_, 
          nnz_, 
          descr_a_, 
          a_v_, 
          a_i_,  
          a_j_,
          info_, 
          &internal_data_in_bytes, 
          &worksepace_in_bytes));
    allocateBufferOnDevice(&buffer_gpu_, worksepace_in_bytes);
  }

  void CholeskyClass::set_pivot_tolerance(const double tol)
  {
    tol_ = tol;
  }

  void CholeskyClass::set_matrix_values(double* a_v)
  {
    a_v_ = a_v;
  }

  void CholeskyClass::set_nnz(int nnz)
  {
    nnz_ = nnz;
  }

// Numerical factorization of $H_\gamma$ in (6)
  void CholeskyClass::numerical_factorization()
  {
    int singularity = 0;
    checkCudaErrors(cusolverSpDcsrcholFactor(handle_cusolver_,
          n_,
          nnz_,
          descr_a_,
          a_v_, 
          a_i_, 
          a_j_, 
          info_, 
          buffer_gpu_));
    checkCudaErrors(cusolverSpDcsrcholZeroPivot(handle_cusolver_,
          info_, 
          tol_, 
          &singularity)); 
    if (singularity >= 0) {
      fprintf(stderr, "Error: H not invertible, singularity=%d\n", singularity);
    }
  }
 
//Used for application of $S$ in solving (7) with CG and solving (8)
  void CholeskyClass::solve(double* x, double* b)
  {
    checkCudaErrors(cusolverSpDcsrcholSolve(handle_cusolver_, 
                                            n_, 
                                            b, 
                                            x, 
                                            info_, 
                                            buffer_gpu_));
  }
