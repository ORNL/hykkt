// @file SchurComplementConjugateGradient.hpp

// forward declaration of cusolverSpHandle_t

// struct cusolverSpHandle_t; //not sure what this is for

namespace hykkt {

class SchurComplementConjugateGradient()
{
public:
  // default constructor
  SchurComplementConjugateGradient();

  // parametrized constructor
  SchurComplementConjugateGradient(cusparseSpMatDescr_t matJC, cusparseSpMatDescr_t matJCt, 
      csrcholInfo_t dH, double* x0, double* b, int n, int m, int nnz, void* buffer_gpu);

  // destructor
  ~SchurComplementConjugateGradient();

  // solver API
  int allocate();
  int setup();
  int solve();
  void set_solver_tolerance(double tol);
  void set_solver_itmax(double itmax);

private:
  // member variables
  int itmax_=100, n_, m_, nnz_;
  double tol_=1e-12;

  cusparseSpMatDescr_t matJC_; 
  cusparseSpMatDescr_t matJCt_; 
  csrcholInfo_t dH_;
  double* x0;
  double* b;
  void* buffer_gpu_;

  cusolverSpHandle_t handle_cusolver_;
  cusolverHandle_t handle;
  cublasHandle_t handle_cublas;

  double               one      = 1.0;
  double               zero     = 0.0;
  double               minusone = -1.0;

  cusparseDnVecDescr_t vecx     = NULL;
  cusparseDnVecDescr_t vecb = NULL;
  cusparseDnVecDescr_t vecy = NULL;
  cusparseDnVecDescr_t vecz = NULL;
  cusparseDnVecDescr_t vecr = NULL;
  cusparseDnVecDescr_t vecw = NULL;
  cusparseDnVecDescr_t vecp = NULL;
  cusparseDnVecDescr_t vecs = NULL;
 
  double gam_i, beta = 0, delta, alpha, minalpha, gam_i1;
  double* ycp, *y, *z, *r, *p, *s, *w;
  
  double timeIO = 0.0;
  struct timeval t1, t2;

};

} // namespace hykkt

