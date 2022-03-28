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
      csrcholInfo_t dH, double* x0, double* b, int n, int m, int nnz, void* buffer_gpu):
      matJC_(matJC), matJCt_(matJCt), dH_(dH), x0_(x0), b_(b), n_(n), m_(m),
      nnz_(nnz), buffer_gpu_(buffer_gpu){}

  // destructor
  ~SchurComplementConjugateGradient();

  // solver API
  int allocate();
  int setup();
  int solve();
  void set_solver_tolerance(double tol)
  {
    tol_ = tol;
  }
  void set_solver_itmax(double itmax)
  {
    itmax_ = itmax;
  }
};

} // namespace hykkt

