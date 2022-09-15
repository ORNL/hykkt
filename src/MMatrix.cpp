#include "MMatrix.hpp"

MMatrix::MMatrix(int n, int m, int nnz) 
  : n_(n),
  m_(m),
  nnz_(nnz)
{
    coo_rows = new int[nnz];
    coo_cols = new int[nnz];
    coo_vals = new double[nnz];
    csr_rows = new int[n + 1];
    csr_cols = new int[nnz];
    csr_vals = new double[nnz];
}

MMatrix::~MMatrix()
{
    delete [] coo_rows;
    delete [] coo_cols;
    delete [] coo_vals;
    delete [] csr_rows;
    delete [] csr_cols;
    delete [] csr_vals;
}
  
void MMatrix::populate(int n, int m, int nnz)
{
  n_ = n;
  m_ = m;
  nnz_= nnz;

  coo_rows = new int[nnz];
  coo_cols = new int[nnz];
  coo_vals = new double[nnz];

  csr_rows = new int[n + 1];
  csr_cols = new int[nnz];
  csr_vals = new double[nnz];
}
