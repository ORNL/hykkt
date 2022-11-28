#include <iostream>
#include <iomanip>
#include <cassert>
#include "MMatrix.hpp"

MMatrix::MMatrix()
  : n_(0),
    m_(0),
    nnz_(0),
    owns_csr_data_(true)
{
  coo_rows = nullptr;
  coo_cols = nullptr;
  coo_vals = nullptr;
  csr_rows = nullptr;
  csr_cols = nullptr;
  csr_vals = nullptr;
}

MMatrix::MMatrix(int n, int m, int nnz) 
  : n_(n),
    m_(m),
    nnz_(nnz),
    owns_csr_data_(true)
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

  if(owns_csr_data_)
  {
    delete [] csr_rows;
    delete [] csr_cols;
    delete [] csr_vals;
  }
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

void MMatrix::print_csr()
{
  std::cout << "Printing " << n_ << "x" << m_ << " matrix with " 
            << nnz_ << " nonzeros:\n";
  for(int i=0; i<nnz_; ++i)
  {
    if(i < n_ + 1)
      std::cout << std::setw(10) << csr_rows[i] << " ";
    std::cout << std::setw(10) << csr_cols[i] << " "
              << std::setw(10) << csr_vals[i] 
              << std::endl;
  }
}

void MMatrix::clear()
{
  if(coo_cols)
  {
    delete [] coo_cols;
    coo_cols = nullptr;
  }
  if(coo_rows)
  {
    delete [] coo_rows;
    coo_rows = nullptr;
  }
  if(coo_vals)
  {
    delete [] coo_vals;
    coo_vals = nullptr;
  }

  if(owns_csr_data_)
  {
    if(csr_cols)
    {
      delete [] csr_cols;
      csr_cols = nullptr;
    }
    if(csr_rows)
    {
      delete [] csr_rows;
      csr_rows = nullptr;
    }
    if(csr_vals)
    {
      delete [] csr_vals;
      csr_vals = nullptr;
    }
  }
  
  n_   = 0;
  m_   = 0;
  nnz_ = 0;
}

void MMatrix::owns_csr_data(bool flag)
{
  owns_csr_data_ = flag;
}

bool MMatrix::owns_csr_data()
{
  return owns_csr_data_;
}

void copy_mmatrix(const MMatrix& src, MMatrix& dst)
{
  assert(src.n_ == dst.n_);
  assert(src.m_ == dst.m_);
  assert(src.nnz_ == dst.nnz_);

  for(int i=0; i<(dst.n_ + 1); ++i)
    dst.csr_rows[i] = src.csr_rows[i];

  for(int i=0; i<dst.nnz_; ++i)
  {
    dst.csr_cols[i] = src.csr_cols[i];
    dst.csr_vals[i] = src.csr_vals[i];
  }
}

void copy_vector(const double* src, double* dst, int n)
{
  for(int i=0; i<n; ++i)
    dst[i] = src[i];
}
