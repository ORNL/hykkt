#pragma once

struct MMatrix
  {

  MMatrix() = default;

  //constructor
  MMatrix(int n, int m, int nnz) 
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
 
  //destructor
  ~MMatrix()
  {
    delete [] coo_rows;
    delete [] coo_cols;
    delete [] coo_vals;
    delete [] csr_rows;
    delete [] csr_cols;
    delete [] csr_vals;
  }
  
  /*
   * @brief loads structured values of matrix
   *
   * @param n,m - dimensions of matrix
   * nnz - number of nonzeros of matrix 
   *
   * @pre 
   * @post Coordinate format and CSR format for matrix are allocated and
   *       the dimensions and number of nonzeros for CSR formate are set
  */
  void populate(int n, int m, int nnz)
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

  int* coo_rows; // row pointers of coo storage
  int* coo_cols; // column pointers of coo storage
  double* coo_vals; // values of matrix

  int* csr_rows; // row offsets of csr storage
  int* csr_cols; // column pointers of csr storage
  double* csr_vals; // nonzero values of matrix

  int n_; // number of rows
  int m_; // number of cols
  int nnz_; // number of nonzeros
  int nnz_unpacked_;
};
