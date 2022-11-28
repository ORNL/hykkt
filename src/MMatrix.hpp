#pragma once

struct MMatrix
{

  //constructors
  MMatrix();
  MMatrix(int n, int m, int nnz);
 
  //destructor
  ~MMatrix();
  
  /// @brief Set whether MMatrix object owns matrix data arrays.
  /// @param[in] flag - if true, matrix owns data.
  void owns_csr_data(bool flag);

  /// @brief Returns whether MMatrix object owns matrix data.
  /// @return true if matrix object owns matrix data.
  bool owns_csr_data();

  /**
   * @brief loads structured values of matrix
   *
   * @param n, m - dimensions of matrix
   * nnz - number of nonzeros of matrix 
   *
   * @pre  n, m, nnz are positive integers
   * @post Coordinate format and CSR format for matrix are allocated and
   *       the dimensions and number of nonzeros for CSR formate are set
   */
  void populate(int n, int m, int nnz);
  void print_csr();
  void clear();

  int* coo_rows; // row pointers of coo storage
  int* coo_cols; // column pointers of coo storage
  double* coo_vals; // values of matrix

  int* csr_rows; // row offsets of csr storage
  int* csr_cols; // column pointers of csr storage
  double* csr_vals; // nonzero values of matrix

  int n_; // number of rows
  int m_; // number of cols
  int nnz_; // number of nonzeros
  int nnz_unpacked_; // nnz after unpacking implicit symmetric format

private:
  bool owns_csr_data_;
};

void copy_mmatrix(const MMatrix& src, MMatrix& dst);

void copy_vector(const double* src, double* dst, int n);
