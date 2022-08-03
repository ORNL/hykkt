#pragma once
#include "MMatrix.hpp"

struct indexPlusValue
{
  double value;
  int    idx;
};

/*
 * @brief compares indeces of values for use in sorting methods
 *
 * @param a - first indexPlusValue to be compared
 * b - second indexPlusValue to be compared
 * @return -1 if a->idx < b->idx, 
 *         1 if a->idx > b->idx,
 *         0 if a->idx == b->idx
*/
static int indexPlusValue_comp(const void* a, const void* b);

typedef struct
{
  int*    coo_rows;
  int*    coo_cols;
  double* coo_vals;

  int*    csr_rows;
  int*    csr_cols;
  double* csr_vals;

  int n;
  int m;
  int nnz;
  int nnz_unpacked;
} mmatrix;

/*
 * @brief reads a matrix stored in matrix_file_name into a
 *        structure mat_a of COO format
 * 
 * @param matrix_file_name - name of file holding matrix to be read
 * mat_a - MMatrix structure to be written to
 * lines - number of lines in file to be read
 *
 * @pre matrix_file_name is a valid file
 * @post mat_a now holds the values for the loaded matrix using COO format
*/
void read_mm_file_into_coo(const char* matrix_file_name, 
                           MMatrix* mat_a, 
                           int lines);

/*
 * @brief takes matrix stored in implicit symmetric format COO mat_a and fills 
 *        out the entries and converts it to CSR format within the structure
 * 
 * @param mat_a - MMatrix structure where CSR format will be created
 *
 * @pre matrix is already in COO format in mat_a
 * @post matrix is now also stored in MMatrix in CSR format
*/
void sym_coo_to_csr(MMatrix* mat_a);

/*
 * @brief takes a matrix stored in COO format in mat_a and fills out the entries
 *        and converts it to csr format within the structure
 *
 * @param mat_a - MMatrid structure where CSR format will be created
 *
 * @pre matrix is already in COO format in mat_a
 * @post matrix is now also stored in MMatrix in CSR format
*/
void coo_to_csr(MMatrix*);

/*
 * @brief reads a 1-index based permutation array stored in rhs_file_name into
 *        an array rhs that is 0 index based
 * 
 * @param rhs_file_name - name of permutation array file that is being read
 * rhs - array where the permutation is stored as 0 index based
 *
 * @pre rhs_file_name is a valid file
 * @post rhs now stores the permutation array as a 0 index based vector
*/
void read_1idx_perm(const char* rhs_file_name, int* rhs);

/*
 * @brief reads vector stored in rhs_file_name into an array rhs
 *
 * @param rhs_file_name - name of vector file that is being read
 * rhs - array where the vector will be stored
 *
 * @pre rhs_file_name is a valid file
 * @post rhs now stores the vector from the file
*/
void read_rhs(const char* rhs_file_name, double* rhs);

