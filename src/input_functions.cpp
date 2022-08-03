#include <stdio.h>
#include <assert.h>

#include "input_functions.hpp"
#include "MMatrix.hpp"

static int indexPlusValue_comp(const void* a, const void* b)
{
  const struct indexPlusValue* da = (indexPlusValue*) a;
  const struct indexPlusValue* db = (indexPlusValue*) b;

  return da->idx < db->idx ? -1 : da->idx > db->idx;
}

void read_mm_file_into_coo(const char* matrix_file_name, 
                           MMatrix* mat_a, 
                           int lines)
{
  // this reads triangular matrix but expands into full as it goes (important)
  // lines indicates the number of the first useful line
  FILE* fpm = fopen(matrix_file_name, "r");

  assert(fpm != NULL);
  char line_buffer[256];
  for(int i = 0; i < lines; i++){
    fgets(line_buffer, sizeof(line_buffer), fpm);
  }
  
  // first line is size and nnz, need this info to allocate memory
  sscanf(line_buffer, "%ld %ld %ld", &(mat_a->n_), &(mat_a->m_), &(mat_a->nnz_));
  // allocate
  printf("allocating COO structures %d %d %d\n", mat_a->n_, mat_a->m_, mat_a->nnz_);

  mat_a->coo_vals = new double[mat_a->nnz_];
  mat_a->coo_rows = new int[mat_a->nnz_];
  mat_a->coo_cols = new int[mat_a->nnz_]; 

  // read
  int r;
  int c;
  double val;
  int    i = 0;
  while(fgets(line_buffer, sizeof(line_buffer), fpm) != NULL) {
    sscanf(line_buffer, "%d %d %lf", &r, &c, &val);
    mat_a->coo_rows[i] = r - 1;
    mat_a->coo_cols[i] = c - 1;
    mat_a->coo_vals[i] = val;
    i++;
  }
  fclose(fpm);
}

void sym_coo_to_csr(MMatrix* mat_a)
{
  // this is diffucult
  // first, decide how many nnz we have in each row
  int* nnz_counts = new int[mat_a->n_]{0};
  int nnz_unpacked = 0;
  for(int i = 0; i < mat_a->nnz_; ++i) {
    nnz_counts[mat_a->coo_rows[i]]++;
    nnz_unpacked++;
    if(mat_a->coo_rows[i] != mat_a->coo_cols[i]) {
      nnz_counts[mat_a->coo_cols[i]]++;
      nnz_unpacked++;
    }
  }
  // allocate full CSR structure
  mat_a->nnz_unpacked_    = nnz_unpacked;
  mat_a->csr_vals         = new double[mat_a->nnz_unpacked_];
  mat_a->csr_cols         = new int[mat_a->nnz_unpacked_];
  mat_a->csr_rows         = new int[(mat_a->n_) + 1];
  indexPlusValue* tmp = new indexPlusValue [mat_a->nnz_unpacked_];
  // create IA (row starts)
  mat_a->csr_rows[0] = 0;
  for(int i = 1; i < mat_a->n_ + 1; ++i)
  {
    mat_a->csr_rows[i] = mat_a->csr_rows[i - 1] + nnz_counts[i - 1];
  }

  int* nzz_shifts = new int[mat_a->n_]{0};
  int  r;
  int  start;

  for(int i = 0; i < mat_a->nnz_; ++i) {
    // which row
    r     = mat_a->coo_rows[i];
    start = mat_a->csr_rows[r];
    if((start + nzz_shifts[r]) > mat_a->nnz_unpacked_) {
      printf("index out of bounds\n");
    }
    tmp[start + nzz_shifts[r]].idx   = mat_a->coo_cols[i];
    tmp[start + nzz_shifts[r]].value = mat_a->coo_vals[i];
    nzz_shifts[r]++;
    if(mat_a->coo_rows[i] != mat_a->coo_cols[i]) {
      r     = mat_a->coo_cols[i];
      start = mat_a->csr_rows[r];
      if((start + nzz_shifts[r]) > mat_a->nnz_unpacked_) {
        printf("index out of bounds 2\n");
      }
      tmp[start + nzz_shifts[r]].idx   = mat_a->coo_rows[i];
      tmp[start + nzz_shifts[r]].value = mat_a->coo_vals[i];
      nzz_shifts[r]++;
    }
  }
  
  for(int i = 0; i < mat_a->n_; ++i) {
    int col_start = mat_a->csr_rows[i];
    int col_end   = mat_a->csr_rows[i + 1];
    int length   = col_end - col_start;
    qsort(&tmp[col_start], length, sizeof(indexPlusValue), indexPlusValue_comp);
  }

  for(int i = 0; i < mat_a->nnz_unpacked_; ++i) {
    mat_a->csr_cols[i]   = tmp[i].idx;
    mat_a->csr_vals[i] = tmp[i].value;
  }
  mat_a->nnz_ = mat_a->nnz_unpacked_;
  delete [] nnz_counts;
}

void coo_to_csr(MMatrix* mat_a)
{
  // this is diffucult
  // first, decide how many nnz we have in each row
  int* nnz_counts = new int[mat_a->n_]{0};
  for(int i = 0; i < mat_a->nnz_; ++i) {
    nnz_counts[mat_a->coo_rows[i]]++;
  }
  // allocate full CSR structure
  mat_a->csr_rows     = new int[(mat_a->n_) + 1];
  indexPlusValue* tmp = new indexPlusValue [mat_a->nnz_];
  // create IA (row starts)
  mat_a->csr_rows[0] = 0;
  for(int i = 1; i < mat_a->n_ + 1; ++i) {
    mat_a->csr_rows[i] = mat_a->csr_rows[i - 1] + nnz_counts[i - 1];
  }

  int* nzz_shifts = new int[mat_a->n_]{0};
  int  r;
  int  start;

  for(int i = 0; i < mat_a->nnz_; ++i) {
    // which row
    r     = mat_a->coo_rows[i];
    start = mat_a->csr_rows[r];
    if((start + nzz_shifts[r]) > mat_a->nnz_)
      printf("index out of boubds\n");
    tmp[start + nzz_shifts[r]].idx   = mat_a->coo_cols[i];
    tmp[start + nzz_shifts[r]].value = mat_a->coo_vals[i];

    nzz_shifts[r]++;
  }
  // now sort whatever is inside rows

  for(int i = 0; i < mat_a->n_; ++i) {
    // now sorting (and adding 1)
    int col_start = mat_a->csr_rows[i];
    int col_end   = mat_a->csr_rows[i + 1];
    int length    = col_end - col_start;
    qsort(&tmp[col_start], length, sizeof(indexPlusValue), indexPlusValue_comp);
  }

  // and copy
  for(int i = 0; i < mat_a->nnz_; ++i) {
    mat_a->coo_cols[i] = tmp[i].idx;
    mat_a->coo_vals[i] = tmp[i].value;
  }
  delete [] nnz_counts;
}

void read_1idx_perm(const char* rhs_file_name, int* rhs)
{
  FILE* fpr = fopen(rhs_file_name, "r");
  char  line_buffer[256];
  fgets(line_buffer, sizeof(line_buffer), fpr);
  fgets(line_buffer, sizeof(line_buffer), fpr);
  int n;
  int m;
  sscanf(line_buffer, "%ld %ld", &n, &m);
  printf("N = %d, m=%d\n", n, m);
  int i = 0;
  int val;

  while(fgets(line_buffer, sizeof(line_buffer), fpr) != NULL) {
    sscanf(line_buffer, "%ld", &val);
    rhs[i] = val - 1;
    i++;
  }
  fclose(fpr);
}

void read_rhs(const char* rhs_file_name, double* rhs)
{
  FILE* fpr = fopen(rhs_file_name, "r");
  char  line_buffer[256];

  fgets(line_buffer, sizeof(line_buffer), fpr);
  fgets(line_buffer, sizeof(line_buffer), fpr);
  int n, m;
  sscanf(line_buffer, "%ld %ld", &n, &m);
  int    i = 0;
  double val;
  // allocate

  while(fgets(line_buffer, sizeof(line_buffer), fpr) != NULL) {
    sscanf(line_buffer, "%lf", &val);
    rhs[i] = val;
    i++;
  }
  fclose(fpr);
}

