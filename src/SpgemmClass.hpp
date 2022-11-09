#pragma once

#include <cusparse.h>
//C = alpha_p_*A*B
//E = alpha_s_*D + beta_s_*C

class SpgemmClass
{
public:
  // constructor
  SpgemmClass(int n,
      int m,
      cusparseHandle_t handle, 
      double alpha_p, 
      double alpha_s, 
      double beta_s); 

  // destructor
  ~SpgemmClass();
  
  /*
   * @brief loads matrices used for SpGEMM 
   *
   * @param[in] a_desc - sparse matrix desriptor for first product matrix
   * @param[in] b_desc - sparse matrix descriptor for second product matrix
   *
   * @post a_desc_ now equal to a_desc, b_desc_ now equal to b_desc
  */
  void load_product_matrices(cusparseSpMatDescr_t a_desc, 
      cusparseSpMatDescr_t b_desc);

  /*
   * @brief loads matrices used for sum E = alpha_s_*D + beta_s_*C
   *
   * @param[in] d_i - row offsets for CSR format of matrix D
   * @param[in] d_j - column pointers for CSR format of matrix D
   * @param[in] d_v - nonzero values for CSR format of matrix D
   * @param[in] nnz_d - number of nonzero values in matrix D
   *
   * @post d_i_ now equals d_i, d_j_ now equals d_j, d_v_ now equals d_v, 
   *       nnz_d_ now equals nnz_d
  */
  void load_sum_matrices(int* d_i, 
      int* d_j, 
      double* d_v,
      int nnz_d);

  /*
   * @brief loads pointers to CSR format of result matrix E
   *
   * @param[in] e_i - a pointer for the row offsets of E
   * @param[in] e_j - a pointer for the column pointers of E
   * @param[in] e_v - a pointer for the nonzero values of E
   * @param[in] nnz_e - a pointer to the number of nonzero values in E
   *
   * @post e_i_ now equals e_i, e_j_ now equals e_j, e_v_ now equals e_v,
   *       nnz_e_ now equals nnz_e
  */
  void load_result_matrix(int** e_i, int** e_j, double** e_v, int* nnz_e);
  
  /*
   * @brief computes SpGEMM and sum, only allocating for product and sum on 
   *        first call of the function
   *
   * @pre all member variables for matrices used in the product and sum CSR
   *      formats are properly initialized using the load functions
   * 
   * @post matrix E is now equal to alpha_s_*alpha_p_*A*B + beta_s_*D 
   *       in CSR format where e_i is the row offsets, e_j is the 
   *       column pointers, e_v is the nonzero values, and nnz_e is 
   *       the number of nonzeros. After the first spGEMM_reuse()
   *       call, spgemm_desc_ contains the information needed to 
   *       repeat the product and sum methods without allocation
  */
  void spGEMM_reuse();

private:

  /*
   * @brief creates matrix descriptors used for SpGEMM and sum and setups
   *        resulting matrices of SpGEMM and sum
   *
   * @post spgemm_desc_ and descr_d_ are now properly initialized matrix
   *       descriptors for executing SpGEMM and sum, c_desc_ is now an empty
   *       matrix in CSR format, the
   *       number of nonzeros in the resulting matrix 
   *       E = alpha_s_*alpha_p_*A*B + beta_s_*D
  */
  void allocate_workspace();
 
  /*
   * @brief allocates memory used for computing SpGEMM
   *
   * @pre variables used in computing SpGEMM are properly initialized using
   *      load_product_matrices() function
   *
   * @post c_desc_ now contains properly allocated CSR pointers c_i_, c_j_,
   *       and c_v_; nnz_c_ contains the nonzeros of the result of SpGEMM;
   *       spgemm_desc_ now contains information so that compute product
   *       function can be repeated without repeated allocation; d_buffer1_ 
   *       and d_buffer2 now have properly allocated memory for computing 
   *       the product
  */
  void allocate_spGEMM_product();
 
  /*
   * @brief computes SpGEMM C = alpha_p_*A*B
   *
   * @pre product matrices properly initialized with load_product_matrices(),
   *      result descriptor c_desc_ and spgemm_desc_ are properly initialized
   *      with allocate_spGEMM_product()
   *
   * @post c_desc_ now has the resulting product alpha_p_*A*B in CSR format 
   *       stored in the pointers c_i_, c_j_, and c_v_
  */
  void compute_spGEMM_product();
 
  /*
   * @brief allocates memory used for computing matrix sum
   *
   * @pre matrices used for matrix sum are properly initialized using
   *      load_sum_matrices()
   *
   * @post CSR format for result matrix E are properly loaded in e_i_,
   *       e_j_, e_v_, and nnz_e_;
   *       buffer_add_ has the proper amount of memory allocated for the sum
  */
  void allocate_spGEMM_sum();

  /*
   * @brief computes sum E = alpha_s_*C + beta_s_*D
   *
   * @pre sum matrices from SpGEMM properly intialized and memory 
   *      allocated for sum
   *
   * @post the sum alpha_s_*C + beta_s_*D is stored in CSR format 
   *       in e_i_, e_j_, e_v_,
  */
  void compute_spGEMM_sum();
  
  // member variables
  cusparseHandle_t handle_; //handle to the cuSPARSE library context
  cusparseSpGEMMDescr_t spgemm_desc_; //descriptor for SpGEMM computation
  cusparseMatDescr_t descr_d_; //descriptor for matrix D used in sum

  bool spgemm_allocated_ = false; //check if allocation needed for spGEMM_reuse

  //dimensions of matrix D in sum alpha_s_*alpha_p_*A*B + beta_s_*D
  int n_;
  int m_; 

  double alpha_p_; //scalar for first product matrix
  double alpha_s_; //scalar for first sum matrix
  double beta_s_; //scalar for second sum matrix
 
  cusparseSpMatDescr_t a_desc_; //matrix descriptor for first product matrix
  cusparseSpMatDescr_t b_desc_; //matrix descriptor for second product matrix
  cusparseSpMatDescr_t c_desc_; //matrix descriptor for product result matrix

  //buffers used for memory use in product computation
  void* d_buffer1_;
  void* d_buffer2_;

  void* buffer_add_; //buffer for matrix sum computation

  int* c_i_; //row offsets of product result matrix
  int* c_j_; //column pointers of product result matrix
  double* c_v_; //nonzero values of product result matrix
  
  int* d_i_; //row offsets for sum matrix D
  int* d_j_; //column pointers for sum matrix D
  double* d_v_; //nonzero values for sum matrix D

  int** e_i_; //pointer to row offsets for result matrix E
  int** e_j_; //pointer to column pointers for result matrix E
  double** e_v_; //pointer to nonzero values for result matrix E

  int64_t nnz_c_; //number of nonzeros in result product matrix
  int nnz_d_; //number of nonzeros in sum matrix D
  int* nnz_e_; //pointer to number of nonzeros in result matrix E
};
