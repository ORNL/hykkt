#pragma once

class RuizClass
{
public:

  // constructor
  RuizClass(int ruiz_its, int n, int totn);

  // destructor
  ~RuizClass();

/*
 * @brief loads CSR format of matrix H
 *
 * @param[in] h_v - nonzero values for CSR format of matrix H
 * @param[in] h_i - row offsets for CSR format of H
 * @param[in] h_j - column pointers for CSR format of H
 *
 * @post h_v_ set to h_v, h_i_ set to h_i, h_j_ set to h_j
*/
  void add_block11(double* h_v, int* h_i, int* h_j);

/*
 * @brief loads CSR format of matrix J tranpose
 *
 * @param[in] jt_v - nonzero values for CSR format of matrix J tranpose 
 * @param[in] jt_i - row offsets for CSR format of J tranpose 
 * @param[in] jt_j - column pointers for CSR format of J tranpose
 *
 * @post jt_v_ set to jt_v, jt_i_ set to jt_i, jt_j_ set to jt_j
*/
  void add_block12(double* jt_v, int* jt_i, int* jt_j);

/*
 * @brief loads CSR format of matrix J
 *
 * @param[in] j_v - nonzero values for CSR format of matrix J
 * @param[in] j_i - row offsets for CSR format of J
 * @param[in] j_j - column pointers for CSR format of J
 *
 * @post j_v_ set to j_v, j_i_ set to j_i, j_j_ set to j_j
*/
  void add_block21(double* j_v, int* j_i, int* j_j);

/*
 * @brief Set values of right hand side of 2x2 system
 *
 * @param[in] rhs1 - first vector of rhs
 * @param[in] rhs2 - second vector of rhs
 *
 * @post rhs1_ set to rhs1, rhs2_ set to rhs2
*/
  void add_rhs1(double* rhs1);
  
  void add_rhs2(double* rhs2);

/*
 * @brief implements Ruiz scaling and aggregates scaling in max_d
 *
 * @pre Member variables n_, totn_, h_v_, h_i_, h_j_, j_v_, j_i_, j_j_,
 *      jt_v_, jt_i_, jt_j_, rhs1_, rhs2_, scale_, max_d have
 *      been initialized to the size of matrix H, the number of rows in
 *      the matrix, the CSR format of H, the CSR format of J, the CSR
 *      format of Jt, the right hand side of the 2x2 system, a scaling vector 
 *      of size totn_, and a scaling vector of size totn containing all 1s
 *
 * @post max_d_ is now the aggregated ruiz scaling of the 2x2 system
*/
  void ruiz_scale();

/*
 * @brief returns scaling pointer
 * 
 * @param[out] - max_d_ the aggregated scaling vector
 *
 * @return max_d_ - the aggregated scaling vector after iterations of scaling
*/
  double* get_max_d() const;

  void reset_scaling();
private:

/*
 * @brief allocates scaling vectors on device
 *
 * @pre Member variable totn_ intialized to the number of rows in the matrix
 * @post scale_ and max_d allocated on the device with size totn_,
 *       max_d_ scaling vector initalized to 1
*/
  void allocate_workspace();

  int ruiz_its_; // number of iterations of scaling
  int n_; // size n of the matrix H
  int totn_; // total rows in matrix

  // H is block 1,1 of 2x2 system
  double* h_v_; // nonzero values of csr storage of H 
  int* h_i_; // row offsets of csr storage of H
  int* h_j_; // column pointers of csr storage of H

  // J is block 2,1 of 2x2 system
  double* j_v_; // nonzero values of csr storage of J
  int* j_i_; // row offsets of csr storage of J
  int* j_j_; // column pointers of csr storage of J

  // J tranpose is block 1,2 of 2x2 system
  double *jt_v_; // nonzero values of csr storage of J tranpose
  int* jt_i_; // row offsets of csr storage of J tranpose 
  int* jt_j_; // column pointers of csr storage of J tranpose

  // right hand side of 2x2 system
  double* rhs1_; // first vector in rhs
  double* rhs2_; // second vector in rhs

  double* scale_; // scaling vector representing diagonal matrix
  double* max_d_; // aggregate of Ruiz scaling iterations
};

