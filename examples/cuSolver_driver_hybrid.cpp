#include <iostream>
#include <string>
#include "input_functions.hpp"
#include "SchurComplementConjugateGradient.hpp"
#include "RuizClass.hpp"
#include "matrix_vector_ops.hpp"
#include "vector_vector_ops.hpp"
#include "MMatrix.hpp"
#include "CholeskyClass.hpp"
#include "PermClass.hpp"
#include "SpgemmClass.hpp"
#include "cuda_memory_utils.hpp"

// this version reads NORMAL mtx matrices; dont have to be sorted.
/**
 * @brief Driver file demonstrates hybrid use of Cholesky
 * decomposition, Schur complement, Ruiz scaling,
 * and Permutation using the respective classes.
 * 
 * @pre Only NORMAL mtx matrices are read; don't have to be sorted
 * 
 */

int main(int argc, char* argv[])
{
  if(argc != 11)
  {
    std::cout << "Incorrect number of inputs. Exiting ...\n";
    return -1;
  }

  char const* const h_file_name  = argv[1];
  char const* const ds_file_name = argv[2];
  char const* const jc_file_name = argv[3];
  char const* const jd_file_name = argv[4];

  // Get rhs block files
  char const* const rx_file_name  = argv[5];
  char const* const rs_file_name  = argv[6];
  char const* const ry_file_name  = argv[7];
  char const* const ryd_file_name = argv[8];
 // char const* const permFileName = argv[11];
  int skip_lines = atoi(argv[9]);
  double gamma   = atof(argv[10]);

  // Start of block: reading matrices from files and allocating structures for
  // them, to be replaced by HiOp structures
  const int ruiz_its = 2;
  const double norm_tol = 1e-2;
  const double tol = 1e-12;
  /*** cuda stuff ***/

  cusparseHandle_t handle = NULL;
  createSparseHandle(handle);
 
  cusparseMatDescr_t descr_a;
  createSparseMatDescr(descr_a);
  
  cublasHandle_t handle_cublas;
  createCublasHandle(handle_cublas);

  // Get matrix block files
  // Matix structure allocations
  // Start block
  MMatrix mat_h = MMatrix();
  MMatrix mat_ds = MMatrix();
  MMatrix mat_jc = MMatrix();
  MMatrix mat_jd = MMatrix();
  // Vector allocations
  double* rx{nullptr};
  double* rs{nullptr};
  double* ry{nullptr};
  double* ryd{nullptr};

  // read matrices
  read_mm_file_into_coo(h_file_name, mat_h, skip_lines);
  sym_coo_to_csr(mat_h);

  read_mm_file_into_coo(ds_file_name, mat_ds, skip_lines);
  coo_to_csr(mat_ds);

  read_mm_file_into_coo(jc_file_name, mat_jc, skip_lines);
  coo_to_csr(mat_jc);

  read_mm_file_into_coo(jd_file_name, mat_jd, skip_lines);
  coo_to_csr(mat_jd);
  int jd_flag = (mat_jd.nnz_ > 0);
  // read right hand side
  rx = new double[mat_h.n_];
  read_rhs(rx_file_name, rx);
  rs = new double[mat_ds.n_];
  read_rhs(rs_file_name, rs);
  ry = new double[mat_jc.n_];
  read_rhs(ry_file_name, ry);
  ryd = new double[mat_jd.n_];
  read_rhs(ryd_file_name, ryd);
 
  // now copy data to GPU and format convert
  double* d_rx{nullptr};
  double* d_rs{nullptr};
  double* d_ry{nullptr};
  double* d_ry_c{nullptr};
  double* d_ryd{nullptr};
  double* d_ryd_s{nullptr};
  double* h_v{nullptr};
  double* ds_v{nullptr};
  double* jc_v{nullptr};
  double* jd_v{nullptr};
  double* jd_vs{nullptr};

  // columns and rows of H, JC, JD
  int* h_j{nullptr};
  int* h_i{nullptr};
  int* jc_j{nullptr};
  int* jc_i{nullptr};
  int* jd_j{nullptr};
  int* jd_i{nullptr};   
  
  // allocate space for rhs and copy it to device
  cloneVectorToDevice(mat_h.n_, &rx, &d_rx);
  cloneVectorToDevice(mat_ds.n_, &rs, &d_rs);
  cloneVectorToDevice(mat_jc.n_, &ry, &d_ry);
  cloneVectorToDevice(mat_jd.n_, &ryd, &d_ryd);

  allocateVectorOnDevice(mat_jd.n_, &d_ryd_s);

  cloneDeviceVector(mat_jc.n_, &d_ry, &d_ry_c);

  //allocate space for matrix and copy it to device
  cloneSymmetricMatrixToDevice(&mat_h, &h_i, &h_j, &h_v);
  cusparseSpMatDescr_t h_desc;
  createCsrMat(&h_desc, 
      mat_h.n_, 
      mat_h.m_, 
      mat_h.nnz_, 
      h_i, 
      h_j, 
      h_v);

  cloneVectorToDevice(mat_ds.nnz_, &mat_ds.coo_vals, &ds_v);
  cloneMatrixToDevice(&mat_jc, &jc_i, &jc_j, &jc_v);

  if(jd_flag)
  {
    allocateVectorOnDevice(mat_jd.nnz_, &jd_vs);
    cloneMatrixToDevice(&mat_jd, &jd_i, &jd_j, &jd_v);
  }
  // malloc initial guess (potentially supplied by HiOp)
  // could change at each iteration, but might only happen once
  double* h_x  = new double[mat_h.m_];
  double* h_s  = new double[mat_ds.m_];
  double* h_y  = new double[mat_jc.n_];
  double* h_yd = new double[mat_jd.n_];

  double* d_x{nullptr};
  double* d_s{nullptr};
  double* d_y{nullptr};
  double* d_yd{nullptr};

  for(int i = 0; i < mat_h.m_; i++)
  {
    h_x[i] = 0;
  }

  for(int i = 0; i < mat_ds.m_; i++)
  {
    h_s[i] = 0;
  }

  for(int i = 0; i < mat_jc.n_; i++)
  {
    h_y[i] = 0;
  }

  for(int i = 0; i < mat_jd.n_; i++)
  {
    h_yd[i] = 0;
  }
  
  cloneVectorToDevice(mat_h.m_, &h_x, &d_x);
  cloneVectorToDevice(mat_ds.m_, &h_s, &d_s);
  cloneVectorToDevice(mat_jc.n_, &h_y, &d_y);
  cloneVectorToDevice(mat_jd.n_, &h_yd, &d_yd);

  cusparseSpMatDescr_t jc_desc;
  createCsrMat(&jc_desc, 
      mat_jc.n_, 
      mat_jc.m_, 
      mat_jc.nnz_, 
      jc_i, 
      jc_j, 
      jc_v);
  // set up vectors to store products

  double* d_rx_til{nullptr};
  double* d_rs_til{nullptr};

  cloneDeviceVector(mat_h.n_, &d_rx, &d_rx_til);
  cloneDeviceVector(mat_ds.n_, &d_rs, &d_rs_til);
  
  allocateVectorOnDevice(mat_jd.n_, &d_ryd_s);

  cusparseDnVecDescr_t vec_d_ryd    = NULL;
  cusparseDnVecDescr_t vec_d_rs_til = NULL;
  cusparseDnVecDescr_t vec_d_ryd_s  = NULL;
  cusparseDnVecDescr_t vec_d_rx_til = NULL;
  
  createDnVec(&vec_d_ryd, mat_jd.n_, d_ryd);
  createDnVec(&vec_d_rs_til, mat_ds.n_, d_rs_til);
  createDnVec(&vec_d_ryd_s, mat_jd.n_, d_ryd_s);
  createDnVec(&vec_d_rx_til, mat_h.n_, d_rx_til);
  // Start of block: Setting up eq (4) from the paper
  // start products
  double* htil_v{nullptr};
  int*    htil_j{nullptr};
  int*    htil_i{nullptr};
  int nnz_htil{0};
 
  cusparseSpGEMMDescr_t spgemm_desc;
  createSpGEMMDescr(&spgemm_desc);
  cusparseSpMatDescr_t jd_desc = NULL; //create once, overwrite each iteration
  createCsrMat(&jd_desc,
      mat_jd.n_,
      mat_jd.m_,
      mat_jd.nnz_,
      jd_i,
      jd_j,
      jd_v);

  cusparseSpMatDescr_t jd_t_desc = NULL;
  double* jd_t_v{nullptr};
  int*    jd_t_j{nullptr};
  int*    jd_t_i{nullptr};
  
  if(jd_flag)   // if JD is not all zeros (otherwise computation is saved)
  { // Creating a CSR matrix and buffer for transposing - done only once
    allocateMatrixOnDevice(mat_jd.m_, 
        mat_jd.nnz_, 
        &jd_t_i, 
        &jd_t_j, 
        &jd_t_v);
   
    void* buffer;
    transposeMatrixOnDevice(handle,
        mat_jd.n_, 
        mat_jd.m_, 
        mat_jd.nnz_, 
        jd_i, 
        jd_j, 
        jd_v, 
        jd_t_i, 
        jd_t_j, 
        jd_t_v,
        &buffer,
        false);
    deleteOnDevice(buffer);

    createCsrMat(&jd_t_desc,
        mat_jd.m_,
        mat_jd.n_,
        mat_jd.nnz_,
        jd_t_i,
        jd_t_j,
        jd_t_v);

    // math ops for eq (4) done at every iteration
    fun_row_scale(mat_jd.n_, jd_v, jd_i, jd_j, jd_vs, d_ryd, d_ryd_s, ds_v);
    cusparseSpMatDescr_t jd_s_desc = NULL;   //(except this part)
    createCsrMat(&jd_s_desc,
        mat_jd.n_,
        mat_jd.m_,
        mat_jd.nnz_,
        jd_i,
        jd_j,
        jd_vs);
    
    fun_add_vecs(mat_jd.n_, d_ryd_s, ONE, d_rs);
    // create buffer for matvec - done once
    // matvec done every iteration
    fun_SpMV_full(handle, ONE, jd_t_desc, vec_d_ryd_s, ONE, vec_d_rx_til);

    // Compute H_til= H+J_d^T * D_s * J_d
    SpgemmClass* sc = new SpgemmClass(mat_h.n_, 
        mat_h.n_, 
        handle, 
        ONE, 
        ONE, 
        ONE);
    
    sc->load_product_matrices(jd_t_desc, jd_s_desc);
    sc->load_sum_matrices(h_i, h_j, h_v, mat_h.nnz_);
    sc->load_result_matrix(&htil_i, &htil_j, &htil_v, &nnz_htil);

    sc->spGEMM_reuse();
    delete sc;
    // This closes the if J_d!=0 statement
  }else{   // overwite H with Htil if JD==0
     
    allocateMatrixOnDevice(mat_h.n_, mat_h.nnz_, &htil_i, &htil_j, &htil_v);
    matrixDeviceToDeviceCopy(mat_h.n_, 
        mat_h.nnz_, 
        h_i, 
        h_j, 
        h_v, 
        htil_i, 
        htil_j, 
        htil_v);
   
    nnz_htil = mat_h.nnz_;
  }
  // Start of block: Ruiz scaling
  // Allocation - happens once
  int     n_hj = mat_h.n_ + mat_jc.n_;
  double* jc_t_v{nullptr};
  int*    jc_t_j{nullptr};
  int*    jc_t_i{nullptr};

  allocateMatrixOnDevice(mat_jc.m_, mat_jc.nnz_, &jc_t_i, &jc_t_j, &jc_t_v);
 
  void* buffer;
  transposeMatrixOnDevice(handle,
      mat_jc.n_, 
      mat_jc.m_, 
      mat_jc.nnz_, 
      jc_i, 
      jc_j, 
      jc_v, 
      jc_t_i, 
      jc_t_j, 
      jc_t_v,
      &buffer,
      false);
  deleteOnDevice(buffer);

  cusparseSpMatDescr_t jc_t_desc = NULL;
  createCsrMat(&jc_t_desc, 
      mat_jc.m_, 
      mat_jc.n_, 
      mat_jc.nnz_, 
      jc_t_i, 
      jc_t_j, 
      jc_t_v);
#if 1 //this block is only activated to check solution (requires more copying)
  // saves the original JC and JCt  
  double* jc_v_c{nullptr};
  int*    jc_i_c{nullptr}; 
  int*    jc_j_c{nullptr};
  
  allocateMatrixOnDevice(mat_jc.n_, mat_jc.nnz_, &jc_i_c, &jc_j_c, &jc_v_c);
  matrixDeviceToDeviceCopy(mat_jc.n_, 
      mat_jc.nnz_, 
      jc_i, 
      jc_j, 
      jc_v, 
      jc_i_c, 
      jc_j_c, 
      jc_v_c);
  
  cusparseSpMatDescr_t jc_c_desc = NULL;
  createCsrMat(&jc_c_desc, 
      mat_jc.n_, 
      mat_jc.m_, 
      mat_jc.nnz_, 
      jc_i_c, 
      jc_j_c, 
      jc_v_c);

  double* jct_v_c{nullptr};
  int*    jct_i_c{nullptr};
  int*    jct_j_c{nullptr};
  
  allocateMatrixOnDevice(mat_jc.m_, 
      mat_jc.nnz_, 
      &jct_i_c, 
      &jct_j_c, 
      &jct_v_c);
  matrixDeviceToDeviceCopy(mat_jc.m_, 
      mat_jc.nnz_, 
      jc_t_i, 
      jc_t_j, 
      jc_t_v, 
      jct_i_c, 
      jct_j_c, 
      jct_v_c);

  cusparseSpMatDescr_t jc_t_desc_c = NULL;
  createCsrMat(&jc_t_desc_c, 
      mat_jc.m_, 
      mat_jc.n_, 
      mat_jc.nnz_, 
      jct_i_c, 
      jct_j_c, 
      jct_v_c);
#endif
  double* max_d{nullptr};
  allocateVectorOnDevice(n_hj, &max_d);
  RuizClass* rz = new RuizClass(ruiz_its, mat_h.n_, n_hj);
  rz->add_block11(htil_v, htil_i, htil_j);
  rz->add_block12(jc_t_v, jc_t_i, jc_t_j);
  rz->add_block21(jc_v, jc_i, jc_j);
  rz->add_rhs1(d_rx_til);
  rz->add_rhs2(d_ry);
  rz->ruiz_scale();
  max_d = rz->get_max_d();
 

  double* hgam_v{nullptr};
  int*    hgam_j{nullptr};
  int*    hgam_i{nullptr};
  int     nnz_hgam{0};

  //Hgamma= Htilde + gamma(J_c^TJ_c) 
  SpgemmClass* sc_gamma = new SpgemmClass(mat_h.n_, 
      mat_h.n_, 
      handle, 
      gamma, 
      ONE, 
      ONE);
    
  sc_gamma->load_product_matrices(jc_t_desc, jc_desc);
  sc_gamma->load_sum_matrices(htil_i,
      htil_j,
      htil_v,
      nnz_htil);
  sc_gamma->load_result_matrix(&hgam_i, &hgam_j, &hgam_v, &nnz_hgam);
  
  sc_gamma->spGEMM_reuse();
  double* d_rx_hat{nullptr};
  
  cloneDeviceVector(mat_h.n_, &d_rx_til, &d_rx_hat);
  cusparseDnVecDescr_t vec_d_rx_hat = NULL;
  createDnVec(&vec_d_rx_hat, mat_h.n_, d_rx_hat);
  cusparseDnVecDescr_t vec_d_ry = NULL;
  createDnVec(&vec_d_ry, mat_jc.n_, d_ry);
  fun_SpMV_full(handle, gamma, jc_t_desc, vec_d_ry, ONE, vec_d_rx_hat);
  // Start of block: permutation calculation (happens once)
  int* hgam_h_i = new int[mat_h.n_ + 1];
  int* hgam_h_j = new int[nnz_hgam];
  
  copyVectorToHost(mat_h.n_ + 1, hgam_i, hgam_h_i);
  copyVectorToHost(nnz_hgam, hgam_j, hgam_h_j);
  
  int* hgam_p_i = new int[mat_h.n_ + 1];
  int* hgam_p_j = new int[nnz_hgam];
  int* jc_p_j = new int[mat_jc.nnz_];
  int* jct_p_j = new int[mat_jc.nnz_];
  int* jct_p_i = new int[mat_jc.m_ + 1];
  int* jct_j = new int [mat_jc.nnz_];
  int* jct_i = new int [mat_jc.m_ + 1];
   
  PermClass* pc = new PermClass(mat_h.n_, nnz_hgam, mat_jc.nnz_);
  pc->add_h_info(hgam_h_i, hgam_h_j);
  pc->add_j_info(mat_jc.csr_rows, mat_jc.coo_cols, mat_jc.n_, mat_jc.m_); 
  pc->add_jt_info(jct_i, jct_j);
  pc->symamd(); 
  pc->invert_perm();
  pc->vec_map_rc(hgam_p_i, hgam_p_j);
  pc->vec_map_c(jc_p_j);
  
  copyVectorToHost(mat_jc.m_ + 1, jc_t_i, jct_i);
  copyVectorToHost(mat_jc.nnz_, jc_t_j, jct_j);
  pc->vec_map_r(jct_p_i, jct_p_j);
  
  copyVectorToDevice(mat_h.n_ + 1, hgam_p_i, hgam_i);
  copyVectorToDevice(nnz_hgam, hgam_p_j, hgam_j);
  
  copyVectorToDevice(mat_jc.nnz_, jct_p_j, jc_t_j);
  copyVectorToDevice(mat_jc.m_ + 1, jct_p_i, jc_t_i);
  copyVectorToDevice(mat_jc.nnz_, jc_p_j, jc_j);
  
  double* hgam_p_val{nullptr};
  double* jc_p_val{nullptr};
  double* jct_p_val{nullptr};
  
  allocateVectorOnDevice(nnz_hgam, &hgam_p_val);
  allocateVectorOnDevice(mat_jc.nnz_, &jc_p_val);
  allocateVectorOnDevice(mat_jc.nnz_, &jct_p_val);
  
  // Start of block: permutation application - happens every iteration
  pc->map_index(perm_h_v, hgam_v, hgam_p_val);
  pc->map_index(perm_j_v, jc_v, jc_p_val);
  pc->map_index(perm_jt_v, jc_t_v, jct_p_val);

  cusparseSpMatDescr_t jc_p_desc = NULL;
  createCsrMat(&jc_p_desc,
      mat_jc.n_,
      mat_jc.m_,
      mat_jc.nnz_,
      jc_i,
      jc_j,
      jc_p_val);
  
  fun_add_diag(mat_h.n_, ZERO, hgam_i, hgam_j, hgam_p_val);
  cusparseSpMatDescr_t jc_t_descp = NULL;
  createCsrMat(&jc_t_descp,
      mat_jc.m_,
      mat_jc.n_,
      mat_jc.nnz_,
      jc_t_i,
      jc_t_j,
      jct_p_val);

  double* d_rxp{nullptr};
  allocateVectorOnDevice(mat_h.n_, &d_rxp);
  pc->map_index(perm_v, d_rx_hat, d_rxp);
  //  Start of block: Factorization of Hgamma
  //  Symbolic analysis: Happens once
  
  CholeskyClass* cc = new CholeskyClass(mat_h.n_, 
      nnz_hgam, 
      hgam_p_val,
      hgam_i, 
      hgam_j);
  
  cc->symbolic_analysis();
  cc->set_pivot_tolerance(tol);
  cc->numerical_factorization();
  //  Start of block : setting up the right hand side for equation 7
  //  Allocation - happens once
  double* d_hrxp{nullptr};
  double* d_schur{nullptr};
  allocateVectorOnDevice(mat_h.n_, &d_hrxp);
  allocateVectorOnDevice(mat_jc.n_, &d_schur);
  //  Solve and copy - happen every iteration
  cc->solve(d_hrxp, d_rxp);
  copyDeviceVector(mat_jc.n_, d_ry, d_schur);
  // Allocation - happens once
  cusparseDnVecDescr_t vec_d_schur = NULL;
  createDnVec(&vec_d_schur, mat_jc.n_, d_schur);
  cusparseDnVecDescr_t vec_d_hrxp = NULL;
  createDnVec(&vec_d_hrxp, mat_h.n_, d_hrxp);
  fun_SpMV_full(handle, ONE, jc_p_desc, vec_d_hrxp, MINUS_ONE, vec_d_schur);
  // class implementation
  SchurComplementConjugateGradient* sccg = 
    new SchurComplementConjugateGradient(jc_p_desc, 
        jc_t_descp, 
        d_y, 
        d_schur, 
        mat_jc.n_, 
        mat_jc.m_, 
        cc, 
        handle, 
        handle_cublas);
  
  sccg->setup();
  sccg->solve();
  // Start of block - recovering the solution to the original system by parts
  // this part is to recover delta_x
  // Allocation - happens once
  cusparseDnVecDescr_t vec_d_y = NULL;
  createDnVec(&vec_d_y, mat_jc.n_, d_y);
  cusparseDnVecDescr_t vec_d_rxp = NULL;
  createDnVec(&vec_d_rxp, mat_h.n_, d_rxp);
  // Matrix-vector product - happens every iteration
  fun_SpMV_full(handle, MINUS_ONE, jc_t_descp, vec_d_y, ONE, vec_d_rxp);
  //  Allocation - happens once
  double* d_z{nullptr};
  allocateVectorOnDevice(mat_h.n_, &d_z);
  //  Solve - happens every iteration
  cc->solve(d_z, d_rxp);
  pc->map_index(rev_perm_v, d_z, d_x);
  // scale back delta_y and delta_x (every iteration)
  fun_vec_scale(mat_h.n_, d_x, max_d);
  fun_vec_scale(mat_jc.n_, d_y, &max_d[mat_h.n_]);
  cusparseDnVecDescr_t vec_d_x = NULL;
  createDnVec(&vec_d_x, mat_h.n_, d_x);
  cusparseDnVecDescr_t vec_d_s = NULL;
  createDnVec(&vec_d_s, mat_ds.n_, d_s);
  copyDeviceVector(mat_ds.m_, d_ryd, d_s);
  if(jd_flag)
  {
    fun_SpMV_full(handle, ONE, jd_desc, vec_d_x, MINUS_ONE, vec_d_s);
  }else{   //  Math operations - happens every iteration
    fun_mult_const(mat_ds.n_, MINUS_ONE, d_s);
  }
  //  Math operations - happens every iteration
  copyDeviceVector(mat_ds.m_, d_s, d_yd);
  fun_vec_scale(mat_ds.n_, d_yd, ds_v);
  fun_add_vecs(mat_ds.n_, d_yd, MINUS_ONE, d_rs);
  
  //  Start of block, calculate error of Ax-b 
  //  Calculate error in rx
  double norm_rx_sq   = 0;
  double norm_rs_sq   = 0;
  double norm_ry_sq   = 0;
  double norm_ryd_sq  = 0;
  double norm_resx_sq = 0;
  double norm_resy_sq = 0; 
  // This will aggregate the squared norms of the residual and rhs
  // Note that by construction the residuals of rs and ryd are 0
  dotProduct(handle_cublas, mat_h.n_, d_rx, d_rx, &norm_rx_sq);
  dotProduct(handle_cublas, mat_ds.n_, d_rs, d_rs, & norm_rs_sq);
  dotProduct(handle_cublas, mat_jc.n_, d_ry_c, d_ry_c, &norm_ry_sq);
  dotProduct(handle_cublas, mat_jd.n_, d_ryd, d_ryd, &norm_ryd_sq);
    
  norm_rx_sq += norm_rs_sq + norm_ry_sq + norm_ryd_sq;
  cusparseDnVecDescr_t vec_d_rx = NULL;
  createDnVec(&vec_d_rx, mat_h.n_, d_rx);
  cusparseDnVecDescr_t vec_d_yd = NULL;
  createDnVec(&vec_d_yd, mat_jd.n_, d_yd);
  fun_SpMV_full(handle, MINUS_ONE, h_desc, vec_d_x, ONE, vec_d_rx);
  dotProduct(handle_cublas, mat_h.n_, d_rx, d_rx, &norm_resx_sq);
 
  if (jd_flag){
    fun_SpMV_full(handle, MINUS_ONE, jd_t_desc, vec_d_yd, ONE, vec_d_rx);
    dotProduct(handle_cublas, mat_h.n_, d_rx, d_rx, &norm_resx_sq);
  }
  
  fun_SpMV_full(handle, MINUS_ONE, jc_t_desc_c, vec_d_y, ONE, vec_d_rx);
  dotProduct(handle_cublas, mat_h.n_, d_rx, d_rx, &norm_resx_sq);
  //  Calculate error in ry
  cusparseDnVecDescr_t vec_d_ry_c = NULL;
  createDnVec(&vec_d_ry_c, mat_jc.n_, d_ry_c);
  fun_SpMV_full(handle, MINUS_ONE, jc_c_desc, vec_d_x, ONE, vec_d_ry_c);
  dotProduct(handle_cublas, mat_jc.n_, d_ry_c, d_ry_c, &norm_resy_sq);
  // Calculate final relative norm
  norm_resx_sq += norm_resy_sq;
  double norm_res = sqrt(norm_resx_sq)/sqrt(norm_rx_sq);
  printf("||Ax-b||/||b|| = %32.32g\n", norm_res);
  //  Start of block - free memory
  
  delete rz;
  delete sc_gamma;
  delete pc;
  delete cc;
  delete sccg; 

  deleteMatrixOnDevice(htil_i, htil_j, htil_v);
  deleteMatrixOnDevice(hgam_i, hgam_j, hgam_v);
  deleteMatrixOnDevice(h_i, h_j, h_v);
  deleteMatrixOnDevice(jc_i, jc_j, jc_v);
  deleteMatrixOnDevice(jc_t_i, jc_t_j, jc_t_v);
  deleteMatrixOnDevice(jc_i_c, jc_j_c, jc_v_c);
  deleteMatrixOnDevice(jct_i_c, jct_j_c, jct_v_c);
  deleteMatrixOnDevice(jd_i, jd_j, jd_v);
  deleteMatrixOnDevice(jd_t_i, jd_t_j, jd_t_v);
  
  deleteOnDevice(d_x);
  deleteOnDevice(d_s);
  deleteOnDevice(d_y);
  deleteOnDevice(d_yd);
  deleteOnDevice(d_z);
  deleteOnDevice(d_rx);
  deleteOnDevice(d_rxp);
  deleteOnDevice(d_hrxp);
  deleteOnDevice(d_schur);
  deleteOnDevice(d_rs);
  deleteOnDevice(d_ry);
  deleteOnDevice(d_ry_c);
  deleteOnDevice(d_ryd);
  deleteOnDevice(d_ryd_s);
  deleteOnDevice(d_rx_til);
  deleteOnDevice(d_rx_hat);
  deleteOnDevice(d_rs_til);
  deleteOnDevice(ds_v);
  deleteOnDevice(jd_vs);
  deleteOnDevice(hgam_p_val);
  deleteOnDevice(jc_p_val);
  deleteOnDevice(jct_p_val);
  
  delete [] rx;
  delete [] rs;
  delete [] ry;
  delete [] ryd;
  delete [] h_x;
  delete [] h_s;
  delete [] h_y;
  delete [] h_yd;
  delete [] hgam_h_i;
  delete [] hgam_h_j;
  delete [] hgam_p_i;
  delete [] hgam_p_j;
  delete [] jc_p_j;
  delete [] jct_p_j;
  delete [] jct_i;
  delete [] jct_j;
  delete [] jct_p_i;
 
  if (norm_res<norm_tol){
    printf("Residual test passed ");
  }else{
    printf("Residual test failed ");
    return 1;
  }
  return 0;
}
