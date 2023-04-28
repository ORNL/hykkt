#include <vector>
#include "HykktSolver.hpp"
#include <input_functions.hpp>

int main(int argc, char* argv[])
{
  if(argc != 19)
  {
    std::cout << "Incorrect number of inputs. Exiting ...\n";
    return -1;
  }
  std::cout << "Starting ...\n";

  // Get matrix block files for KKT system 1
  char const* const h_file_name1  = argv[1];
  char const* const ds_file_name1 = argv[2];
  char const* const jc_file_name1 = argv[3];
  char const* const jd_file_name1 = argv[4];

  // Get rhs block files for KKT system 1
  char const* const rx_file_name1  = argv[5];
  char const* const rs_file_name1  = argv[6];
  char const* const ry_file_name1  = argv[7];
  char const* const ryd_file_name1 = argv[8];
  
  // Get matrix block files for KKT system 2
  char const* const h_file_name2  = argv[9];
  char const* const ds_file_name2 = argv[10];
  char const* const jc_file_name2 = argv[11];
  char const* const jd_file_name2 = argv[12];

  // Get rhs block files for KKT system 2
  char const* const rx_file_name2  = argv[13];
  char const* const rs_file_name2  = argv[14];
  char const* const ry_file_name2  = argv[15];
  char const* const ryd_file_name2 = argv[16];
  
  int skip_lines = atoi(argv[17]);
  double gamma   = atof(argv[18]);
  
  // Load matrix blocks of KKT system 1 from input files.
  MMatrix H1;
  MMatrix Ds1;
  MMatrix Jc1;
  MMatrix Jd1;
  read_mm_file_into_coo(h_file_name1,  H1,  skip_lines);  
  read_mm_file_into_coo(ds_file_name1, Ds1, skip_lines);  
  read_mm_file_into_coo(jc_file_name1, Jc1, skip_lines);  
  read_mm_file_into_coo(jd_file_name1, Jd1, skip_lines);
  sym_coo_to_csr(H1);
  coo_to_csr(Ds1);
  coo_to_csr(Jc1);
  coo_to_csr(Jd1);

  // Load vector blocks of KKT system 1 from input files.
  std::vector<double> rx1(H1.n_, 0.0);
  std::vector<double> rs1(Ds1.n_, 0.0);
  std::vector<double> ry1(Jc1.n_, 0.0);
  std::vector<double> ryd1(Jd1.n_, 0.0);
  read_rhs(rx_file_name1,  &rx1[0]);
  read_rhs(rs_file_name1,  &rs1[0]);
  read_rhs(ry_file_name1,  &ry1[0]);
  read_rhs(ryd_file_name1, &ryd1[0]);

  // Load matrix blocks of KKT system 2 from input files.
  MMatrix H2;
  MMatrix Ds2;
  MMatrix Jc2;
  MMatrix Jd2;
  read_mm_file_into_coo(h_file_name2,  H2,  skip_lines);  
  read_mm_file_into_coo(ds_file_name2, Ds2, skip_lines);  
  read_mm_file_into_coo(jc_file_name2, Jc2, skip_lines);  
  read_mm_file_into_coo(jd_file_name2, Jd2, skip_lines);
  sym_coo_to_csr(H2);
  coo_to_csr(Ds2);
  coo_to_csr(Jc2);
  coo_to_csr(Jd2);

  // Load vector blocks of KKT system 1 from input files.
  std::vector<double> rx2(H2.n_, 0.0);
  std::vector<double> rs2(Ds2.n_, 0.0);
  std::vector<double> ry2(Jc2.n_, 0.0);
  std::vector<double> ryd2(Jd2.n_, 0.0);
  read_rhs(rx_file_name2,  &rx2[0]);
  read_rhs(rs_file_name2,  &rs2[0]);
  read_rhs(ry_file_name2,  &ry2[0]);
  read_rhs(ryd_file_name2, &ryd2[0]);

  // Create solution vector blocks
  std::vector<double> x(H1.n_, 0.0);
  std::vector<double> s(Ds1.n_, 0.0);
  std::vector<double> y(Jc1.n_, 0.0);
  std::vector<double> yd(Jd1.n_, 0.0);

  // Create solver
  HykktSolver* hs = new HykktSolver(gamma);
  int status = 0;

  // Set KKT matrix blocks
  hs->set_H(H1.csr_rows, H1.csr_cols, H1.csr_vals, H1.n_, H1.m_, H1.nnz_);
  hs->set_Ds(Ds1.csr_rows, Ds1.csr_cols, Ds1.csr_vals, Ds1.n_, Ds1.m_, Ds1.nnz_);
  hs->set_Jc(Jc1.csr_rows, Jc1.csr_cols, Jc1.csr_vals, Jc1.n_, Jc1.m_, Jc1.nnz_);
  hs->set_Jd(Jd1.csr_rows, Jd1.csr_cols, Jd1.csr_vals, Jd1.n_, Jd1.m_, Jd1.nnz_);

  // Set residual vectors
  hs->set_rx(&rx1[0],   H1.n_);
  hs->set_rs(&rs1[0],   Ds1.n_);
  hs->set_ry(&ry1[0],   Jc1.n_);
  hs->set_ryd(&ryd1[0], Jd1.n_);

  // Set solution vectors
  hs->set_x_host(&x[0], H1.n_);
  hs->set_s_host(&s[0], Ds1.n_);
  hs->set_y_host(&y[0], Jc1.n_);
  hs->set_yd_host(&yd[0], Jd1.n_);

  // Run solver
  status = hs->execute();
  if(status == 1)
  {
    delete hs;
    return 1;
  }

  // Update matrix blocks
  copy_mmatrix(H2, H1);
  copy_mmatrix(Ds2, Ds1);
  copy_mmatrix(Jc2, Jc1);
  copy_mmatrix(Jd2, Jd1);

  // Update rhs vector blocks
  copy_vector(&rx2[0], &rx1[0],   static_cast<int>(rx1.size()));
  copy_vector(&rs2[0], &rs1[0],   static_cast<int>(rs1.size()));
  copy_vector(&ry2[0], &ry1[0],   static_cast<int>(ry1.size()));
  copy_vector(&ryd2[0], &ryd1[0], static_cast<int>(ryd1.size()));

  status = hs->execute();
  delete hs;

  return status;
}
