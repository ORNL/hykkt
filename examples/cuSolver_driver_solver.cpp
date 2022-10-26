#include "HykktSolver.hpp"

int main(int argc, char* argv[])
{
  if(argc != 19)
  {
    std::cout << "Incorrect number of inputs. Exiting ...\n";
    return -1;
  }

  char const* const h_file_name1  = argv[1];
  char const* const ds_file_name1 = argv[2];
  char const* const jc_file_name1 = argv[3];
  char const* const jd_file_name1 = argv[4];

  // Get rhs block files
  char const* const rx_file_name1  = argv[5];
  char const* const rs_file_name1  = argv[6];
  char const* const ry_file_name1  = argv[7];
  char const* const ryd_file_name1 = argv[8];
  
  char const* const h_file_name2  = argv[9];
  char const* const ds_file_name2 = argv[10];
  char const* const jc_file_name2 = argv[11];
  char const* const jd_file_name2 = argv[12];

  // Get rhs block files
  char const* const rx_file_name2  = argv[13];
  char const* const rs_file_name2  = argv[14];
  char const* const ry_file_name2  = argv[15];
  char const* const ryd_file_name2 = argv[16];
  
  int skip_lines = atoi(argv[17]);
  double gamma   = atof(argv[18]);
  
  HykktSolver* hs = new HykktSolver(gamma);
  int status = 0;

  //test iteration 1
  hs->read_matrix_files(h_file_name1,
      ds_file_name1,
      jc_file_name1,
      jd_file_name1,
      rx_file_name1,
      rs_file_name1,
      ry_file_name1,
      ryd_file_name1,
      skip_lines);
  status = hs->execute();
  if(status == 1) return 1;
  
  //test iteration 2
  hs->read_matrix_files(h_file_name2,
      ds_file_name2,
      jc_file_name2,
      jd_file_name2,
      rx_file_name2,
      rs_file_name2,
      ry_file_name2,
      ryd_file_name2,
      skip_lines);
  status = hs->execute();
  if(status == 1) return 1;

  delete hs;
  return 0;
}
