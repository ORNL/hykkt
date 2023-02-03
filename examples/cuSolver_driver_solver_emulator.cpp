#include "HykktSolver.hpp"
#include <input_functions.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>
using namespace std;

string get_filename(string preamble, string basename, int seq)
{
    ostringstream stream;
    stream << preamble << "_" << basename << "_" << setfill('0') << setw(2) << seq << ".mtx";
    return stream.str();
}

inline bool file_exists(const std::string &name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

int main(int argc, char *argv[])
{
    if (argc != 7)
    {
        std::cout << "Incorrect number of inputs. Exiting ...\n";
        return -1;
    }
    std::cout << "Starting ...\n";

    string base_dir = string(argv[1]);
    string base_name = string(argv[2]);
    int starting_sequence = atoi(argv[3]);
    int ending_sequence = atof(argv[4]);
    int skip_lines = atoi(argv[5]);
    double gamma = atof(argv[6]);

    HykktSolver *hs = new HykktSolver(gamma);
    int status = 0;

    for (int i = starting_sequence; i <= ending_sequence; i++)
    {
        // Get KKT matrix block files
        string h_file_name1 = base_dir + "/" + get_filename("block_H_matrix", base_name, i);
        string ds_file_name1 = base_dir + "/" + get_filename("block_Dd_matrix", base_name, i);
        string jc_file_name1 = base_dir + "/" + get_filename("block_J_matrix", base_name, i);
        string jd_file_name1 = base_dir + "/" + get_filename("block_Jd_matrix", base_name, i);

        // Get rhs block files
        string rx_file_name1 = base_dir + "/" + get_filename("block_rx", base_name, i);
        string rs_file_name1 = base_dir + "/" + get_filename("block_rs", base_name, i);
        string ry_file_name1 = base_dir + "/" + get_filename("block_ry", base_name, i);
        string ryd_file_name1 = base_dir + "/" + get_filename("block_ryd", base_name, i);

        cout << file_exists(h_file_name1) << endl;
        cout << file_exists(ds_file_name1) << endl;
        cout << file_exists(jc_file_name1) << endl;
        cout << file_exists(jd_file_name1) << endl;
        cout << file_exists(rx_file_name1) << endl;
        cout << file_exists(rs_file_name1) << endl;
        cout << file_exists(ry_file_name1) << endl;
        cout << file_exists(ryd_file_name1) << endl;

        hs->read_matrix_files(h_file_name1.c_str(),
                              ds_file_name1.c_str(),
                              jc_file_name1.c_str(),
                              jd_file_name1.c_str(),
                              rx_file_name1.c_str(),
                              rs_file_name1.c_str(),
                              ry_file_name1.c_str(),
                              ryd_file_name1.c_str(),
                              skip_lines);
        status = hs->execute();
        if (status == 1)
            return 1;
    }
    delete hs;
    return 0;
}
