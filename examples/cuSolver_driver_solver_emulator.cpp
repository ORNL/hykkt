#include "HykktSolver.hpp"
#include <input_functions.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std;

string get_filename(string preamble, string basename, int seq)
{
    ostringstream stream;
    stream << preamble << "_" << basename << "_" << setfill('0') << setw(2) << seq << ".mtx";
    return stream.str();
}

string get_filename(string preamble, string basename, string seq)
{
    ostringstream stream;
    stream << preamble << "_" << basename << "_" << seq << ".mtx";
    return stream.str();
}

inline bool file_exists(const std::string &name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

// for string delimiter
vector<string> split(string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        std::cout << "Incorrect number of inputs. Exiting ...\n";
        return -1;
    }
    std::cout << "Starting ...\n";

    string base_dir = string(argv[1]);
    string base_name = string(argv[2]);
    string str = string(argv[3]);

    string delimiter = ",";
    vector<string> v = split(str, delimiter);

    int skip_lines = atoi(argv[4]);
    double gamma = atof(argv[5]);

    HykktSolver *hs = new HykktSolver(gamma);
    int status = 0;

    for(auto i: v) {
	    cout << "Working on Sequence: " << i << endl << endl;
        // Get KKT matrix block files
        string h_file_name1 = base_dir + "/" + get_filename("block_H_matrix", base_name, i);
        string ds_file_name1 = base_dir + "/" + get_filename("block_Ds_matrix", base_name, i);
        string jc_file_name1 = base_dir + "/" + get_filename("block_Jc_matrix", base_name, i);
        string jd_file_name1 = base_dir + "/" + get_filename("block_Jd_matrix", base_name, i);

        // Get rhs block files
        string rx_file_name1 = base_dir + "/" + get_filename("block_rx_rhs", base_name, i);
        string rs_file_name1 = base_dir + "/" + get_filename("block_rs_rhs", base_name, i);
        string ry_file_name1 = base_dir + "/" + get_filename("block_ry_rhs", base_name, i);
        string ryd_file_name1 = base_dir + "/" + get_filename("block_rd_rhs", base_name, i);

        hs->read_matrix_files(h_file_name1.c_str(),
                              ds_file_name1.c_str(),
                              jc_file_name1.c_str(),
                              jd_file_name1.c_str(),
                              rx_file_name1.c_str(),
                              rs_file_name1.c_str(),
                              ry_file_name1.c_str(),
                              ryd_file_name1.c_str(),
                              skip_lines);
        //std::cout << "Finished Reading Files" << std::endl;
        status = hs->execute();
        if (status == 1)
            return 1;
    }
    delete hs;
    return 0;
}
