[![pipeline status](https://gitlab.pnnl.gov/exasgd/frameworks/exago/badges/master/pipeline.svg)](https://gitlab.pnnl.gov/exasgd/solvers/hykkt/-/commits/develop)


# HyKKT

## Description
HyKKT is a package for solving systems of equations and unknowns resulting from an iterative solution of an optimization
problem, for example optimal power flow, which uses hardware accelerators (GPUs) efficiently.

The HyKKT package contains a linear solver tailored for Karush Kuhn Tucker (KKT) linear systems and
deployment on hardware accelerator hardware such as GPUs. The solver requires
all blocks of the $`4\times 4`$ block system: 

```math
\begin{bmatrix}
    H + D_x     & 0         & J_c^T     & J_d^T \\
      0         & D_s       & 0           & -I  \\
     J_c        & 0         & 0           & 0   \\
     J_d        & -I        & 0           & 0
\end{bmatrix}
\begin{bmatrix}
  \Delta x \\ \Delta s \\ \Delta y_c \\ \Delta y_d
\end{bmatrix} =
\begin{bmatrix}
  \tilde{r}_x \\ r_s \\ r_c \\ r_d
\end{bmatrix}
```

separately and solves the system to a desired
numerical precision exactly via block reduction and conjugate gradient on the
schur complement.

## Installation and build instructions
Clone the repository 
``` 
git clone https://gitlab.pnnl.gov/exasgd/solvers/hykkt.git
``` 
and cd into the root directory (hykkt). The build dependencies are
* C++ compiler supporting C++11 standard or higher
* CMake >= 3.19
* CUDA >= 11.0
```
./buildsystem/build.sh // to make
cd build
ctest -VV // to test - this must be done from a compute node with a GPU or in a batch script as shown [here](https://gitlab.pnnl.gov/exasgd/solvers/hykkt/-/blob/README/deception_test.sbatch)
```

## Usage

The executable ```hybrid_solver``` is built in build/src by make
This executable can be run with an appropriate batch script with 10 arguments

1. `h_file_name` - contains the sparse symmetric $`H+D_x`$ matrix block in matrix market format
2. `ds_file_name` - contains the diagonal $`D_s`$ matrix block in matrix market format
3. `jc_file_name` - contains the sparse $`J_c`$ matrix block in matrix market format
4. `jd_file_name` - contains the sparse $`J_d`$ matrix block in matrix market format
5. `rx_file_name` - contains the $`r_{x}`$ vector block in matrix market format
6. `rs_file_name` - contains the $`r_{s}`$ vector block in matrix market format
7. `ryc_file_name` - contains the $`r_{yc}`$ vector block in matrix market format
8. `ryd_file_name` - contains the $`r_{yd}`$ vector block in matrix market format
9. `skip` - number of header lines to ignore in the .mtx matrix files
10. `gamma` - constant to make $`H_\gamma= H + D_x + J_d^T D_s J_d + \gamma J_c^T J_c`$ more positive definite (typically $`10^4-10^6`$)

In the case of solution of multiple systems, the first 8 arguments are repated for the next matrix in the series. Both matrices must have the same sparsity structure.

Examples of this script can be found in [`src/old_scripts`](./src/old_scripts)
## Support
To receive support or ask a question, submit an issue on [Gitlab](https://gitlab.pnnl.gov/exasgd/solvers/hykkt/-/issues).

## Contributing
Please see [the developer guidelines](https://gitlab.pnnl.gov/exasgd/solvers/hykkt/-/blob/README/README_hiop_developers.md) before attempting to contribute.

## Authors
* Shaked Regev
* Maksudul Alam
* Ryan Danahy
* Cameron Rutherford
* Kasia Swirydowicz
* Slaven Peles 

## Acknowledgement
This package is developed as a part of [ExaSGD](https://www.exascaleproject.org/research-project/exasgd/) project under the [Exascale computing project](https://www.exascaleproject.org/).

## License
Copyright &copy; 2022, Battelle Memorial Institute.

ExaGO<sup>TM</sup> is a free software distributed under a BSD 2-clause license. You may reuse, modify, and redistribute the software. See the [license](https://gitlab.pnnl.gov/exasgd/solvers/hykkt/-/blob/README/LICENSE) file for details.
