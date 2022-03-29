# HyKKT

## Description
A linear solver tailored for Karush Kuhn Tucker (KKT) linear systems and 
deployment on hardware accelerator hardware such as GPUs. The solver requires
all blocks of the 4X4 block system separately and solves the system to a desired
numerical precision exactly via block reduction and conjugate gradient on the
schur complement. 

## Installation
No installation is required

## make
From the root directory run
```
source buildsystem/deception-env.sh // or alternatively load these modules
./buildsystem/build.sh // to make
sbatch deception_test.sbatch // to run, or use as template for batch script
```

## Usage
The executable ```hybrid_solver``` is built in build/src by make
This executable can be run with an appropriate batch script with 10 arguments
```
Hfile #represents the $H+D_x$ matrix block
Dsfile #represents the $D_s$ matrix block
Jfile #represents the $J$ matrix block
Jdfile #represents the $J_d$ matrix block
rxfile #represents the $r_x$ vector block
rsfile #represents the $r_s$ vector block
ryfile #represents the $r_y$ vector block
rydfile #represents the $r_{yd}$ vector block
skip #number of lines to ignore in the .mtx matrix files
gamma #constant to make system more PD in eq(6)
```
Examples of this script can be found in ```src/old_scripts```
## Support
Email Shaked Regev at sregev@stanford.edu or submit an issue

## Contributing
Contributions are welcome and are should ideally be in their own .cu or .cpp 
files with an explanation of their functionality. For improvements in efficiency
in the main code, please have a comment explaining the difference between the 
approaches and the advantages (and disadvantages) of your suggested one.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.
