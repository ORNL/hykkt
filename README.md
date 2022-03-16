# HyKKT

## Description
A linear solver tailored for Karush Kuhn Tucker (KKT) linear systems and 
deployment on hardware accelerator hardware such as GPUs. The solver requires
all blocks of the 4X4 block system separately and solves the system to a desired
numerical precision exactly via block reduction and conjugate gradient on the
schur complement. 

## Installation
No installation is required

## Usage
```
cd src
./hybrid_compile_run test$i //for example
./hybrid_compile_run test2
```
Where ```$i = {1,2,3,4,5}``` represents test cases of increasing size.
This calls the scripts ```hybrid_batch_$x```. The scripts also show how one
would call the solver on a general problem. The 9th argument is the line number
where the data starts in the matrix files. The 10th is gamma. Increasing gamma
improves CG convergence, but makes the problem more ill-conditioned and the 
original solution recovered may be of worse quality.

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
