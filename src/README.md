# HyKKT

## Description
This file contains descriptions of the various utility files and classes used 
by the HyKKT solver

## CholeskyClass
This class computes and stores the symbolic and numeric factorization of 
$`H_\gamma`$ (or $`H_\delta = H_\gamma + \delta_1 I`$ when regularized). The 
factorization is then used to solve systems of the form $`H_\gamma x = b`$ such
as the inner solve of the Schur Complement conjugate gradient solve or the 
system $`H_\gamma \Delta x = \hat{r_x} - J_c^T \Delta y_c`$. The numeric 
factorization is reused for all these systems, such that only a triangular 
solve is performed each time a system is solved. The symbolic factorization is
reused between iterations of the optimization solver, since the sparsity 
structure is maintained. All actions are performed on the device.

## HyKKTSolver
This class contains all the other classes and the relevant data to use HyKKT
to solve the system
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
Allocation operations are performed during the first optimization solver 
iteration on the host. Subsequently, all computations are performed on the
device including in the first iteration.

## MMatrix
This structure can store a matrix in both COO and CSR format. It is useful to 
HyKKT because matrix market matrices are in COO format, so this is the format
to read them into. If the matrix is symmetric, its upper triangular part is
stored implicitly. This is all unpacked into the full explicit CSR format which
is necessary for storing permuted matrices (permutation is done to reduce 
fill-in) and using them efficiently, The structure is on the host.

## PermClass
This class computes and stores the permutation for the system
```math
\begin{bmatrix}
    H_\gamma    & J_c^T     \\
      J_c         & 0       
\end{bmatrix}
\begin{bmatrix}
  \Delta x \\ \Delta y_c
\end{bmatrix} =
\begin{bmatrix}
  \hat{r}_x \\ r_c 
\end{bmatrix}
```
The permutation is calculated based on minimizing the fill-in of the Cholesky
factorization of $`H_\gamma`$ with symmetric approximate minimum degree. The 
equivalent permutations to the row and column arrays of the sparse blocks are
calculated and the column permutation is stored to use on the value array. 
These actions are performed on the host and only during the first iteration of 
the optimization solver. At each iteration, only the value arrays of the block
matrices are permuted, since the other arrays are unchanged due to a constant 
sparsity pattern. This is performed on the device. Permutation is essential
to reduce fill-in of the Cholesky factorization, which reduces the time and 
storage required for its computation.

## RuizClass
This class performs Ruiz scaling of the system
```math
\begin{bmatrix}
    H_\gamma    & J_c^T     \\
      J_c         & 0       
\end{bmatrix}
\begin{bmatrix}
  \Delta x \\ \Delta y_c
\end{bmatrix} =
\begin{bmatrix}
  \hat{r}_x \\ r_c 
\end{bmatrix}
```
on the device. This is done to ensure all entries in the matrix are $`O(1)`$ to
aid in parameter selection portability between different problems.

## SchurComplementConjugateGradient
This class calculates the solution to the system
```math
S\Delta y_c = J_c H_\gamma^{-1} \hat{r}_x-r_{c},
```
where $`S = J_c H_\gamma^{-1} J_c^T`$, via conjugate gradient. $`S`$ is applied
by multiplying by $`J_c^T`$, solving a system with $`H_\gamma`$ (using its
precomputed Cholesky factorization), and finally multiplying by $`J_c`$. 
Operations are performed on the device.

## SpgemmClass
This class allocates the memory necessary for matrix-matrix sums and products,
which are used at various stages of the HyKKT algorithm. The sparsity structure
is repeated between optimization solver iterations so the symbolic sum or 
product is computed only once and then reused. The numerical sums and products 
are computed at each iteration on the device.

## constants
Defines numerical constants useful in matrix and vector operations.

## cuda\_check\_errors
Defines a wrapper function for CUDA calls to check they are successful
before continuing with the program.

## cuda\_memory\_utils
Contains wrapper functions to allocate, copy, or display matrices and vectors.

## cusparse\_params
Defines parameters used by cusparse in function calls such as data, index,
and algorithm types.

## cusparse\_utils
Defines functions to create native cusparse handle, vector, and matrix objects
and ones to transpose or display cusparseSpMatDescr\_t matrices.

## input\_functions
Defines functions used to populate an MMatrix from a matrix market file. This
reading and populating is performed on the host.

## matrix\_matrix\_ops
Defines matrix-matrix functions for products and sums used by Spgemm class.
These operations are carried out on the device.

## matrix\_vector\_ops
Defines wrappers to CUDA functions for matrix-vector products, and adds 
functions used by RuizClass, for finding maximum row entries in a block matrix,
and performing the appropriate Ruiz scaling. Additionally, it provides functions
for scaling a matrix by a diagonal matrix, regularizing it, or multiplying by a
constant. All calculations are performed on the device.

## permcheck
Contains functions implementing the methods of PermClass to permute block 
systems, to reduce fill-in during factorization, to limit storage usage and
compute time.

## vector\_vector\_ops
Contains wrappers for methods to sum two vectors, take their dot product,
or scale a vector on the device. All these operations are essential at various
stages of the HyKKT solver.
