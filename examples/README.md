# HyKKT

## Description
This folder contains drivers in increasing order of complexity that test different components of the HyKKT solver.

## perm\_driver
This driver tests the code to compute a permutation for an array stored in CSR format given its dense format permutation.
The test code is built in to the driver.

## Ruiz\_driver
This driver tests the code to compute the Ruiz scaling of a block $2\times 2$ system given the blocks of that system.
The test code is built in to the driver.

## cuSolver\_driver\_cholesky
This driver tests the code that solves a linear system $Hx=r$, where $H$ is symmetric positive definite and sparse, via cholesky factorization.
The inputs are:

* h_file_name - file containing sparse SPD matrix $H$ in matrix market format
* r_file_name - file containing vector $r$ in matrix market format


## cuSolver\_driver\_schur\_cg
This driver tests the code that performs conjugate gradient on the Schur complement of a block $2\times 2$ system:

```math
\begin{bmatrix}
    H_\gamma  & J_c^T \\
    J_c       & 0
\end{bmatrix}
\begin{bmatrix}
  \Delta x \\ \Delta y_c
\end{bmatrix} =
\begin{bmatrix}
  \hat{r}_x \\ r_c
\end{bmatrix}
```

The functionality of the cuSolver\_driver\_cholesky is wholly included in it.
The inputs are:

* jc_file_name - file containing sparse matrix $J_c$ in matrix market format
* h_file_name - file containing sparse SPD matrix $H$ in matrix market format
* r_file_name - file containing vector $r$ in matrix market format


## cuSolver\_driver\_hybrid
This driver tests the code that solves one linear system arising from an optimization problem,
given the blocks of the $4\times 4$ block system:

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

The functionality of all drivers above is included in it.
The inputs are:

1. `h_file_name` - contains the sparse symmetric $H+D_x$ matrix block in matrix market format
2. `ds_file_name` - contains the diagonal $D_s$ matrix block in matrix market format
3. `jc_file_name` - contains the sparse $J_c$ matrix block in matrix market format
4. `jd_file_name` - contains the sparse $J_d$ matrix block in matrix market format
5. `rx_file_name` - contains the $r_{x}$ vector block in matrix market format
6. `rs_file_name` - contains the $r_{s}$ vector block in matrix market format
7. `ryc_file_name` - contains the $r_{yc}$ vector block in matrix market format
8. `ryd_file_name` - contains the $r_{yd}$ vector block in matrix market format
9. `skip` - number of header lines to ignore in the .mtx matrix files
10. `gamma` - constant to make $H_\gamma= H + D_x + J_d^T D_s J_d + \gamma J_c^T J_c$ more positive definite (typically $10^4-10^6$)

## cuSolver\_driver\_solver
This drivers tests the code that solves multiple linear systems (with the same nonzero structure) 
arising from an underlying optimization problem,
given the blocks of the $4 \times 4$ block system. The functionality of all drivers above is included in it.
The inputs are similar to cuSolver\_driver\_hybrid, except the first 8 arguments are repated for the next matrix in the series. Both matrices must have the same sparsity structure.
