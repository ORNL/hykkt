

# HyKKT utilities

## Description
The HyKKT solver requires all blocks of the $`4\times 4`$ block system: 

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
## getallparts
Extracts these relevant blocks from a full matrix

## cholesky\_matrix\_extractor
Extracts $`H_\gamma`$, the matrix used for Cholesky solves in HyKKT, with a corresponding rhs 

## findblocks
A function which finds the blocks of the matrix based on the assumed structure

## mmread
Reads a matrix market file which contains a matrix or vector

## mmwrite
Writes to a matrix market file which contains a matrix or vector

## SymRuizScaling
Equilibriates a matrix so all entries have magnitude $`O(1)`$

## Usage
Run getallparts.m or cholesky\_matrix\_extractor.m in MATLAB (changing the matrix and vector prefices as needed) 
