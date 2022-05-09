#pragma once

/*
  @brief: maps the values in old_val to new_val based on perm

  @inputs: Size n of the matrix, perm - desired permutation,
  and old_val - the array to be permuted

  @outputs: new_val contains the permuted old_val
*/
void fun_map_idx(int, int*, double*, double*);

/*
  @brief: Selection sorts arr1 and arr2 w/indices
  based on increasing value in arr1
  
  @inputs: Size n of the matrix, arr1 - the array that determines the sorting order,
  arr2- sorted based on arr1
  
  @outputs: arr1 and arr2 are sorted based on increasing values in arr1
*/
void selection_sort2(int, int, int);

/*
@brief: Permutes the columns in a matrix represented by rows and cols

@inputs: Size n of the matrix, rows and cols - representing the matrix,
rev_perm - the permutation to be applied

@outputs: perm_cols is now the permuted column array and perm_map stores
the corresponding indices to facilitate permuting the values
 */
void make_vec_map_c(int, int*, int*, int*, int*, int*);

/*
@brief: Creates a reverse permutate based on a given permutation

@inputs: Size n of the vector, perm - original permutation

@outputs: rev_perm now contains the reverse permutation
 */
void reverse_perm(int, int*, int*);

/*
@brief: Permutes the rows in a matrix represented by rows and cols

@inputs: Size n of the matrix, rows and cols - representing the matrix,
perm - the permutation to be applied

@outputs: perm_rows and perm_cols are now the permuted rows and column arrays
and perm_map stores the corresponding indices to facilitate permuting the values
*/
void make_vec_map_r(int, int*, int*, int*, int*, int*, int*);

/*
@brief: Permutes the rows and columns in a matrix represented by rows and cols

@inputs: Size n of the matrix, rows and cols - representing the matrix,
perm - the permutation to be applied on rows, rev_perm permutation to be applied
on the columns

@outputs: perm_rows and perm_cols are now the permuted rows and column arrays
and perm_map stores the corresponding indices to facilitate permuting the values
*/
void make_vec_map_rc(int, int*, int*, int*, int*, int*, int*, int*);

