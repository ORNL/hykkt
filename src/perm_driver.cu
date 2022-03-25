#include <stdio.h>
#include <stdlib.h>
#include "permcheck.hpp"
int main(int argc, char* argv[])
{
  int n = 4;
  int A_rows[5]    = {0, 2, 5, 7, 9};
  int A_cols[9]    = {0, 2, 0, 1, 2, 1, 2, 1, 3};
  int Aprc_rows[5] = {0, 2, 4, 6, 9};
  int Aprc_cols[9] = {0, 3, 0, 1, 2, 3, 0, 1, 3};
  int Apr_rows[5]  = {0, 2, 4, 6, 9};
  int Apr_cols[9]  = {1, 2, 0, 2, 1, 3, 0, 1, 2};
  int Apc_rows[5]  = {0, 2, 5, 7, 9};
  int Apc_cols[9]  = {0, 1, 0, 1, 3, 0, 3, 2, 3};
  int B_rows[5];
  int B_cols[9];
  int perm[4] = {2, 0, 3, 1};
  int rev_perm[4];
  int perm_map[9];
  int flagrc = 0, flagr = 0, flagc = 0;
  reverse_perm(n, perm, rev_perm);
  make_vec_map_rc(n, A_rows, A_cols, perm, rev_perm, B_rows, B_cols, perm_map);
  printf("Comparing RC permutation\n");
  for(int i = 0; i < n; i++)
  {
    if(Aprc_rows[i] != B_rows[i])
    {
      printf("Missmatch in row pointer %d \n", i);
      flagrc = 1;
    }
    for(int j = B_rows[i]; j < B_rows[i + 1]; j++)
    {
      if(Aprc_cols[j] != B_cols[j])
      {
        printf("Missmatch in row %d column  %d \n", i, B_cols[j]);
        flagrc = 1;
      }
    }
  }
  if(flagrc)
    printf("RC permute failed\n");
  else
    printf("RC permute passed\n");
  make_vec_map_r(n, A_rows, A_cols, perm, B_rows, B_cols, perm_map);
  printf("Comparing R permutation\n");
  for(int i = 0; i < n; i++)
  {
    if(Apr_rows[i] != B_rows[i])
    {
      printf("Missmatch in row pointer %d \n", i);
      flagr = 1;
    }
    for(int j = B_rows[i]; j < B_rows[i + 1]; j++)
    {
      if(Apr_cols[j] != B_cols[j])
      {
        printf("Missmatch in row %d column  %d \n", i, B_cols[j]);
        flagr = 1;
      }
    }
  }
  if(flagr)
    printf("R permute failed\n");
  else
    printf("R permute passed\n");
  make_vec_map_c(n, A_rows, A_cols, rev_perm, B_cols, perm_map);
  printf("Comparing C permutation\n");
  for(int i = 0; i < n; i++)
  {
    if(Apc_rows[i] != A_rows[i])
    {
      printf("Missmatch in row pointer %d \n", i);
      flagc = 1;
    }
    for(int j = B_rows[i]; j < B_rows[i + 1]; j++)
    {
      if(Apc_cols[j] != B_cols[j])
      {
        printf("Missmatch in row %d column  %d \n", i, B_cols[j]);
        flagc = 1;
      }
    }
  }
  if(flagc)
    printf("C permute failed\n");
  else
    printf("C permute passed\n");
  return flagrc + flagr + flagc;
}
