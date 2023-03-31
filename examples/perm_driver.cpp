#include <stdio.h>
#include <stdlib.h>
#include "permcheck.hpp"
#include "PermClass.hpp"

/*
  *@brief Driver demonstrates use of Permutation class functions
*/

int main(int /* argc */, char** /* argv */)
{
  int n = 4;
  int m = 4;
  int nnz = 9;
  int a_i[5]    = {0, 2, 5, 7, 9};
  int a_j[9]    = {0, 2, 0, 1, 2, 1, 2, 1, 3};
  int a_prc_i[5] = {0, 2, 4, 6, 9};
  int a_prc_j[9] = {0, 3, 0, 1, 2, 3, 0, 1, 3};
  int a_pr_i[5]  = {0, 2, 4, 6, 9};
  int a_pr_j[9]  = {1, 2, 0, 2, 1, 3, 0, 1, 2};
  int a_pc_i[5]  = {0, 2, 5, 7, 9};
  int a_pc_j[9]  = {0, 1, 0, 1, 3, 0, 3, 2, 3};
  int b_i[5];
  int b_j[9];
  int perm[4] = {2, 0, 3, 1};
  int flagrc = 0;
  int flagr = 0;
  int flagc = 0;
 
  PermClass pc(n, nnz, nnz);
  pc.add_h_info(a_i, a_j);
  pc.add_j_info(a_i, a_j, n, m);
  pc.add_jt_info(a_i, a_j);
  pc.add_perm(perm);
  pc.invert_perm();
  pc.vec_map_rc(b_i, b_j);
  
  printf("Comparing RC permutation\n");
  for(int i = 0; i < n; i++)
  {
    if(a_prc_i[i] != b_i[i])
    {
      printf("Missmatch in row pointer %d \n", i);
      flagrc = 1;
    }
    for(int j = b_i[i]; j < b_i[i + 1]; j++)
    {
      if(a_prc_j[j] != b_j[j])
      {
        printf("Missmatch in row %d column  %d \n", i, b_j[j]);
        flagrc = 1;
      }
    }
  }
  if(flagrc){
    printf("RC permute failed\n");
  } else{
    printf("RC permute passed\n");
  }

  pc.vec_map_r(b_i, b_j);
  printf("Comparing R permutation\n");
  for(int i = 0; i < n; i++)
  {
    if(a_pr_i[i] != b_i[i])
    {
      printf("Missmatch in row pointer %d \n", i);
      flagr = 1;
    }
    for(int j = b_i[i]; j < b_i[i + 1]; j++)
    {
      if(a_pr_j[j] != b_j[j])
      {
        printf("Missmatch in row %d column  %d \n", i, b_j[j]);
        flagr = 1;
      }
    }
  }
  if(flagr){
    printf("R permute failed\n");
  } else{
    printf("R permute passed\n");
  }
  
  pc.vec_map_c(b_j);
  printf("Comparing C permutation\n");
  for(int i = 0; i < n; i++)
  {
    if(a_pc_i[i] != a_i[i])
    {
      printf("Missmatch in row pointer %d \n", i);
      flagc = 1;
    }
    for(int j = b_i[i]; j < b_i[i + 1]; j++)
    {
      if(a_pc_j[j] != b_j[j])
      {
        printf("Missmatch in row %d column  %d \n", i, b_j[j]);
        flagc = 1;
      }
    }
  }
  if(flagc){
    printf("C permute failed\n");
  }
  else{
    printf("C permute passed\n");
  }
  printf("Infinte loop start for testing CI");

  return flagrc + flagr + flagc;
}
