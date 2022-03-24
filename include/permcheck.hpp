#ifndef SCALE__H__
#define SCALE__H__

__global__ void map_idx(int, int*, double*, double*);

void selection_sort2(int, int, int);

void make_vec_map_c(int, int*, int*, int*, int*, int*);

void reverse_perm(int, int*, int*);

void make_vec_map_r(int, int*, int*, int*, int*, int*, int*);

void make_vec_map_rc(int, int*, int*, int*, int*, int*, int*, int*);

#endif
