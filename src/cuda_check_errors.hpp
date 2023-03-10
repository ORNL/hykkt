#pragma once

template <typename T>
void check(T result, 
           char const *const func, 
           const char *const file,
           int const line) 
{
  if (result) {
    printf("CUDA error in function %s at %s:%d, error# %d\n", func, file, line, result);
    exit(EXIT_FAILURE);
  }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
