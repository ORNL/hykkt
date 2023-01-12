
# Exports target `hykkt_cuda` which finds all cuda libraries needed by hykkt.


add_library(hykkt_cuda INTERFACE)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(hykkt_cuda INTERFACE
  CUDA::cusolver 
  CUDA::cublas
  CUDA::cusparse
  CUDA::cudart
  )

install(TARGETS hykkt_cuda EXPORT HyKKT-targets)


