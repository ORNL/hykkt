set(HYBRID_SRC
  Timer.cpp
  MMatrix.cpp
  input_functions.cpp
  vector_vector_ops.cpp
  cusparse_utils.cpp
  matrix_matrix_ops.cpp
  PermClass.cpp
  RuizClass.cpp
  SchurComplementConjugateGradient.cpp
  CholeskyClass.cpp
  SpgemmClass.cpp
  HykktSolver-Timing.cpp
)

set(HYBRID_SRC_CUDA
  cuda_memory_utils.cu
  matrix_vector_ops.cu
  permcheck.cu
)

set_source_files_properties(${HYBRID_SRC_CUDA} PROPERTIES LANGUAGE CUDA)
add_library(hybrid SHARED ${HYBRID_SRC} ${HYBRID_SRC_CUDA})
target_link_libraries(hybrid PRIVATE cuda_libs)

if(NOT AMD_LIBRARY)
  message(STATUS "Cannot find AMD, disabling SuiteSparse AMD module ...")
  set(HYKKT_USE_AMD OFF CACHE BOOL "Build without AMD module." FORCE)
else()  
  target_link_libraries(hybrid PUBLIC AMD)
endif()  
