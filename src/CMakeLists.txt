set(HYBRID_SRC
  MMatrix.cpp
  input_functions.cpp
  PermClass.cpp
  RuizClass.cpp
)

set(HYBRID_SRC_CUDA
  cuda_memory_utils.cu
  cusparse_utils.cu
  matrix_matrix_ops.cu
  matrix_vector_ops.cu
  vector_vector_ops.cu
  permcheck.cu
  SchurComplementConjugateGradient.cu
  CholeskyClass.cu
  SpgemmClass.cu
  HykktSolver.cu
)

set_source_files_properties(${HYBRID_SRC_CUDA} PROPERTIES LANGUAGE CUDA)
add_library(hybrid OBJECT ${HYBRID_SRC} ${HYBRID_SRC_CUDA})

