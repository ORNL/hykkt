set(HYBRID_SRC
  cuda_memory_utils.cu
  MMatrix.cpp
  PermClass.cpp
  RuizClass.cpp
  matrix_matrix_ops.cu
  matrix_vector_ops.cu
  vector_vector_ops.cu
  input_functions.cpp
  permcheck.cu
  SchurComplementConjugateGradient.cu
  CholeskyClass.cu
  SpgemmClass.cu
  HykktSolver.cu
)


set_source_files_properties(${HYBRID_SRC} PROPERTIES LANGUAGE CUDA)
add_library(hybrid OBJECT ${HYBRID_SRC} )
set_target_properties(hybrid PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

