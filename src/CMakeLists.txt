set(HYBRID_SRC
  input_functions.cu
  SchurComplementConjugateGradient.cu
  permcheck.cu
  matrix_vector_ops.cu
  vector_vector_ops.cu
  cuda_memory_utils.cu
)


set_source_files_properties(${HYBRID_SRC} PROPERTIES LANGUAGE CUDA)
add_library(hybrid OBJECT ${HYBRID_SRC} )
set_target_properties(hybrid PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

