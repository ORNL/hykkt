set(HYBRID_SRC
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
  HykktSolver.cpp
)

set(HYBRID_SRC_CUDA
  cuda_memory_utils.cu
  matrix_vector_ops.cu
  permcheck.cu
)

# define list of headers and source files
set(header MMatrix.hpp
    PermClass.hpp
    RuizClass.hpp
    SchurComplementConjugateGradient.hpp
    CholeskyClass.hpp
    SpgemmClass.hpp
    HykktSolver.hpp)

install(FILES ${header} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

set_source_files_properties(${HYBRID_SRC_CUDA} PROPERTIES LANGUAGE CUDA)
add_library(hykkt_libs SHARED ${HYBRID_SRC} ${HYBRID_SRC_CUDA})
target_link_libraries(hykkt_libs PUBLIC hykkt_cuda)

# install rules
install(TARGETS hykkt_libs EXPORT HyKKT-targets)

add_library(HyKKT::HyKKT ALIAS hykkt_libs)


if (HYKKT_USE_AMD)
  if(NOT AMD_LIBRARY)
    message(STATUS "Cannot find AMD, disabling SuiteSparse AMD module ...")
    set(HYKKT_USE_AMD OFF CACHE BOOL "Build without SuiteSparse AMD module." FORCE)
  else()
    target_link_libraries(hykkt_libs PUBLIC AMD)
    install(TARGETS AMD EXPORT HyKKT-targets)
  endif()
else()
  message(STATUS "Not using SuiteSparse AMD")
endif()
