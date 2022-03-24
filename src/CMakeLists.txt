set(HYBRID_SOLVER_SRC
  input_functions.cu
  schur_complement_cg.cu)

set(PERM_SRC
  permcheck.cu)

set(RUIZ_SRC
  adapted_Ruiz_scaling.cu
  Ruiz_scaling.cu)

# set_source_files_properties(${HYBRID_SOLVER_SRC} PROPERTIES LANGUAGE CUDA)
# 
# set_source_files_properties(${PERM_SRC} PROPERTIES LANGUAGE CUDA)

set_source_files_properties(${RUIZ_SRC} PROPERTIES LANGUAGE CUDA)
add_library(ruiz_scaling OBJECT ${RUIZ_SRC})
set_target_properties(ruiz_scaling PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# add_executable(cuSolver_driver cuSolver_driver_hybrid.cu)
# target_sources(cuSolver_driver PRIVATE ${HYBRID_SOLVER_SRC} ${PERM_SRC} ${RUIZ_SRC})
# # target_link_libraries(cuSolver_driver hybrid_solver perm_check ruiz_scaling)
# 
# add_executable(perm_driver perm_driver.cu)
# target_sources(perm_driver PRIVATE ${HYBRID_SOLVER_SRC} ${PERM_SRC} ${RUIZ_SRC})
# # target_link_libraries(perm_driver perm_check)

add_executable(ruiz_driver Ruiz_driver.cu)
target_link_libraries(ruiz_driver cuda_libs ruiz_scaling)
add_test(NAME ruiz_scaling_test
         COMMAND $<TARGET_FILE:ruiz_driver>)

