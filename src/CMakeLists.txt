set(a200DIR 
  ${CMAKE_CURRENT_SOURCE_DIR}/mats/a200/)

set(a2000DIR 
  ${CMAKE_CURRENT_SOURCE_DIR}/mats/a2000/)

set(HYBRID_SRC
  input_functions.cu
  schur_complement_cg.cu)

set(PERM_SRC
  permcheck.cu)

set(RUIZ_SRC
  Ruiz_scaling.cu)

set(test1_args
  ${a200DIR}/block_H_matrix_ACTIVSg200_AC_09.mtx
  ${a200DIR}/block_Dd_matrix_ACTIVSg200_AC_09.mtx
  ${a200DIR}/block_J_matrix_ACTIVSg200_AC_09.mtx
  ${a200DIR}/block_Jd_matrix_ACTIVSg200_AC_09.mtx
  ${a200DIR}/block_rx_ACTIVSg200_AC_09.mtx
  ${a200DIR}/block_rs_ACTIVSg200_AC_09.mtx
  ${a200DIR}/block_ry_ACTIVSg200_AC_09.mtx
  ${a200DIR}/block_ryd_ACTIVSg200_AC_09.mtx
  2
  10000.0)

set(test2_args
  ${a2000DIR}/block_H_matrix_ACTIVSg2000_AC_09.mtx
  ${a2000DIR}/block_Dd_matrix_ACTIVSg2000_AC_09.mtx
  ${a2000DIR}/block_J_matrix_ACTIVSg2000_AC_09.mtx
  ${a2000DIR}/block_Jd_matrix_ACTIVSg2000_AC_09.mtx
  ${a2000DIR}/block_rx_ACTIVSg2000_AC_09.mtx
  ${a2000DIR}/block_rs_ACTIVSg2000_AC_09.mtx
  ${a2000DIR}/block_ry_ACTIVSg2000_AC_09.mtx
  ${a2000DIR}/block_ryd_ACTIVSg2000_AC_09.mtx
  3 
  10000.0)

set_source_files_properties(${HYBRID_SRC} ${PERM_SRC} ${RUIZ_SRC} PROPERTIES LANGUAGE CUDA)
add_library(hybrid OBJECT ${HYBRID_SRC} ${PERM_SRC} ${RUIZ_SRC})
set_target_properties(hybrid PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

set_source_files_properties(${PERM_SRC} PROPERTIES LANGUAGE CUDA)
add_library(permutation OBJECT ${PERM_SRC})
set_target_properties(permutation PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

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

add_executable(perm_driver perm_driver.cu)
target_link_libraries(perm_driver cuda_libs permutation)
add_test(NAME permutation
         COMMAND $<TARGET_FILE:perm_driver>)

add_executable(ruiz_driver Ruiz_driver.cu)
target_link_libraries(ruiz_driver cuda_libs ruiz_scaling)
add_test(NAME ruiz_scaling_test
         COMMAND $<TARGET_FILE:ruiz_driver>)

add_executable(hybrid_driver cuSolver_driver_hybrid.cu)
target_link_libraries(hybrid_driver cuda_libs hybrid)
add_test(NAME hybrid1 COMMAND $<TARGET_FILE:hybrid_driver> ${test1_args})
add_test(NAME hybrid2 COMMAND $<TARGET_FILE:hybrid_driver> ${test2_args})
