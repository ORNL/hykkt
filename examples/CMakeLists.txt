set(a200DIR 
  ${PROJECT_SOURCE_DIR}/src/mats/a200/)

set(a2000DIR 
  ${PROJECT_SOURCE_DIR}/src/mats/a2000/)

set(a200_chol_DIR 
  ${PROJECT_SOURCE_DIR}/src/mats/a200_chol/)

set(a200_schur_DIR 
  ${PROJECT_SOURCE_DIR}/src/mats/a200_schur/)

set(testchol_args
  ${a200_chol_DIR}/H_matrix_ACTIVSg200_AC_00.mtx
  ${a200_chol_DIR}/rhs_chol_ACTIVSg200_AC_00.mtx)

set(testcg_args
  ${a200_schur_DIR}/JC_matrix_ACTIVSg200_AC_00.mtx
  ${a200_schur_DIR}/H_matrix_ACTIVSg200_AC_00.mtx
  ${a200_schur_DIR}/CG_rhs_ACTIVSg200_AC_00.mtx)

set(test1_args
  ${a200DIR}/block_H_matrix_ACTIVSg200_AC_10.mtx
  ${a200DIR}/block_Dd_matrix_ACTIVSg200_AC_10.mtx
  ${a200DIR}/block_J_matrix_ACTIVSg200_AC_10.mtx
  ${a200DIR}/block_Jd_matrix_ACTIVSg200_AC_10.mtx
  ${a200DIR}/block_rx_ACTIVSg200_AC_10.mtx
  ${a200DIR}/block_rs_ACTIVSg200_AC_10.mtx
  ${a200DIR}/block_ry_ACTIVSg200_AC_10.mtx
  ${a200DIR}/block_ryd_ACTIVSg200_AC_10.mtx
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

set(test_solver1_args
  ${a200DIR}/block_H_matrix_ACTIVSg200_AC_09.mtx
  ${a200DIR}/block_Dd_matrix_ACTIVSg200_AC_09.mtx
  ${a200DIR}/block_J_matrix_ACTIVSg200_AC_09.mtx
  ${a200DIR}/block_Jd_matrix_ACTIVSg200_AC_09.mtx
  ${a200DIR}/block_rx_ACTIVSg200_AC_09.mtx
  ${a200DIR}/block_rs_ACTIVSg200_AC_09.mtx
  ${a200DIR}/block_ry_ACTIVSg200_AC_09.mtx
  ${a200DIR}/block_ryd_ACTIVSg200_AC_09.mtx
  
  ${a200DIR}/block_H_matrix_ACTIVSg200_AC_10.mtx
  ${a200DIR}/block_Dd_matrix_ACTIVSg200_AC_10.mtx
  ${a200DIR}/block_J_matrix_ACTIVSg200_AC_10.mtx
  ${a200DIR}/block_Jd_matrix_ACTIVSg200_AC_10.mtx
  ${a200DIR}/block_rx_ACTIVSg200_AC_10.mtx
  ${a200DIR}/block_rs_ACTIVSg200_AC_10.mtx
  ${a200DIR}/block_ry_ACTIVSg200_AC_10.mtx
  ${a200DIR}/block_ryd_ACTIVSg200_AC_10.mtx
  2
  10000.0)

set(test_solver2_args
  ${a2000DIR}/block_H_matrix_ACTIVSg2000_AC_09.mtx
  ${a2000DIR}/block_Dd_matrix_ACTIVSg2000_AC_09.mtx
  ${a2000DIR}/block_J_matrix_ACTIVSg2000_AC_09.mtx
  ${a2000DIR}/block_Jd_matrix_ACTIVSg2000_AC_09.mtx
  ${a2000DIR}/block_rx_ACTIVSg2000_AC_09.mtx
  ${a2000DIR}/block_rs_ACTIVSg2000_AC_09.mtx
  ${a2000DIR}/block_ry_ACTIVSg2000_AC_09.mtx
  ${a2000DIR}/block_ryd_ACTIVSg2000_AC_09.mtx
  
  ${a2000DIR}/block_H_matrix_ACTIVSg2000_AC_10.mtx
  ${a2000DIR}/block_Dd_matrix_ACTIVSg2000_AC_10.mtx
  ${a2000DIR}/block_J_matrix_ACTIVSg2000_AC_10.mtx
  ${a2000DIR}/block_Jd_matrix_ACTIVSg2000_AC_10.mtx
  ${a2000DIR}/block_rx_ACTIVSg2000_AC_10.mtx
  ${a2000DIR}/block_rs_ACTIVSg2000_AC_10.mtx
  ${a2000DIR}/block_ry_ACTIVSg2000_AC_10.mtx
  ${a2000DIR}/block_ryd_ACTIVSg2000_AC_10.mtx
  3
  10000.0)

if(HYKKT_TEST_WITH_BSUB)
  set(RUNCMD "jsrun" "-n" "1" "-g" "1")
else()
  set(RUNCMD "cuda-memcheck") # No special command is needed to run this program  
endif()

add_executable(perm_driver perm_driver.cpp)
target_link_libraries(perm_driver PUBLIC cuda_libs $<TARGET_OBJECTS:hybrid>)
add_test(NAME permutation
  COMMAND ${RUNCMD} $<TARGET_FILE:perm_driver>)

add_executable(ruiz_driver Ruiz_driver.cpp)
target_link_libraries(ruiz_driver PUBLIC cuda_libs $<TARGET_OBJECTS:hybrid>)
add_test(NAME ruiz_scaling_test
  COMMAND ${RUNCMD} $<TARGET_FILE:ruiz_driver>)
#
add_executable(chol_driver cuSolver_driver_cholesky.cpp)
target_link_libraries(chol_driver PUBLIC $<TARGET_OBJECTS:hybrid>
  cuda_libs)
add_test(NAME cholesky 
 COMMAND ${RUNCMD} $<TARGET_FILE:chol_driver> ${testchol_args})

add_executable(schur_cg_driver cuSolver_driver_schur_cg.cpp)
target_link_libraries(schur_cg_driver PUBLIC $<TARGET_OBJECTS:hybrid>
  cuda_libs)
add_test(NAME schur_cg 
  COMMAND ${RUNCMD} $<TARGET_FILE:schur_cg_driver> ${testcg_args})

add_executable(hybrid_driver cuSolver_driver_hybrid.cpp)
target_link_libraries(hybrid_driver PUBLIC
  cuda_libs $<TARGET_OBJECTS:hybrid>)
add_test(NAME hybrid1 
  COMMAND ${RUNCMD} $<TARGET_FILE:hybrid_driver> ${test1_args})
add_test(NAME hybrid2
  COMMAND ${RUNCMD} $<TARGET_FILE:hybrid_driver> ${test2_args})
#
add_executable(solver_driver cuSolver_driver_solver.cpp)
target_link_libraries(solver_driver PUBLIC cuda_libs $<TARGET_OBJECTS:hybrid>)
add_test(NAME solver1
  COMMAND ${RUNCMD} $<TARGET_FILE:solver_driver> ${test_solver1_args})
add_test(NAME solver2
  COMMAND ${RUNCMD} $<TARGET_FILE:solver_driver> ${test_solver2_args})

add_executable(solver_driver_fileio cuSolver_driver_solver_fileio.cpp)
target_link_libraries(solver_driver_fileio PUBLIC cuda_libs $<TARGET_OBJECTS:hybrid>)
add_test(NAME solver_fileio1
  COMMAND ${RUNCMD} $<TARGET_FILE:solver_driver_fileio> ${test_solver1_args})
add_test(NAME solver_fileio2
  COMMAND ${RUNCMD} $<TARGET_FILE:solver_driver_fileio> ${test_solver2_args})
  