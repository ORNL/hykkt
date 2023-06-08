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

set(applierDIR
  ${PROJECT_SOURCE_DIR}/src/mats/applier/)

set(lpqpDIR 
  ${PROJECT_SOURCE_DIR}/src/mats/lpqp/)

set(lbhelpersDIR
  ${PROJECT_SOURCE_DIR}/src/mats/lbhelpers/)

#set(logbarriersmallDIR
#  ${PROJECT_SOURCE_DIR}/src/mats/smallproblem/)

#set(logbarrierlargeDIR
#  ${PROJECT_SOURCE_DIR}/src/mats/largeproblem/)

set(logbarriersimpleDIR
  ${PROJECT_SOURCE_DIR}/src/mats/logbarrier/)

set(sandboxDIR
  ${PROJECT_SOURCE_DIR}/src/mats/sandbox/)

#set(BiELLDIR
#  ${PROJECT_SOURCE_DIR}/src/mats/BiELL/)

set(test_applier_args
  ${applierDIR}/q.mtx
  ${applierDIR}/a.mtx
  ${applierDIR}/h.mtx
  ${applierDIR}/vin.mtx
  ${applierDIR}/exp_apply.mtx
  ${applierDIR}/exp_extract.mtx
  ${applierDIR}/exp_diag.mtx
  1.7)

set(test_lp_args
  ${lpqpDIR}/lp_a_test.mtx
  ${lpqpDIR}/lp_h1_test.mtx
  ${lpqpDIR}/lp_h2_test.mtx
  ${lpqpDIR}/lp_res1_test.mtx
  ${lpqpDIR}/lp_res2_test.mtx)

set(test_qp_args
  ${lpqpDIR}/qp_q_test.mtx
  ${lpqpDIR}/qp_a_test.mtx
  ${lpqpDIR}/qp_h1_test.mtx
  ${lpqpDIR}/qp_h2_test.mtx
  ${lpqpDIR}/qp_res1_test.mtx
  ${lpqpDIR}/qp_res2_test.mtx
  2.3)

set(lbhelper_args
  ${lbhelpersDIR}/q.mtx
  ${lbhelpersDIR}/a.mtx
  ${lbhelpersDIR}/b.mtx
  ${lbhelpersDIR}/c.mtx
  ${lbhelpersDIR}/x.mtx
  -7.062633655790155 #qp objective expected
  -19.785143970396057 #lp objective expected
  ${lbhelpersDIR}/qp_gradient.mtx
  ${lbhelpersDIR}/lp_gradient.mtx
  ${lbhelpersDIR}/inv_hessian.mtx
  1.3) #t

set(log_barrier0_args
  ${logbarriersimpleDIR}/a.mtx
  ${logbarriersimpleDIR}/b.mtx
  ${logbarriersimpleDIR}/c.mtx)

set(log_barrier1_args
  ${logbarriersmallDIR}/a.mtx
  ${logbarriersmallDIR}/b.mtx
  ${logbarriersmallDIR}/c.mtx)

set(log_barrier2_args
  ${logbarrierlargeDIR}/a.mtx
  ${logbarrierlargeDIR}/b.mtx
  ${logbarrierlargeDIR}/c.mtx)

set(sandbox_args
  ${sandboxDIR}/v1.mtx
  ${sandboxDIR}/v2.mtx
  100
  -12.60027265479004
  ${sandboxDIR}/a.mtx
  ${sandboxDIR}/x.mtx
  #${logbarriersmallDIR}/a.mtx
  #${logbarriersmallDIR}/c.mtx
  ${sandboxDIR}/y.mtx)

set(BiELL1_args
  ${BiELLDIR}/a1.mtx
  ${BiELLDIR}/x1.mtx
  ${BiELLDIR}/y1.mtx)

if(HYKKT_TEST_WITH_BSUB)
  set(RUNCMD "jsrun" "-n" "1" "-g" "1")
else()
  set(RUNCMD "cuda-memcheck") # No special command is needed to run this program  
endif()

add_executable(perm_driver perm_driver.cpp)
target_link_libraries(perm_driver PUBLIC HyKKT::HyKKT)
#add_test(NAME permutation
#  COMMAND ${RUNCMD} $<TARGET_FILE:perm_driver>)

add_executable(ruiz_driver Ruiz_driver.cpp)
target_link_libraries(ruiz_driver PUBLIC HyKKT::HyKKT)
#add_test(NAME ruiz_scaling_test
#  COMMAND ${RUNCMD} $<TARGET_FILE:ruiz_driver>)

add_executable(chol_driver cuSolver_driver_cholesky.cpp)
target_link_libraries(chol_driver PUBLIC HyKKT::HyKKT)
#add_test(NAME cholesky 
# COMMAND ${RUNCMD} $<TARGET_FILE:chol_driver> ${testchol_args})

add_executable(schur_cg_driver cuSolver_driver_schur_cg.cpp)
target_link_libraries(schur_cg_driver PUBLIC HyKKT::HyKKT)
#add_test(NAME schur_cg 
#  COMMAND ${RUNCMD} $<TARGET_FILE:schur_cg_driver> ${testcg_args})

add_executable(hybrid_driver cuSolver_driver_hybrid.cpp)
target_link_libraries(hybrid_driver PUBLIC HyKKT::HyKKT)
#add_test(NAME hybrid1 
#  COMMAND ${RUNCMD} $<TARGET_FILE:hybrid_driver> ${test1_args})
#add_test(NAME hybrid2
#  COMMAND ${RUNCMD} $<TARGET_FILE:hybrid_driver> ${test2_args})

add_executable(solver_driver cuSolver_driver_solver.cpp)
target_link_libraries(solver_driver PUBLIC HyKKT::HyKKT)
#add_test(NAME solver1
#  COMMAND ${RUNCMD} $<TARGET_FILE:solver_driver> ${test_solver1_args})
#add_test(NAME solver2
#  COMMAND ${RUNCMD} $<TARGET_FILE:solver_driver> ${test_solver2_args})
  
add_executable(solver_driver_emulator cuSolver_driver_solver_emulator.cpp)
target_link_libraries(solver_driver_emulator PUBLIC HyKKT::HyKKT)

#find_package(benchmark REQUIRED)
#include_directories(${benchmark_INCLUDE_DIRS})
#add_executable(sandbox sandbox_test.cpp)
#target_link_libraries(sandbox PUBLIC benchmark::benchmark HyKKT::HyKKT)
#add_test(NAME sandbox1
# COMMAND ${RUNCMD} $<TARGET_FILE:sandbox>) #${sandbox_args})

add_executable(operator_driver driver_operator_applier.cpp)
target_link_libraries(operator_driver PUBLIC HyKKT::HyKKT)
add_test(NAME lq_applier
  COMMAND ${RUNCMD} $<TARGET_FILE:operator_driver> ${test_applier_args})

add_executable(lpqp_solver driver_lpqp_solver.cpp)
target_link_libraries(lpqp_solver PUBLIC HyKKT::HyKKT)
add_test(NAME lp_solve
  COMMAND ${RUNCMD} $<TARGET_FILE:lpqp_solver> ${test_lp_args})
add_test(NAME qp_solve
  COMMAND ${RUNCMD} $<TARGET_FILE:lpqp_solver> ${test_qp_args})

add_executable(lb_helper driver_lb_helper.cpp)
target_link_libraries(lb_helper PUBLIC HyKKT::HyKKT)
add_test(NAME lb_helper
  COMMAND ${RUNCMD} $<TARGET_FILE:lb_helper> ${lbhelper_args})

add_executable(log_barrier driver_log_barrier.cpp)
target_link_libraries(log_barrier PUBLIC HyKKT::HyKKT)
add_test(NAME log_barrier0
  COMMAND ${RUNCMD} $<TARGET_FILE:log_barrier> ${log_barrier0_args})
#add_test(NAME log_barrier1
#  COMMAND ${RUNCMD} $<TARGET_FILE:log_barrier> ${log_barrier1_args})
#add_test(NAME log_barrier2
#  COMMAND ${RUNCMD} $<TARGET_FILE:log_barrier> ${log_barrier2_args})

#add_executable(BiELL driver_BiELL.cpp)
#target_link_libraries(BiELL PUBLIC HyKKT::HyKKT benchmark::benchmark)
#add_test(NAME BiELL1
#  COMMAND ${RUNCMD} $<TARGET_FILE:BiELL>) #${BiELL1_args})

# install examples into user specified dir; default is /usr/local
install(TARGETS perm_driver DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(TARGETS ruiz_driver DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(TARGETS chol_driver DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(TARGETS schur_cg_driver DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(TARGETS hybrid_driver DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(TARGETS solver_driver DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(TARGETS solver_driver_emulator DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(TARGETS operator_driver DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(TARGETS lpqp_solver DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(TARGETS log_barrier DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(TARGETS lb_helper DESTINATION "${CMAKE_INSTALL_PREFIX}")
#install(TARGETS sandbox DESTINATION "${CMAKE_INSTALL_PREFIX}")
#install(TARGETS BiELL DESTINATION "${CMAKE_INSTALL_PREFIX}")

if (HYKKT_USE_AMD)
  if(NOT AMD_LIBRARY)
    message(STATUS "Cannot find AMD, disabling SuiteSparse AMD module ...")
    set(HYKKT_USE_AMD OFF CACHE BOOL "Build without SuiteSparse AMD module." FORCE)
  else()
    target_link_libraries(perm_driver PUBLIC AMD)
    target_link_libraries(ruiz_driver PUBLIC AMD)
    target_link_libraries(chol_driver PUBLIC AMD)
    target_link_libraries(schur_cg_driver PUBLIC AMD)
    target_link_libraries(hybrid_driver PUBLIC AMD)
    target_link_libraries(solver_driver PUBLIC AMD)
    target_link_libraries(solver_driver_emulator PUBLIC AMD)
    target_link_libraries(driver_operator_applier PUBLIC AMD)
    target_link_libraries(lpqp_solver PUBLIC AMD)
    target_link_libraries(log_barrier PUBLIC AMD)
    target_link_libraries(lb_helper PUBLIC AMD)
    #target_link_libraries(sandbox PUBLIC AMD)
    #target_link_libraries(BiELL PUBLIC AMD)
    endif()
else()
  message(STATUS "Not using SuiteSparse AMD")
endif()
