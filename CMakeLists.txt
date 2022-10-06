cmake_minimum_required(VERSION 3.15)

project(HyKKT VERSION "0.0.1")

set(CMAKE_CXX_STANDARD 11)

option(HYKKT_TEST_WITH_BSUB "Use `jsrun` instead of `mpirun` commands when running tests" OFF)
set(HYKKT_CTEST_OUTPUT_DIR ${PROJECT_BINARY_DIR} CACHE PATH "Directory where CTest outputs are saved")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH)

# Configure CUDA
include(CheckLanguage)
enable_language(CUDA)
check_language(CUDA)

#find_package(CUDA REQUIRED)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "Selects CUDA architectures")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# HiOp requires this since we require static
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# Link in required cuda dependencies
find_package(CUDAToolkit REQUIRED)

add_library(cuda_libs INTERFACE)
target_link_libraries(cuda_libs INTERFACE
                      CUDA::cusolver
                      CUDA::cublas
                      CUDA::cusparse
                      CUDA::cudart)

# Enable testing
enable_testing()

# TODO - use target based includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src
                    ${CMAKE_CURRENT_SOURCE_DIR}/examples)

# Add code directory
add_subdirectory(src)
add_subdirectory(examples)

