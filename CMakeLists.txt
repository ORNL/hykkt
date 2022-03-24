cmake_minimum_required(VERSION 3.15)

project(HyKKT VERSION "0.0.1")

set(CMAKE_CXX_STANDARD 11)

# Configure CUDA
include(CheckLanguage)
enable_language(CUDA)
check_language(CUDA)

find_package(CUDA REQUIRED)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# HiOp requires this since we require static
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Link in required cuda dependencies
find_package(CUDAToolkit REQUIRED)

add_library(cuda_libs INTERFACE)
target_link_libraries(cuda_libs INTERFACE
                      CUDA::cusolver
                      CUDA::cublas
                      CUDA::cusparse)

# Enable testing
enable_testing()

# TODO - use target based includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add code directory
add_subdirectory(src)

