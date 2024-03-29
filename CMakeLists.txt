cmake_minimum_required(VERSION 3.15)

# Adds version settings and set variable CMAKE_PROJECT_VERSION
project(HyKKT VERSION "0.1.0")

set(CMAKE_CXX_STANDARD 11)

set(PACKAGE_NAME  "HyKKT")
set(PACKAGE_TARNAME "hykkt")

option(HYKKT_TEST_WITH_BSUB "Use `jsrun` instead of `mpirun` commands when running tests" OFF)
option(HYKKT_USE_AMD "Use AMD from SuiteSparse" OFF)
set(HYKKT_CTEST_OUTPUT_DIR ${PROJECT_BINARY_DIR} CACHE PATH "Directory where CTest outputs are saved")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# For Unix CMAKE_INSTALL_PREFIX defaults to /usr/local
# If user doesn't have permission to write to /usr/local than the below line is recommended
# set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "${CMAKE_SOURCE_DIR}/install"FORCE)

message(STATUS "Installation Destination is ${CMAKE_INSTALL_PREFIX}")

# Including clang-format cmake files to do automatic checking of formating
include(./cmake/clang-format)

# Configure CUDA
include(CheckLanguage)
enable_language(CUDA)
check_language(CUDA)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
configure_file(
  "${CMAKE_SOURCE_DIR}/src/hykkt_defs.hpp.in"
  "${CMAKE_BINARY_DIR}/hykkt_defs.hpp")
include_directories(${CMAKE_BINARY_DIR})

if (HYKKT_USE_AMD)
  include(FindAMD)
  if(NOT AMD_LIBRARY)
    message(STATUS "Cannot find AMD, disabling SuiteSparse AMD module ...")
    set(HYKKT_USE_AMD OFF CACHE BOOL "Build without SuiteSparse AMD module." FORCE)
  endif()
else()
  message(STATUS "Not using SuiteSparse AMD")
endif()

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

include(./cmake/FindHykktCudaLibraries)

# Enable testing
enable_testing()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src
                    ${CMAKE_CURRENT_SOURCE_DIR}/examples)

# Add code directory
add_subdirectory(src)

# create package version file
include(CMakePackageConfigHelpers)

# Creates a version file for the package
write_basic_package_version_file(HyKKTConfigVersion.cmake
                                 VERSION ${CMAKE_PROJECT_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
# Generate install rules for targets
install(EXPORT HyKKT-targets
        FILE HyKKTTargets.cmake
        NAMESPACE HyKKT::
        DESTINATION 
        share/hykkt/cmake)

# Creates a config file
configure_package_config_file(./cmake/HyKKTConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/HyKKTConfig.cmake
                              INSTALL_DESTINATION share/hykkt/cmake)

# Generates install rules for cmake config files
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/HyKKTConfig.cmake"
        "${PROJECT_SOURCE_DIR}/build/HyKKTConfigVersion.cmake"
        DESTINATION share/hykkt/cmake)

add_subdirectory(examples)


