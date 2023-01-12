cmake_minimum_required(VERSION 3.15)

project(HyKKT VERSION "0.0.1")

set(CMAKE_CXX_STANDARD 11)

set(PACKAGE_NAME  "HyKKT")
set(PACKAGE_TARNAME "hykkt")

option(HYKKT_TEST_WITH_BSUB "Use `jsrun` instead of `mpirun` commands when running tests" OFF)
set(HYKKT_CTEST_OUTPUT_DIR ${PROJECT_BINARY_DIR} CACHE PATH "Directory where CTest outputs are saved")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# For Unix CMAKE_INSTALL_PREFIX defaults to /usr/local
# If user doesn't have permission to write to /usr/local than the below line is recommended
#set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "${CMAKE_SOURCE_DIR}/install"FORCE)

message(STATUS "Installation Destination is ${CMAKE_INSTALL_PREFIX}")

# Configure CUDA
include(CheckLanguage)
enable_language(CUDA)
check_language(CUDA)

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

include(FindHykktCudaLibraries)

# Enable testing
enable_testing()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src
                    ${CMAKE_CURRENT_SOURCE_DIR}/examples)

# Add code directory
add_subdirectory(src)

# create package version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(HyKKTConfigVersion.cmake
                                 VERSION ${CMAKE_PROJECT_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(EXPORT HyKKT-targets
        FILE HyKKTTargets.cmake
        NAMESPACE HyKKT::
        DESTINATION 
        share/hykkt/cmake)

configure_package_config_file(HyKKTConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/HyKKTConfig.cmake
                              INSTALL_DESTINATION share/hykkt/cmake)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/HyKKTConfig.cmake"
        "${PROJECT_SOURCE_DIR}/build/HyKKTConfigVersion.cmake"
        DESTINATION share/hykkt/cmake)

add_subdirectory(examples)


