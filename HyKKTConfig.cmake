# Author(s):
# - Ryan Danehy <ryan.danehy@pnnl.gov>


include("${CMAKE_CURRENT_LIST_DIR}/HyKKTTargets.cmake")

include(CheckLanguage)
# This must come before enable_language(CUDA)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.6/bin/nvcc)
enable_language(CUDA)
check_language(CUDA)
set(CMAKE_CUDA_FLAGS " --expt-extended-lambda")
find_package(CUDAToolkit REQUIRED)

