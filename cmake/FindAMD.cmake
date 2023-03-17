find_library(AMD_LIBRARY
  NAMES
  amd
  PATHS
  ${AMD_DIR} $ENV{AMD_DIR}
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  PATH_SUFFIXES
  lib64 lib)

if(AMD_LIBRARY)
  get_filename_component(AMD_LIBRARY_DIR ${AMD_LIBRARY} DIRECTORY)
endif()

find_path(AMD_INCLUDE_DIR
  NAMES
  amd.h
  PATHS
  ${AMD_DIR} $ENV{AMD_DIR} ${AMD_LIBRARY_DIR}/..
  PATH_SUFFIXES
  include
  include/suitesparse
  include/amd)

if(AMD_LIBRARY)
  message(STATUS "Found AMD include: ${AMD_INCLUDE_DIR}")
  message(STATUS "Found AMD library: ${AMD_LIBRARY}")
  add_library(AMD INTERFACE)
  target_link_libraries(AMD INTERFACE ${AMD_LIBRARY})
  target_include_directories(AMD INTERFACE ${AMD_INCLUDE_DIR})
  get_filename_component(AMD_LIB_DIR ${AMD_LIBRARY} DIRECTORY)
  set(CMAKE_INSTALL_RPATH "${AMD_LIB_DIR}")
else()
  message(STATUS "AMD was not found.")
endif()

set(AMD_INCLUDE_DIR CACHE PATH "Path to amd.h")
set(AMD_LIBRARY CACHE PATH "Path to amd library")


