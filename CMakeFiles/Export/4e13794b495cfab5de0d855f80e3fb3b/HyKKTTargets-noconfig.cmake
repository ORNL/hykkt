#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "HyKKT::hykkt_libs" for configuration ""
set_property(TARGET HyKKT::hykkt_libs APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(HyKKT::hykkt_libs PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libhykkt_libs.so"
  IMPORTED_SONAME_NOCONFIG "libhykkt_libs.so"
  )

list(APPEND _cmake_import_check_targets HyKKT::hykkt_libs )
list(APPEND _cmake_import_check_files_for_HyKKT::hykkt_libs "${_IMPORT_PREFIX}/lib/libhykkt_libs.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
