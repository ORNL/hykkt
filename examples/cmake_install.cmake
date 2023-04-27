# Install script for directory: /home/shakedregev/hykkt/examples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/usr/local/perm_driver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/perm_driver")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/usr/local/perm_driver"
         RPATH "/usr/local/cuda-11.6/targets/x86_64-linux/lib")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/perm_driver")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local" TYPE EXECUTABLE FILES "/home/shakedregev/hykkt/examples/perm_driver")
  if(EXISTS "$ENV{DESTDIR}/usr/local/perm_driver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/perm_driver")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/usr/local/perm_driver"
         OLD_RPATH "/home/shakedregev/hykkt/src:/usr/local/cuda-11.6/targets/x86_64-linux/lib:"
         NEW_RPATH "/usr/local/cuda-11.6/targets/x86_64-linux/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/usr/local/perm_driver")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/usr/local/ruiz_driver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/ruiz_driver")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/usr/local/ruiz_driver"
         RPATH "/usr/local/cuda-11.6/targets/x86_64-linux/lib")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/ruiz_driver")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local" TYPE EXECUTABLE FILES "/home/shakedregev/hykkt/examples/ruiz_driver")
  if(EXISTS "$ENV{DESTDIR}/usr/local/ruiz_driver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/ruiz_driver")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/usr/local/ruiz_driver"
         OLD_RPATH "/home/shakedregev/hykkt/src:/usr/local/cuda-11.6/targets/x86_64-linux/lib:"
         NEW_RPATH "/usr/local/cuda-11.6/targets/x86_64-linux/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/usr/local/ruiz_driver")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/usr/local/chol_driver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/chol_driver")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/usr/local/chol_driver"
         RPATH "/usr/local/cuda-11.6/targets/x86_64-linux/lib")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/chol_driver")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local" TYPE EXECUTABLE FILES "/home/shakedregev/hykkt/examples/chol_driver")
  if(EXISTS "$ENV{DESTDIR}/usr/local/chol_driver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/chol_driver")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/usr/local/chol_driver"
         OLD_RPATH "/home/shakedregev/hykkt/src:/usr/local/cuda-11.6/targets/x86_64-linux/lib:"
         NEW_RPATH "/usr/local/cuda-11.6/targets/x86_64-linux/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/usr/local/chol_driver")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/usr/local/schur_cg_driver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/schur_cg_driver")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/usr/local/schur_cg_driver"
         RPATH "/usr/local/cuda-11.6/targets/x86_64-linux/lib")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/schur_cg_driver")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local" TYPE EXECUTABLE FILES "/home/shakedregev/hykkt/examples/schur_cg_driver")
  if(EXISTS "$ENV{DESTDIR}/usr/local/schur_cg_driver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/schur_cg_driver")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/usr/local/schur_cg_driver"
         OLD_RPATH "/home/shakedregev/hykkt/src:/usr/local/cuda-11.6/targets/x86_64-linux/lib:"
         NEW_RPATH "/usr/local/cuda-11.6/targets/x86_64-linux/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/usr/local/schur_cg_driver")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/usr/local/hybrid_driver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/hybrid_driver")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/usr/local/hybrid_driver"
         RPATH "/usr/local/cuda-11.6/targets/x86_64-linux/lib")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/hybrid_driver")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local" TYPE EXECUTABLE FILES "/home/shakedregev/hykkt/examples/hybrid_driver")
  if(EXISTS "$ENV{DESTDIR}/usr/local/hybrid_driver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/hybrid_driver")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/usr/local/hybrid_driver"
         OLD_RPATH "/home/shakedregev/hykkt/src:/usr/local/cuda-11.6/targets/x86_64-linux/lib:"
         NEW_RPATH "/usr/local/cuda-11.6/targets/x86_64-linux/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/usr/local/hybrid_driver")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/usr/local/solver_driver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/solver_driver")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/usr/local/solver_driver"
         RPATH "/usr/local/cuda-11.6/targets/x86_64-linux/lib")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/solver_driver")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local" TYPE EXECUTABLE FILES "/home/shakedregev/hykkt/examples/solver_driver")
  if(EXISTS "$ENV{DESTDIR}/usr/local/solver_driver" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/solver_driver")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/usr/local/solver_driver"
         OLD_RPATH "/home/shakedregev/hykkt/src:/usr/local/cuda-11.6/targets/x86_64-linux/lib:"
         NEW_RPATH "/usr/local/cuda-11.6/targets/x86_64-linux/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/usr/local/solver_driver")
    endif()
  endif()
endif()

