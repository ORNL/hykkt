# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/1288/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1288/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shakedregev/hykkt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shakedregev/hykkt

# Include any dependencies generated for this target.
include src/CMakeFiles/hykkt_libs.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/hykkt_libs.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/hykkt_libs.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/hykkt_libs.dir/flags.make

src/CMakeFiles/hykkt_libs.dir/MMatrix.cpp.o: src/CMakeFiles/hykkt_libs.dir/flags.make
src/CMakeFiles/hykkt_libs.dir/MMatrix.cpp.o: src/MMatrix.cpp
src/CMakeFiles/hykkt_libs.dir/MMatrix.cpp.o: src/CMakeFiles/hykkt_libs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/hykkt_libs.dir/MMatrix.cpp.o"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/hykkt_libs.dir/MMatrix.cpp.o -MF CMakeFiles/hykkt_libs.dir/MMatrix.cpp.o.d -o CMakeFiles/hykkt_libs.dir/MMatrix.cpp.o -c /home/shakedregev/hykkt/src/MMatrix.cpp

src/CMakeFiles/hykkt_libs.dir/MMatrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hykkt_libs.dir/MMatrix.cpp.i"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shakedregev/hykkt/src/MMatrix.cpp > CMakeFiles/hykkt_libs.dir/MMatrix.cpp.i

src/CMakeFiles/hykkt_libs.dir/MMatrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hykkt_libs.dir/MMatrix.cpp.s"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shakedregev/hykkt/src/MMatrix.cpp -o CMakeFiles/hykkt_libs.dir/MMatrix.cpp.s

src/CMakeFiles/hykkt_libs.dir/input_functions.cpp.o: src/CMakeFiles/hykkt_libs.dir/flags.make
src/CMakeFiles/hykkt_libs.dir/input_functions.cpp.o: src/input_functions.cpp
src/CMakeFiles/hykkt_libs.dir/input_functions.cpp.o: src/CMakeFiles/hykkt_libs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/hykkt_libs.dir/input_functions.cpp.o"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/hykkt_libs.dir/input_functions.cpp.o -MF CMakeFiles/hykkt_libs.dir/input_functions.cpp.o.d -o CMakeFiles/hykkt_libs.dir/input_functions.cpp.o -c /home/shakedregev/hykkt/src/input_functions.cpp

src/CMakeFiles/hykkt_libs.dir/input_functions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hykkt_libs.dir/input_functions.cpp.i"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shakedregev/hykkt/src/input_functions.cpp > CMakeFiles/hykkt_libs.dir/input_functions.cpp.i

src/CMakeFiles/hykkt_libs.dir/input_functions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hykkt_libs.dir/input_functions.cpp.s"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shakedregev/hykkt/src/input_functions.cpp -o CMakeFiles/hykkt_libs.dir/input_functions.cpp.s

src/CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.o: src/CMakeFiles/hykkt_libs.dir/flags.make
src/CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.o: src/vector_vector_ops.cpp
src/CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.o: src/CMakeFiles/hykkt_libs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.o"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.o -MF CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.o.d -o CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.o -c /home/shakedregev/hykkt/src/vector_vector_ops.cpp

src/CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.i"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shakedregev/hykkt/src/vector_vector_ops.cpp > CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.i

src/CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.s"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shakedregev/hykkt/src/vector_vector_ops.cpp -o CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.s

src/CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.o: src/CMakeFiles/hykkt_libs.dir/flags.make
src/CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.o: src/cusparse_utils.cpp
src/CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.o: src/CMakeFiles/hykkt_libs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.o"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.o -MF CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.o.d -o CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.o -c /home/shakedregev/hykkt/src/cusparse_utils.cpp

src/CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.i"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shakedregev/hykkt/src/cusparse_utils.cpp > CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.i

src/CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.s"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shakedregev/hykkt/src/cusparse_utils.cpp -o CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.s

src/CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.o: src/CMakeFiles/hykkt_libs.dir/flags.make
src/CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.o: src/matrix_matrix_ops.cpp
src/CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.o: src/CMakeFiles/hykkt_libs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.o"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.o -MF CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.o.d -o CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.o -c /home/shakedregev/hykkt/src/matrix_matrix_ops.cpp

src/CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.i"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shakedregev/hykkt/src/matrix_matrix_ops.cpp > CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.i

src/CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.s"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shakedregev/hykkt/src/matrix_matrix_ops.cpp -o CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.s

src/CMakeFiles/hykkt_libs.dir/PermClass.cpp.o: src/CMakeFiles/hykkt_libs.dir/flags.make
src/CMakeFiles/hykkt_libs.dir/PermClass.cpp.o: src/PermClass.cpp
src/CMakeFiles/hykkt_libs.dir/PermClass.cpp.o: src/CMakeFiles/hykkt_libs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/CMakeFiles/hykkt_libs.dir/PermClass.cpp.o"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/hykkt_libs.dir/PermClass.cpp.o -MF CMakeFiles/hykkt_libs.dir/PermClass.cpp.o.d -o CMakeFiles/hykkt_libs.dir/PermClass.cpp.o -c /home/shakedregev/hykkt/src/PermClass.cpp

src/CMakeFiles/hykkt_libs.dir/PermClass.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hykkt_libs.dir/PermClass.cpp.i"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shakedregev/hykkt/src/PermClass.cpp > CMakeFiles/hykkt_libs.dir/PermClass.cpp.i

src/CMakeFiles/hykkt_libs.dir/PermClass.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hykkt_libs.dir/PermClass.cpp.s"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shakedregev/hykkt/src/PermClass.cpp -o CMakeFiles/hykkt_libs.dir/PermClass.cpp.s

src/CMakeFiles/hykkt_libs.dir/RuizClass.cpp.o: src/CMakeFiles/hykkt_libs.dir/flags.make
src/CMakeFiles/hykkt_libs.dir/RuizClass.cpp.o: src/RuizClass.cpp
src/CMakeFiles/hykkt_libs.dir/RuizClass.cpp.o: src/CMakeFiles/hykkt_libs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/CMakeFiles/hykkt_libs.dir/RuizClass.cpp.o"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/hykkt_libs.dir/RuizClass.cpp.o -MF CMakeFiles/hykkt_libs.dir/RuizClass.cpp.o.d -o CMakeFiles/hykkt_libs.dir/RuizClass.cpp.o -c /home/shakedregev/hykkt/src/RuizClass.cpp

src/CMakeFiles/hykkt_libs.dir/RuizClass.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hykkt_libs.dir/RuizClass.cpp.i"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shakedregev/hykkt/src/RuizClass.cpp > CMakeFiles/hykkt_libs.dir/RuizClass.cpp.i

src/CMakeFiles/hykkt_libs.dir/RuizClass.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hykkt_libs.dir/RuizClass.cpp.s"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shakedregev/hykkt/src/RuizClass.cpp -o CMakeFiles/hykkt_libs.dir/RuizClass.cpp.s

src/CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.o: src/CMakeFiles/hykkt_libs.dir/flags.make
src/CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.o: src/SchurComplementConjugateGradient.cpp
src/CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.o: src/CMakeFiles/hykkt_libs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.o"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.o -MF CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.o.d -o CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.o -c /home/shakedregev/hykkt/src/SchurComplementConjugateGradient.cpp

src/CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.i"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shakedregev/hykkt/src/SchurComplementConjugateGradient.cpp > CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.i

src/CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.s"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shakedregev/hykkt/src/SchurComplementConjugateGradient.cpp -o CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.s

src/CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.o: src/CMakeFiles/hykkt_libs.dir/flags.make
src/CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.o: src/CholeskyClass.cpp
src/CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.o: src/CMakeFiles/hykkt_libs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.o"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.o -MF CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.o.d -o CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.o -c /home/shakedregev/hykkt/src/CholeskyClass.cpp

src/CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.i"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shakedregev/hykkt/src/CholeskyClass.cpp > CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.i

src/CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.s"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shakedregev/hykkt/src/CholeskyClass.cpp -o CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.s

src/CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.o: src/CMakeFiles/hykkt_libs.dir/flags.make
src/CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.o: src/SpgemmClass.cpp
src/CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.o: src/CMakeFiles/hykkt_libs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object src/CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.o"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.o -MF CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.o.d -o CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.o -c /home/shakedregev/hykkt/src/SpgemmClass.cpp

src/CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.i"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shakedregev/hykkt/src/SpgemmClass.cpp > CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.i

src/CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.s"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shakedregev/hykkt/src/SpgemmClass.cpp -o CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.s

src/CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.o: src/CMakeFiles/hykkt_libs.dir/flags.make
src/CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.o: src/HykktSolver.cpp
src/CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.o: src/CMakeFiles/hykkt_libs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object src/CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.o"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.o -MF CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.o.d -o CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.o -c /home/shakedregev/hykkt/src/HykktSolver.cpp

src/CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.i"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shakedregev/hykkt/src/HykktSolver.cpp > CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.i

src/CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.s"
	cd /home/shakedregev/hykkt/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shakedregev/hykkt/src/HykktSolver.cpp -o CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.s

src/CMakeFiles/hykkt_libs.dir/cuda_memory_utils.cu.o: src/CMakeFiles/hykkt_libs.dir/flags.make
src/CMakeFiles/hykkt_libs.dir/cuda_memory_utils.cu.o: src/CMakeFiles/hykkt_libs.dir/includes_CUDA.rsp
src/CMakeFiles/hykkt_libs.dir/cuda_memory_utils.cu.o: src/cuda_memory_utils.cu
src/CMakeFiles/hykkt_libs.dir/cuda_memory_utils.cu.o: src/CMakeFiles/hykkt_libs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CUDA object src/CMakeFiles/hykkt_libs.dir/cuda_memory_utils.cu.o"
	cd /home/shakedregev/hykkt/src && /usr/local/cuda-11.6/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT src/CMakeFiles/hykkt_libs.dir/cuda_memory_utils.cu.o -MF CMakeFiles/hykkt_libs.dir/cuda_memory_utils.cu.o.d -x cu -c /home/shakedregev/hykkt/src/cuda_memory_utils.cu -o CMakeFiles/hykkt_libs.dir/cuda_memory_utils.cu.o

src/CMakeFiles/hykkt_libs.dir/cuda_memory_utils.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/hykkt_libs.dir/cuda_memory_utils.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/CMakeFiles/hykkt_libs.dir/cuda_memory_utils.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/hykkt_libs.dir/cuda_memory_utils.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

src/CMakeFiles/hykkt_libs.dir/matrix_vector_ops.cu.o: src/CMakeFiles/hykkt_libs.dir/flags.make
src/CMakeFiles/hykkt_libs.dir/matrix_vector_ops.cu.o: src/CMakeFiles/hykkt_libs.dir/includes_CUDA.rsp
src/CMakeFiles/hykkt_libs.dir/matrix_vector_ops.cu.o: src/matrix_vector_ops.cu
src/CMakeFiles/hykkt_libs.dir/matrix_vector_ops.cu.o: src/CMakeFiles/hykkt_libs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CUDA object src/CMakeFiles/hykkt_libs.dir/matrix_vector_ops.cu.o"
	cd /home/shakedregev/hykkt/src && /usr/local/cuda-11.6/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT src/CMakeFiles/hykkt_libs.dir/matrix_vector_ops.cu.o -MF CMakeFiles/hykkt_libs.dir/matrix_vector_ops.cu.o.d -x cu -c /home/shakedregev/hykkt/src/matrix_vector_ops.cu -o CMakeFiles/hykkt_libs.dir/matrix_vector_ops.cu.o

src/CMakeFiles/hykkt_libs.dir/matrix_vector_ops.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/hykkt_libs.dir/matrix_vector_ops.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/CMakeFiles/hykkt_libs.dir/matrix_vector_ops.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/hykkt_libs.dir/matrix_vector_ops.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

src/CMakeFiles/hykkt_libs.dir/permcheck.cu.o: src/CMakeFiles/hykkt_libs.dir/flags.make
src/CMakeFiles/hykkt_libs.dir/permcheck.cu.o: src/CMakeFiles/hykkt_libs.dir/includes_CUDA.rsp
src/CMakeFiles/hykkt_libs.dir/permcheck.cu.o: src/permcheck.cu
src/CMakeFiles/hykkt_libs.dir/permcheck.cu.o: src/CMakeFiles/hykkt_libs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CUDA object src/CMakeFiles/hykkt_libs.dir/permcheck.cu.o"
	cd /home/shakedregev/hykkt/src && /usr/local/cuda-11.6/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT src/CMakeFiles/hykkt_libs.dir/permcheck.cu.o -MF CMakeFiles/hykkt_libs.dir/permcheck.cu.o.d -x cu -c /home/shakedregev/hykkt/src/permcheck.cu -o CMakeFiles/hykkt_libs.dir/permcheck.cu.o

src/CMakeFiles/hykkt_libs.dir/permcheck.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/hykkt_libs.dir/permcheck.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/CMakeFiles/hykkt_libs.dir/permcheck.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/hykkt_libs.dir/permcheck.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target hykkt_libs
hykkt_libs_OBJECTS = \
"CMakeFiles/hykkt_libs.dir/MMatrix.cpp.o" \
"CMakeFiles/hykkt_libs.dir/input_functions.cpp.o" \
"CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.o" \
"CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.o" \
"CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.o" \
"CMakeFiles/hykkt_libs.dir/PermClass.cpp.o" \
"CMakeFiles/hykkt_libs.dir/RuizClass.cpp.o" \
"CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.o" \
"CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.o" \
"CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.o" \
"CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.o" \
"CMakeFiles/hykkt_libs.dir/cuda_memory_utils.cu.o" \
"CMakeFiles/hykkt_libs.dir/matrix_vector_ops.cu.o" \
"CMakeFiles/hykkt_libs.dir/permcheck.cu.o"

# External object files for target hykkt_libs
hykkt_libs_EXTERNAL_OBJECTS =

src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/MMatrix.cpp.o
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/input_functions.cpp.o
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/vector_vector_ops.cpp.o
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/cusparse_utils.cpp.o
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/matrix_matrix_ops.cpp.o
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/PermClass.cpp.o
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/RuizClass.cpp.o
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/SchurComplementConjugateGradient.cpp.o
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/CholeskyClass.cpp.o
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/SpgemmClass.cpp.o
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/HykktSolver.cpp.o
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/cuda_memory_utils.cu.o
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/matrix_vector_ops.cu.o
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/permcheck.cu.o
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/build.make
src/libhykkt_libs.so: /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcusolver.so
src/libhykkt_libs.so: /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcublas.so
src/libhykkt_libs.so: /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcublasLt.so
src/libhykkt_libs.so: /usr/local/cuda-11.6/targets/x86_64-linux/lib/libculibos.a
src/libhykkt_libs.so: /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcusparse.so
src/libhykkt_libs.so: /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcudart.so
src/libhykkt_libs.so: src/CMakeFiles/hykkt_libs.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shakedregev/hykkt/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Linking CXX shared library libhykkt_libs.so"
	cd /home/shakedregev/hykkt/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hykkt_libs.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/hykkt_libs.dir/build: src/libhykkt_libs.so
.PHONY : src/CMakeFiles/hykkt_libs.dir/build

src/CMakeFiles/hykkt_libs.dir/clean:
	cd /home/shakedregev/hykkt/src && $(CMAKE_COMMAND) -P CMakeFiles/hykkt_libs.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/hykkt_libs.dir/clean

src/CMakeFiles/hykkt_libs.dir/depend:
	cd /home/shakedregev/hykkt && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shakedregev/hykkt /home/shakedregev/hykkt/src /home/shakedregev/hykkt /home/shakedregev/hykkt/src /home/shakedregev/hykkt/src/CMakeFiles/hykkt_libs.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/hykkt_libs.dir/depend

