set(CLANG_FORMAT_VERSION "11.0.0")

# Formatting and format checking using clang-format.
find_program(CLANG_FORMAT clang-format)

# Checking for the right clang-format version
if (NOT CLANG_FORMAT STREQUAL "CLANG_FORMAT-NOTFOUND")
  # Is this the blessed version? If not, we create targets that warn the user  
  # to obtain the right version.  
  execute_process(COMMAND clang-format --version    OUTPUT_VARIABLE CF_VERSION)
  string(STRIP ${CF_VERSION} CF_VERSION)
  if (NOT ${CF_VERSION} MATCHES ${CLANG_FORMAT_VERSION})
    add_custom_target(clangformat
      echo "You have clang-format version ${CF_VERSION}, but ${CLANG_FORMAT_VERSION} is required." "Please make sure this version appears in your path.")
  else()

# Creates a make rule that when 'make clangformat' is called warnings are outputted about style issues within src code
    add_custom_target(clangformat
      find ${PROJECT_SOURCE_DIR}/src -name "*.[hc]pp" -exec ${CLANG_FORMAT} --style=file -n -Werror -ferror-limit=1 {} \+;
      VERBATIM
      COMMENT "Checking C++ formatting...")
# Create a make rule that when 'make clangformat-fix' is called style changes are automated implemented in the code
    add_custom_target(clangformat-fix 
      find ${PROJECT_SOURCE_DIR}/src -name "*.[hc]pp" -exec ${CLANG_FORMAT} --style=file -i {} \+;
      VERBATIM
      COMMENT "Reformatting Code ....")
  endif()
endif()

 make 