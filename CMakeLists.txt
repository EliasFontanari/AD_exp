cmake_minimum_required(VERSION 3.16)
project(EigenLesson CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/executables")
file(MAKE_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# Eigen
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eigen")

# TinyAD
set(TINYAD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/TinyAD/include")

find_package(Doxygen)

# Suppress common third-party warnings
add_compile_options(-Wno-deprecated-declarations -Wno-unused-variable)

# Dual-number validation executable
add_executable(test_AD src/AD_example.cpp)
target_include_directories(test_AD PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${TINYAD_INCLUDE_DIR}
)

target_compile_options(test_AD PRIVATE -O2 -Wall -Wextra)

# Doxygen documentation
if(DOXYGEN_FOUND)
  set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs")
  set(DOXYGEN_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
    "${DOXYGEN_CONFIG_FILE}"
    @ONLY
  )

  add_custom_target(doc
    COMMAND "${DOXYGEN_EXECUTABLE}" "${DOXYGEN_CONFIG_FILE}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating Doxygen HTML documentation"
    VERBATIM
  )
else()
  message(STATUS "Doxygen not found - documentation generation disabled")
endif()

message(STATUS "Eigen: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "TinyAD: ${TINYAD_INCLUDE_DIR}")
message(STATUS "CppAD: ${CPPAD_INCLUDE_DIR}")
message(STATUS "CppAD library: ${CPPAD_LIB}")
message(STATUS "autodiff: ${AUTODIFF_INCLUDE_DIR}")
message(STATUS "{fmt}: ${LOCAL_FMT_DIR}")
message(STATUS "Doxygen: ${DOXYGEN_EXECUTABLE}")