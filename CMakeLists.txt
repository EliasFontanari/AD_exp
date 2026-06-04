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