cmake_minimum_required(VERSION 3.20)
project(qsim LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(QSIM_NATIVE  "Compile with -march=native" ON)
option(QSIM_OPENMP  "Enable OpenMP (gate loops, state initialisation)" OFF)
option(QSIM_TESTS   "Build unit tests" ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
add_compile_options(-Wall -Wextra)
if(QSIM_NATIVE)
  add_compile_options(-march=native)
endif()

add_library(qsim_core
  src/qsim/circuit.cpp
  src/qsim/gates.cpp
  src/qsim/state.cpp
  src/qsim/simulator.cpp
  src/qsim/fusion.cpp
)
target_include_directories(qsim_core PUBLIC src)

if(QSIM_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(qsim_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(qsim_core PUBLIC QSIM_USE_OPENMP=1)
endif()

add_executable(qsim src/apps/qsim_main.cpp)
target_link_libraries(qsim PRIVATE qsim_core)

if(QSIM_TESTS)
  enable_testing()
  add_executable(test_kernels tests/test_kernels.cpp)
  target_link_libraries(test_kernels PRIVATE qsim_core)
  add_test(NAME kernels COMMAND test_kernels)
endif()

message(STATUS "qsim: build=${CMAKE_BUILD_TYPE} native=${QSIM_NATIVE} openmp=${QSIM_OPENMP}")
