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
option(QSIM_CUDA    "Build the single-GPU backend (qsim_cuda)" OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# C++ only: nvcc does not accept these flags for .cu files.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
if(QSIM_NATIVE)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
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

# Memory-bandwidth probe for the roofline (slurm/stream.sbatch). Always built
# with OpenMP if the compiler supports it, independent of QSIM_OPENMP.
add_executable(stream_triad src/bench/stream_triad.cpp)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(stream_triad PRIVATE OpenMP::OpenMP_CXX)
endif()

if(QSIM_CUDA)
  # Default arch 89 = RTX 6000 Ada (haskell). Override with -DCMAKE_CUDA_ARCHITECTURES=...
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
  add_library(qsim_cuda_backend src/qsim/cuda/cuda_sim.cu)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(qsim_cuda_backend PUBLIC qsim_core CUDA::cudart)
  add_executable(qsim_cuda src/apps/qsim_cuda_main.cpp)
  target_link_libraries(qsim_cuda PRIVATE qsim_cuda_backend)
endif()

if(QSIM_TESTS)
  enable_testing()
  add_executable(test_kernels tests/test_kernels.cpp)
  target_link_libraries(test_kernels PRIVATE qsim_core)
  add_test(NAME kernels COMMAND test_kernels)
endif()

message(STATUS "qsim: build=${CMAKE_BUILD_TYPE} native=${QSIM_NATIVE} openmp=${QSIM_OPENMP} cuda=${QSIM_CUDA}")
