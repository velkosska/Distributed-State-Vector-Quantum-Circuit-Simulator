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
option(QSIM_MPI     "Build the distributed backend (qsim_mpi)" OFF)
option(QSIM_LIKWID  "Build the LIKWID CPU profiling benchmark" OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# C++ only: nvcc does not accept these flags for .cu files.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
if(QSIM_NATIVE)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

# Run-time libraries of the compiler we build with. On the cluster (EESSI) the
# GCC module does not put its own libstdc++/libgomp on the loader path, so a
# binary would pick up an older system libstdc++ at run time and fail with
# "GLIBCXX_3.4.29 not found". Embedding the compiler's lib dir fixes that for
# every binary, independent of the environment the job runs in.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
                  OUTPUT_VARIABLE _qsim_libstdcxx OUTPUT_STRIP_TRAILING_WHITESPACE)
  get_filename_component(_qsim_gcc_libdir "${_qsim_libstdcxx}" DIRECTORY)
  if(IS_ABSOLUTE "${_qsim_gcc_libdir}" AND IS_DIRECTORY "${_qsim_gcc_libdir}")
    get_filename_component(_qsim_gcc_libdir "${_qsim_gcc_libdir}" REALPATH)
    add_link_options("LINKER:-rpath,${_qsim_gcc_libdir}")
    message(STATUS "qsim: rpath to compiler runtime ${_qsim_gcc_libdir}")
  endif()
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

if(QSIM_LIKWID)
  if(NOT QSIM_OPENMP)
    message(FATAL_ERROR "QSIM_LIKWID requires QSIM_OPENMP")
  endif()
  find_path(LIKWID_INCLUDE_DIR likwid-marker.h
            HINTS "${LIKWID_ROOT}" ENV EBROOTLIKWID PATH_SUFFIXES include REQUIRED)
  find_library(LIKWID_LIBRARY likwid
               HINTS "${LIKWID_ROOT}" ENV EBROOTLIKWID PATH_SUFFIXES lib lib64 REQUIRED)
  add_executable(qsim_cpu_profile src/bench/qsim_cpu_profile.cpp)
  target_include_directories(qsim_cpu_profile PRIVATE "${LIKWID_INCLUDE_DIR}")
  target_compile_definitions(qsim_cpu_profile PRIVATE LIKWID_PERFMON)
  target_link_libraries(qsim_cpu_profile PRIVATE qsim_core "${LIKWID_LIBRARY}")
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

if(QSIM_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  add_library(qsim_mpi_backend src/qsim/mpi/dist_sim.cpp)
  target_link_libraries(qsim_mpi_backend PUBLIC qsim_core MPI::MPI_CXX)
  add_executable(qsim_mpi src/apps/qsim_mpi_main.cpp)
  target_link_libraries(qsim_mpi PRIVATE qsim_mpi_backend)
  if(QSIM_TESTS)
    enable_testing()
    add_executable(test_mpi tests/test_mpi.cpp)
    target_link_libraries(test_mpi PRIVATE qsim_mpi_backend)
    foreach(np 1 2 4 8)
      add_test(NAME mpi_np${np} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
               ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi> ${MPIEXEC_POSTFLAGS})
    endforeach()
  endif()
endif()

if(QSIM_TESTS)
  enable_testing()
  add_executable(test_kernels tests/test_kernels.cpp)
  target_link_libraries(test_kernels PRIVATE qsim_core)
  add_test(NAME kernels COMMAND test_kernels)
  add_executable(test_fusion tests/test_fusion.cpp)
  target_link_libraries(test_fusion PRIVATE qsim_core)
  add_test(NAME fusion COMMAND test_fusion)
endif()

message(STATUS "qsim: build=${CMAKE_BUILD_TYPE} native=${QSIM_NATIVE} openmp=${QSIM_OPENMP} mpi=${QSIM_MPI} cuda=${QSIM_CUDA}")
