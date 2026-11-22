# Build qsim_cuda for the GPU on this node. Source it from a job script
# after `module load hpc-course-cuda/12.8.1` (run on a GPU node only).
#
# Sets:
#   CUDA_ARCH      compute capability of the GPU, e.g. 89 (RTX 6000 Ada), 120 (RTX PRO 6000 Blackwell)
#   CUDA_BUILD     build directory (default build-cuda, the path the docs and job scripts expect)
#   QSIM_CUDA_BIN  command that runs qsim_cuda: the binary itself, or a cuda-exec
#                  wrapper if it only starts inside the container
#
# CUDA on this cluster is an Apptainer container: CUDA tools run via cuda-exec.
# The container has nvcc but no cmake, so the CMake path is only a fallback.

if [ -z "${CUDA_ARCH:-}" ]; then
  CUDA_ARCH=$(cuda-exec nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
fi
CUDA_BUILD=${CUDA_BUILD:-build-cuda}
echo "--- building qsim_cuda for sm_$CUDA_ARCH in $CUDA_BUILD"

mkdir -p "$CUDA_BUILD"
if cuda-exec bash -c 'command -v cmake' >/dev/null 2>&1; then
  cuda-exec cmake -S . -B "$CUDA_BUILD" -DQSIM_CUDA=ON -DQSIM_NATIVE=OFF -DQSIM_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
  cuda-exec cmake --build "$CUDA_BUILD" -j 4 --target qsim_cuda
else
  cuda-exec nvcc -O3 -DNDEBUG -std=c++17 -arch=sm_"$CUDA_ARCH" -Isrc \
    src/apps/qsim_cuda_main.cpp src/qsim/cuda/cuda_sim.cu \
    src/qsim/circuit.cpp src/qsim/gates.cpp src/qsim/state.cpp src/qsim/simulator.cpp src/qsim/fusion.cpp \
    -o "$CUDA_BUILD/qsim_cuda"
fi

_qsim_root=$(pwd)
cat > "$CUDA_BUILD/qsim_cuda_wrapped.sh" <<EOF
#!/usr/bin/env bash
exec cuda-exec "$_qsim_root/$CUDA_BUILD/qsim_cuda" "\$@"
EOF
chmod +x "$CUDA_BUILD/qsim_cuda_wrapped.sh"
if "$CUDA_BUILD/qsim_cuda" --help >/dev/null 2>&1; then
  QSIM_CUDA_BIN="$CUDA_BUILD/qsim_cuda"
else
  QSIM_CUDA_BIN="$CUDA_BUILD/qsim_cuda_wrapped.sh"
fi
echo "--- qsim_cuda command: $QSIM_CUDA_BIN"
