# shellcheck shell=bash
# Shared run harness for every qsim job script (serial, OpenMP, MPI, CUDA).
#
# Source it from a job script, then call the functions below:
#
#   source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/slurm/lib/run_harness.sh"
#   harness_init openmp_strong            # -> results/openmp_strong/<jobid>/
#   harness_guard_local 24                # refuse >24 qubits outside a Slurm job
#   circ=$(harness_gen_circuit qft 24 --seed 1)
#   harness_run kind=qft qubits=24 repeat=1 threads=8 circuit="$circ"
#   harness_finish
#
# Layout written by one job:
#   results/<experiment>/<jobid>/meta.txt   provenance (git, modules, nodes, CPU, env, commands)
#   results/<experiment>/<jobid>/runs.csv   one row per qsim run (schema: HARNESS_COLUMNS)
#   results/<experiment>/<jobid>/raw/       per-gate timing CSVs, stdout/stderr of every run
#
# Settings read from the environment (all optional):
#   QSIM_BIN        binary to run                       (default build/qsim)
#   LAUNCHER        command prefix, e.g. "srun" for MPI (default: none)
#   BACKEND         serial|openmp|mpi|cuda, stored per row (default: serial)
#   QSIM_ARGS       extra qsim arguments for every run  (e.g. "--fusion 1q")
#   GATE_TIMING     1 = write per-gate CSVs (default), 0 = pass --no-gate-timing
#   ALLOW_LOCAL     1 = allow any size outside Slurm (you are NOT on a login node)
#   QSIM_MODULES_EXTRA  extra module sets for env/modules.sh, e.g. "mpi" or "cuda"
#   HARNESS_SKIP_MODULES 1 = do not source env/modules.sh
#
# The harness never aborts the job when a single run fails: the row is written
# with status=fail:<exit code> and the next run starts.

HARNESS_VERSION=2
HARNESS_COLUMNS="run_id,experiment,job_id,backend,kind,qubits,nodes,ranks,threads,gpus,repeat,seed,depth,iterations,global_fraction,global_qubits,gates_in,gates_run,fusion,total_s,mean_gate_s,norm_err,status,tag,circuit,timing_csv"

# ------------------------------------------------------------------ helpers
_h_log()  { echo "[harness $(date +%T)] $*" >&2; }
_h_die()  { echo "[harness] ERROR: $*" >&2; exit 1; }
_h_meta() { echo "$*" >> "$HARNESS_OUT/meta.txt"; }
_h_csv()  { # quote a CSV field only if needed
  local v=$1
  if [[ $v == *[,\"]* ]]; then v=${v//\"/\"\"}; printf '"%s"' "$v"; else printf '%s' "$v"; fi
}
_h_section() { _h_meta ""; _h_meta "--- $*"; }

# ------------------------------------------------------------------ init
# harness_init EXPERIMENT
harness_init() {
  [ $# -eq 1 ] || _h_die "usage: harness_init EXPERIMENT"
  HARNESS_EXPERIMENT=$1
  [[ $HARNESS_EXPERIMENT =~ ^[A-Za-z0-9_.-]+$ ]] || _h_die "experiment name must be [A-Za-z0-9_.-]"

  HARNESS_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
  cd "$HARNESS_ROOT" || _h_die "cannot cd to $HARNESS_ROOT"
  [ -f src/tools/gen_circuits.py ] || _h_die "$HARNESS_ROOT is not the repo root (submit jobs from the repo root)"

  if [ "${HARNESS_SKIP_MODULES:-0}" != 1 ] && [ -f env/modules.sh ]; then
    # modules + venv. Explicit empty argument: otherwise `source` would see harness_init's arguments
    set +u; source env/modules.sh ""; set -u
  fi

  QSIM_BIN=${QSIM_BIN:-build/qsim}
  LAUNCHER=${LAUNCHER:-}
  BACKEND=${BACKEND:-serial}
  QSIM_ARGS=${QSIM_ARGS:-}
  GATE_TIMING=${GATE_TIMING:-1}
  HARNESS_JOBID=${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)-$$}
  HARNESS_OUT="results/$HARNESS_EXPERIMENT/$HARNESS_JOBID"
  HARNESS_DATA="data/$HARNESS_EXPERIMENT"
  HARNESS_RUN_COUNT=0
  HARNESS_FAIL_COUNT=0
  HARNESS_T0=$(date +%s)
  mkdir -p "$HARNESS_OUT/raw" "$HARNESS_DATA"

  [ -x "$QSIM_BIN" ] || _h_die "missing $QSIM_BIN: build it first (see REPRODUCE.md)"

  : > "$HARNESS_OUT/meta.txt"
  _h_meta "harness_version: $HARNESS_VERSION"
  _h_meta "experiment:      $HARNESS_EXPERIMENT"
  _h_meta "job_id:          $HARNESS_JOBID"
  _h_meta "job_name:        ${SLURM_JOB_NAME:-}"
  _h_meta "job_script:      ${HARNESS_SCRIPT:-$0}"
  _h_meta "start:           $(date -Iseconds)"
  _h_meta "submit_host:     ${SLURM_SUBMIT_HOST:-}"
  _h_meta "batch_host:      $(hostname)"
  _h_meta "user:            ${USER:-$(id -un)}"

  _h_section "git"
  _h_meta "git_commit:      $(git rev-parse --verify -q HEAD 2>/dev/null || echo unknown)"
  _h_meta "git_branch:      $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  _h_meta "git_describe:    $(git describe --always --dirty --tags 2>/dev/null || echo unknown)"
  # dirty = modified tracked files or untracked source files (results/ excluded:
  # earlier jobs' output must not make every later job "dirty")
  local dirty=no
  git status --porcelain -- . ':(exclude)results' 2>/dev/null | grep -q . && dirty=yes
  _h_meta "git_dirty:       $dirty"
  if [ "$dirty" = yes ]; then
    _h_meta "git_dirty_files:"
    git status --porcelain -- . ':(exclude)results' 2>/dev/null | sed 's/^/  /' >> "$HARNESS_OUT/meta.txt"
    git diff HEAD > "$HARNESS_OUT/raw/git_diff.patch" 2>/dev/null || true
    _h_meta "git_diff:        raw/git_diff.patch"
  fi

  _h_section "binary"
  _h_meta "backend:         $BACKEND"
  _h_meta "qsim_bin:        $QSIM_BIN"
  _h_meta "qsim_bin_sha256: $(sha256sum "$QSIM_BIN" 2>/dev/null | cut -d' ' -f1)"
  _h_meta "qsim_bin_mtime:  $(date -Iseconds -r "$QSIM_BIN" 2>/dev/null || echo unknown)"
  _h_meta "launcher:        ${LAUNCHER:-none}"
  _h_meta "qsim_args:       ${QSIM_ARGS:-none}"
  _h_meta "gate_timing:     $GATE_TIMING"
  local cache
  cache="$(dirname "$QSIM_BIN")/CMakeCache.txt"
  if [ -f "$cache" ]; then
    _h_meta "cmake_cache:     $cache"
    grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CXX_FLAGS|CMAKE_CXX_FLAGS_RELEASE|QSIM_[A-Z_]+|MPI_CXX_COMPILER|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_COMPILER):' "$cache" \
      | sed 's/^/  /' >> "$HARNESS_OUT/meta.txt" || true
  fi
  _h_meta "cxx:             $( (${CXX:-c++} --version 2>/dev/null || echo unknown) | head -1)"
  command -v mpirun >/dev/null 2>&1 && _h_meta "mpi:             $(mpirun --version 2>&1 | head -1)"
  command -v nvcc >/dev/null 2>&1   && _h_meta "nvcc:            $(nvcc --version 2>&1 | tail -1)"
  _h_meta "python:          $(python3 --version 2>&1)"

  _h_section "nodes"
  _h_meta "nodelist:        ${SLURM_JOB_NODELIST:-$(hostname)}"
  if [ -n "${SLURM_JOB_NODELIST:-}" ] && command -v scontrol >/dev/null 2>&1; then
    _h_meta "nodes_expanded:  $(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd' ')"
  fi
  _h_meta "num_nodes:       ${SLURM_JOB_NUM_NODES:-1}"
  _h_meta "partition:       ${SLURM_JOB_PARTITION:-}"
  _h_meta "account:         ${SLURM_JOB_ACCOUNT:-}"
  _h_meta "kernel:          $(uname -r)"

  _h_section "cpu (batch host)"
  if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E '^(Model name|Architecture|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|CPU\(s\)|NUMA node\(s\)|NUMA node[0-9]+ CPU\(s\)|CPU max MHz|L1d cache|L2 cache|L3 cache|Flags)' \
      | grep -v '^Flags' | sed 's/^/  /' >> "$HARNESS_OUT/meta.txt" || true
    # only the SIMD-relevant flags, the full list is huge
    _h_meta "  simd_flags: $(lscpu | sed -n 's/^Flags:[[:space:]]*//p' | tr ' ' '\n' | grep -E '^(avx|avx2|avx512f|avx512dq|fma|sve|asimd)$' | paste -sd' ')"
  fi
  _h_meta "  mem_total: $(awk '/MemTotal/ {printf "%.1f GiB", $2/1048576}' /proc/meminfo 2>/dev/null)"
  _h_meta "  cpus_allowed: $(grep -s Cpus_allowed_list /proc/self/status | awk '{print $2}')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    _h_section "gpu"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,clocks.max.sm,clocks.max.mem --format=csv 2>/dev/null \
      | sed 's/^/  /' >> "$HARNESS_OUT/meta.txt" || true
  fi

  _h_section "module list"
  if command -v module >/dev/null 2>&1; then
    module list 2>&1 | sed 's/^/  /' >> "$HARNESS_OUT/meta.txt" || true
  else
    _h_meta "  module command not available"
  fi

  _h_section "environment (OMP_*, KMP_*, GOMP_*, SLURM_*, MPI, UCX, CUDA, QSIM_*)"
  env | LC_ALL=C sort | grep -E '^(OMP_|KMP_|GOMP_|SLURM_|SRUN_|SBATCH_|OMPI_|PMIX_|I_MPI_|MPICH_|UCX_|FI_|CUDA_|NVIDIA_|QSIM_|QUBITS=|KINDS=|REPEATS=|SEED=|LAUNCHER=|BACKEND=)' \
    | grep -vE '^(SLURM_JOB_USER|OMPI_MCA_.*password|.*TOKEN|.*SECRET|.*KEY)=' \
    | sed 's/^/  /' >> "$HARNESS_OUT/meta.txt" || true

  _h_section "circuits (generator command, sha256)"
  echo "$HARNESS_COLUMNS" > "$HARNESS_OUT/runs.csv"
  _h_log "experiment=$HARNESS_EXPERIMENT job=$HARNESS_JOBID out=$HARNESS_OUT"
}

# ------------------------------------------------------------------ guard
# harness_guard_local MAX_QUBITS_OF_THIS_JOB
# Outside a Slurm allocation (i.e. most likely on a login node) refuse jobs
# above LOCAL_MAX_QUBITS (default 18 = 4 MiB state) unless ALLOW_LOCAL=1.
harness_guard_local() {
  local want=$1 limit=${LOCAL_MAX_QUBITS:-18}
  [ -n "${SLURM_JOB_ID:-}" ] && return 0
  [ "${ALLOW_LOCAL:-0}" = 1 ] && { _h_log "ALLOW_LOCAL=1: running ${want}q outside Slurm"; return 0; }
  if [ "$want" -gt "$limit" ]; then
    _h_die "not inside a Slurm job and the largest run has $want qubits (> $limit).
  Submit with sbatch, or for a quick check use QUBITS=\"10 12\". Never run heavy jobs on the login node."
  fi
}

# ------------------------------------------------------------------ circuits
# harness_gen_circuit KIND N [gen_circuits.py args...]  -> prints circuit path
# Circuits are deterministic, so an existing file with the same name is reused.
harness_gen_circuit() {
  local kind=$1 n=$2; shift 2
  local tag
  tag=$(printf '%s' "$*" | tr -s ' =' '__' | tr -cd 'A-Za-z0-9_.-' | sed 's/--//g')
  local circ="$HARNESS_DATA/${kind}_${n}${tag:+_$tag}.txt"
  local cmd=(python3 src/tools/gen_circuits.py "$kind" -n "$n" "$@" -o "$circ")
  if [ ! -s "$circ" ]; then
    "${cmd[@]}" 2>>"$HARNESS_OUT/raw/gen_circuits.log" || _h_die "generator failed: ${cmd[*]}"
  fi
  _h_meta "$(sha256sum "$circ" | cut -c1-16)  ${cmd[*]}"
  echo "$circ"
}

# ------------------------------------------------------------------ run
# harness_run key=value ... [-- extra qsim args]
# Keys: kind qubits circuit repeat seed ranks threads nodes gpus depth
#       iterations global_fraction global_qubits tag
# `ranks` > 1 or LAUNCHER set: the command is "$LAUNCHER $QSIM_BIN ..."; put
# the srun options (e.g. -n RANKS) into LAUNCHER or into the `launch` key.
harness_run() {
  local -A k=([kind]="" [qubits]="" [circuit]="" [repeat]=1 [seed]="" [ranks]=1 [threads]="${OMP_NUM_THREADS:-1}"
              [nodes]="${SLURM_JOB_NUM_NODES:-1}" [gpus]=0 [depth]="" [iterations]="" [global_fraction]=""
              [global_qubits]="" [tag]="" [launch]="$LAUNCHER")
  local extra=()
  while [ $# -gt 0 ]; do
    case $1 in
      --) shift; extra=("$@"); break ;;
      *=*) local key=${1%%=*}; [[ -v "k[$key]" ]] || _h_die "harness_run: unknown key $key"; k[$key]=${1#*=} ;;
      *) _h_die "harness_run: expected key=value, got '$1'" ;;
    esac
    shift
  done
  [ -n "${k[circuit]}" ] && [ -f "${k[circuit]}" ] || _h_die "harness_run: circuit '${k[circuit]}' not found"

  HARNESS_RUN_COUNT=$((HARNESS_RUN_COUNT + 1))
  local id
  id=$(printf 'r%04d' "$HARNESS_RUN_COUNT")
  local stem="${id}_${k[kind]}_${k[qubits]}q_p${k[ranks]}_t${k[threads]}_rep${k[repeat]}"
  local timing="" qargs=()
  if [ "$GATE_TIMING" = 1 ]; then timing="$HARNESS_OUT/raw/$stem.gates.csv"; qargs+=(--timing "$timing"); else qargs+=(--no-gate-timing); fi
  # shellcheck disable=SC2206  # word splitting of QSIM_ARGS / launch is intended
  local cmd=(${k[launch]} "$QSIM_BIN" "${k[circuit]}" "${qargs[@]}" $QSIM_ARGS "${extra[@]}")

  _h_meta "$id  OMP_NUM_THREADS=${OMP_NUM_THREADS:-} ${cmd[*]}"
  local t0 t1 rc=0
  t0=$(date +%s.%N)
  "${cmd[@]}" > "$HARNESS_OUT/raw/$stem.out" 2> "$HARNESS_OUT/raw/$stem.err" || rc=$?
  t1=$(date +%s.%N)

  # With MPI every rank may print; the first SUMMARY line (rank 0) wins.
  local line
  line=$(grep -m1 '^SUMMARY' "$HARNESS_OUT/raw/$stem.out" || true)
  _get() { sed -n "s/.* $1=\([^ ]*\).*/\1/p" <<<" ${line#SUMMARY}"; }
  local status=ok
  [ "$rc" -ne 0 ] && status="fail:$rc"
  [ -z "$line" ] && status="${status/ok/nosummary}"
  [ "$status" != ok ] && HARNESS_FAIL_COUNT=$((HARNESS_FAIL_COUNT + 1))
  # prefer what the binary reports (an MPI binary may print ranks=/threads=)
  local ranks threads
  ranks=$(_get ranks); ranks=${ranks:-${k[ranks]}}
  threads=$(_get threads); threads=${threads:-${k[threads]}}
  [ "$GATE_TIMING" = 1 ] && [ ! -s "$timing" ] && timing=""

  local f row=()
  for f in "$id" "$HARNESS_EXPERIMENT" "$HARNESS_JOBID" "$BACKEND" "${k[kind]}" "${k[qubits]:-$(_get qubits)}" \
           "${k[nodes]}" "$ranks" "$threads" "${k[gpus]}" "${k[repeat]}" "${k[seed]}" "${k[depth]}" \
           "${k[iterations]}" "${k[global_fraction]}" "${k[global_qubits]}" \
           "$(_get gates_in)" "$(_get gates_run)" "$(_get fusion)" "$(_get total_s)" "$(_get mean_gate_s)" \
           "$(_get norm_err)" "$status" "${k[tag]}" "${k[circuit]}" "${timing#"$HARNESS_OUT"/}"; do
    row+=("$(_h_csv "$f")")
  done
  (IFS=,; echo "${row[*]}") >> "$HARNESS_OUT/runs.csv"
  _h_log "$id ${k[kind]} n=${k[qubits]} P=$ranks T=$threads rep=${k[repeat]} total_s=$(_get total_s) wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')s $status"
  return 0
}

# ------------------------------------------------------------------ finish
harness_finish() {
  local t1
  t1=$(date +%s)
  _h_section "end"
  _h_meta "end:             $(date -Iseconds)"
  _h_meta "elapsed_s:       $((t1 - HARNESS_T0))"
  _h_meta "runs:            $HARNESS_RUN_COUNT"
  _h_meta "failed_runs:     $HARNESS_FAIL_COUNT"
  _h_log "done: $HARNESS_OUT ($HARNESS_RUN_COUNT runs, $HARNESS_FAIL_COUNT failed)"
  [ "$HARNESS_FAIL_COUNT" -eq 0 ]
}

# ------------------------------------------------------------------ utilities
# harness_list VAR_DEFAULT: split a space/comma separated list
harness_list() { tr ',' ' ' <<<"$1"; }

# harness_pow2_upto N: 1 2 4 ... <= N
harness_pow2_upto() { local p=1; while [ "$p" -le "$1" ]; do printf '%s ' "$p"; p=$((p * 2)); done; echo; }
