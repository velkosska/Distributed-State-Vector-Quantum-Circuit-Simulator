#!/usr/bin/env bash
#SBATCH --job-name=qsim-omp-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%x-%j.out
#
# Set QSIM_MODULES_EXTRA to the site's LIKWID module if needed.
# Course QOS "student": at most 64 CPUs per user (32 physical cores with 2 hardware
# threads each) and no --exclusive, or the job waits forever (QOSMaxCpuPerUserLimit).
# Submit from the repo root. See REPRODUCE.md for the outputs.
set -euo pipefail

N=${N:-28}
ARRAY_MIB=${ARRAY_MIB:-2048}
PROFILE_THREADS=${PROFILE_THREADS:-32}   # physical cores of a 64-CPU allocation
PROFILE_PASSES=${PROFILE_PASSES:-20}
PROFILE_REPEATS=${PROFILE_REPEATS:-3}
export OMP_DYNAMIC=false OMP_PLACES=cores OMP_DISPLAY_AFFINITY=true
export LC_ALL=C
unset OMP_THREAD_LIMIT GOMP_CPU_AFFINITY

HARNESS_SCRIPT=slurm/omp_profile.sbatch
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/slurm/lib/run_harness.sh"
[[ -n ${SLURM_JOB_ID:-} ]] || _h_die "submit this job with sbatch"
cd "$ROOT"
harness_load_modules
command -v likwid-perfctr >/dev/null || _h_die "load the site's LIKWID module with QSIM_MODULES_EXTRA"

cores=$(python3 slurm/lib/physical_cores.py)
IFS=, read -r -a places <<<"$cores"
for t in $PROFILE_THREADS; do
  if (( t > ${#places[@]} )); then
    _h_die "$t threads requested, but only ${#places[@]} physical cores are available"
  fi
done

build="build-omp-profile-$SLURM_JOB_ID"
configure=(cmake -S . -B "$build" -DQSIM_OPENMP=ON -DQSIM_LIKWID=ON)
if [[ -n ${LIKWID_ROOT:-} ]]; then configure+=("-DLIKWID_ROOT=$LIKWID_ROOT"); fi
compile=(cmake --build "$build" --parallel 4 --target qsim_cpu_profile)
"${configure[@]}"
"${compile[@]}"
export QSIM_BIN="$build/qsim_cpu_profile" BACKEND=openmp OMP_PROC_BIND=close
harness_init omp_profile
rm "$HARNESS_OUT/runs.csv"
_h_meta "configure: ${configure[*]}"
_h_meta "build:     ${compile[*]}"
_h_meta "settings:  qubits=$N array_mib=$ARRAY_MIB threads=($PROFILE_THREADS) passes=$PROFILE_PASSES repeats=$PROFILE_REPEATS"
likwid-perfctr --version >> "$HARNESS_OUT/meta.txt" 2>&1
likwid-perfctr -a > "$HARNESS_OUT/raw/groups.txt" 2>&1
group=${LIKWID_GROUP:-}
if [[ -z $group ]]; then
  group=$(awk '$1 == "MEM_DP" {print $1; exit}' "$HARNESS_OUT/raw/groups.txt")
  group=${group:-MEM}
fi
[[ $group == MEM_DP || $group == MEM ]] || _h_die "LIKWID_GROUP must be MEM_DP or MEM"
_h_meta "likwid_group: $group"
echo "omp_proc_bind,threads,repeat,kernel,qubits,array_mib,cpu_ids,likwid_file" > "$HARNESS_OUT/profile_manifest.csv"

for t in $PROFILE_THREADS; do
  export OMP_NUM_THREADS=$t
  for bind in close spread; do
    export OMP_PROC_BIND=$bind
    probe=(taskset --cpu-list "$cores" "$QSIM_BIN" --cpus)
    _h_meta "OMP_NUM_THREADS=$t OMP_PROC_BIND=$bind OMP_PLACES=cores ${probe[*]}"
    cpus=$("${probe[@]}" 2>> "$HARNESS_OUT/raw/affinity.log")
    IFS=, read -r -a workers <<<"$cpus"
    [[ ${#workers[@]} -eq $t ]] || _h_die "expected $t assigned workers, got $cpus"
    for r in $(seq 1 "$PROFILE_REPEATS"); do
      for kernel in h rz cx stream; do
        qubits=$N; array_mib=; size=$N
        if [[ $kernel == stream ]]; then qubits=; array_mib=$ARRAY_MIB; size=$ARRAY_MIB; fi
        stem="${bind}_t${t}_r${r}_${kernel}"
        raw="raw/$stem.likwid.csv"
        # LIKWID's N domain here contains only the sampled CPUs. Restore
        # the full set of OpenMP places for the application itself.
        cmd=(taskset --cpu-list "$cpus" likwid-perfctr -c N -g "$group" -m -O
             taskset --cpu-list "$cores" "$QSIM_BIN" "$kernel" "$size" "$PROFILE_PASSES")
        printf -v command_line '%q ' "${cmd[@]}"
        _h_meta "OMP_NUM_THREADS=$t OMP_PROC_BIND=$bind OMP_PLACES=cores $command_line"
        "${cmd[@]}" > "$HARNESS_OUT/$raw" 2> "$HARNESS_OUT/raw/$stem.err"
        printf '%s,%s,%s,%s,%s,%s,"%s",%s\n' "$bind" "$t" "$r" "$kernel" "$qubits" "$array_mib" "$cpus" "$raw" >> "$HARNESS_OUT/profile_manifest.csv"
        HARNESS_RUN_COUNT=$((HARNESS_RUN_COUNT + 1))
      done
    done
  done
done

python3 analysis/cpu_roofline.py "$HARNESS_OUT"
harness_finish
