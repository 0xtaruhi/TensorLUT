#!/usr/bin/env bash
# Verilator single-thread cycle-accurate baseline (cyc*batch/s). Verilator runs one
# instance; independent stimuli parallelize near-linearly across cores (multiply by
# the core count for the multi-thread figure). Set OSS_CAD_SUITE to use an oss-cad-suite install;
# otherwise verilator is taken from PATH.
set -euo pipefail
if [ -n "${OSS_CAD_SUITE:-}" ]; then
  export PATH="$OSS_CAD_SUITE/bin:$PATH" VERILATOR_ROOT="$OSS_CAD_SUITE/share/verilator"
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/bench_verilator"
BUILD="$ROOT/build/verilator_baseline"
mkdir -p "$BUILD/src"
cp "$ROOT/scripts/tb_lfsr.cpp" "$BUILD/src/tb_lfsr.cpp"
cp "$ROOT/scripts/tb_counter.cpp" "$BUILD/src/tb_counter.cpp"
cd "$BUILD"
B="${1:-65536}"; C="${2:-512}"
verilator --cc --exe --build -O3 -CFLAGS "-O3 -march=native" --Mdir obj_lfsr \
  "$ROOT/benchmarks/lfsr16_free.v" src/tb_lfsr.cpp -o vsim_lfsr
verilator --cc --exe --build -O3 -CFLAGS "-O3 -march=native" --Mdir obj_counter \
  "$ROOT/benchmarks/counter8.v" src/tb_counter.cpp -o vsim_counter
./obj_lfsr/vsim_lfsr "$B" "$C"
./obj_counter/vsim_counter "$B" "$C"
