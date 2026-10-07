#!/usr/bin/env bash
# Build the RTLflow Rocket testbench for one batch size (RTLflow fixes it at compile time).
#
#   RTLFLOW_ROOT=<RTLflow checkout, built with autoconf/configure/make>
#   CUDA12_HOME=<CUDA 12.x toolkit>   (RTLflow's bundled Taskflow does not compile with CUDA 13)
#   CCBIN=<dir with a gcc/g++ <= 13>   (CUDA 12.x rejects newer host compilers)
#   scripts/rtlflow/build_rocket.sh <netlist.v> <GPU_THREADS> <outdir>
#
# Fixes applied to RTLflow (github.com/dian-lun-lin/RTLflow) for this design and GPU:
#   1. bin/verilator_includer is missing from the fork; restored from upstream Verilator 4.x.
#   2. include/verilated.mk targets sm_80; retargeted to sm_89 (RTX 4090).
#   3. --output-split 0 / --output-split-cfuncs 0: the emitter crashes ("Underflow of indentation")
#      when splitting Rocket's slow-path file.
#   4. fix_change_request.py: the generated _change_request kernel stores __req before declaring it.
set -euo pipefail
SRC=$(realpath "$1"); N=$2; OUT=$3
: "${RTLFLOW_ROOT:?}" "${CUDA12_HOME:?}" "${CCBIN:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
export VERILATOR_ROOT=$RTLFLOW_ROOT PATH=$CUDA12_HOME/bin:$PATH
[ -x "$RTLFLOW_ROOT/bin/verilator_includer" ] || install -m 755 "$HERE/verilator_includer" "$RTLFLOW_ROOT/bin/"
sed -i 's/-arch=sm_80/-arch=sm_89/g' "$RTLFLOW_ROOT/include/verilated.mk"
mkdir -p "$OUT" && cd "$OUT"
rm -rf obj_$N
"$RTLFLOW_ROOT/bin/rtlflow" --threads 2 -CFLAGS "-O2 -DGPU_THREADS=$N" --Mdir obj_$N -Wno-fatal \
  -Wno-WIDTH -Wno-UNOPTFLAT --output-split 0 --output-split-cfuncs 0 --top-module Rocket -cc "$SRC"
python3 "$HERE/fix_change_request.py" obj_$N/VRocket.cu
NV="nvcc -ccbin $CCBIN"
( cd obj_$N && make -f VRocket.mk OBJCACHE= NVCC="$NV" )
python3 "$HERE/gen_rocket_harness.py" obj_$N/VRocket.h > main_$N.cu
$NV -arch=sm_89 -std=c++17 -O2 -rdc=true -DGPU_THREADS=$N -DVL_THREADED \
  -I"$RTLFLOW_ROOT/include" -I"$RTLFLOW_ROOT/include/vltstd" -I"$RTLFLOW_ROOT/include/taskflow" \
  -Iobj_$N -Xcompiler -fopenmp --extended-lambda "$RTLFLOW_ROOT/include/rf_verilated.cu" \
  obj_$N/VRocket__ALL.a main_$N.cu -lgomp -o tb_$N
echo "built $OUT/tb_$N  (run: tb_$N <cycles> <warmup>; validate: tb_$N <C> 0 8 > dump.txt, then check_rtlflow_rocket.py)"
