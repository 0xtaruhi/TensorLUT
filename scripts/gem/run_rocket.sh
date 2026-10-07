#!/usr/bin/env bash
# Synthesize, map, simulate and validate Rocket with GEM (github.com/NVlabs/GEM).
#
#   GEM_ROOT=<GEM checkout>   built with: UCC_CUDA_PTX=89 UCC_CUDA_GENCODE=89 CUDA_PATH=... \
#       CUDA_LIBRARY_PATH=<cuda root> cargo build -r --features cuda --bin cut_map_interactive --bin cuda_test
#   Build fixes needed with CUDA 13 / GCC 15: eda-infra-rs/ucc/src/compile.rs uses -std=c++14 (libcu++
#   needs C++17), and mt-kahypar's utils/memory_tree.h needs #include <cstdint>.
#   scripts/gem/run_rocket.sh <rocket_Rocket_lut.json> <outdir> [cycles]
set -euo pipefail
NL=$(realpath "$1"); OUT=$2; C=${3:-20000}
: "${GEM_ROOT:?}"
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
mkdir -p "$OUT" && cd "$OUT"
sed -e "s#NETLIST_JSON#$NL#" -e "s#AIGPDK_NOMEM_LIB#$GEM_ROOT/aigpdk/aigpdk_nomem.lib#g" \
  "$HERE/synth_aigpdk.ys" > synth.ys
yosys -q synth.ys
"$GEM_ROOT/target/release/cut_map_interactive" gatelevel.gv rocket.gemparts
# Vector VCD with zero-padded values: GEM maps value characters MSB-first without left-extension.
python3 "$HERE/make_random_vcd.py" --json-netlist "$NL" --cycles 200 --out input_200.vcd
python3 "$HERE/make_random_vcd.py" --json-netlist "$NL" --cycles "$C" --out input_$C.vcd
"$GEM_ROOT/target/release/cuda_test" gatelevel.gv rocket.gemparts input_200.vcd output_200.vcd 256 \
  --input-vcd-scope Rocket --output-vcd-scope Rocket --check-with-cpu
PYTHONPATH="$ROOT" python3 "$HERE/check_gem_rocket.py" --json-netlist "$NL" \
  --input-vcd input_200.vcd --output-vcd output_200.vcd --cycles 200
"$GEM_ROOT/target/release/cuda_test" gatelevel.gv rocket.gemparts input_$C.vcd output_$C.vcd 256 \
  --input-vcd-scope Rocket --output-vcd-scope Rocket   # prints "simulation, Elapsed=..."
