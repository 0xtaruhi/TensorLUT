# Benchmark Suite

TensorLUT uses three benchmark tiers instead of treating ISCAS'89 as the whole
evaluation.

## Measured in this artifact

- Algebraic microbenchmarks: `lfsr16_free`, `crc8_serial`, `nfsr16`, `lfsr8`,
  and `counter8`. These isolate the affine, ANF, and levelized LUT-ANF
  tensor-core routes.
- ISCAS'89 sequential circuits: the repository contains 24 synthesized RTL
  inputs under `benchmarks/iscas89/`. They are used for compiler statistics,
  monomial/chunk characterization, Verilator baselines, and Icarus
  primary-output differential tests on the validated subset.

## Large-RTL target tier

Recent RTL-simulation papers use larger open designs and software workloads:

- RTeAAL Sim evaluates Chipyard RocketChip, BOOM, Gemmini, and SHA3 with
  dhrystone, `matrix_add-baremetal`, and `sha3-rocc`, plus RocketChip scale
  points.
- GSIM evaluates Rocket, BOOM, and XiangShan with CoreMark, Linux boot, and
  SPEC CPU2006 checkpoints.
- Parendi evaluates `mc`, VTA, and Chipyard Rocket mesh designs (`srN`/`lrN`);
  its largest reported sr/lr points are about 20M gates excluding SRAMs.
- Manticore evaluates single-clock RTL benchmarks on a 225-core FPGA/DSA
  static-BSP simulator.

The machine-readable version is `benchmarks/benchmark_manifest.json`. Local
checkouts live under `third_party/large_rtl/` and `third_party/simulators/`;
those directories are git-ignored so we can keep exact commits locally without
vendoring large external trees into this repository.

Large-RTL status recorded today:

- RTeAAL's bundled 8-core RocketChip DUT (`ExampleRocketSystem`) parses
  through Yosys hierarchy and produces
  `build/large_rtl/rteaal_rocket8_dut_raw_stat.json`: 168,439 cells, 5.26M
  wire bits, 1,088 memories, and 2.25M memory bits.
- The full RTeAAL `TestHarness` parses separately and produces
  `build/large_rtl/rteaal_rocket8_raw_stat.json`: 169,278 cells and 2.15B
  memory bits because it includes the simulated AXI memory model.
- The Rocket core itself lowers to TensorLUT's `$lut`/`$_DFF_P_` form:
  13,406 LUTs, 4,116 DFFs, 31 layers, 68,606 distinct monomials, and 150
  tensor chunks at `tau=512`. Reproduce with
  `python3 scripts/rocket_lut_stats.py --json`.
- UltraEmbedded `core_jpeg` parses through Yosys `proc`/`flatten`/`async2sync`
  with memory cells preserved and produces `build/large_rtl/core_jpeg_stat.json`.

## Performance Results

The current measured performance file is `benchmarks/performance_results.json`.
It records the exact commands, aggregate cycle-stimuli/s, per-seed Hz, Verilator
baselines, and ESOP-vs-LUT-ANF ablations.

Headline numbers on one RTX 4090:

- Portable tensor-core GEMM path: 2.99e9 cycle-stimuli/s peak on CRC8, i.e.
  2.99 GHz aggregate simulation frequency.
- Nonlinear ANF/LUT-ANF: 4.95e8 to 1.29e9 cycle-stimuli/s across the measured
  workloads.
- At batch 65536, per-seed frequency is 39.15 kHz for CRC8 and 7.55 kHz for
  Counter8.
- Strict same-machine Verilator comparison: LFSR16 is 285x faster than one
  Verilator process; Counter8 is 57.5x faster than one Verilator process.
- RTeAAL RocketChip 8-core Verilator baseline on Dhrystone: three runs of
  550,352 simulated cycles take 133.93s, 133.62s, and 150.48s, for a mean
  3.96 kcycles/s single-instance simulation rate.
- Rocket core TensorLUT backend differential: the generated RTeAAL `Rocket`
  core netlist matches the CPU netlist interpreter and GPU gather backend with
  zero state/primary-output mismatches for batch 8 and 8 cycles. Reproduce with
  `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python scripts/check_rocket_core.py --batch 8 --cycles 8 --device cuda --json`.
Prior-work reported numbers are recorded separately in
`performance_results.json` under `prior_work_reported_results`; they are not
mixed into the same-machine speedup rows because the substrates and workloads
are different.

## Camera-ready results (ASP-DAC 2027)

These files back every Rocket number in the paper. They supersede the earlier Rocket fast-path
entries in `performance_results.json`, which predate the final multi-tile layer kernel.
All runs use one server (2x Xeon Gold 6148, 80 hardware threads, 251 GB; RTX 4090). Rocket
runs use `build/large_rtl/rocket_Rocket_lut.json`, produced by `scripts/rocket_lut_stats.py`.

- Batch sweep, Rocket, B = 2^0..2^19, C = 64 (final kernel, tiles per warp chosen from B):
  `rocket_batch_sweep_b1.json` and the CUDA-core baseline `rocket_batch_sweep_cuda.json`, from
  `RTLGEMM_B1_LAYER_WPB=8 scripts/bench_rocket_backends.py --backends b1_layerreplayv824|cuda_lutreplay --batches ... --cycles 64 --runs 3 --warmup 1 --skip-reference`.
  Every-cycle PO capture at 2^18: `rocket_b1_final_capturepo_b262144.json` (add `--capture-po`).
- Multi-process Verilator at the same total batches (P = min(80, B), wall time from launch to
  last exit): `verilator_batch_sweep_rocket.json` (C = 64) and `verilator_batch_sweep_small.json`
  (LFSR16, Counter8, s349, s1488; C = 512), from `scripts/bench_verilator_batch_sweep.py`.
  Single-process scaling: `rocket_verilator_scale.json` (`scripts/bench_rocket_verilator_scale.py`).
- GPU batch sweeps for microbenchmarks and ISCAS'89 ESOP: `micro_batch_sweep.json`
  (`scripts/bench.py --batch B --cycles 512 --backend gemm`) and `iscas_batch_sweep.json`
  (`scripts/esop_bench.py --designs s349_bench,s1488_bench`).
- On-device stimulus generation inside the timed loop: `rocket_ondevice_stimulus.json`
  (`scripts/bench_rocket_ondevice_stimulus.py`).
- Compile-time breakdown: `rocket_compile_overhead.json` (`scripts/compile_overhead_rocket.py`).
- LUT width k = 4..7, B = 2^16: `rocket_lutk_sweep.json`, per-k runs in `rocket_lutk{4,5,6,7}_b65536.json`.
- Multi-GPU with on-device stimuli: `rocket_layer_multigpu_{1x65536,2x65536,4x65536,4x16384}.json`
  (`scripts/bench_rocket_b1_multigpu.py --backend layer --chunk-outputs 24`).
- RTLflow head-to-head: `rocket_rtlflow_sweep.json`. Build with `scripts/rtlflow/build_rocket.sh`
  (it documents the four fixes RTLflow needs), run `tb_<B> 64 4`, and validate with
  `tb_<B> <C> 0 8 > dump.txt` plus `scripts/rtlflow/check_rtlflow_rocket.py`.
- GEM head-to-head: `rocket_gem.json`, from `scripts/gem/run_rocket.sh` (synthesis, mapping,
  validation with `scripts/gem/check_gem_rocket.py`, and timing).
- Figures: `scripts/mk_eval_figs.py` writes `paper/figs/batch_scaling.pdf` and
  `paper/figs/eval_summary.pdf`.
