# TensorLUT

TensorLUT runs batched, 2-state, single-clock RTL simulation on NVIDIA tensor cores. A batch of
independent stimuli forms the rows of a matrix, and each LUT layer of the synthesized netlist is
evaluated as two single-bit (`b1` `and.popc`) matrix products in algebraic normal form (ANF): the
first builds AND monomials by comparing popcounts with monomial degrees, the second combines them
with a parity. Results are bit-exact with the LUT/DFF netlist at every cycle boundary.

This repository is the artifact for:

> Zhengyi Zhang, Sijing Yang, Yuwei Qu, Zixuan Xiao, and Lingli Wang. *TensorLUT: Exact Batched RTL
> Simulation via ANF-Lifted Tensor-Core GEMM.* ASP-DAC 2027.

## Scope

Exact here means bit-exact under 2-state, single-clock, cycle-boundary semantics after Yosys
normalization (`proc; flatten; async2sync`). X/Z values, multiple clocks, and internal memory arrays
are outside the current tensor plan; Sec. 8 of the paper discusses each.

## Layout

| Path | Contents |
|------|----------|
| `rtlgemm/frontend/` | Yosys LUT/DFF and AIG flows and netlist parsers |
| `rtlgemm/ir/` | ANF construction, affine composition, ESOP routing, plan builder |
| `rtlgemm/kernels/` | `b1_anf.cu` (binary tensor-core layer kernels), GF(2) GEMM, CUDA-core LUT baseline |
| `rtlgemm/runtime/` | simulators (`b1_sim.py`: packed tensor-core replay; `simulate.py`; `lut_cuda_sim.py`) |
| `rtlgemm/reference/` | Icarus Verilog golden flow and a NumPy netlist interpreter |
| `benchmarks/` | Verilog benchmarks (microbenchmarks, ISCAS'89) and all recorded measurements |
| `scripts/` | benchmark drivers, Verilator baselines, RTLflow/GEM setup, figure generation |
| `tests/` | differential tests against Icarus Verilog and the interpreter |
| `paper/` | LaTeX source and figures |

## Setup

Requirements: an NVIDIA GPU with binary tensor cores (measured on RTX 4090, `sm_89`), Yosys,
Icarus Verilog, and [uv](https://docs.astral.sh/uv/). Verilator is needed only for the CPU baselines.

```bash
bash scripts/setup_env.sh               # Python 3.12 venv, torch (CUDA 13), pip-packaged nvcc
.venv/bin/python -m pytest -q tests     # bit-exact differential tests
```

If Yosys and Verilator come from an [oss-cad-suite](https://github.com/YosysHQ/oss-cad-suite-build)
install that is not on `PATH`, set `OSS_CAD_SUITE` to its directory.

## Reproducing the paper

`benchmarks/README.md` maps every number in the paper to a result file and the command that
produced it. The main steps:

```bash
# RocketChip core netlist (RTeAAL Sim's generated RTL under third_party/), LUT/DFF form
PYTHONPATH=. .venv/bin/python scripts/rocket_lut_stats.py --src <OctaCoreConfig.v>

# Throughput vs. batch, tensor-core path and CUDA-core baseline
PYTHONPATH=. .venv/bin/python scripts/bench_rocket_backends.py \
    --backends b1_layerreplayv824,cuda_lutreplay --batches 64,1024,16384,262144 --cycles 64 --skip-reference

# Multi-process Verilator at the same batches
PYTHONPATH=. .venv/bin/python scripts/bench_verilator_batch_sweep.py --exe Rocket=<vsim_Rocket_lut> --cycles 64

# Figures
.venv/bin/python scripts/mk_eval_figs.py
cd paper && latexmk aspdac2027.tex
```

RTLflow and GEM comparisons: `scripts/rtlflow/build_rocket.sh` and `scripts/gem/run_rocket.sh`
document the toolchains and the fixes each simulator needs for this design.

## License

MIT; see `LICENSE`.
