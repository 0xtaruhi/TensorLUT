"""End-to-end differential tests: the GPU tensor path == iverilog golden, bit-exact
per cycle per register, across a batch of random stimuli.

Coverage spans all three tensor routes, auto-selected by `compile_design` from
algebraic degree: GF2_AFFINE (deg<=1), whole-cone ANF monomial lifting, and
levelized LUT_TENSOR repair.
"""
import numpy as np
import pytest

from rtlgemm.frontend import synth, parse_netlist
from rtlgemm.ir import compile_design, build_plan
from rtlgemm.reference import (simulate_netlist, read_ports, state_from_port,
                               golden_iverilog)
from rtlgemm.runtime import simulate_gpu
from rtlgemm.runtime.simulate import CompiledSim

torch = pytest.importorskip("torch")
CUDA = torch.cuda.is_available()

# (verilog, top, output/state register port, expected routed mode)
BENCHMARKS = [
    ("benchmarks/lfsr16_free.v", "lfsr16_free", "state", "GF2_AFFINE"),
    ("benchmarks/crc8_serial.v", "crc8_serial", "crc",   "GF2_AFFINE"),
    ("benchmarks/nfsr16.v",      "nfsr16",      "s",     "ANF"),
    ("benchmarks/lfsr8.v",       "lfsr8",       "state", "ANF"),
    ("benchmarks/counter8.v",    "counter8",    "cnt",   "LUT_TENSOR"),
]


def _stimuli(nl, port, B, C, seed):
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 1 << nl.n_state, size=B)
    x0 = state_from_port(nl, port, seeds)
    u = (rng.integers(0, 2, (C, B, nl.n_input)).astype(np.uint8)
         if nl.n_input > 0 else np.zeros((C, B, 0), np.uint8))
    return seeds, x0, u


@pytest.mark.parametrize("src,top,port,mode", BENCHMARKS, ids=[b[1] for b in BENCHMARKS])
def test_differential(src, top, port, mode):
    plan = compile_design(src, top)
    assert plan.mode == mode, plan.summary()
    if not CUDA:
        pytest.skip("CUDA required")
    B, C = 64, 200
    seeds, x0, u = _stimuli(plan.nl, port, B, C, 1234)
    gold = golden_iverilog(src, top, plan.nl, seeds, u, C, state_reg=port)[port]
    gpu = read_ports(plan.nl, simulate_gpu(plan, x0, u, C).cpu().numpy().astype(np.uint8))[port]
    assert np.array_equal(gold, gpu)


@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_lut_path_matches_golden():
    """Force the LUT_TENSOR backend (via the LUT frontend) and check it too is exact,
    independent of the router's choice."""
    src, top, port = "benchmarks/lfsr8.v", "lfsr8", "state"
    nl = parse_netlist(synth(src, top).json_path)
    plan = build_plan(nl)               # affine on LUT netlist fails -> LUT_TENSOR
    assert plan.mode == "LUT_TENSOR"
    B, C = 64, 200
    seeds, x0, u = _stimuli(nl, port, B, C, 77)
    gold = golden_iverilog(src, top, nl, seeds, u, C, state_reg=port)[port]
    gpu = read_ports(nl, simulate_gpu(plan, x0, u, C).cpu().numpy().astype(np.uint8))[port]
    assert np.array_equal(gold, gpu)


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("src,top,port", [
    ("benchmarks/lfsr16_free.v", "lfsr16_free", "state"),
    ("benchmarks/crc8_serial.v", "crc8_serial", "crc"),
])
def test_affine_triton_matches_gemm(src, top, port):
    """The fused Triton recurrence matches the INT8-GEMM affine path bit-for-bit."""
    plan = compile_design(src, top)
    assert plan.mode == "GF2_AFFINE"
    B, C = 4096, 128
    _, x0, u = _stimuli(plan.nl, port, B, C, 5)
    tri = CompiledSim.build(plan, B, C, "cuda", backend="triton").run(x0, u).cpu().numpy()
    gem = CompiledSim.build(plan, B, C, "cuda", backend="gemm").run(x0, u).cpu().numpy()
    assert np.array_equal(tri, gem)


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("src,top,port", [
    ("benchmarks/nfsr16.v", "nfsr16", "s"),
    ("benchmarks/counter8.v", "counter8", "cnt"),
])
def test_cuda_graph_matches_eager(src, top, port):
    """CUDA-graph replay produces identical results to eager execution (ANF + LUT)."""
    plan = compile_design(src, top)
    B, C = 4096, 64
    _, x0, u = _stimuli(plan.nl, port, B, C, 9)
    g = CompiledSim.build(plan, B, C, "cuda", use_cuda_graph=True).run(x0, u).cpu().numpy()
    e = CompiledSim.build(plan, B, C, "cuda", use_cuda_graph=False).run(x0, u).cpu().numpy()
    assert np.array_equal(g, e)


@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_lut_onehot_gemm_matches_gather():
    """The tensor-core one-hot LUT GEMM equals the gather truth-table lookup."""
    from rtlgemm.kernels import lut_onehot_gemm, lut_gather
    dev = "cuda"
    w, G, batch = 5, 4, 1024
    rng = np.random.default_rng(0)
    tables = torch.tensor(rng.integers(0, 2, size=(1 << w, G)), dtype=torch.int8, device=dev)
    idx = torch.tensor(rng.integers(0, 1 << w, size=batch), device=dev)
    gemm = lut_onehot_gemm(idx, tables)
    for g in range(G):
        ref = lut_gather(tables[:, g].to(torch.uint8), idx).to(torch.int8)
        assert torch.equal(gemm[:, g], ref)
