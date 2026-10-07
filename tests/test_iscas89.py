"""Real-benchmark differential tests: ISCAS'89 sequential circuits.

These have async reset, many internal registers, and combinational primary outputs,
so the observable is the primary-output trace (frontend-independent). We drive all PIs
(reset asserted for the first R cycles), then compare the GPU LUT path's PO trace
against Icarus Verilog on the original RTL, bit-exact, for cycles >= R.
"""
import os

import numpy as np
import pytest

from rtlgemm.frontend import synth, parse_netlist
from rtlgemm.ir import build_plan
from rtlgemm.reference import golden_iverilog_po, po_bits
from rtlgemm.runtime.simulate import CompiledSim

torch = pytest.importorskip("torch")
CUDA = torch.cuda.is_available()

ISCAS = ["s27", "s349", "s386", "s510", "s1488", "s5378"]


def _reset_col(nl):
    flat = [pn for pn, bits in nl.inputs for b in bits if b not in nl.clock_bits]
    return flat.index("blif_reset_net") if "blif_reset_net" in flat else None


def _po_ports(nl, po):
    res = {}
    for k, (port, pos, net) in enumerate(po_bits(nl)):
        res.setdefault(port, np.zeros(po.shape[:2], np.int64))
        res[port] |= (po[:, :, k].astype(np.int64) << pos)
    return res


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("name", ISCAS)
def test_iscas89_primary_outputs(name):
    src = f"benchmarks/iscas89/{name}.v"
    if not os.path.exists(src):
        pytest.skip(f"{src} not present (run the ISCAS'89 download)")
    top = f"{name}_bench"
    nl = parse_netlist(synth(src, top).json_path, top)
    plan = build_plan(nl)
    assert plan.mode == "LUT_TENSOR"

    B, C, R = 64, 64, 8
    rng = np.random.default_rng(0)
    u = rng.integers(0, 2, (C, B, nl.n_input)).astype(np.uint8)
    rc = _reset_col(nl)
    if rc is not None:
        u[:R, :, rc] = 1   # assert reset first
        u[R:, :, rc] = 0
    x0 = np.zeros((B, nl.n_state), np.uint8)

    # tensor-core per-layer ANF backend (default) and the gather backend must both
    # match the golden bit-for-bit.
    tc = CompiledSim.build(plan, B, C, "cuda", use_cuda_graph=False, backend="auto")
    tc.run(x0, u); torch.cuda.synchronize()
    ga = CompiledSim.build(plan, B, C, "cuda", use_cuda_graph=False, backend="gather")
    ga.run(x0, u); torch.cuda.synchronize()
    tcp = _po_ports(nl, tc.po_out.cpu().numpy().astype(np.uint8))
    gap = _po_ports(nl, ga.po_out.cpu().numpy().astype(np.uint8))
    # fused tensor-core Triton LUT backend (tl.dot) must also match the golden
    from rtlgemm.kernels.triton_lut import TritonLutSim, HAVE_TRITON
    gold = golden_iverilog_po(src, top, nl, u, C, clk="blif_clk_net")
    for p in gold:
        assert np.array_equal(gold[p][R:], tcp[p][R:]), f"TC PO {p} mismatch in {name}"
        assert np.array_equal(tcp[p], gap[p]), f"TC vs gather PO {p} mismatch in {name}"
    if HAVE_TRITON:
        tri = TritonLutSim(plan, B, C).run(u).cpu().numpy().astype(np.uint8)
        trip = _po_ports(nl, tri)
        for p in gold:
            assert np.array_equal(gold[p][R:], trip[p][R:]), f"Triton-TC PO {p} mismatch in {name}"
