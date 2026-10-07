"""ESOP single-layer vs current multi-layer LUT-ANF: throughput + bit-exactness on 4090.

Flow: RTL --yosys--> sequential AIG --abc(comb + &exorcism)--> minimal single-level ESOP
(XOR-of-AND-products) mapping (PI, latch-Q) -> (PO, latch-D). ABC's `comb` turns the L latches
into appended PI/PO in latch-index order, so latch k has current-state = input I+k and
next-state = output O+k (clean pairing, no name matching). We build ONE ANF layer, handling
complemented ESOP literals via an augmented input [X | ~X], run it on the existing b1 and.popc
kernel, and drive a per-cycle sequential loop. Correctness is checked against a direct evaluation
of the same synthesized AIG (ground truth). Throughput is compared with B1AnfSim (per-layer
LUT->ANF, the current path).
"""
import argparse
import os, re, subprocess, sys, time
import numpy as np
import torch

sys.path.insert(0, ".")
from rtlgemm.frontend import synth, parse_netlist
from rtlgemm.ir.plan import build_plan
from rtlgemm.ir.anf import lut_anf
from rtlgemm.runtime.b1_sim import B1AnfSim
from rtlgemm.kernels.b1_anf import B1Anf

SUITE = os.environ.get("OSS_CAD_SUITE", "")  # optional; tools come from PATH when unset
YOSYS = os.path.join(SUITE, "bin", "yosys") if SUITE else "yosys"
ABC = os.path.join(SUITE, "bin", "yosys-abc") if SUITE else "yosys-abc"
_FLOW = ("proc; flatten; opt; techmap; opt; async2sync; "
         "dfflegalize -cell $_DFF_P_ x; aigmap; opt_clean")


def yosys_seq(src, top, outdir="/tmp/esop"):
    os.makedirs(outdir, exist_ok=True)
    aig, aag = f"{outdir}/{top}.aig", f"{outdir}/{top}.aag"
    for out, extra in ((aig, ""), (aag, "-ascii ")):
        s = f"read_verilog {src}; hierarchy -top {top}; {_FLOW}; write_aiger {extra}{out}"
        r = subprocess.run([YOSYS, "-q", "-p", s], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"yosys failed:\n{r.stderr[-1500:]}")
    return aig, aag


def parse_aag_seq(path):
    """Parse ASCII aag. Returns I,L,O, in_lits, latches[(cur,next)], out_lits, ands{lhs:(r0,r1)}."""
    lines = [l.rstrip("\n") for l in open(path)]
    M, I, L, O, A = (int(x) for x in lines[0].split()[1:6])
    p = 1
    in_lits = [int(lines[p + i]) for i in range(I)]; p += I
    latches = [tuple(int(x) for x in lines[p + i].split()[:2]) for i in range(L)]; p += L
    out_lits = [int(lines[p + i]) for i in range(O)]; p += O
    ands = {}
    for i in range(A):
        lhs, r0, r1 = (int(x) for x in lines[p + i].split()); ands[lhs] = (r0, r1)
    return I, L, O, in_lits, latches, out_lits, ands


def aig_eval_seq(I, L, in_lits, latches, out_lits, ands, u_t, state):
    """One combinational eval of the sequential AIG. u_t:(B,I) state:(B,L) -> (PO(B,O), nextD(B,L))."""
    B = state.shape[0]
    val = {0: np.zeros(B, np.uint8)}
    for j, lit in enumerate(in_lits):
        val[lit] = u_t[:, j].astype(np.uint8); val[lit ^ 1] = 1 - val[lit]
    for k, (cur, _nxt) in enumerate(latches):
        val[cur] = state[:, k].astype(np.uint8); val[cur ^ 1] = 1 - val[cur]
    for lhs in sorted(ands):
        r0, r1 = ands[lhs]
        v = _lit(val, r0) & _lit(val, r1); val[lhs] = v; val[lhs ^ 1] = 1 - v
    po = np.stack([_lit(val, o) for o in out_lits], axis=1) if out_lits else np.zeros((B, 0), np.uint8)
    nxt = np.stack([_lit(val, nx) for (_c, nx) in latches], axis=1) if latches else np.zeros((B, 0), np.uint8)
    return po.astype(np.uint8), nxt.astype(np.uint8)


def _lit(val, lit):
    if lit in val:
        return val[lit]
    val[lit] = 1 - val[lit & ~1]
    return val[lit]


def exorcism(aig):
    esop = aig.replace(".aig", ".esop")
    subprocess.run([ABC, "-c", f"read_aiger {aig}; comb; &get; &exorcism {esop}"],
                   capture_output=True, text=True)
    return esop


def parse_esop(path):
    ni = no = None; cubes = []
    for l in open(path):
        l = l.strip()
        if l.startswith(".i "):
            ni = int(l.split()[1])
        elif l.startswith(".o "):
            no = int(l.split()[1])
        elif l and l[0] in "01-":
            a, b = l.split(); cubes.append((a, b))
    return ni, no, cubes


class EsopSim:
    """Single-layer ESOP sequential simulator on the b1 kernel. Layer inputs = [PI | latch-Q];
    layer outputs = [PO | latch-D]; complemented literals via augmented [X | ~X].

    Passthrough stripping: a next-state D_k that is a constant, a direct copy of some PI/latch-Q,
    or its negation is routed as a FREE gather (no monomial). Only genuinely nonlinear latch-D go
    through EXORCISM, so shift registers / wire routing don't inflate the cube count F."""

    def __init__(self, src, top, device="cuda", strip=True):
        self.device = device
        aig, aag = yosys_seq(src, top)
        self.I, self.L, self.O, il, self.latches, ol, self.ands = parse_aag_seq(aag)
        self.in_lits, self.out_lits = il, ol
        self.nin = self.I + self.L
        self.nout = self.O + self.L
        # literal -> layer-input column (PI i -> i ; latch k -> I+k)
        lit2col = {}
        for i, lit in enumerate(il):
            lit2col[lit & ~1] = i
        for k, (cur, _n) in enumerate(self.latches):
            lit2col[cur & ~1] = self.I + k
        # classify each latch's next-state
        self.const = []        # (k, value)
        self.copy = []         # (k, src_col, neg)
        nonlin = []            # latch indices going through ESOP
        for k, (_cur, nxt) in enumerate(self.latches):
            if not strip:
                nonlin.append(k); continue
            if nxt == 0:
                self.const.append((k, 0))
            elif nxt == 1:
                self.const.append((k, 1))
            elif (nxt & ~1) in lit2col:
                self.copy.append((k, lit2col[nxt & ~1], nxt & 1))
            else:
                nonlin.append(k)
        self.nonlin = nonlin
        ni, no, cubes = parse_esop(exorcism(aig))
        assert ni == self.nin and no == self.nout, f"ESOP {ni}/{no} vs {self.nin}/{self.nout}"
        self.F_full = len(cubes)
        # keep only cubes feeding the nonlinear latch-D outputs (output index O+k)
        keep_out = [self.O + k for k in nonlin]
        A_full = np.zeros((len(cubes), 2 * self.nin), np.uint8)
        Csel = np.zeros((len(nonlin), len(cubes)), np.uint8)
        for f, (a, b) in enumerate(cubes):
            for j, ch in enumerate(a):
                if ch == "1":
                    A_full[f, j] = 1
                elif ch == "0":
                    A_full[f, self.nin + j] = 1
            for oi, g in enumerate(keep_out):
                if b[g] == "1":
                    Csel[oi, f] = 1
        used = Csel.any(axis=0)                          # drop cubes not feeding any kept output
        self.A = A_full[used]; self.C = Csel[:, used]
        self.F = int(used.sum())
        self.b1 = B1Anf(self.A, self.C, device=device) if self.F and len(nonlin) else None
        # precompute passthrough index tensors
        self.nl_dst = torch.tensor(nonlin, dtype=torch.long, device=device)
        self.cp_dst = torch.tensor([k for k, _s, _n in self.copy], dtype=torch.long, device=device)
        self.cp_src = torch.tensor([s for _k, s, _n in self.copy], dtype=torch.long, device=device)
        self.cp_neg = torch.tensor([n for _k, _s, n in self.copy], dtype=torch.int8, device=device)
        self.ct_dst = torch.tensor([k for k, _v in self.const], dtype=torch.long, device=device)
        self.ct_val = torch.tensor([v for _k, v in self.const], dtype=torch.int8, device=device)

    def run(self, u_seq, cycles, sample=True):
        """u_seq:(cycles,B,I) int8. Latches start at 0. Returns state traj (cycles,B,L) or None."""
        u = torch.as_tensor(u_seq, dtype=torch.int8, device=self.device)
        B = u.shape[1]
        state = torch.zeros((B, self.L), dtype=torch.int8, device=self.device)
        traj = torch.empty((cycles, B, self.L), dtype=torch.int8, device=self.device) if sample else None
        for t in range(cycles):
            X = torch.cat([u[t], state], dim=1) if self.I else state
            nxt = torch.empty((B, self.L), dtype=torch.int8, device=self.device)
            if self.b1 is not None:
                Xaug = torch.cat([X, 1 - X], dim=1).contiguous()
                Ynl = self.b1.run_into(Xaug) & 1                # (B, |nonlin|)
                nxt[:, self.nl_dst] = Ynl
            if self.cp_dst.numel():
                v = X[:, self.cp_src]
                nxt[:, self.cp_dst] = torch.where(self.cp_neg.bool(), 1 - v, v)
            if self.ct_dst.numel():
                nxt[:, self.ct_dst] = self.ct_val
            state = nxt
            if sample:
                traj[t] = state
        torch.cuda.synchronize()
        return traj

    def golden(self, u_seq, cycles):
        B = u_seq.shape[1]
        state = np.zeros((B, self.L), np.uint8)
        traj = np.empty((cycles, B, self.L), np.uint8)
        for t in range(cycles):
            _po, nxt = aig_eval_seq(self.I, self.L, self.in_lits, self.latches,
                                    self.out_lits, self.ands, u_seq[t], state)
            state = nxt; traj[t] = state
        return traj


def time_sim(fn, warmup=2, runs=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs


DESIGNS = [("benchmarks/nfsr16.v", "nfsr16"),
           ("benchmarks/iscas89/s349.v", "s349_bench"),
           ("benchmarks/iscas89/s386.v", "s386_bench"),
           ("benchmarks/iscas89/s510.v", "s510_bench"),
           ("benchmarks/iscas89/s1488.v", "s1488_bench")]


def _parse_batches(s: str):
    vals = []
    for x in s.split(","):
        x = x.strip()
        vals.append(1 << int(x[2:]) if x.startswith("2^") else int(x))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default="nfsr16,s349_bench,s1488_bench")
    ap.add_argument("--batches", default="2^16,2^18,2^20")
    ap.add_argument("--cycles", type=int, default=512)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    selected = set(x.strip() for x in args.designs.split(",") if x.strip())
    C = args.cycles
    BATCHES = _parse_batches(args.batches)
    for src, top in DESIGNS:
        if top not in selected:
            continue
        print(f"\n[build] {top}", flush=True)
        es = EsopSim(src, top)
        npi = es.I
        nl = parse_netlist(synth(src, top).json_path, top)
        plan = build_plan(nl)
        summ = 0
        for layer in plan.layers:
            mset = set()
            for lut in layer:
                mset |= set(lut_anf(lut))
            summ += len(mset)
        sim = B1AnfSim(plan)
        # correctness (small batch)
        rng = np.random.default_rng(0)
        Bc = 64
        ug = (rng.integers(0, 2, (C, Bc, npi)).astype(np.int8) if npi else np.zeros((C, Bc, 0), np.int8))
        ok = bool((es.run(ug, C).cpu().numpy() == es.golden(ug, C)).all())
        print(f"\n=== {top}: ESOP 1 layer / F={es.F} cubes (full {es.F_full}; "
              f"{len(es.nonlin)} nonlinear latch-D, {len(es.copy)} copy, {len(es.const)} const of {es.L})"
              f"  vs  current {len(plan.layers)} layers / {summ} total monomials"
              f"  (bit-exact={'YES' if ok else 'NO'})", flush=True)
        print(f"{'batch':>10}{'ESOP cyc*b/s':>15}{'cur cyc*b/s':>14}{'speedup':>9}", flush=True)
        for B in BATCHES:
            print(f"[run] {top} batch={B}", flush=True)
            u = (rng.integers(0, 2, (C, B, npi)).astype(np.int8) if npi else np.zeros((C, B, 0), np.int8))
            uc = (rng.integers(0, 2, (C, B, nl.n_input)).astype(np.int8) if nl.n_input else np.zeros((C, B, 0), np.int8))
            e_tp = B * C / time_sim(lambda: es.run(u, C, sample=False),
                                    warmup=args.warmup, runs=args.runs)
            c_tp = B * C / time_sim(lambda: sim.run(uc, C),
                                    warmup=args.warmup, runs=args.runs)
            print(f"{B:>10}{e_tp:>15.3e}{c_tp:>14.3e}{e_tp / c_tp:>9.2f}", flush=True)


if __name__ == "__main__":
    main()
