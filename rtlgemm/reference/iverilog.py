"""Independent golden reference via Icarus Verilog on the *original* RTL.

This is a fully independent oracle: an event-driven simulator running the source
Verilog, with no dependence on Yosys synthesis or the numpy interpreter. The whole
batch is looped inside one generated testbench (one vvp process) for speed.

Assumptions (MVP): single clock; the initial state is loaded by hierarchically
assigning one state register (``state_reg``) to a per-stimulus seed; outputs are
sampled while the clock is low (register-backed outputs). Signal widths <= 64.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

from ..frontend.netlist import Netlist


def _port_width(bits) -> int:
    return len(bits)


def _input_port_columns(nl: Netlist):
    """Map each non-clock input port to (port_width, [(bit_pos_in_port, u_column)])."""
    col = 0
    ports = {}
    for pname, bits in nl.inputs:
        entries = []
        for pos, b in enumerate(bits):
            if b in nl.clock_bits:
                continue
            entries.append((pos, col))
            col += 1
        if entries:
            ports[pname] = (len(bits), entries)
    return ports


def golden_iverilog_po(src: str, top: str, nl, u_seq: np.ndarray, cycles: int, *,
                       clk: str = "blif_clk_net", workdir: str = "build/iv_po",
                       iverilog: str = "iverilog", vvp: str = "vvp") -> dict:
    """Generic golden: drive ALL primary inputs (in nl.input_bits order, grouped by
    port) from ``u_seq`` and record ALL primary outputs each cycle, sampled from the
    combinational logic of (state_t, u_t) before the clock edge. No register seeding —
    the reset PI is driven via ``u_seq``. Returns {port: (cycles, batch) int}."""
    batch = u_seq.shape[1]
    out_w = {p: len(b) for p, b in nl.outputs}
    out_ports = [p for p, _ in nl.outputs]
    in_ports = _input_port_columns(nl)
    wd = os.path.join(workdir, top)
    os.makedirs(wd, exist_ok=True)

    for pname, (w, entries) in in_ports.items():
        stream = np.zeros((cycles, batch), dtype=np.int64)
        for pos, c in entries:
            stream |= (u_seq[:, :, c].astype(np.int64) & 1) << pos
        with open(os.path.join(wd, f"in_{pname}.hex"), "w") as f:
            for b in range(batch):
                for t in range(cycles):
                    f.write(f"{int(stream[t, b]):x}\n")

    L = ["`timescale 1ns/1ps", "module tb;", f"  reg {clk};"]
    for pname, (w, _) in in_ports.items():
        L.append(f"  reg [{w-1}:0] {pname};")
    for p in out_ports:
        L.append(f"  wire [{out_w[p]-1}:0] {p};")
    L.append("  integer b, t, fo;")
    for pname, (w, _) in in_ports.items():
        L.append(f"  reg [{w-1}:0] mem_{pname} [0:{batch*cycles-1}];")
    conns = [f".{clk}({clk})"] + [f".{p}({p})" for p in in_ports] + [f".{p}({p})" for p in out_ports]
    L.append(f"  {top} dut({', '.join(conns)});")
    L.append("  initial begin")
    for pname in in_ports:
        L.append(f'    $readmemh("in_{pname}.hex", mem_{pname});')
    L.append('    fo = $fopen("out.txt", "w");')
    L.append(f"    {clk} = 0;")
    L.append(f"    for (b = 0; b < {batch}; b = b + 1) begin")
    L.append(f"      for (t = 0; t < {cycles}; t = t + 1) begin")
    for pname in in_ports:
        L.append(f"        {pname} = mem_{pname}[b*{cycles}+t];")
    L.append("        #1;")
    fmt = " ".join(["%0d"] * len(out_ports)); args = ", ".join(out_ports)
    L.append(f'        $fwrite(fo, "{fmt}\\n", {args});')     # sample PO before edge
    L.append(f"        {clk} = 1; #1; {clk} = 0; #1;")
    L.append("      end")
    L.append("    end")
    L.append("    $fclose(fo); $finish;")
    L.append("  end\nendmodule")
    with open(os.path.join(wd, "tb.v"), "w") as f:
        f.write("\n".join(L) + "\n")

    r = subprocess.run([iverilog, "-g2012", "-o", "sim.vvp", "tb.v", os.path.abspath(src)],
                       cwd=wd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"iverilog compile failed:\n{r.stdout}\n{r.stderr}")
    r = subprocess.run([vvp, "sim.vvp"], cwd=wd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"vvp failed:\n{r.stdout}\n{r.stderr}")

    rows = [ln.split() for ln in open(os.path.join(wd, "out.txt")) if ln.strip()]
    assert len(rows) == batch * cycles, (len(rows), batch * cycles)
    result = {p: np.zeros((cycles, batch), dtype=np.int64) for p in out_ports}
    i = 0
    for b in range(batch):
        for t in range(cycles):
            for pi, p in enumerate(out_ports):
                result[p][t, b] = int(rows[i][pi])
            i += 1
    return result


def golden_iverilog(src: str, top: str, nl: Netlist, seeds: np.ndarray,
                    u_seq: np.ndarray, cycles: int, *, clk: str = "clk",
                    state_reg: str = "state", out_ports: list | None = None,
                    workdir: str = "build/iv", iverilog: str = "iverilog",
                    vvp: str = "vvp") -> dict:
    """Run the golden simulation.

    seeds: (batch,) int initial value of ``state_reg``
    u_seq: (cycles, batch, n_input) uint8 in nl.input_bits order
    returns {port_name: (cycles+1, batch) int} for each requested output port.
    """
    batch = seeds.shape[0]
    if out_ports is None:
        out_ports = [p for p, _ in nl.outputs]
    out_w = {p: _port_width(b) for p, b in nl.outputs}
    in_ports = _input_port_columns(nl)
    state_w = out_w.get(state_reg, max((w for w, _ in in_ports.values()), default=32))

    wd = os.path.join(workdir, top)
    os.makedirs(wd, exist_ok=True)

    # seed memory
    with open(os.path.join(wd, "seed.hex"), "w") as f:
        for b in range(batch):
            f.write(f"{int(seeds[b]):x}\n")

    # per-input-port stimulus memories (row order: b*cycles + t)
    port_streams = {}
    for pname, (w, entries) in in_ports.items():
        stream = np.zeros((cycles, batch), dtype=np.int64)
        for pos, c in entries:
            stream |= (u_seq[:, :, c].astype(np.int64) & 1) << pos
        port_streams[pname] = (w, stream)
        with open(os.path.join(wd, f"in_{pname}.hex"), "w") as f:
            for b in range(batch):
                for t in range(cycles):
                    f.write(f"{int(stream[t, b]):x}\n")

    # --- generate testbench ---
    L = []
    L.append("`timescale 1ns/1ps")
    L.append("module tb;")
    L.append(f"  reg {clk};")
    for pname, (w, _) in in_ports.items():
        L.append(f"  reg [{w-1}:0] {pname};")
    for p in out_ports:
        L.append(f"  wire [{out_w[p]-1}:0] {p};")
    L.append("  integer b, t, fo;")
    L.append(f"  reg [{state_w-1}:0] seed_mem [0:{batch-1}];")
    for pname, (w, _) in in_ports.items():
        L.append(f"  reg [{w-1}:0] mem_{pname} [0:{batch*cycles-1}];")

    conns = [f".{clk}({clk})"]
    conns += [f".{p}({p})" for p in in_ports]
    conns += [f".{p}({p})" for p in out_ports]
    L.append(f"  {top} dut({', '.join(conns)});")

    L.append("  initial begin")
    L.append('    $readmemh("seed.hex", seed_mem);')
    for pname in in_ports:
        L.append(f'    $readmemh("in_{pname}.hex", mem_{pname});')
    L.append('    fo = $fopen("out.txt", "w");')
    L.append(f"    {clk} = 0;")
    L.append(f"    for (b = 0; b < {batch}; b = b + 1) begin")
    L.append(f"      {clk} = 0; #1;")
    L.append(f"      dut.{state_reg} = seed_mem[b]; #1;")
    fmt = " ".join(["%0d"] * len(out_ports))
    args = ", ".join(out_ports)
    L.append(f'      $fwrite(fo, "{fmt}\\n", {args});')
    L.append(f"      for (t = 0; t < {cycles}; t = t + 1) begin")
    for pname in in_ports:
        L.append(f"        {pname} = mem_{pname}[b*{cycles}+t];")
    L.append("        #1;")
    L.append(f"        {clk} = 1; #1;")
    L.append(f'        $fwrite(fo, "{fmt}\\n", {args});')
    L.append(f"        {clk} = 0; #1;")
    L.append("      end")
    L.append("    end")
    L.append("    $fclose(fo); $finish;")
    L.append("  end")
    L.append("endmodule")
    tb_path = os.path.join(wd, "tb.v")
    with open(tb_path, "w") as f:
        f.write("\n".join(L) + "\n")

    # compile + run
    src_abs = os.path.abspath(src)
    vvp_out = os.path.join(wd, "sim.vvp")
    r = subprocess.run([iverilog, "-g2012", "-o", "sim.vvp", "tb.v", src_abs],
                       cwd=wd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"iverilog compile failed:\n{r.stdout}\n{r.stderr}")
    r = subprocess.run([vvp, "sim.vvp"], cwd=wd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"vvp run failed:\n{r.stdout}\n{r.stderr}")

    # parse out.txt: batch blocks of (cycles+1) lines, each line = space-sep ints
    rows = [ln.split() for ln in open(os.path.join(wd, "out.txt")) if ln.strip()]
    expected = batch * (cycles + 1)
    assert len(rows) == expected, (len(rows), expected)
    result = {p: np.zeros((cycles + 1, batch), dtype=np.int64) for p in out_ports}
    r_i = 0
    for b in range(batch):
        for t in range(cycles + 1):
            vals = rows[r_i]; r_i += 1
            for pi, p in enumerate(out_ports):
                result[p][t, b] = int(vals[pi])
    return result
