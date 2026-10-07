import json

import numpy as np

from rtlgemm.frontend import parse_netlist
from rtlgemm.ir import build_plan
from rtlgemm.runtime.simulate import CompiledSim


def test_lut_backends_sample_constant_and_missing_po_bits(tmp_path):
    design = {
        "modules": {
            "top": {
                "ports": {
                    "clk": {"direction": "input", "bits": [2]},
                    "o": {"direction": "output", "bits": ["0", "1", 999, 10]},
                },
                "cells": {
                    "hold": {
                        "type": "$lut",
                        "connections": {"A": [10, 10], "Y": [11]},
                        "parameters": {"WIDTH": "00000000000000000000000000000010", "LUT": "1000"},
                    },
                    "q": {
                        "type": "$_DFF_P_",
                        "connections": {"C": [2], "D": [11], "Q": [10]},
                        "parameters": {},
                    }
                },
            }
        }
    }
    path = tmp_path / "const_po.json"
    path.write_text(json.dumps(design))

    nl = parse_netlist(str(path), "top")
    plan = build_plan(nl)
    x0 = np.ones((3, nl.n_state), dtype=np.uint8)
    u = np.zeros((2, 3, nl.n_input), dtype=np.uint8)
    expected = np.array([0, 1, 0, 1], dtype=np.uint8)

    for backend in ("auto", "gather"):
        sim = CompiledSim.build(plan, 3, 2, "cpu", use_cuda_graph=False, backend=backend)
        sim.run(x0, u)
        po = sim.po_out.numpy().astype(np.uint8)
        assert np.array_equal(po, np.broadcast_to(expected, po.shape))
