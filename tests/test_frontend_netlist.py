import json

from rtlgemm.frontend import parse_netlist


def test_parse_netlist_ignores_scopeinfo(tmp_path):
    design = {
        "modules": {
            "top": {
                "ports": {
                    "clk": {"direction": "input", "bits": [2]},
                    "i": {"direction": "input", "bits": [3]},
                    "o": {"direction": "output", "bits": [4]},
                },
                "cells": {
                    "scope": {"type": "$scopeinfo", "connections": {}, "parameters": {}},
                    "lut": {
                        "type": "$lut",
                        "connections": {"A": [3], "Y": [4]},
                        "parameters": {"WIDTH": "00000000000000000000000000000001", "LUT": "10"},
                    },
                },
            }
        }
    }
    path = tmp_path / "scopeinfo.json"
    path.write_text(json.dumps(design))

    nl = parse_netlist(str(path), "top")

    assert len(nl.luts) == 1
    assert nl.luts[0].name == "lut"
