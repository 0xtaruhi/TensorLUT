"""Layer-local LUT chunking policies for b1 Tensor-Core execution."""
from __future__ import annotations

import os

from ..frontend.netlist import _const_val
from .anf import lut_anf

WM, WN, WK = 8, 8, 128


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def b1_chunk_bmma_cost(f: int, nin: int, g: int) -> int:
    """BMMA tiles used by the packed-v8 two-stage LUT chunk."""
    stage1 = ceildiv(f, WN) * ceildiv(max(nin, 1), WK)
    stage2 = ceildiv(g, WN) * ceildiv(f, WK)
    return stage1 + stage2


def input_order_key(lut, col: dict):
    vals = []
    for bit in lut.inputs:
        c = _const_val(bit)
        vals.append(-1 if c is not None else col[bit])
    return tuple(sorted(vals))


def ordered_layer_luts(layer, col: dict, strategy: str | None = None):
    strategy = strategy or os.environ.get("RTLGEMM_B1_CHUNK_ORDER", "input")
    if strategy in {"none", "original"}:
        return list(layer)
    if strategy in {"input", "greedy", "tc_greedy"}:
        return sorted(layer, key=lambda lut: input_order_key(lut, col))
    raise ValueError(f"unknown LUT chunk order strategy: {strategy}")


def _mono_support(monos: set[tuple]) -> set:
    return {v for mono in monos for v in mono}


def _greedy_layer_chunks(layer, col: dict, chunk_outputs: int,
                         window: int) -> list[list]:
    """Greedily cluster nearby LUTs by the runtime BMMA tile score.

    The search is intentionally windowed after the stable input-order sort. Full
    all-pairs clustering is unnecessary for these layers and makes host-side
    compilation noticeably heavier than the kernel work we are trying to tune.
    """
    ordered_items = []
    for lut in sorted(layer, key=lambda x: input_order_key(x, col)):
        monos = set(lut_anf(lut))
        ordered_items.append({
            "lut": lut,
            "monos": monos,
            "support": _mono_support(monos),
            "key": input_order_key(lut, col),
        })

    chunks: list[list] = []
    window = max(chunk_outputs, (window // chunk_outputs) * chunk_outputs)
    for offset in range(0, len(ordered_items), window):
        items = list(ordered_items[offset:offset + window])
        while items:
            seed_i = max(
                range(len(items)),
                key=lambda i: (len(items[i]["monos"]), len(items[i]["support"]))
            )
            seed = items.pop(seed_i)
            chunk = [seed]
            mono_union = set(seed["monos"])
            support_union = set(seed["support"])

            while len(chunk) < chunk_outputs and items:
                best_i = 0
                best_score = None
                for i, cand in enumerate(items):
                    new_monos = mono_union | cand["monos"]
                    new_support = support_union | cand["support"]
                    score = (
                        b1_chunk_bmma_cost(len(new_monos), len(new_support),
                                           len(chunk) + 1),
                        len(new_monos),
                        len(new_support),
                        -len(cand["monos"] & mono_union),
                        cand["key"],
                    )
                    if best_score is None or score < best_score:
                        best_i = i
                        best_score = score
                cand = items.pop(best_i)
                chunk.append(cand)
                mono_union.update(cand["monos"])
                support_union.update(cand["support"])

            chunks.append([item["lut"] for item in chunk])
    return chunks


def chunk_layer_luts(layer, col: dict, chunk_outputs: int,
                     strategy: str | None = None,
                     greedy_window: int | None = None) -> list[list]:
    strategy = strategy or os.environ.get("RTLGEMM_B1_CHUNK_ORDER", "input")
    if chunk_outputs <= 0:
        raise ValueError("chunk_outputs must be positive")
    if strategy in {"greedy", "tc_greedy"}:
        if greedy_window is None:
            greedy_window = int(os.environ.get("RTLGEMM_B1_CHUNK_GREEDY_WINDOW", "256"))
        return _greedy_layer_chunks(layer, col, chunk_outputs, max(1, greedy_window))

    ordered = ordered_layer_luts(layer, col, strategy)
    return [ordered[i:i + chunk_outputs] for i in range(0, len(ordered), chunk_outputs)]
