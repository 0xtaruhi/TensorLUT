#!/usr/bin/env python3
"""Patch an RTLflow code-generation bug in the emitted `_change_request` kernel.

RTLflow (Verilator 4.203 fork) emits, for designs that need change detection,
    change[tid] = __req;          // before __req is declared
    IData __req = false; __req |= ...;
    ...
    return __req;                 // inside a __global__ void kernel
which does not compile. The intent is to compute __req and publish it in change[tid]. This script moves
the store after the final assignments and drops the return; the logic is otherwise unchanged.
Usage: fix_change_request.py obj_dir/VRocket.cu
"""
import re
import sys

path = sys.argv[1]
src = open(path).read()
pat = re.compile(
    r"(void _change_request\(.*?\{)(.*?)(\n\s*change\[blockDim\.x \* blockIdx\.x \+ threadIdx\.x\] = __req;)(.*?)\n(\s*)return __req;",
    re.S)
m = pat.search(src)
if not m:
    sys.exit("pattern not found (already patched?)")
head, pre, store, body, indent = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
fixed = head + pre + body + "\n" + indent + "change[blockDim.x * blockIdx.x + threadIdx.x] = __req;"
src = src[:m.start()] + fixed + src[m.end():]
open(path, "w").write(src)
print("patched _change_request")
