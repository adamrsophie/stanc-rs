#!/usr/bin/env python3
"""Find the Stan Math signatures that support the struct of arrays (SoA) memory pattern.

stanc3's signature table records, for each signature, whether Stan Math accepts SoA matrices
there; `stanc --dump-stan-math-signatures` omits it. This script recovers it by compiling probe
models with `stanc --O1 --debug-mem-patterns`:

* With an autodiff vector or matrix argument, the call is reduced to a scalar by indexing,
  `target += f(p...)[1]`, so the only reason to demote the parameters is the call itself.
* Otherwise, with a real vector or matrix result, all arguments are data and the result is
  assigned to a local, whose pattern shows the support.

`_jacobian` functions are probed the same way in transformed parameters, the only block that
allows them.

A probe answers whether some signature the arguments convert to supports SoA, which is what the
optimizer asks of a call, so the answers compose. Signatures that no probe can observe are left
out; the optimizer never needs them.

Usage: soa_signatures.py STANC SIGNATURES > soa_signatures.txt
"""

import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

STANC, SIGS = sys.argv[1], sys.argv[2]
REAL_EIGEN = {"vector", "row_vector", "matrix"}
COMPLEX_EIGEN = {"complex_vector", "complex_row_vector", "complex_matrix"}
DIMS = {"vector": 1, "row_vector": 1, "matrix": 2,
        "complex_vector": 1, "complex_row_vector": 1, "complex_matrix": 2}
BAR_SUFFIXES = ("_lpdf", "_lpmf", "_lcdf", "_lccdf", "_cdf")


def split_top(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse(line):
    name, rest = line.split("(", 1)
    depth = 1
    for i, ch in enumerate(rest):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0:
            break
    return name, split_top(rest[:i]), rest[i + 1:].split("=>", 1)[1].strip()


def base(t):
    """(array dimensions, element type) of an unsized type."""
    m = re.match(r"array\[(,*)\]\s*(.*)", t)
    return (len(m.group(1)) + 1, m.group(2)) if m else (0, t)


def sized(t):
    if t.startswith("tuple("):
        return "tuple(" + ", ".join(sized(x) for x in split_top(t[6:-1])) + ")"
    n, b = base(t)
    inner = {1: b + "[2]", 2: b + "[2, 2]"}.get(DIMS.get(b, 0), b)
    return f"array[{', '.join(['2'] * n)}] {inner}" if n else inner


def contains(t, kinds):
    return any(re.search(rf"\b{k}\b", t) for k in kinds)


def to_scalar(expr, t):
    """An expression indexing `expr` of type `t` down to a real."""
    if t.startswith("tuple("):
        return to_scalar(f"({expr}).1", split_top(t[6:-1])[0])
    n, b = base(t)
    k = n + DIMS.get(b, 0)
    if k:
        expr = f"({expr})[{', '.join(['1'] * k)}]"
    return f"get_real({expr})" if b.startswith("complex") else expr


def probe(i, name, args, ret):
    """The declarations and statement of probe `i`, and the variable that shows the pattern;
    None if no probe can observe the pattern."""
    ad_eigen = any(not a.startswith("data ") and base(a)[1] in REAL_EIGEN for a in args)
    if not ad_eigen and not (contains(ret, REAL_EIGEN) and not contains(ret, COMPLEX_EIGEN)):
        return None
    data, params, vals = [], [], []
    for j, a in enumerate(args):
        t = a.removeprefix("data ")
        b = base(t)[1]
        is_param = ad_eigen and not a.startswith("data ") and b in REAL_EIGEN | {"real"}
        v = f"{'p' if is_param else 'd'}{i}_{j}"
        (params if is_param else data).append(f"  {sized(t)} {v};")
        vals.append(v)
    if len(vals) > 1 and name.endswith(BAR_SUFFIXES):
        call = f"{name}({vals[0]} | {', '.join(vals[1:])})"
    else:
        call = f"{name}({', '.join(vals)})"
    if name.endswith("_jacobian"):
        # Only allowed in transformed parameters, whose variables are kept.
        decl = f"real t{i} = {to_scalar(call, ret)}" if ad_eigen else f"{sized(ret)} l{i} = {call}"
        stmt = f"  {decl};"
        watch = next((f"p{i}_{j}" for j, a in enumerate(args)
                      if not a.startswith("data ") and base(a)[1] in REAL_EIGEN), f"l{i}")
        return data, params, stmt, watch, True
    if ad_eigen:
        stmt = f"  target += {to_scalar(call, ret)};"
        watch = next(f"p{i}_{j}" for j, a in enumerate(args)
                     if not a.startswith("data ") and base(a)[1] in REAL_EIGEN)
    else:
        stmt = f"  {{\n    {sized(ret)} l{i} = {call};\n    print(l{i});\n  }}"
        watch = f"l{i}"
    return data, params, stmt, watch, False


def run(probes):
    """The memory pattern of each watched variable, or None if the model does not compile."""
    block = "transformed parameters" if probes[0][4] else "model"
    src = "data {\n%s\n}\nparameters {\n%s\n}\n%s {\n%s\n}\n" % (
        "\n".join(d for p in probes for d in p[0]),
        "\n".join(d for p in probes for d in p[1]),
        block,
        "\n".join(p[2] for p in probes))
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "probe.stan"
        f.write_text(src)
        r = subprocess.run([STANC, "--O1", "--debug-mem-patterns", str(f),
                            "--o", str(Path(tmp) / "probe.hpp")],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode:
        return None
    pats = dict(re.findall(r" (\w+): (SoA|AoS)$", r.stdout, re.M))
    return pats


groups = defaultdict(list)
for line in Path(SIGS).read_text().splitlines():
    name, args, ret = parse(line)
    if name.endswith("_rng") or any("=>" in a for a in args):
        continue
    groups[name].append((line, args, ret))

failed = 0
for name in sorted(groups):
    items = []
    for line, args, ret in groups[name]:
        p = probe(len(items), name, args, ret)
        if p:
            items.append((line, p))
    if not items:
        continue
    pats = run([p for _, p in items])
    if pats is None:
        results = []
        for line, (_, args, ret) in [(l, next(g for g in groups[name] if g[0] == l))
                                     for l, _ in items]:
            p = probe(0, name, args, ret)
            r = run([p])
            if r is None:
                failed += 1
                print(f"cannot probe: {line}", file=sys.stderr)
            results.append((line, r and r.get(p[3])))
    else:
        results = [(line, pats.get(p[3])) for line, p in items]
    for line, pat in results:
        if pat == "SoA":
            print(line)
print(f"{failed} signatures could not be probed", file=sys.stderr)
