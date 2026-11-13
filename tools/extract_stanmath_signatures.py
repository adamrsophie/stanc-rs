#!/usr/bin/env python3
"""Writes the signatures of the stanmath-rs functions that implement Stan library functions.

Usage (from the repository root):

    for c in autodiff core distributions linalg solvers transforms; do
      RUSTC_BOOTSTRAP=1 cargo rustdoc -p stanmath-rs-$c -- -Z unstable-options --output-format json
    done
    python3 tools/extract_stanmath_signatures.py > crates/stanc-rs-codegen/data/stanmath_rs_signatures.txt

Each output line is `crate stan_name rust_name<generic bounds>(param types) -> output`. Only free
functions named after a Stan library function (or its `_lupdf`/`_lupmf` form), possibly with one of
the overload suffixes below, are kept.
"""
import json
import re
import sys

CRATES = ["autodiff", "core", "distributions", "linalg", "solvers", "transforms"]
SIGS = "crates/stanc-rs-semantic/data/stan_math_signatures_2.40.0.txt"


def stan_names():
    names = set()
    for line in open(SIGS):
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\(", line)
        if m:
            names.add(m.group(1))
    for n in list(names):
        for a, b in [("_lpdf", "_lupdf"), ("_lpmf", "_lupmf")]:
            if n.endswith(a):
                names.add(n[: -len(a)] + b)
    return names


def ty(t):
    if t is None:
        return "()"
    k, v = next(iter(t.items()))
    if k == "primitive":
        return v
    if k == "generic":
        return v
    if k == "borrowed_ref":
        return "&" + ("mut " if v["is_mutable"] else "") + ty(v["type"])
    if k == "slice":
        return "[" + ty(v) + "]"
    if k == "array":
        return "[" + ty(v["type"]) + "; " + v["len"] + "]"
    if k == "tuple":
        return "(" + ", ".join(ty(x) for x in v) + ")"
    if k == "resolved_path":
        name = v["path"].split("::")[-1]
        args = v.get("args") or {}
        ab = args.get("angle_bracketed")
        if ab and ab["args"]:
            parts = []
            for a in ab["args"]:
                if "type" in a:
                    parts.append(ty(a["type"]))
                elif "const" in a:
                    parts.append(a["const"]["expr"])
                elif "lifetime" in a:
                    continue
            if parts:
                return name + "<" + ", ".join(parts) + ">"
        return name
    if k == "impl_trait":
        bs = [b["trait_bound"]["trait"]["path"].split("::")[-1] for b in v if "trait_bound" in b]
        return "impl " + "+".join(bs)
    if k == "dyn_trait":
        return "dyn " + "+".join(t["trait"]["path"].split("::")[-1] for t in v["traits"])
    if k == "qualified_path":
        return ty(v["self_type"]) + "::" + v["name"]
    if k == "function_pointer":
        return "fn"
    return k


def bound_names(bs):
    return [("?" if b["trait_bound"]["modifier"] == "maybe" else "")
            + b["trait_bound"]["trait"]["path"].split("::")[-1]
            for b in bs if "trait_bound" in b]


def bounds(g):
    extra = {}
    for w in g.get("where_predicates", []):
        bp = w.get("bound_predicate")
        if bp and "generic" in bp["type"]:
            extra.setdefault(bp["type"]["generic"], []).extend(bound_names(bp["bounds"]))
    out = []
    for p in g["params"]:
        kind = p["kind"]
        if "type" not in kind:
            continue
        bs = bound_names(kind["type"]["bounds"]) + extra.get(p["name"], [])
        out.append(p["name"] + (": " + "+".join(bs) if bs else ""))
    return out


# Suffixes that stanmath-rs uses for the overloads of one Stan function.
SUFFIXES = [
    "complex", "array", "row", "vector", "int", "row_vector", "mu_array", "matrix", "slice",
    "from_array_2d", "from_array_3d", "matrix_vector", "vector_matrix", "vectors",
    "matrix_row_vector", "row_vector_matrix", "row_vectors", "alpha_array", "vector_v", "2d", "3d",
    "nested", "from_int_array_2d", "from_row_vector", "from_vector", "from_vector_array", "outer",
    "vector_complex", "row_vector_complex", "scalar_y", "vector_x", "ard", "cross", "ard_cross",
]


def stan_name_of(name, want):
    if name in want:
        return name
    for suf in sorted(SUFFIXES, key=len, reverse=True):
        if name.endswith("_" + suf) and name[: -len(suf) - 1] in want:
            return name[: -len(suf) - 1]
    return None


def main():
    want = stan_names()
    seen = {}
    stans = set()
    for c in CRATES:
        d = json.load(open(f"target/doc/stanmath_rs_{c}.json"))
        idx = d["index"]
        for iid, p in d["paths"].items():
            if p["kind"] != "function" or p["crate_id"] != 0:
                continue
            item = idx.get(iid)
            if not item or item.get("visibility") != "public":
                continue
            name = item["name"]
            stan = stan_name_of(name, want)
            if stan is None:
                continue
            f = item["inner"]["function"]
            sig = f["sig"]
            params = ", ".join(ty(t) for _, t in sig["inputs"])
            gen = bounds(f["generics"])
            line = f"{c} {stan} {name}<{', '.join(gen)}>({params}) -> {ty(sig['output'])}"
            seen.setdefault(name, set()).add(line)
            stans.add(stan)
    print("# stanmath-rs function signatures for Stan library names; written by")
    print("# tools/extract_stanmath_signatures.py. Do not edit by hand.")
    for name in sorted(seen):
        lines = sorted(seen[name])
        # The prelude exports the generic autodiff version over the f64 one in stanmath-rs-core.
        if len(lines) > 1:
            lines = [x for x in lines if not x.startswith("core ")]
        for line in lines:
            print(line)
    missing = sorted(want - stans)
    print(f"# {len(seen)} names found, {len(missing)} Stan names without a free function", file=sys.stderr)
    print(" ".join(missing), file=sys.stderr)


main()
