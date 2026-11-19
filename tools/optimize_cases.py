#!/usr/bin/env python3
"""Extracts stanc3's unit tests of single optimization passes into a fixture for stanc-rs.

Usage (from the repository root):

    python3 tools/optimize_cases.py STANC3_DIR > crates/stanc-rs/tests/data/optimize_cases.txt

Each case of stanc3's `test/unit/Optimize.ml` that compiles a Stan program, applies passes with
`let mir = PASS mir in`, and prints the result with `Program.Typed.pp` becomes a record of the
fixture: a `@@@ case` line with the name, a `@@@ passes` line, then the program and stanc3's
expected text, each after its own `@@@` line. Cases that build MIR by hand or print
s-expressions are skipped.
"""
import os
import re
import sys

src = open(os.path.join(sys.argv[1], "test", "unit", "Optimize.ml")).read()
cases = []
for m in re.finditer(r'let%expect_test "([^"]*)" =(.*?)\[%expect\s*\{\|(.*?)\|\}\]', src, re.S):
    name, body, expected = m.group(1), m.group(2), m.group(3)
    prog = re.search(r"reset_and_mir_of_string\s*\{\|(.*?)\|\}", body, re.S)
    if not prog or "Program.Typed.pp" not in body:
        continue
    passes = re.findall(r"let mir =\s*([a-z_]+(?: ~[a-z_]+:[a-z]+)*) mir in", body)
    if not passes:
        continue
    cases.append({"name": name, "program": prog.group(1), "passes": passes,
                  "expected": expected})
for c in cases:
    print(f"@@@ case {c['name']}")
    print(f"@@@ passes {' '.join(c['passes'])}")
    print("@@@ program")
    print(c["program"].strip("\n"))
    print("@@@ expected")
    print(c["expected"].strip("\n"))
