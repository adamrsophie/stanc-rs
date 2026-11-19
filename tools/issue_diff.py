#!/usr/bin/env python3
"""Compares stanc-rs and stanc3 on the Stan programs posted in stanc3's GitHub issues.

Usage (from the repository root, after `cargo build --release -p stanc-rs`):

    python3 tools/issue_diff.py fetch OUT_DIR
    python3 tools/issue_diff.py diff OUT_DIR STANC [REPORT] [-- extra flags]

`fetch` downloads every issue of stan-dev/stanc3 with its comments (through the `gh` CLI) and
writes each fenced code block that looks like a Stan program to OUT_DIR/repro/iN_K.stan, where N
is the issue number. `diff` runs both compilers on those programs and reports where they
disagree on acceptance or on the text of errors and warnings. STANC is a stanc3 binary of the
version stanc-rs follows; most old issues are fixed there, so a disagreement points at a bug
stanc3 had and stanc-rs may still have. Flags after `--` go to both compilers (for example
`--warn-pedantic`). With REPORT, both outputs of every disagreement are written there.
"""
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys

REPO = "stan-dev/stanc3"
BLOCK = re.compile(r"```[ \t]*([A-Za-z0-9+]*)[^\n]*\n(.*?)```", re.S)
STAN = re.compile(r"\b(model|parameters|data|functions|generated quantities|transformed)\s*\{")


def gh(*args):
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def fetch(out):
    os.makedirs(os.path.join(out, "repro"), exist_ok=True)
    issues = json.loads(gh("issue", "list", "-R", REPO, "--state", "all", "--limit", "5000",
                           "--json", "number,title,state,labels,body"))
    with open(os.path.join(out, "issues.json"), "w") as f:
        json.dump(issues, f)

    def texts(i):
        c = json.loads(gh("issue", "view", str(i["number"]), "-R", REPO, "--json", "comments"))
        return i["number"], [i["body"] or ""] + [x["body"] or "" for x in c["comments"]]

    n = 0
    with cf.ThreadPoolExecutor(6) as ex:
        for num, bodies in ex.map(texts, issues):
            k = 0
            for body in bodies:
                for lang, code in BLOCK.findall(body):
                    if lang.lower() in ("", "stan") and STAN.search(code):
                        k += 1
                        with open(os.path.join(out, "repro", f"i{num}_{k}.stan"), "w") as f:
                            f.write(code)
            n += k
    print(f"{len(issues)} issues, {n} programs")


def diff(out, st, report, extra):
    rs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "target", "release", "stanc-rs")
    rs_extra = [x for e in extra for x in e.split("=", 1)]
    d = os.path.join(out, "repro")
    files = sorted(os.listdir(d), key=lambda f: [int(x) for x in re.findall(r"\d+", f)])

    def one(f):
        p = os.path.join(d, f)
        run = lambda cmd: subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                                         stdin=subprocess.DEVNULL)
        try:
            a = run([st, "--o=/dev/null", *extra, p])
            b = run([rs, "check", p, *rs_extra])
        except subprocess.TimeoutExpired:
            return f, "timeout", "", ""
        # stanc-rs prints the path it was given; stanc3 prints it too, so the texts compare.
        return f, (a.returncode == 0) == (b.returncode == 0), a.stderr.strip(), b.stderr.strip()

    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, files))
    status = [r for r in res if r[1] is not True]
    text = [r for r in res if r[1] is True and r[2] != r[3]]
    if report:
        with open(report, "w") as f:
            for name, ok, a, b in status + text:
                kind = "ACCEPTANCE" if ok is not True else "TEXT"
                f.write(f"=== {name} {kind} {ok}\n--- stanc3\n{a}\n--- stanc-rs\n{b}\n")
    print(f"{len(res)} programs: {len(status)} disagree on acceptance or time out, "
          f"{len(text)} differ only in message text")


args = sys.argv[1:]
extra = args[args.index("--") + 1:] if "--" in args else []
args = args[: args.index("--")] if "--" in args else args
if args[0] == "fetch":
    fetch(args[1])
else:
    diff(args[1], os.path.abspath(args[2]), args[3] if len(args) > 3 else None, extra)
