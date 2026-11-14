#!/usr/bin/env python3
"""Compiles stanc3's `test/integration/good` programs with stanc-rs and type checks the Rust.

Usage (from the repository root, after `cargo build --release -p stanc-rs`):

    python3 tools/corpus.py STANC3_DIR OUT_DIR [--check]

STANC3_DIR is a checkout of stanc3 (for example the v2.40.0 release archive). OUT_DIR receives
`codegen_ok.txt` and `codegen_fail.txt` (one `path | message` line per program stanc-rs rejects),
a `gen/` directory with the generated modules, and a Cargo workspace of shard crates that include
them. With `--check`, the workspace is type checked and `check_errors.txt` lists each error with
the program it came from, and `check_bad.txt` the programs whose Rust does not build.
"""
import concurrent.futures as cf
import os
import re
import subprocess
import sys

SHARD = 25


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stanc3, out = os.path.abspath(args[0]), os.path.abspath(args[1])
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    binary = os.path.join(repo, "target", "release", "stanc-rs")
    root = os.path.join(stanc3, "test", "integration")
    good = sorted(
        os.path.relpath(os.path.join(d, f), root)
        for d, _, fs in os.walk(os.path.join(root, "good"))
        for f in fs
        if f.endswith(".stan")
    )
    gen = os.path.join(out, "gen")
    os.makedirs(gen, exist_ok=True)
    for f in os.listdir(gen):
        os.remove(os.path.join(gen, f))

    def compile_one(item):
        i, p = item
        r = subprocess.run(
            [binary, "compile", p, "-o", f"{gen}/m{i}.rs",
             "--include-paths", f"{root}/included,{os.path.dirname(os.path.join(root, p))}"],
            cwd=root, capture_output=True, text=True,
        )
        if r.returncode == 0:
            return i, p, None
        lines = [x for x in r.stderr.strip().splitlines() if x.strip()]
        return i, p, lines[-1] if lines else f"exit {r.returncode}"

    with cf.ThreadPoolExecutor(8) as ex:
        results = list(ex.map(compile_one, enumerate(good)))
    ok = [(i, p) for i, p, e in results if e is None]
    with open(f"{out}/codegen_ok.txt", "w") as f:
        f.writelines(f"m{i} {p}\n" for i, p in ok)
    with open(f"{out}/codegen_fail.txt", "w") as f:
        f.writelines(f"{p} | {e}\n" for _, p, e in results if e is not None)
    print(f"generated {len(ok)} of {len(good)}")

    shards = [ok[k:k + SHARD] for k in range(0, len(ok), SHARD)]
    members = []
    for n, chunk in enumerate(shards):
        d = f"{out}/corpus/s{n}"
        os.makedirs(f"{d}/src", exist_ok=True)
        with open(f"{d}/Cargo.toml", "w") as f:
            f.write(f'[package]\nname = "s{n}"\nversion = "0.0.0"\nedition = "2024"\n'
                    "[dependencies]\nstanc-rs-runtime.workspace = true\n")
        with open(f"{d}/src/lib.rs", "w") as f:
            for i, _ in chunk:
                f.write(f'#[path = "../../../gen/m{i}.rs"]\npub mod m{i};\n')
        members.append(f'"s{n}"')
    runtime = os.path.join(repo, "crates", "stanc-rs-runtime")
    with open(f"{out}/corpus/Cargo.toml", "w") as f:
        f.write(f"[workspace]\nresolver = \"3\"\nmembers = [{', '.join(members)}]\n"
                f'[workspace.dependencies]\nstanc-rs-runtime = {{ path = "{runtime}" }}\n')

    if "--check" not in sys.argv:
        return
    env = dict(os.environ, CARGO_TARGET_DIR=f"{out}/target")
    r = subprocess.run(
        ["cargo", "check", "--keep-going", "--workspace", "--message-format", "short"],
        cwd=f"{out}/corpus", env=env, capture_output=True, text=True,
    )
    path_of = {f"m{i}": p for i, p in ok}
    errors = []
    for line in r.stderr.splitlines():
        m = re.search(r"gen/(m\d+)\.rs:\d+:\d+: error", line)
        if m:
            errors.append(f"{path_of[m.group(1)]} | {line.split(': error', 1)[1]}")
    with open(f"{out}/check_errors.txt", "w") as f:
        f.writelines(e + "\n" for e in errors)
    bad = sorted({e.split(" | ")[0] for e in errors})
    with open(f"{out}/check_bad.txt", "w") as f:
        f.writelines(b + "\n" for b in bad)
    print(f"type checked: {len(ok) - len(bad)} of {len(ok)} generated programs build")


main()
