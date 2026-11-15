# Compatibility

stanc-rs follows stanc3 v2.40.0. This page records what has been measured against stanc3's own
test corpus and against a stanc3 v2.40.0 binary, and what is known not to work. It is an
independent implementation; nothing here is a claim of full equivalence with stanc3 or CmdStan.

All numbers below were measured against the `test/integration` programs of the stanc3 v2.40.0
release, with stanmath-rs at revision `04dc87184161bf29cbc035a32eb0407b14169900`.

## Summary

| Check | Result |
| --- | --- |
| `good` programs accepted by the type checker | 1244 of 1244 |
| `bad` programs rejected | 1191 of 1191 (stanc3 rejects all 1191) |
| Crashes on `bad` programs | 0 |
| Error reported on the same line as stanc3 | 1191 of 1191 |
| Same error category as stanc3 (syntax or semantic) | 1173 of 1191 |
| `good` programs translated to Rust | 1244 of 1244 |
| Translated programs whose Rust builds (`cargo check`) | 1244 of 1244 |
| `stanc-rs info` equal to `stanc --info` (as JSON values) | 1244 of 1244 |
| Fixture models whose log density and gradient agree with CmdStan's `log_prob` | 24 of 24 |
| `stanc-rs format` output identical to `stanc --auto-format` | 1245 of 1245 (1244 programs and one `.stanfunctions` file) |
| The same with `--max-line-length` 12 and 40, and with each `--canonicalize` option and `--print-canonical` | 1245 of 1245 each |

"Builds" means the generated module type checks against the runtime and stanmath-rs. It does not
mean the model's numbers were checked; see [Numerical validation](#numerical-validation) for the
models whose values are tested.

## How to reproduce

```sh
cargo build --release
python3 tools/corpus.py STANC3_DIR OUT_DIR --check   # translation and cargo check
python3 tools/info_diff.py STANC3_DIR STANC          # info against stanc --info
python3 tools/bad_diff.py STANC3_DIR STANC           # rejections and error locations
python3 tools/format_diff.py STANC3_DIR STANC        # format against stanc --auto-format
python3 tools/cmdstan_diff.py CMDSTAN_DIR WORK_DIR   # fixture densities against CmdStan
```

`STANC3_DIR` is a stanc3 v2.40.0 checkout, `CMDSTAN_DIR` a CmdStan 2.40.0 installation, and
`STANC` a stanc3 v2.40.0 binary (CmdStan ships one as `bin/stanc`). `corpus.py` writes `codegen_fail.txt`, `check_errors.txt`, and
`check_bad.txt` to `OUT_DIR`.

## Extensions beyond Stan

stanc-rs accepts every program stanc3 accepts. Where a Stan function is defined mathematically
for more inputs than Stan Math evaluates, stanc-rs evaluates them:

* `laplace_marginal_neg_binomial_2_log_lpmf`, its `_tol` and `_lupmf` forms, and
  `laplace_latent_neg_binomial_2_log_rng` take the dispersion as a vector. Stan Math 5.x
  evaluates it only when the observations, latent groups, and dispersions all have length one.
  stanc-rs accepts one dispersion shared by every observation or one per observation, and the
  fixture `laplace_nb_vector` checks both against a closed form. The fully scalar case also
  agrees with CmdStan.

## Diagnostics

Every `bad` program is rejected, and the reported line always agrees with stanc3. The text of
the messages differs:

* stanc3 reports a column range (`column 23 to column 37`); stanc-rs reports the start column
  and underlines the whole span.
* For ill-typed calls, stanc3 lists every available signature and explains the first mismatch;
  stanc-rs prints the supplied types and the reason the closest signature fails.
* stanc-rs labels lexer errors `Syntax error (lexical)`; stanc3 calls 18 of them plain
  `Syntax error`.

Warnings that stanc3 emits during type checking (deprecated `_lp` calls in transformed
parameters, `_jacobian` calls without parameter arguments, unreachable statements,
self-assignment in a declaration, and others) are emitted. stanc3's `--warn-pedantic` checks
are not implemented.

## Numerical validation

The models in `crates/stanc-rs-fixtures/models` are compiled at build time, and their tests
compare log densities, gradients, Jacobian adjustments, constrained and unconstrained values,
generated quantities, and error behavior with closed forms worked out independently, or with
central finite differences where a closed form would be long. They cover scalar and container
parameters; every constraint transform, with each Jacobian adjustment checked; user-defined
functions (including `_lp`, `_jacobian`, and `_rng` functions); truncation; control flow; tuples;
complex numbers; ODE, DAE, and algebraic solvers; `integrate_1d`; `reduce_sum`; `map_rect`; and
the Laplace approximation functions.

`tools/cmdstan_diff.py` also compiles every fixture model with CmdStan 2.40.0 and compares the
log density (up to a constant) and its gradient at a fixed unconstrained point, with and
without the Jacobian adjustment. All 24 models with parameters agree to a relative 1e-8, which
also confirms that the unconstrained parameter layout is Stan's. The one looser comparison is
`vector_truncate`, whose gradients differ from CmdStan's by about 1e-6: Stan Math approximates
the derivatives of `normal_lcdf` and `normal_lccdf`, while stanmath-rs differentiates them
exactly. The fixture test checks the stanc-rs gradient against the exact derivative.

For corpus programs outside the fixtures, only the build is checked, since the corpus has no
data.

## Known differences and gaps

* **Random numbers.** `_rng` functions draw from stanmath-rs's generators, so the streams differ
  from CmdStan's even with the same seed. Distributions are the same; individual draws are not.
* **Formatting.** `stanc-rs format` ports stanc3's formatter and OCaml's `Format` line
  breaking, so its layout and comment placement are stanc3's (see the summary). The programs
  with `#include` in the stanc3 test directories also format identically, with and without
  `--canonicalize includes`. Where stanc3 stops with an internal error because an `#include`
  or comment cannot be kept in place, stanc-rs reports an ordinary error instead.
* **Not implemented command line features:** `--warn-pedantic`, optimization levels (`-O0`,
  `-O1`), `--standalone-functions`, and JSON diagnostics.
* **Optimization.** The typed AST is translated directly. There are no optimization passes.
* **Code generation target.** Generated models implement the `stanc_rs_runtime::GeneratedModel`
  trait; they are not C++ and cannot be used by CmdStan.
