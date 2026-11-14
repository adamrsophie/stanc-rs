# Architecture

stanc-rs is a compiler from the Stan language (stanc3 v2.40.0 syntax and semantics) to Rust.
The generated Rust evaluates log densities and gradients with
[stanmath-rs](https://github.com/adamrsophie/stanmath-rs), a Rust port of the Stan Math library.
stanc-rs is an independent project. It is not an official Stan compiler.

## Crates

| Crate | Role |
| --- | --- |
| `stanc-rs-syntax` | Lexer (with `#include`), parser, untyped AST, pretty printer, diagnostics |
| `stanc-rs-semantic` | Stan Math signature table, type checker, typed AST |
| `stanc-rs-codegen` | Rust code generation from the typed AST |
| `stanc-rs-runtime` | Support library linked by generated code: model traits, JSON data, transforms, indexing |
| `stanc-rs` | Library entry points and the `stanc-rs` command line tool |
| `stanc-rs-fixtures` | Stan models compiled at build time, with numerical tests (not published) |

## Pipeline

1. **Lexing.** `stanc_rs_syntax::lexer` tokenizes the source and splices `#include` files through
   an `IncludeResolver`. Spans carry a file index, so diagnostics point into included files.
2. **Parsing.** A hand-written recursive descent parser (`parser.rs`) builds the AST in `ast.rs`.
   Error messages and locations follow stanc3 where the corpus checks them.
3. **Semantic analysis.** `stanc_rs_semantic::check_program` resolves names, checks block rules
   (for example, no assignment to data, `_rng` only where allowed, `target` only in the model
   block and `_lp` functions), resolves overloads against the Stan Math signatures, and computes
   the autodiff level of every expression. The result is the typed AST in `typed.rs`. Implicit
   conversions are explicit `Promotion` nodes.
4. **Code generation.** `stanc_rs_codegen::generate` walks the typed AST and prints one Rust
   module (see [runtime-contract.md](runtime-contract.md)).

There is no separate optimizing IR yet. The typed AST is the intermediate representation; the
`ir` CLI command prints it.

## Code generation

* `program.rs` lays out the module: user functions, the `Model` struct, the constructor (data
  reading, transformed data, constraint checks, output shapes) and the `GeneratedModel` methods.
* `stmt.rs` lowers statements: assignment through `rt::index` (one index at a time for any mix
  of single and multiple indexes), `~` statements (including truncation through `_lcdf` and
  `_lccdf`, element by element for container outcomes), loops, `print`, `reject`,
  `fatal_error`. Each top-level statement of a program block runs in an immediately called
  closure, because rustc's borrow checking time grows faster than linearly with the size of a
  function body.
* `expr.rs` lowers expressions: conversions between Stan types, operators (`rt::ops` for
  container arithmetic), indexing, and calls.
* `solvers.rs` lowers the higher-order functions (ODE, DAE, adjoint ODE, algebraic solvers,
  1D integrators, `reduce_sum`, `map_rect`): each function argument becomes a local struct that
  implements the stanmath-rs system trait, with the autodiff arguments flattened into one
  vector and rebuilt inside the trait method. `laplace.rs` does the same for the Laplace
  approximation functions.
* `library.rs` matches a Stan library call to a stanmath-rs function. The table
  `data/stanmath_rs_signatures.txt` lists every stanmath-rs free function by Stan name with its
  Rust signature. It is generated from rustdoc JSON by `tools/extract_stanmath_signatures.py`.
  Calls of scalar functions on containers are vectorized elementwise when stanmath-rs has no
  container overload. Functions stanmath-rs does not have for some element type (slicing,
  sorting, and reshaping of integer, complex, or nested containers; complex linear algebra)
  go to `rt::generic`, and the `_constrain`, `_jacobian`, and `_unconstrain` functions to
  `rt::transform`.

### Scalar types

The log density is generic: `fn log_prob_generic<T: Scalar>(&self, theta: &[T], ...)`. Values
that can depend on parameters have type `T`; data stay `f64`. With `T = f64` the same code
computes plain values, and with stanmath-rs's reverse-mode `Var` it computes gradients. Data used
where an autodiff value is required is lifted with `T::from_f64` or `rt::stan::lift_*`.

### Errors

Generated code never panics on bad input. Every fallible operation returns
`rt::Result` and is propagated with `?`. Indexing is checked by `rt::index`, which reports
one-based indices like Stan.

## Updating the stanmath-rs signature table

When stanmath-rs gains functions, bump the pinned revision in the workspace `Cargo.toml` and run

```sh
python3 tools/extract_stanmath_signatures.py
```

which builds rustdoc JSON for the stanmath-rs crates and rewrites
`crates/stanc-rs-codegen/data/stanmath_rs_signatures.txt`. New functions become callable without
codegen changes when their parameter types are among those listed in `library::Param`.

## Developer notes

* **New syntax:** add tokens in `lexer.rs`, AST nodes in `ast.rs`, parsing in `parser.rs`, and
  printing in `pretty.rs`. The corpus tests compare acceptance with stanc3.
* **New types:** extend `UnsizedType` and `SizedType`, the checker's conversion rules in
  `types.rs`, and `rust_type`, `convert`, and `default_value` in the code generator.
* **New library functions:** add the Stan signatures in `stanc-rs-semantic/src/signatures.rs`
  (if stanc3 has them) and the implementation in stanmath-rs; then regenerate the table.
* **New code generation rules:** special cases for calls live in `Gen::special_call`; operator
  lowering in `Gen::binop`.
