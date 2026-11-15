# stanc-rs

A compiler from the [Stan](https://mc-stan.org) probabilistic programming language to Rust.

stanc-rs parses and type checks Stan programs (following stanc3 v2.40.0) and translates them into
Rust modules whose log densities and gradients are computed by
[stanmath-rs](https://github.com/adamrsophie/stanmath-rs), a Rust port of the Stan Math library.
It is part of a series of Rust ports of the Stan stack by Adam R Sophie.

stanc-rs is an independent, unofficial implementation. It is not affiliated with or endorsed by
the Stan Development Team, and it is not a drop-in replacement for stanc3 or CmdStan.

## Status

Early development. See [docs/compatibility.md](docs/compatibility.md) for measured coverage and
known gaps.

* Parsing and type checking: all 1244 programs in stanc3's `test/integration/good` corpus are
  accepted and all 1191 programs in the `bad` corpus are rejected, with the error on the same
  line as stanc3.
* Code generation: 1242 of the 1244 good programs are translated, and the Rust of all 1242
  builds. The two that are not use a Laplace function variant stanmath-rs does not have yet.
* Numerical tests: the models in `crates/stanc-rs-fixtures/models` are compiled at build time and
  their log densities, gradients, Jacobians, constrained values, and errors are checked against
  independent closed forms. Their log densities and gradients also agree with CmdStan 2.40.0.

## Build

Rust 1.88 or newer.

```sh
cargo build --release
cargo test --workspace
```

## Command line

```sh
stanc-rs check   model.stan            # parse and type check
stanc-rs compile model.stan -o model.rs  # generate Rust
stanc-rs ast     model.stan            # print the syntax tree
stanc-rs ir      model.stan            # print the typed program
stanc-rs format  model.stan            # as stanc --auto-format
stanc-rs info    model.stan            # variables and library calls as JSON, as stanc3 --info
```

Options: `--name <model name>`, `--include-paths <dir,dir>`, `--allow-undefined`.

## Using a generated model

Add `stanc-rs-runtime` as a dependency, include the generated file, and use the `Model` trait:

```rust
mod normal { include!("normal.rs"); }

use stanc_rs_runtime::{data::Data, DensityOptions, Model};

let data = Data::from_json_str(r#"{"y": 1.5}"#)?;
let model = normal::Model::new(&data, 1234)?;
let (lp, grad) = model.log_density_gradient(&[0.3], DensityOptions::default())?;
```

The generated interface is described in [docs/runtime-contract.md](docs/runtime-contract.md) and
the compiler design in [docs/architecture.md](docs/architecture.md).

## License

GPL-3.0-only. See [LICENSE](LICENSE).
