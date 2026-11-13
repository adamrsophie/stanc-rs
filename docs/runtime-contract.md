# Generated model contract

`stanc-rs compile model.stan -o model.rs` writes one Rust module. It depends on the
`stanc-rs-runtime` crate (imported as `rt`) and, through it, on stanmath-rs (imported as `sm`).

## What the module defines

```rust
pub struct Model { /* data and transformed data */ }

impl Model {
    /// Reads JSON data, runs transformed data, checks every declared constraint.
    pub fn new(data: &rt::data::Data, seed: u64) -> rt::Result<Model>;
}

impl rt::GeneratedModel for Model { /* ... */ }
```

Every `GeneratedModel` also implements the object-safe `rt::Model` trait:

| Method | Meaning |
| --- | --- |
| `name()` | `<file stem>_model`, or the `--name` option |
| `num_unconstrained()` | Length of the unconstrained parameter vector |
| `log_density(theta, opts)` | Log density at unconstrained `theta` |
| `log_density_gradient(theta, opts)` | Value and gradient (reverse mode) |
| `write_array(theta, include_tp, include_gq, rng)` | Constrained parameters, then transformed parameters and generated quantities |
| `unconstrain(values)` | Unconstrained vector from constrained values given as JSON |
| `param_names(include_tp, include_gq)` | Flattened names such as `beta.1`, `Sigma.2.1`, `z.real` |
| `variables()` | Shape of each output variable |

`DensityOptions { jacobian, propto }`:

* `jacobian`: add the log absolute Jacobian determinant of each constraining transform.
* `propto`: drop constant terms. `~` statements and `_lupdf`/`_lupmf` calls drop terms that do
  not depend on parameters. Because only autodiff values know whether they are parameters,
  `log_density` with `propto` evaluates with autodiff.

## Data

`rt::data::Data::from_json_str` reads CmdStan's JSON data format: arrays and vectors as JSON
arrays, matrices as arrays of rows, complex numbers as `[re, im]`, tuples as objects with keys
`"1"`, `"2"`, .... Zero-size containers may be omitted. Sizes and constraints are checked; any
problem is an `ErrorKind::Data` error naming the variable.

## Unconstrained parameter order

Parameters are read in declaration order. Within a parameter, array elements come in row-major
order, vectors are contiguous, and matrices are column-major. Each constrained type uses the
number of free values of its transform (for example `K - 1` for `simplex[K]` and
`K (K - 1) / 2` for `cholesky_factor_corr[K]`). This matches stanc3's deserializer.

## Output order

`write_array` flattens each variable in CmdStan's order: column-major over all dimensions (the
first index varies fastest), with a complex value written as its real then imaginary part.

## Errors

`rt::ModelError { kind, message }` with `kind`:

| Kind | Raised when |
| --- | --- |
| `Data` | Data missing, malformed, wrong size, or violating its declared constraint |
| `Domain` | A function argument is out of its domain, or a parameter block constraint fails |
| `Index` | An index is out of range |
| `Size` | Container sizes disagree |
| `Reject` | The program executed `reject` |
| `Fatal` | The program executed `fatal_error` |
| `Parameters` | The parameter vector has the wrong length |

Generated code does not panic on bad input.

## Random numbers

`_rng` functions draw from the `rand::Rng` passed to `write_array` (generated quantities) or from
a `StdRng` seeded with `seed` (transformed data). Streams are not identical to CmdStan's.
