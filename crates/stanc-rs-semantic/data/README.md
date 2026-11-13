# Stan Math signature table

`stan_math_signatures_2.40.0.txt` lists every Stan Math library signature known to stanc3 v2.40.0,
one per line, as `name(argument types) => return type`.

It was produced in two steps:

1. `stanc --dump-stan-math-signatures` from the stanc3 v2.40.0 release (BSD-3-Clause, Stan
   Development Team).
2. That dump omits the `data` qualifier on top-level arguments. The script
   `tools/add_signature_data_qualifiers.py` reads the `add_qualified` entries of stanc3's
   `src/stan_math_signatures/Generate.ml` (plus the `to_int` and Wiener loops there) and restores the
   qualifiers:

   ```sh
   python3 tools/add_signature_data_qualifiers.py stanc3/src/stan_math_signatures/Generate.ml \
       dump.txt > stan_math_signatures_2.40.0.txt
   ```

Variadic higher-order functions (ODE and DAE solvers, `solve_newton`, `solve_powell`,
`integrate_1d_*`, `reduce_sum`) and the embedded Laplace functions are not in the dump; they are
described in `src/signatures.rs`.
