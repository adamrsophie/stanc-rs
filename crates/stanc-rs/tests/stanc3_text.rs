//! Output pinned to that of stanc 2.40.0 on the same program.

const SRC: &str = r#"functions {
  real f(real x);
  real f(real x) { return ((x)); }
}
transformed data {
  int j = 3 / 2;
  int c = 1 < 2 < 3;
  matrix[2, 2] A;
  matrix[2, 2] B = A ^ 2;
  real y;
  if (j > 0) y = (1 + 2) * 3; else if (j < 0) y = 2; else y = 3;
  while (j > 0) j -= 1;
}
"#;

#[test]
fn warnings_match_stanc3() {
    let c = stanc_rs::check_source("w.stan", SRC, &stanc_rs::Options::default()).unwrap();
    assert_eq!(
        c.warnings.concat(),
        r#"Warning in 'w.stan', line 6, column 10 to column 11:
    Found int division:
        3 / 2
    Values will be rounded towards zero. If rounding is not desired you can
    write the division as
        3.0 / 2
    If rounding is intended please use the integer division operator %/%.
Warning in 'w.stan', line 7, column 10 to column 19:
    Found chained comparison
        1 < 2 < 3
    This is interpreted as
        (1 < 2) < 3
    Consider if the intended meaning was
        1 < 2 && 2 < 3
    instead. You can silence this warning by adding explicit parentheses.
    This can be automatically changed using the canonicalize flag for stanc
Warning in 'w.stan', line 9, column 19 to column 20:
    Found matrix^scalar:
        A ^ 2
    matrix ^ number is interpreted as element-wise exponentiation. If this is
    intended, you can silence this warning by using elementwise operator .^
    If you intended matrix exponentiation, use
        matrix_power(A, 2)
    instead.
Warning in 'w.stan', line 2, column 7 to column 8:
    Functions do not need to be declared before definition; all user defined
    function names are always in scope regardless of definition order.
"#
    );
}

#[test]
fn print_canonical_matches_stanc3() {
    let (out, _) = stanc_rs::format_source(
        "w.stan",
        SRC,
        &stanc_rs::Options::default(),
        &stanc_rs::syntax::autoformat::Options::default(),
        &stanc_rs::syntax::canonicalize::Settings::LEGACY,
    )
    .unwrap();
    // stanc prints the program followed by a newline.
    assert_eq!(
        format!("{out}\n"),
        r#"functions {
  real f(real x) {
    return x;
  }
}
transformed data {
  int j = 3 / 2;
  int c = (1 < 2) < 3;
  matrix[2, 2] A;
  matrix[2, 2] B = A ^ 2;
  real y;
  if (j > 0) {
    y = (1 + 2) * 3;
  } else if (j < 0) {
    y = 2;
  } else {
    y = 3;
  }
  while (j > 0) {
    j -= 1;
  }
}

"#
    );
}

const PEDANTIC: &str = r#"data {
  int N;
  vector[N] y;
}
parameters {
  real mu;
  real sigma;
  real<lower=0, upper=5> tau;
  real unused;
}
model {
  real k = 2;
  if (mu > 0) k = 3;
  y ~ normal(mu, sigma);
  mu ~ normal(0, 100);
  tau ~ gamma(0.5, 0.5);
  exp(tau) ~ normal(0, k);
}
"#;

#[test]
fn pedantic_warnings_match_stanc3() {
    let opts = stanc_rs::Options {
        warn_pedantic: true,
        ..Default::default()
    };
    let c = stanc_rs::check_source("p.stan", PEDANTIC, &opts).unwrap();
    assert_eq!(
        c.warnings.concat(),
        r#"Warning in 'p.stan', line 6, column 2 to column 10:
    The parameter mu has 2 priors.
Warning in 'p.stan', line 7, column 2 to column 13:
    The parameter sigma has no priors. This means either no prior is
    provided, or the prior(s) depend on data variables. In the later case,
    this may be a false positive.
Warning in 'p.stan', line 8, column 2 to column 29:
    Your Stan program has a parameter tau with a lower and upper bound in its
    declaration. These hard constraints are not recommended, for two reasons:
    (a) Except when there are logical or physical constraints, it is very
    unusual for you to be sure that a parameter will fall inside a specified
    range, and (b) The infinite gradient induced by a hard constraint can
    cause difficulties for Stan's sampling algorithm. As a consequence, we
    recommend soft constraints rather than hard constraints; for example,
    instead of constraining an elasticity parameter to fall between 0, and 1,
    leave it unconstrained and give it a normal(0.5,0.5) prior distribution.
Warning in 'p.stan', line 9, column 2 to column 14:
    The parameter unused was declared but was not used in the density
    calculation.
Warning in 'p.stan', line 13, column 2 to column 20:
    A control flow statement depends on parameter(s): mu.
Warning in 'p.stan', line 14, column 17 to column 22:
    A normal distribution is given parameter sigma as a scale parameter
    (argument 2), but sigma was not constrained to be strictly positive.
Warning in 'p.stan', line 15, column 17 to column 20:
    Argument 100 suggests there may be parameters that are not unit scale;
    consider rescaling with a multiplier, see:
    https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html#standardizing-predictors
Warning in 'p.stan', line 16, column 14 to column 17:
    There is a gamma or inverse-gamma distribution with parameters that are
    equal to each other and set to values less than 1. This is mathematically
    acceptable and can make sense in some problems, but typically we see this
    model used as an attempt to assign a noninformative prior distribution.
    In fact, priors such as inverse-gamma(.001,.001) can be very strong, as
    explained by Gelman (2006). Instead we recommend something like a
    normal(0,1) or student_t(4,0,1), with parameter constrained to be
    positive.
Warning in 'p.stan', line 17, column 2 to column 26:
    Left-hand side of distribution statement (~) may contain a non-linear
    transform of a parameter or local variable. If it does, you need to
    include a target += statement with the log absolute determinant of the
    Jacobian of the transform. You could also consider defining a transformed
    parameter and using jacobian += in the transformed parameters block.
"#
    );
}
