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
