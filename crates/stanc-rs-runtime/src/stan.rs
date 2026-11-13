//! Stan operations used by generated code that stanmath-rs does not provide in the form the code
//! generator needs: elementwise container arithmetic, conversions, and printing.

use stanmath_rs::Matrix;
use stanmath_rs::prelude::Real;

use crate::{ErrorKind, ModelError, Result};

/// Converts an `int` size to `usize`.
pub fn usize_of(n: i64) -> Result<usize> {
    usize::try_from(n).map_err(|_| ModelError::new(ErrorKind::Size, format!("size {n} is negative")))
}

/// Lifts a data value to the autodiff scalar.
pub fn lift<T: Real>(x: f64) -> T {
    T::from_f64(x)
}

/// Lifts a data vector.
pub fn lift_vec<T: Real>(x: &[f64]) -> Vec<T> {
    x.iter().map(|v| T::from_f64(*v)).collect()
}

/// Lifts a data matrix.
pub fn lift_mat<T: Real>(x: &Matrix<f64>) -> Matrix<T> {
    x.map(|v| T::from_f64(*v))
}

/// Integer division with Stan's semantics (truncation toward zero, error on division by zero).
pub fn int_divide(a: i64, b: i64) -> Result<i64> {
    if b == 0 {
        return Err(ModelError::domain("integer division by zero"));
    }
    Ok(a.wrapping_div(b))
}

/// Integer modulus with Stan's semantics.
pub fn modulus(a: i64, b: i64) -> Result<i64> {
    if b == 0 {
        return Err(ModelError::domain("modulus by zero"));
    }
    Ok(a.wrapping_rem(b))
}

/// Formats a value for `print` and `reject`.
pub trait StanDisplay {
    /// Appends the Stan rendering of `self`.
    fn stan_fmt(&self, out: &mut String);
}

/// Formats a real number as Stan does.
fn fmt_real(x: f64, out: &mut String) {
    if x.is_nan() {
        out.push_str("nan");
    } else if x.is_infinite() {
        out.push_str(if x > 0.0 { "inf" } else { "-inf" });
    } else {
        out.push_str(&format!("{x}"));
    }
}

impl StanDisplay for i64 {
    fn stan_fmt(&self, out: &mut String) {
        out.push_str(&self.to_string());
    }
}

impl StanDisplay for f64 {
    fn stan_fmt(&self, out: &mut String) {
        fmt_real(*self, out);
    }
}

impl StanDisplay for stanmath_rs::prelude::Var<'_> {
    fn stan_fmt(&self, out: &mut String) {
        fmt_real(self.value(), out);
    }
}

impl<T: StanDisplay> StanDisplay for stanmath_rs::prelude::Complex<T> {
    fn stan_fmt(&self, out: &mut String) {
        out.push('(');
        self.re.stan_fmt(out);
        out.push(',');
        self.im.stan_fmt(out);
        out.push(')');
    }
}

impl<T: StanDisplay> StanDisplay for Vec<T> {
    fn stan_fmt(&self, out: &mut String) {
        out.push('[');
        for (i, x) in self.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            x.stan_fmt(out);
        }
        out.push(']');
    }
}

impl<T: StanDisplay + Clone> StanDisplay for Matrix<T> {
    fn stan_fmt(&self, out: &mut String) {
        out.push('[');
        for (i, row) in self.to_rows().iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            row.stan_fmt(out);
        }
        out.push(']');
    }
}

impl StanDisplay for &str {
    fn stan_fmt(&self, out: &mut String) {
        out.push_str(self);
    }
}

/// Writes a `print` line to standard output.
pub fn print(s: String) {
    println!("{s}");
}
