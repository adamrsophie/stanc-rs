//! Stan operations used by generated code that stanmath-rs does not provide in the form the code
//! generator needs: elementwise container arithmetic, conversions, and printing.

use stanmath_rs::Matrix;
use stanmath_rs::prelude::{Complex, Dual, Real};

use crate::{ErrorKind, ModelError, Result};

/// Converts an `int` size to `usize`.
pub fn usize_of(n: i64) -> Result<usize> {
    usize::try_from(n)
        .map_err(|_| ModelError::new(ErrorKind::Size, format!("size {n} is negative")))
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

impl<T: Real> StanDisplay for Dual<T> {
    fn stan_fmt(&self, out: &mut String) {
        fmt_real(self.value(), out);
    }
}

impl<T: StanDisplay> StanDisplay for Complex<T> {
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

macro_rules! tuple_display {
    ($($t:ident $i:tt),+) => {
        impl<$($t: StanDisplay),+> StanDisplay for ($($t,)+) {
            fn stan_fmt(&self, out: &mut String) {
                out.push('(');
                $(
                    if $i > 0 {
                        out.push_str(", ");
                    }
                    self.$i.stan_fmt(out);
                )+
                out.push(')');
            }
        }
    };
}

tuple_display!(A 0);
tuple_display!(A 0, B 1);
tuple_display!(A 0, B 1, C 2);
tuple_display!(A 0, B 1, C 2, D 3);
tuple_display!(A 0, B 1, C 2, D 3, E 4);
tuple_display!(A 0, B 1, C 2, D 3, E 4, F 5);
tuple_display!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
tuple_display!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
tuple_display!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
tuple_display!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
tuple_display!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
tuple_display!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);

impl StanDisplay for &str {
    fn stan_fmt(&self, out: &mut String) {
        out.push_str(self);
    }
}

/// Writes a `print` line to the print sink of the current thread (see [`crate::print`]).
pub fn print(s: String) {
    crate::print::line(&s);
}

/// Lifts a data complex number.
pub fn lift_complex<T: Real>(x: Complex<f64>) -> Complex<T> {
    Complex::new(T::from_f64(x.re), T::from_f64(x.im))
}

/// Lifts a data complex vector.
pub fn lift_cvec<T: Real>(x: &[Complex<f64>]) -> Vec<Complex<T>> {
    x.iter().map(|v| lift_complex(*v)).collect()
}

/// Lifts a data complex matrix.
pub fn lift_cmat<T: Real>(x: &Matrix<Complex<f64>>) -> Matrix<Complex<T>> {
    x.map(|v| lift_complex(*v))
}

/// A real number as a complex number.
pub fn to_complex<T: Real>(x: T) -> Complex<T> {
    Complex::new(x, T::from_f64(0.0))
}

/// A real vector as a complex vector.
pub fn to_complex_vec<T: Real>(x: Vec<T>) -> Vec<Complex<T>> {
    x.into_iter().map(to_complex).collect()
}

/// A real matrix as a complex matrix.
pub fn to_complex_mat<T: Real>(x: Matrix<T>) -> Matrix<Complex<T>> {
    x.map(|v| to_complex(*v))
}

/// Maps a fallible function over a vector.
pub fn try_map_vec<A, B>(x: Vec<A>, f: impl FnMut(A) -> Result<B>) -> Result<Vec<B>> {
    x.into_iter().map(f).collect()
}

/// Maps a fallible function over a matrix.
pub fn try_map_mat<A, B>(x: Matrix<A>, f: impl FnMut(A) -> Result<B>) -> Result<Matrix<B>> {
    let (r, c) = (x.nrows(), x.ncols());
    Ok(Matrix::from_column_major(
        r,
        c,
        x.into_vec()
            .into_iter()
            .map(f)
            .collect::<Result<Vec<B>>>()?,
    ))
}

/// The common length of the vector arguments of a vectorized call.
pub fn common_len(function: &str, lens: &[usize]) -> Result<usize> {
    let n = lens.first().copied().unwrap_or(1);
    if lens.iter().any(|l| *l != n) {
        return Err(ModelError::new(
            ErrorKind::Size,
            format!("{function}: arguments have different sizes {lens:?}"),
        ));
    }
    Ok(n)
}

/// An `r` by `c` matrix with every element `x`.
pub fn mat_filled<S: Clone>(r: usize, c: usize, x: S) -> Matrix<S> {
    Matrix::from_column_major(r, c, vec![x; r * c])
}

/// A matrix from its rows (the literal `[[a, b], [c, d]]`).
pub fn matrix_from_rows<S: Clone>(rows: Vec<Vec<S>>) -> Result<Matrix<S>> {
    let r = rows.len();
    let c = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != c) {
        return Err(ModelError::new(
            ErrorKind::Size,
            "matrix literal rows have different lengths",
        ));
    }
    let mut data = Vec::with_capacity(r * c);
    for j in 0..c {
        for row in &rows {
            data.push(row[j].clone());
        }
    }
    Ok(Matrix::from_column_major(r, c, data))
}

/// Stan's `to_int`: truncates toward zero, rejecting values outside the 32 bit integer range.
pub fn to_int(x: f64) -> Result<i64> {
    if x.is_nan() || x < i32::MIN as f64 || x >= i32::MAX as f64 + 1.0 {
        return Err(ModelError::domain(format!(
            "to_int: argument is {x}, but must be in the 32 bit integer range"
        )));
    }
    Ok(x.trunc() as i64)
}

/// Converts library sizes to Stan integers.
pub fn usizes_to_ints(v: Vec<usize>) -> Vec<i64> {
    v.into_iter().map(|x| x as i64).collect()
}

/// Builds a vector of `n` elements, each computed by `f`.
pub fn build<A>(n: usize, mut f: impl FnMut() -> Result<A>) -> Result<Vec<A>> {
    (0..n).map(|_| f()).collect()
}

/// Builds a vector of `n` elements, computing each from its one-based index.
pub fn build_at<A>(n: usize, mut f: impl FnMut(i64) -> Result<A>) -> Result<Vec<A>> {
    (1..=n as i64).map(&mut f).collect()
}

/// Sum of an integer array.
pub fn sum_int(x: &[i64]) -> i64 {
    x.iter().sum()
}

/// Largest element of an integer array (an error when empty).
pub fn max_int(x: &[i64]) -> Result<i64> {
    x.iter()
        .copied()
        .max()
        .ok_or_else(|| ModelError::domain("max: array is empty"))
}

/// Smallest element of an integer array (an error when empty).
pub fn min_int(x: &[i64]) -> Result<i64> {
    x.iter()
        .copied()
        .min()
        .ok_or_else(|| ModelError::domain("min: array is empty"))
}

fn pick<A: Clone>(v: &[Vec<A>], j: usize) -> Vec<A> {
    v.get(j).cloned().unwrap_or_default()
}

/// Stan's `map_rect`: `f(phi, thetas[j], x_r[j], x_i[j])` for every job `j`, concatenated. The
/// number of jobs is the largest of the three array sizes; an empty array stands for an empty
/// argument to every job.
pub fn map_rect<S: Clone>(
    phi: Vec<S>,
    thetas: Vec<Vec<S>>,
    x_r: Vec<Vec<f64>>,
    x_i: Vec<Vec<i64>>,
    mut f: impl FnMut(Vec<S>, Vec<S>, Vec<f64>, Vec<i64>) -> Result<Vec<S>>,
) -> Result<Vec<S>> {
    let jobs = thetas.len().max(x_r.len()).max(x_i.len());
    for (what, n) in [
        ("job parameters", thetas.len()),
        ("real data", x_r.len()),
        ("integer data", x_i.len()),
    ] {
        if n != 0 && n != jobs {
            return Err(ModelError::new(
                crate::ErrorKind::Size,
                format!("map_rect: {what} has {n} jobs but another argument has {jobs}"),
            ));
        }
    }
    let mut out = Vec::new();
    for j in 0..jobs {
        out.extend(f(
            phi.clone(),
            pick(&thetas, j),
            pick(&x_r, j),
            pick(&x_i, j),
        )?);
    }
    Ok(out)
}

/// The next value of `theta` at cursor `k` (used to rebuild solver arguments).
pub fn next_theta<T: Real>(theta: &[T], k: &std::cell::Cell<usize>) -> T {
    let i = k.get();
    k.set(i + 1);
    theta
        .get(i)
        .copied()
        .unwrap_or_else(|| T::from_f64(f64::NAN))
}

/// A matrix shaped like `template` with values read from `theta` in column-major order.
pub fn unflatten_mat<T: Real>(
    template: &Matrix<f64>,
    theta: &[T],
    k: &std::cell::Cell<usize>,
) -> Matrix<T> {
    template.map(|_| next_theta(theta, k))
}

/// Element `[i][j]` (zero based) of a two-dimensional array walked along with a matrix.
pub fn at2<T: Clone>(v: &[Vec<T>], i: usize, j: usize) -> Result<T> {
    v.get(i).and_then(|row| row.get(j)).cloned().ok_or_else(|| {
        ModelError::new(
            ErrorKind::Size,
            format!("array has no element [{}][{}]", i + 1, j + 1),
        )
    })
}

/// Checks that every row of a two-dimensional array has `cols` elements.
pub fn check_rows<T>(function: &str, v: &[Vec<T>], cols: usize) -> Result<()> {
    if let Some(row) = v.iter().find(|row| row.len() != cols) {
        return Err(ModelError::new(
            ErrorKind::Size,
            format!(
                "{function}: array rows have {} elements but the matrix has {cols} columns",
                row.len()
            ),
        ));
    }
    Ok(())
}

/// Stan's compressed sparse row arguments `(m, n, w, v, u)`, with one-based `v` and `u`.
pub fn csr<T: Clone>(
    m: i64,
    n: i64,
    w: &[T],
    v: &[i64],
    u: &[i64],
) -> Result<stanmath_rs::linalg::CsrMatrix<T>> {
    let zero_based =
        |xs: &[i64]| -> Result<Vec<usize>> { xs.iter().map(|x| usize_of(*x - 1)).collect() };
    Ok(stanmath_rs::linalg::CsrMatrix {
        nrows: usize_of(m)?,
        ncols: usize_of(n)?,
        values: w.to_vec(),
        columns: zero_based(v)?,
        row_starts: zero_based(u)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_int_truncates_and_checks_range() {
        assert_eq!(to_int(2.9).unwrap(), 2);
        assert_eq!(to_int(-2.9).unwrap(), -2);
        assert!(to_int(f64::NAN).is_err());
        assert!(to_int(3e9).is_err());
    }

    #[test]
    fn tuples_print_in_parentheses() {
        let mut s = String::new();
        (1_i64, (vec![0.5_f64], 2_i64)).stan_fmt(&mut s);
        assert_eq!(s, "(1, ([0.5], 2))");
    }
}
