//! Validation of declared constraints on data, transformed data, transformed parameters, and
//! generated quantities, and the [`Values`] trait that flattens any model value to `f64`s.

use stanmath_rs::Matrix;
use stanmath_rs::prelude::{Complex, Dual, Real, Var};

use crate::{ModelError, Result};

/// Flattens a value to `f64`s: arrays element by element (row-major), matrices in column-major
/// order, complex numbers as `re, im`.
pub trait Values {
    /// Appends the values of `self` to `out`.
    fn values(&self, out: &mut Vec<f64>);

    /// The values of `self`.
    fn to_values(&self) -> Vec<f64> {
        let mut v = Vec::new();
        self.values(&mut v);
        v
    }
}

impl Values for f64 {
    fn values(&self, out: &mut Vec<f64>) {
        out.push(*self);
    }
}

impl Values for i64 {
    fn values(&self, out: &mut Vec<f64>) {
        out.push(*self as f64);
    }
}

impl Values for Var<'_> {
    fn values(&self, out: &mut Vec<f64>) {
        out.push(self.value());
    }
}

impl<T: Real> Values for Dual<T> {
    fn values(&self, out: &mut Vec<f64>) {
        out.push(self.value());
    }
}

impl<T: Values> Values for Complex<T> {
    fn values(&self, out: &mut Vec<f64>) {
        self.re.values(out);
        self.im.values(out);
    }
}

impl<T: Values> Values for Vec<T> {
    fn values(&self, out: &mut Vec<f64>) {
        for x in self {
            x.values(out);
        }
    }
}

impl<T: Values> Values for Matrix<T> {
    fn values(&self, out: &mut Vec<f64>) {
        for x in self.as_slice() {
            x.values(out);
        }
    }
}

macro_rules! tuple_values {
    ($($t:ident $i:tt),+) => {
        impl<$($t: Values),+> Values for ($($t,)+) {
            fn values(&self, out: &mut Vec<f64>) {
                $(self.$i.values(out);)+
            }
        }
    };
}
tuple_values!(A 0);
tuple_values!(A 0, B 1);
tuple_values!(A 0, B 1, C 2);
tuple_values!(A 0, B 1, C 2, D 3);
tuple_values!(A 0, B 1, C 2, D 3, E 4);
tuple_values!(A 0, B 1, C 2, D 3, E 4, F 5);
tuple_values!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
tuple_values!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);

/// A scalar type usable in generated code: a stanmath-rs [`Real`] that can also be flattened and
/// printed.
pub trait Scalar: Real + Values + crate::stan::StanDisplay {}

impl<T: Real + Values + crate::stan::StanDisplay> Scalar for T {}

fn bound_at(b: &[f64], i: usize, n: usize, name: &str) -> Result<f64> {
    match b.len() {
        1 => Ok(b[0]),
        m if m == n => Ok(b[i]),
        m => Err(ModelError::domain(format!(
            "{name}: bound has {m} elements but the variable has {n}"
        ))),
    }
}

fn violated(name: &str, i: usize, n: usize, x: f64, what: &str) -> ModelError {
    let at = if n > 1 {
        format!("[{}]", i + 1)
    } else {
        String::new()
    };
    ModelError::domain(format!("{name}{at} is {x}, but must be {what}"))
}

/// `x >= lb` elementwise (the bound may be a scalar or have one value per element).
pub fn lower(name: &str, x: &impl Values, lb: &impl Values) -> Result<()> {
    let (x, b) = (x.to_values(), lb.to_values());
    for (i, v) in x.iter().enumerate() {
        let l = bound_at(&b, i, x.len(), name)?;
        if !(*v >= l) {
            return Err(violated(
                name,
                i,
                x.len(),
                *v,
                &format!("greater than or equal to {l}"),
            ));
        }
    }
    Ok(())
}

/// `x <= ub` elementwise.
pub fn upper(name: &str, x: &impl Values, ub: &impl Values) -> Result<()> {
    let (x, b) = (x.to_values(), ub.to_values());
    for (i, v) in x.iter().enumerate() {
        let u = bound_at(&b, i, x.len(), name)?;
        if !(*v <= u) {
            return Err(violated(
                name,
                i,
                x.len(),
                *v,
                &format!("less than or equal to {u}"),
            ));
        }
    }
    Ok(())
}

const TOL: f64 = 1e-8;

/// A vector of non-negative values summing to one.
pub fn simplex(name: &str, x: &[f64]) -> Result<()> {
    if x.is_empty() {
        return Err(ModelError::domain(format!(
            "{name} is not a valid simplex: it has no elements"
        )));
    }
    let s: f64 = x.iter().sum();
    if !((s - 1.0).abs() <= TOL) {
        return Err(ModelError::domain(format!(
            "{name} is not a valid simplex: its sum is {s}"
        )));
    }
    if let Some((i, v)) = x.iter().enumerate().find(|(_, v)| !(**v >= 0.0)) {
        return Err(ModelError::domain(format!(
            "{name} is not a valid simplex: element {} is {v}",
            i + 1
        )));
    }
    Ok(())
}

/// A vector of unit Euclidean length.
pub fn unit_vector(name: &str, x: &[f64]) -> Result<()> {
    let ss: f64 = x.iter().map(|v| v * v).sum();
    if x.is_empty() || !((ss - 1.0).abs() <= TOL) {
        return Err(ModelError::domain(format!(
            "{name} is not a valid unit vector: its squared norm is {ss}"
        )));
    }
    Ok(())
}

/// A vector whose elements sum to zero.
pub fn sum_to_zero(name: &str, x: &[f64]) -> Result<()> {
    let s: f64 = x.iter().sum();
    if !(s.abs() <= TOL) {
        return Err(ModelError::domain(format!(
            "{name} does not sum to zero: its sum is {s}"
        )));
    }
    Ok(())
}

/// A strictly increasing vector.
pub fn ordered(name: &str, x: &[f64]) -> Result<()> {
    for i in 1..x.len() {
        if !(x[i] > x[i - 1]) {
            return Err(ModelError::domain(format!(
                "{name} is not a valid ordered vector: element {} is {} and element {} is {}",
                i,
                x[i - 1],
                i + 1,
                x[i]
            )));
        }
    }
    Ok(())
}

/// A strictly increasing vector of positive values.
pub fn positive_ordered(name: &str, x: &[f64]) -> Result<()> {
    if let Some(v) = x.first()
        && !(*v > 0.0)
    {
        return Err(ModelError::domain(format!(
            "{name} is not a valid positive_ordered vector: its first element is {v}"
        )));
    }
    ordered(name, x)
}

fn symmetric(name: &str, m: &Matrix<f64>) -> Result<()> {
    if !m.is_square() {
        return Err(ModelError::domain(format!("{name} is not square")));
    }
    let n = m.nrows();
    for i in 0..n {
        for j in 0..i {
            let (a, b) = (m[(i, j)], m[(j, i)]);
            if !((a - b).abs() <= TOL * a.abs().max(b.abs()).max(1.0)) {
                return Err(ModelError::domain(format!("{name} is not symmetric")));
            }
        }
    }
    Ok(())
}

fn positive_definite(name: &str, m: &Matrix<f64>) -> Result<()> {
    // Cholesky decomposition succeeds exactly for symmetric positive definite matrices.
    let n = m.nrows();
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut d = m[(j, j)];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if !(d > 0.0) {
            return Err(ModelError::domain(format!(
                "{name} is not positive definite"
            )));
        }
        let d = d.sqrt();
        l[j * n + j] = d;
        for i in j + 1..n {
            let mut s = m[(i, j)];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / d;
        }
    }
    Ok(())
}

/// A symmetric positive definite matrix.
pub fn cov_matrix(name: &str, m: &Matrix<f64>) -> Result<()> {
    symmetric(name, m)?;
    positive_definite(name, m)
}

/// A symmetric positive definite matrix with unit diagonal.
pub fn corr_matrix(name: &str, m: &Matrix<f64>) -> Result<()> {
    cov_matrix(name, m)?;
    for i in 0..m.nrows() {
        if !((m[(i, i)] - 1.0).abs() <= TOL) {
            return Err(ModelError::domain(format!(
                "{name} is not a valid correlation matrix: diagonal element {} is {}",
                i + 1,
                m[(i, i)]
            )));
        }
    }
    Ok(())
}

/// A lower triangular matrix with positive diagonal and at least as many rows as columns.
pub fn cholesky_factor_cov(name: &str, m: &Matrix<f64>) -> Result<()> {
    let (r, c) = m.shape();
    if r < c {
        return Err(ModelError::domain(format!(
            "{name} is not a valid Cholesky factor: it has fewer rows than columns"
        )));
    }
    for j in 0..c {
        for i in 0..j.min(r) {
            if m[(i, j)] != 0.0 {
                return Err(ModelError::domain(format!(
                    "{name} is not lower triangular"
                )));
            }
        }
        if !(m[(j, j)] > 0.0) {
            return Err(ModelError::domain(format!(
                "{name} is not a valid Cholesky factor: diagonal element {} is {}",
                j + 1,
                m[(j, j)]
            )));
        }
    }
    Ok(())
}

/// A Cholesky factor of a correlation matrix: lower triangular, positive diagonal, unit rows.
pub fn cholesky_factor_corr(name: &str, m: &Matrix<f64>) -> Result<()> {
    if !m.is_square() {
        return Err(ModelError::domain(format!("{name} is not square")));
    }
    cholesky_factor_cov(name, m)?;
    for i in 0..m.nrows() {
        let ss: f64 = (0..m.ncols()).map(|j| m[(i, j)] * m[(i, j)]).sum();
        if !((ss - 1.0).abs() <= TOL) {
            return Err(ModelError::domain(format!(
                "{name} is not a valid Cholesky factor of a correlation matrix: row {} has squared norm {ss}",
                i + 1
            )));
        }
    }
    Ok(())
}

/// Every column is a simplex.
pub fn stochastic_column(name: &str, m: &Matrix<f64>) -> Result<()> {
    for j in 0..m.ncols() {
        simplex(&format!("{name} column {}", j + 1), m.column(j))?;
    }
    Ok(())
}

/// Every row is a simplex.
pub fn stochastic_row(name: &str, m: &Matrix<f64>) -> Result<()> {
    stochastic_column(name, &m.transpose())
        .map_err(|e| ModelError::domain(e.message.replace(" column ", " row ")))
}

/// A matrix whose rows and columns all sum to zero.
pub fn sum_to_zero_matrix(name: &str, m: &Matrix<f64>) -> Result<()> {
    for j in 0..m.ncols() {
        sum_to_zero(&format!("{name} column {}", j + 1), m.column(j))?;
    }
    let t = m.transpose();
    for i in 0..t.ncols() {
        sum_to_zero(&format!("{name} row {}", i + 1), t.column(i))?;
    }
    Ok(())
}

/// Values of a vector-like value as `f64`.
pub fn vals<T: Values>(x: &[T]) -> Vec<f64> {
    let mut v = Vec::new();
    for e in x {
        e.values(&mut v);
    }
    v
}

/// Values of a matrix as an `f64` matrix.
pub fn mat_vals<T: Real>(m: &Matrix<T>) -> Matrix<f64> {
    m.map(|x| x.value())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks() {
        assert!(lower("x", &vec![1.0, 2.0], &0.0).is_ok());
        assert!(lower("x", &vec![1.0, -2.0], &0.0).is_err());
        assert!(lower("x", &f64::NAN, &0.0).is_err());
        assert!(upper("n", &3_i64, &3_i64).is_ok());
        assert!(simplex("s", &[0.25, 0.75]).is_ok());
        assert!(simplex("s", &[0.5, 0.75]).is_err());
        assert!(ordered("o", &[1.0, 2.0]).is_ok());
        assert!(ordered("o", &[2.0, 2.0]).is_err());
        let m = Matrix::from_column_major(2, 2, vec![2.0, 0.5, 0.5, 1.0]);
        assert!(cov_matrix("S", &m).is_ok());
        assert!(corr_matrix("S", &m).is_err());
    }
}
