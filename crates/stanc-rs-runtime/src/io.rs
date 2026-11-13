//! Reading unconstrained parameters and applying the constraining transforms, and the inverse
//! (freeing) transforms used to unconstrain initial values.
//!
//! The order of unconstrained values follows stanc3's deserializer: array elements in row-major
//! order, each vector contiguous, each matrix in column-major order, and each constrained type
//! taking the number of unconstrained values its transform needs (for example `K - 1` for a
//! `simplex[K]`).

use stanmath_rs::Matrix;
use stanmath_rs::autodiff::Lift;
use stanmath_rs::prelude::*;

use crate::{ErrorKind, ModelError, Result};

/// Sequential reader of an unconstrained parameter vector.
pub struct Reader<'a, T> {
    theta: &'a [T],
    pos: usize,
}

impl<'a, T: Real> Reader<'a, T> {
    /// Starts reading at the beginning of `theta`.
    pub fn new(theta: &'a [T]) -> Self {
        Reader { theta, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [T]> {
        let end = self.pos + n;
        if end > self.theta.len() {
            return Err(ModelError::new(
                ErrorKind::Parameters,
                format!(
                    "parameter vector too short: needed {end} values, found {}",
                    self.theta.len()
                ),
            ));
        }
        let s = &self.theta[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    /// One value.
    pub fn scalar(&mut self) -> Result<T> {
        Ok(self.take(1)?[0])
    }

    /// `n` values.
    pub fn vec(&mut self, n: usize) -> Result<Vec<T>> {
        Ok(self.take(n)?.to_vec())
    }

    /// An `r` by `c` matrix in column-major order.
    pub fn matrix(&mut self, r: usize, c: usize) -> Result<Matrix<T>> {
        Ok(Matrix::from_column_major(r, c, self.vec(r * c)?))
    }

    /// A complex number.
    pub fn complex(&mut self) -> Result<Complex<T>> {
        let s = self.take(2)?;
        Ok(Complex::new(s[0], s[1]))
    }

    /// Number of values read so far.
    pub fn position(&self) -> usize {
        self.pos
    }
}

/// A bound: one value for every element, or one per element.
#[derive(Clone, Copy, Debug)]
pub enum Bound<'a, L> {
    /// The same bound for every element.
    Scalar(L),
    /// Element `i` has bound `v[i]` (matrices in column-major order).
    Each(&'a [L]),
}

impl<L: Copy> Bound<'_, L> {
    fn at(&self, i: usize, n: usize) -> Result<L> {
        match self {
            Bound::Scalar(l) => Ok(*l),
            Bound::Each(v) => {
                if v.len() != n {
                    return Err(ModelError::new(
                        ErrorKind::Size,
                        format!("bound has {} elements but the variable has {n}", v.len()),
                    ));
                }
                Ok(v[i])
            }
        }
    }
}

/// Lower bound for one value.
pub fn lb<T: Real, L: Lift<T>>(y: T, lb: L, lp: &mut T, jacobian: bool) -> T {
    if jacobian {
        lb_constrain_lp(y, lb, lp)
    } else {
        lb_constrain(y, lb)
    }
}

/// Upper bound for one value.
pub fn ub<T: Real, U: Lift<T>>(y: T, ub: U, lp: &mut T, jacobian: bool) -> T {
    if jacobian {
        ub_constrain_lp(y, ub, lp)
    } else {
        ub_constrain(y, ub)
    }
}

/// Lower and upper bounds for one value.
pub fn lub<T: Real, L: Lift<T>, U: Lift<T>>(
    y: T,
    lb: L,
    ub: U,
    lp: &mut T,
    jacobian: bool,
) -> Result<T> {
    Ok(if jacobian {
        lub_constrain_lp(y, lb, ub, lp)?
    } else {
        lub_constrain(y, lb, ub)?
    })
}

/// Offset and multiplier for one value.
pub fn offset_multiplier<T: Real, M: Lift<T>, S: Lift<T>>(
    y: T,
    mu: M,
    sigma: S,
    lp: &mut T,
    jacobian: bool,
) -> Result<T> {
    Ok(if jacobian {
        offset_multiplier_constrain_lp(y, mu, sigma, lp)?
    } else {
        offset_multiplier_constrain(y, mu, sigma)?
    })
}

/// Lower bound for every element.
pub fn lb_each<T: Real, L: Lift<T> + Copy>(
    ys: Vec<T>,
    b: Bound<L>,
    lp: &mut T,
    jacobian: bool,
) -> Result<Vec<T>> {
    let n = ys.len();
    ys.into_iter()
        .enumerate()
        .map(|(i, y)| Ok(lb(y, b.at(i, n)?, lp, jacobian)))
        .collect()
}

/// Upper bound for every element.
pub fn ub_each<T: Real, U: Lift<T> + Copy>(
    ys: Vec<T>,
    b: Bound<U>,
    lp: &mut T,
    jacobian: bool,
) -> Result<Vec<T>> {
    let n = ys.len();
    ys.into_iter()
        .enumerate()
        .map(|(i, y)| Ok(ub(y, b.at(i, n)?, lp, jacobian)))
        .collect()
}

/// Lower and upper bounds for every element.
pub fn lub_each<T: Real, L: Lift<T> + Copy, U: Lift<T> + Copy>(
    ys: Vec<T>,
    l: Bound<L>,
    u: Bound<U>,
    lp: &mut T,
    jacobian: bool,
) -> Result<Vec<T>> {
    let n = ys.len();
    ys.into_iter()
        .enumerate()
        .map(|(i, y)| lub(y, l.at(i, n)?, u.at(i, n)?, lp, jacobian))
        .collect()
}

/// Offset and multiplier for every element.
pub fn offset_multiplier_each<T: Real, M: Lift<T> + Copy, S: Lift<T> + Copy>(
    ys: Vec<T>,
    mu: Bound<M>,
    sigma: Bound<S>,
    lp: &mut T,
    jacobian: bool,
) -> Result<Vec<T>> {
    let n = ys.len();
    ys.into_iter()
        .enumerate()
        .map(|(i, y)| offset_multiplier(y, mu.at(i, n)?, sigma.at(i, n)?, lp, jacobian))
        .collect()
}

/// Applies an elementwise vector transform to a matrix.
pub fn on_matrix<T: Clone>(
    m: Matrix<T>,
    f: impl FnOnce(Vec<T>) -> Result<Vec<T>>,
) -> Result<Matrix<T>> {
    let (r, c) = m.shape();
    Ok(Matrix::from_column_major(r, c, f(m.into_vec())?))
}

fn positive_size(k: usize, what: &str) -> Result<()> {
    if k == 0 {
        return Err(ModelError::domain(format!(
            "{what} must have at least one element"
        )));
    }
    Ok(())
}

/// `ordered[k]`.
pub fn ordered<T: Real>(r: &mut Reader<T>, k: usize, lp: &mut T, jacobian: bool) -> Result<Vec<T>> {
    let y = r.vec(k)?;
    Ok(if jacobian {
        ordered_constrain_lp(&y, lp)
    } else {
        ordered_constrain(&y)
    })
}

/// `positive_ordered[k]`.
pub fn positive_ordered<T: Real>(
    r: &mut Reader<T>,
    k: usize,
    lp: &mut T,
    jacobian: bool,
) -> Result<Vec<T>> {
    let y = r.vec(k)?;
    Ok(if jacobian {
        positive_ordered_constrain_lp(&y, lp)
    } else {
        positive_ordered_constrain(&y)
    })
}

/// `simplex[k]` from `k - 1` values.
pub fn simplex<T: Real>(r: &mut Reader<T>, k: usize, lp: &mut T, jacobian: bool) -> Result<Vec<T>> {
    positive_size(k, "simplex")?;
    let y = r.vec(k - 1)?;
    Ok(if jacobian {
        simplex_constrain_lp(&y, lp)
    } else {
        simplex_constrain(&y)
    })
}

/// `unit_vector[k]`.
pub fn unit_vector<T: Real>(
    r: &mut Reader<T>,
    k: usize,
    lp: &mut T,
    jacobian: bool,
) -> Result<Vec<T>> {
    positive_size(k, "unit_vector")?;
    let y = r.vec(k)?;
    Ok(if jacobian {
        unit_vector_constrain_lp(&y, lp)?
    } else {
        unit_vector_constrain(&y)?
    })
}

/// `sum_to_zero_vector[k]` from `k - 1` values (no Jacobian term).
pub fn sum_to_zero<T: Real>(r: &mut Reader<T>, k: usize) -> Result<Vec<T>> {
    positive_size(k, "sum_to_zero_vector")?;
    let y = r.vec(k - 1)?;
    Ok(sum_to_zero_constrain(&y))
}

/// `sum_to_zero_matrix[n, m]` from an `n - 1` by `m - 1` matrix (no Jacobian term).
pub fn sum_to_zero_matrix<T: Real>(r: &mut Reader<T>, n: usize, m: usize) -> Result<Matrix<T>> {
    positive_size(n, "sum_to_zero_matrix")?;
    positive_size(m, "sum_to_zero_matrix")?;
    let y = r.matrix(n - 1, m - 1)?;
    Ok(sum_to_zero_matrix_constrain(&y))
}

/// `cholesky_factor_corr[k]` from `k (k - 1) / 2` values.
pub fn cholesky_factor_corr<T: Real>(
    r: &mut Reader<T>,
    k: usize,
    lp: &mut T,
    jacobian: bool,
) -> Result<Matrix<T>> {
    let y = r.vec(k * k.saturating_sub(1) / 2)?;
    Ok(if jacobian {
        cholesky_factor_corr_constrain_lp(&y, k, lp)?
    } else {
        cholesky_factor_corr_constrain(&y, k)?
    })
}

/// `cholesky_factor_cov[m, n]` from `n (n + 1) / 2 + (m - n) n` values.
pub fn cholesky_factor_cov<T: Real>(
    r: &mut Reader<T>,
    m: usize,
    n: usize,
    lp: &mut T,
    jacobian: bool,
) -> Result<Matrix<T>> {
    if m < n {
        return Err(ModelError::domain(format!(
            "cholesky_factor_cov[{m}, {n}] needs at least as many rows as columns"
        )));
    }
    let y = r.vec(n * (n + 1) / 2 + (m - n) * n)?;
    Ok(if jacobian {
        cholesky_factor_cov_constrain_lp(&y, m, n, lp)?
    } else {
        cholesky_factor_cov_constrain(&y, m, n)?
    })
}

/// `corr_matrix[k]` from `k (k - 1) / 2` values.
pub fn corr_matrix<T: Real>(
    r: &mut Reader<T>,
    k: usize,
    lp: &mut T,
    jacobian: bool,
) -> Result<Matrix<T>> {
    let y = r.vec(k * k.saturating_sub(1) / 2)?;
    Ok(if jacobian {
        corr_matrix_constrain_lp(&y, k, lp)?
    } else {
        corr_matrix_constrain(&y, k)?
    })
}

/// `cov_matrix[k]` from `k + k (k - 1) / 2` values.
pub fn cov_matrix<T: Real>(
    r: &mut Reader<T>,
    k: usize,
    lp: &mut T,
    jacobian: bool,
) -> Result<Matrix<T>> {
    let y = r.vec(k + k * k.saturating_sub(1) / 2)?;
    Ok(if jacobian {
        cov_matrix_constrain_lp(&y, k, lp)?
    } else {
        cov_matrix_constrain(&y, k)?
    })
}

/// `column_stochastic_matrix[n, m]` from an `n - 1` by `m` matrix.
pub fn stochastic_column<T: Real>(
    r: &mut Reader<T>,
    n: usize,
    m: usize,
    lp: &mut T,
    jacobian: bool,
) -> Result<Matrix<T>> {
    positive_size(n, "column_stochastic_matrix")?;
    let y = r.matrix(n - 1, m)?;
    Ok(if jacobian {
        stochastic_column_constrain_lp(&y, lp)
    } else {
        stochastic_column_constrain(&y)
    })
}

/// `row_stochastic_matrix[n, m]` from an `n` by `m - 1` matrix.
pub fn stochastic_row<T: Real>(
    r: &mut Reader<T>,
    n: usize,
    m: usize,
    lp: &mut T,
    jacobian: bool,
) -> Result<Matrix<T>> {
    positive_size(m, "row_stochastic_matrix")?;
    let y = r.matrix(n, m - 1)?;
    Ok(if jacobian {
        stochastic_row_constrain_lp(&y, lp)
    } else {
        stochastic_row_constrain(&y)
    })
}

/// Collects unconstrained values while freeing constrained ones.
#[derive(Default)]
pub struct Writer {
    out: Vec<f64>,
}

impl Writer {
    /// An empty writer.
    pub fn new() -> Self {
        Writer::default()
    }

    /// Appends one value.
    pub fn push(&mut self, x: f64) {
        self.out.push(x);
    }

    /// Appends values.
    pub fn extend(&mut self, xs: impl IntoIterator<Item = f64>) {
        self.out.extend(xs);
    }

    /// The unconstrained vector.
    pub fn finish(self) -> Vec<f64> {
        self.out
    }
}

/// Inverse transforms, applied to constrained `f64` values read from JSON.
pub mod free {
    use super::*;

    /// Inverse of the lower bound.
    pub fn lb(x: f64, l: f64) -> Result<f64> {
        Ok(lb_free(x, l)?)
    }

    /// Inverse of the upper bound.
    pub fn ub(x: f64, u: f64) -> Result<f64> {
        Ok(ub_free(x, u)?)
    }

    /// Inverse of the interval bound.
    pub fn lub(x: f64, l: f64, u: f64) -> Result<f64> {
        Ok(lub_free(x, l, u)?)
    }

    /// Inverse of offset and multiplier.
    pub fn offset_multiplier(x: f64, mu: f64, sigma: f64) -> Result<f64> {
        Ok(offset_multiplier_free(x, mu, sigma)?)
    }

    fn each(
        xs: &[f64],
        n: usize,
        b: Bound<f64>,
        f: impl Fn(f64, f64) -> Result<f64>,
    ) -> Result<Vec<f64>> {
        xs.iter()
            .enumerate()
            .map(|(i, x)| f(*x, b.at(i, n)?))
            .collect()
    }

    /// Elementwise inverse of the lower bound.
    pub fn lb_each(xs: &[f64], b: Bound<f64>) -> Result<Vec<f64>> {
        each(xs, xs.len(), b, lb)
    }

    /// Elementwise inverse of the upper bound.
    pub fn ub_each(xs: &[f64], b: Bound<f64>) -> Result<Vec<f64>> {
        each(xs, xs.len(), b, ub)
    }

    /// Elementwise inverse of the interval bound.
    pub fn lub_each(xs: &[f64], l: Bound<f64>, u: Bound<f64>) -> Result<Vec<f64>> {
        let n = xs.len();
        xs.iter()
            .enumerate()
            .map(|(i, x)| lub(*x, l.at(i, n)?, u.at(i, n)?))
            .collect()
    }

    /// Elementwise inverse of offset and multiplier.
    pub fn offset_multiplier_each(xs: &[f64], m: Bound<f64>, s: Bound<f64>) -> Result<Vec<f64>> {
        let n = xs.len();
        xs.iter()
            .enumerate()
            .map(|(i, x)| offset_multiplier(*x, m.at(i, n)?, s.at(i, n)?))
            .collect()
    }

    /// Inverse of `ordered`.
    pub fn ordered(x: &[f64]) -> Result<Vec<f64>> {
        Ok(ordered_free(x)?)
    }

    /// Inverse of `positive_ordered`.
    pub fn positive_ordered(x: &[f64]) -> Result<Vec<f64>> {
        Ok(positive_ordered_free(x)?)
    }

    /// Inverse of `simplex`.
    pub fn simplex(x: &[f64]) -> Result<Vec<f64>> {
        Ok(simplex_free(x)?)
    }

    /// Inverse of `unit_vector`.
    pub fn unit_vector(x: &[f64]) -> Result<Vec<f64>> {
        Ok(unit_vector_free(x)?)
    }

    /// Inverse of `sum_to_zero_vector`.
    pub fn sum_to_zero(x: &[f64]) -> Result<Vec<f64>> {
        Ok(sum_to_zero_free(x)?)
    }

    /// Inverse of `sum_to_zero_matrix`, in column-major order.
    pub fn sum_to_zero_matrix(x: &Matrix<f64>) -> Result<Vec<f64>> {
        Ok(sum_to_zero_matrix_free(x)?.into_vec())
    }

    /// Inverse of `cholesky_factor_corr`.
    pub fn cholesky_factor_corr(x: &Matrix<f64>) -> Result<Vec<f64>> {
        Ok(cholesky_factor_corr_free(x)?)
    }

    /// Inverse of `cholesky_factor_cov`.
    pub fn cholesky_factor_cov(x: &Matrix<f64>) -> Result<Vec<f64>> {
        Ok(cholesky_factor_cov_free(x)?)
    }

    /// Inverse of `corr_matrix`.
    pub fn corr_matrix(x: &Matrix<f64>) -> Result<Vec<f64>> {
        Ok(corr_matrix_free(x)?)
    }

    /// Inverse of `cov_matrix`.
    pub fn cov_matrix(x: &Matrix<f64>) -> Result<Vec<f64>> {
        Ok(cov_matrix_free(x)?)
    }

    /// Inverse of `column_stochastic_matrix`, in column-major order.
    pub fn stochastic_column(x: &Matrix<f64>) -> Result<Vec<f64>> {
        Ok(stochastic_column_free(x)?.into_vec())
    }

    /// Inverse of `row_stochastic_matrix`, in column-major order.
    pub fn stochastic_row(x: &Matrix<f64>) -> Result<Vec<f64>> {
        Ok(stochastic_row_free(x)?.into_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_bound_jacobian() {
        let theta = [0.3_f64];
        let mut r = Reader::new(&theta);
        let mut lp = 0.0;
        let s = lb(r.scalar().unwrap(), 0.0, &mut lp, true);
        assert!((s - 0.3_f64.exp()).abs() < 1e-15);
        assert!((lp - 0.3).abs() < 1e-15);
        let mut lp2 = 0.0;
        let _ = lb(0.3, 0.0, &mut lp2, false);
        assert_eq!(lp2, 0.0);
        assert!((free::lb(s, 0.0).unwrap() - 0.3).abs() < 1e-15);
    }

    #[test]
    fn simplex_sizes() {
        let theta = [0.1, -0.2];
        let mut r = Reader::new(&theta);
        let mut lp = 0.0;
        let s = simplex(&mut r, 3, &mut lp, true).unwrap();
        assert_eq!(s.len(), 3);
        assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        assert_eq!(r.position(), 2);
    }
}
