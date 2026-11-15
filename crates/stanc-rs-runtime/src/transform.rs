//! Stan's constraint functions on values: `<transform>_constrain`, `<transform>_jacobian`, and
//! `<transform>_unconstrain`.
//!
//! Constraining takes `lp`: `None` for `_constrain`, and the Jacobian accumulator for
//! `_jacobian`, which adds the log absolute Jacobian determinant. The transforms are those used
//! for parameters, so `x_unconstrain(x_constrain(y)) == y`.

use stanmath_rs::Matrix;
use stanmath_rs::autodiff::Lift;
use stanmath_rs::prelude::*;

use crate::io::{self, Bound, Reader};
use crate::{ErrorKind, ModelError, Result};

/// Runs `f` with the accumulator, or with a scratch one when there is none.
fn with_lp<T: Real, R>(lp: Option<&mut T>, f: impl FnOnce(&mut T, bool) -> R) -> R {
    match lp {
        Some(lp) => f(lp, true),
        None => f(&mut T::from_f64(0.0), false),
    }
}

/// Reads exactly all of `y` with `f`.
fn read_all<T: Real, R>(
    name: &str,
    y: &[T],
    f: impl FnOnce(&mut Reader<T>) -> Result<R>,
) -> Result<R> {
    let mut r = Reader::new(y);
    let out = f(&mut r)?;
    if r.position() != y.len() {
        return Err(ModelError::new(
            ErrorKind::Size,
            format!(
                "{name}: {} unconstrained values were given but {} are needed",
                y.len(),
                r.position()
            ),
        ));
    }
    Ok(out)
}

/// Lower bound, elementwise with `Bound`.
pub fn lower_bound<T: Real, L: Lift<T> + Copy>(
    y: Vec<T>,
    b: Bound<L>,
    lp: Option<&mut T>,
) -> Result<Vec<T>> {
    with_lp(lp, |lp, j| io::lb_each(y, b, lp, j))
}

/// Upper bound, elementwise.
pub fn upper_bound<T: Real, U: Lift<T> + Copy>(
    y: Vec<T>,
    b: Bound<U>,
    lp: Option<&mut T>,
) -> Result<Vec<T>> {
    with_lp(lp, |lp, j| io::ub_each(y, b, lp, j))
}

/// Lower and upper bounds, elementwise.
pub fn lower_upper_bound<T: Real, L: Lift<T> + Copy, U: Lift<T> + Copy>(
    y: Vec<T>,
    l: Bound<L>,
    u: Bound<U>,
    lp: Option<&mut T>,
) -> Result<Vec<T>> {
    with_lp(lp, |lp, j| io::lub_each(y, l, u, lp, j))
}

/// Offset and multiplier, elementwise.
pub fn offset_multiplier<T: Real, M: Lift<T> + Copy, S: Lift<T> + Copy>(
    y: Vec<T>,
    m: Bound<M>,
    s: Bound<S>,
    lp: Option<&mut T>,
) -> Result<Vec<T>> {
    with_lp(lp, |lp, j| io::offset_multiplier_each(y, m, s, lp, j))
}

macro_rules! vector_transform {
    ($(#[$doc:meta])* $name:ident, $extra:expr) => {
        $(#[$doc])*
        pub fn $name<T: Real>(y: &[T], lp: Option<&mut T>) -> Result<Vec<T>> {
            let k = y.len() + $extra;
            with_lp(lp, |lp, j| read_all(stringify!($name), y, |r| io::$name(r, k, lp, j)))
        }
    };
}

vector_transform! {
    /// `ordered`.
    ordered, 0
}
vector_transform! {
    /// `positive_ordered`.
    positive_ordered, 0
}
vector_transform! {
    /// `simplex`: `K - 1` values to a `K`-simplex.
    simplex, 1
}
vector_transform! {
    /// `unit_vector`.
    unit_vector, 0
}

/// `sum_to_zero_vector`: `K - 1` values to `K`.
pub fn sum_to_zero<T: Real>(y: &[T], _lp: Option<&mut T>) -> Result<Vec<T>> {
    read_all("sum_to_zero", y, |r| io::sum_to_zero(r, y.len() + 1))
}

/// `sum_to_zero_matrix`: an `N - 1` by `M - 1` matrix to `N` by `M`.
pub fn sum_to_zero_matrix<T: Real>(y: &Matrix<T>, _lp: Option<&mut T>) -> Result<Matrix<T>> {
    let (n, m) = (y.nrows() + 1, y.ncols() + 1);
    read_all("sum_to_zero_matrix", y.as_slice(), |r| {
        io::sum_to_zero_matrix(r, n, m)
    })
}

/// `cholesky_factor_corr[K]` from `K (K - 1) / 2` values.
pub fn cholesky_factor_corr<T: Real>(y: &[T], k: i64, lp: Option<&mut T>) -> Result<Matrix<T>> {
    let k = crate::stan::usize_of(k)?;
    with_lp(lp, |lp, j| {
        read_all("cholesky_factor_corr", y, |r| {
            io::cholesky_factor_corr(r, k, lp, j)
        })
    })
}

/// `corr_matrix[K]` from `K (K - 1) / 2` values.
pub fn corr_matrix<T: Real>(y: &[T], k: i64, lp: Option<&mut T>) -> Result<Matrix<T>> {
    let k = crate::stan::usize_of(k)?;
    with_lp(lp, |lp, j| {
        read_all("corr_matrix", y, |r| io::corr_matrix(r, k, lp, j))
    })
}

/// `cov_matrix[K]` from `K + K (K - 1) / 2` values.
pub fn cov_matrix<T: Real>(y: &[T], k: i64, lp: Option<&mut T>) -> Result<Matrix<T>> {
    let k = crate::stan::usize_of(k)?;
    with_lp(lp, |lp, j| {
        read_all("cov_matrix", y, |r| io::cov_matrix(r, k, lp, j))
    })
}

/// `cholesky_factor_cov[M, N]`.
pub fn cholesky_factor_cov<T: Real>(
    y: &[T],
    m: i64,
    n: i64,
    lp: Option<&mut T>,
) -> Result<Matrix<T>> {
    let (m, n) = (crate::stan::usize_of(m)?, crate::stan::usize_of(n)?);
    with_lp(lp, |lp, j| {
        read_all("cholesky_factor_cov", y, |r| {
            io::cholesky_factor_cov(r, m, n, lp, j)
        })
    })
}

/// `column_stochastic_matrix`: an `N - 1` by `M` matrix to `N` by `M`.
pub fn stochastic_column<T: Real>(y: &Matrix<T>, lp: Option<&mut T>) -> Result<Matrix<T>> {
    let (n, m) = (y.nrows() + 1, y.ncols());
    with_lp(lp, |lp, j| {
        read_all("stochastic_column", y.as_slice(), |r| {
            io::stochastic_column(r, n, m, lp, j)
        })
    })
}

/// `row_stochastic_matrix`: an `N` by `M - 1` matrix to `N` by `M`.
pub fn stochastic_row<T: Real>(y: &Matrix<T>, lp: Option<&mut T>) -> Result<Matrix<T>> {
    let (n, m) = (y.nrows(), y.ncols() + 1);
    with_lp(lp, |lp, j| {
        read_all("stochastic_row", y.as_slice(), |r| {
            io::stochastic_row(r, n, m, lp, j)
        })
    })
}

/// Inverse transforms on values that may carry derivatives.
pub mod free {
    use super::*;

    fn each<T: Real, L: Copy>(
        xs: Vec<T>,
        b: Bound<L>,
        f: impl Fn(T, L) -> stanmath_rs::prelude::Result<T>,
    ) -> Result<Vec<T>> {
        let n = xs.len();
        xs.into_iter()
            .enumerate()
            .map(|(i, x)| Ok(f(x, bound_at(&b, i, n)?)?))
            .collect()
    }

    fn bound_at<L: Copy>(b: &Bound<L>, i: usize, n: usize) -> Result<L> {
        match b {
            Bound::Scalar(l) => Ok(*l),
            Bound::Each(v) if v.len() == n => Ok(v[i]),
            Bound::Each(v) => Err(ModelError::new(
                ErrorKind::Size,
                format!("bound has {} elements but the value has {n}", v.len()),
            )),
        }
    }

    /// Inverse of the lower bound.
    pub fn lower_bound<T: Real, L: Lift<T> + Copy>(x: Vec<T>, b: Bound<L>) -> Result<Vec<T>> {
        each(x, b, lb_free)
    }

    /// Inverse of the upper bound.
    pub fn upper_bound<T: Real, U: Lift<T> + Copy>(x: Vec<T>, b: Bound<U>) -> Result<Vec<T>> {
        each(x, b, ub_free)
    }

    /// Inverse of lower and upper bounds.
    pub fn lower_upper_bound<T: Real, L: Lift<T> + Copy, U: Lift<T> + Copy>(
        x: Vec<T>,
        l: Bound<L>,
        u: Bound<U>,
    ) -> Result<Vec<T>> {
        let n = x.len();
        x.into_iter()
            .enumerate()
            .map(|(i, v)| Ok(lub_free(v, bound_at(&l, i, n)?, bound_at(&u, i, n)?)?))
            .collect()
    }

    /// Inverse of offset and multiplier.
    pub fn offset_multiplier<T: Real, M: Lift<T> + Copy, S: Lift<T> + Copy>(
        x: Vec<T>,
        m: Bound<M>,
        s: Bound<S>,
    ) -> Result<Vec<T>> {
        let n = x.len();
        x.into_iter()
            .enumerate()
            .map(|(i, v)| {
                Ok(offset_multiplier_free(
                    v,
                    bound_at(&m, i, n)?,
                    bound_at(&s, i, n)?,
                )?)
            })
            .collect()
    }

    /// Inverse of `ordered`.
    pub fn ordered<T: Real>(x: &[T]) -> Result<Vec<T>> {
        Ok(ordered_free(x)?)
    }

    /// Inverse of `positive_ordered`.
    pub fn positive_ordered<T: Real>(x: &[T]) -> Result<Vec<T>> {
        Ok(positive_ordered_free(x)?)
    }

    /// Inverse of `simplex`.
    pub fn simplex<T: Real>(x: &[T]) -> Result<Vec<T>> {
        Ok(simplex_free(x)?)
    }

    /// Inverse of `unit_vector`.
    pub fn unit_vector<T: Real>(x: &[T]) -> Result<Vec<T>> {
        Ok(unit_vector_free(x)?)
    }

    /// Inverse of `sum_to_zero_vector`.
    pub fn sum_to_zero<T: Real>(x: &[T]) -> Result<Vec<T>> {
        Ok(sum_to_zero_free(x)?)
    }

    /// Inverse of `sum_to_zero_matrix`.
    pub fn sum_to_zero_matrix<T: Real>(x: &Matrix<T>) -> Result<Matrix<T>> {
        Ok(sum_to_zero_matrix_free(x)?)
    }

    /// Inverse of `cholesky_factor_corr`.
    pub fn cholesky_factor_corr<T: Real>(x: &Matrix<T>) -> Result<Vec<T>> {
        Ok(cholesky_factor_corr_free(x)?)
    }

    /// Inverse of `cholesky_factor_cov`.
    pub fn cholesky_factor_cov<T: Real>(x: &Matrix<T>) -> Result<Vec<T>> {
        Ok(cholesky_factor_cov_free(x)?)
    }

    /// Inverse of `corr_matrix`.
    pub fn corr_matrix<T: Real>(x: &Matrix<T>) -> Result<Vec<T>> {
        Ok(corr_matrix_free(x)?)
    }

    /// Inverse of `cov_matrix`.
    pub fn cov_matrix<T: Real>(x: &Matrix<T>) -> Result<Vec<T>> {
        Ok(cov_matrix_free(x)?)
    }

    /// Inverse of `column_stochastic_matrix`.
    pub fn stochastic_column<T: Real>(x: &Matrix<T>) -> Result<Matrix<T>> {
        Ok(stochastic_column_free(x)?)
    }

    /// Inverse of `row_stochastic_matrix`.
    pub fn stochastic_row<T: Real>(x: &Matrix<T>) -> Result<Matrix<T>> {
        Ok(stochastic_row_free(x)?)
    }
}

/// Applies a transform of a flat vector to one value.
pub fn on_scalar<T>(x: T, f: impl FnOnce(Vec<T>) -> Result<Vec<T>>) -> Result<T> {
    f(vec![x])?
        .pop()
        .ok_or_else(|| ModelError::new(ErrorKind::Size, "transform returned no value"))
}

/// Element `i` of a container argument that is mapped along with the value.
pub fn at<A>(v: &[A], i: usize) -> Result<&A> {
    v.get(i).ok_or_else(|| {
        ModelError::new(
            ErrorKind::Size,
            format!("argument has {} elements but the value has more", v.len()),
        )
    })
}
