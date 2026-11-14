//! Structural Stan library functions for any element type: slicing, reversing, replicating,
//! sorting, and reshaping arrays, vectors, and matrices of integers, complex numbers, or nested
//! containers. Generated code uses these when stanmath-rs has no version for the element type.
//!
//! Positions and sizes are Stan's one-based `int`s and are checked.

use stanmath_rs::Matrix;
use stanmath_rs::prelude::{Complex, Real};

use crate::stan::usize_of;
use crate::{ErrorKind, ModelError, Result};

fn index_err(function: &str, msg: String) -> ModelError {
    ModelError::new(ErrorKind::Index, format!("{function}: {msg}"))
}

/// The first `n` elements.
pub fn head<T: Clone>(v: &[T], n: i64) -> Result<Vec<T>> {
    let n = usize_of(n)?;
    if n > v.len() {
        return Err(index_err(
            "head",
            format!("size {n} exceeds the length {}", v.len()),
        ));
    }
    Ok(v[..n].to_vec())
}

/// The last `n` elements.
pub fn tail<T: Clone>(v: &[T], n: i64) -> Result<Vec<T>> {
    let n = usize_of(n)?;
    if n > v.len() {
        return Err(index_err(
            "tail",
            format!("size {n} exceeds the length {}", v.len()),
        ));
    }
    Ok(v[v.len() - n..].to_vec())
}

/// `n` elements starting at position `i`.
pub fn segment<T: Clone>(v: &[T], i: i64, n: i64) -> Result<Vec<T>> {
    let n = usize_of(n)?;
    if i < 1 || (i as usize - 1) + n > v.len() {
        return Err(index_err(
            "segment",
            format!("start {i} and size {n} exceed the length {}", v.len()),
        ));
    }
    let s = i as usize - 1;
    Ok(v[s..s + n].to_vec())
}

/// The elements in reverse order.
pub fn reverse<T: Clone>(v: &[T]) -> Vec<T> {
    v.iter().rev().cloned().collect()
}

/// `a` followed by `b`.
pub fn append_array<T: Clone>(a: &[T], b: &[T]) -> Vec<T> {
    let mut v = a.to_vec();
    v.extend_from_slice(b);
    v
}

/// `n` copies of `x`.
pub fn rep_array<T: Clone>(x: T, n: i64) -> Result<Vec<T>> {
    Ok(vec![x; usize_of(n)?])
}

/// An `n` by `m` array of copies of `x`.
pub fn rep_array2<T: Clone>(x: T, n: i64, m: i64) -> Result<Vec<Vec<T>>> {
    rep_array(rep_array(x, m)?, n)
}

/// An `n` by `m` by `k` array of copies of `x`.
pub fn rep_array3<T: Clone>(x: T, n: i64, m: i64, k: i64) -> Result<Vec<Vec<Vec<T>>>> {
    rep_array(rep_array2(x, m, k)?, n)
}

fn by<T: PartialOrd>(a: &T, b: &T) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
}

/// Sorted in ascending order.
pub fn sort_asc<T: Clone + PartialOrd>(v: &[T]) -> Vec<T> {
    let mut v = v.to_vec();
    v.sort_by(by);
    v
}

/// Sorted in descending order.
pub fn sort_desc<T: Clone + PartialOrd>(v: &[T]) -> Vec<T> {
    let mut v = v.to_vec();
    v.sort_by(|a, b| by(b, a));
    v
}

/// One-based positions that sort `v` in ascending order (stable).
pub fn sort_indices_asc<T: PartialOrd>(v: &[T]) -> Vec<i64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|a, b| by(&v[*a], &v[*b]));
    idx.into_iter().map(|i| i as i64 + 1).collect()
}

/// One-based positions that sort `v` in descending order (stable).
pub fn sort_indices_desc<T: PartialOrd>(v: &[T]) -> Vec<i64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|a, b| by(&v[*b], &v[*a]));
    idx.into_iter().map(|i| i as i64 + 1).collect()
}

/// The number of elements of `v` less than `v[s]`.
pub fn rank<T: PartialOrd>(v: &[T], s: i64) -> Result<i64> {
    if s < 1 || s as usize > v.len() {
        return Err(index_err(
            "rank",
            format!("position {s} is outside 1..{}", v.len()),
        ));
    }
    let x = &v[s as usize - 1];
    Ok(v.iter().filter(|e| *e < x).count() as i64)
}

/// A vector of `n` copies of `x`.
pub fn rep_vector<T: Clone>(x: T, n: i64) -> Result<Vec<T>> {
    rep_array(x, n)
}

/// An `r` by `c` matrix of copies of `x`.
pub fn rep_matrix<T: Clone>(x: T, r: i64, c: i64) -> Result<Matrix<T>> {
    Ok(crate::stan::mat_filled(usize_of(r)?, usize_of(c)?, x))
}

fn check_row<T>(function: &str, m: &Matrix<T>, i: i64) -> Result<usize> {
    if i < 1 || i as usize > m.nrows() {
        return Err(index_err(
            function,
            format!("row {i} is outside 1..{}", m.nrows()),
        ));
    }
    Ok(i as usize - 1)
}

fn check_col<T>(function: &str, m: &Matrix<T>, j: i64) -> Result<usize> {
    if j < 1 || j as usize > m.ncols() {
        return Err(index_err(
            function,
            format!("column {j} is outside 1..{}", m.ncols()),
        ));
    }
    Ok(j as usize - 1)
}

/// Row `i`.
pub fn row<T: Clone>(m: &Matrix<T>, i: i64) -> Result<Vec<T>> {
    let i = check_row("row", m, i)?;
    Ok((0..m.ncols()).map(|j| m[(i, j)].clone()).collect())
}

/// Column `j`.
pub fn col<T: Clone>(m: &Matrix<T>, j: i64) -> Result<Vec<T>> {
    let j = check_col("col", m, j)?;
    Ok((0..m.nrows()).map(|i| m[(i, j)].clone()).collect())
}

/// The `nr` by `nc` block starting at row `i`, column `j`.
pub fn block<T: Clone>(m: &Matrix<T>, i: i64, j: i64, nr: i64, nc: i64) -> Result<Matrix<T>> {
    let (nr, nc) = (usize_of(nr)?, usize_of(nc)?);
    let i0 = check_row("block", m, i)?;
    let j0 = check_col("block", m, j)?;
    if i0 + nr > m.nrows() || j0 + nc > m.ncols() {
        return Err(index_err(
            "block",
            "the block extends past the matrix".into(),
        ));
    }
    Ok(Matrix::from_fn(nr, nc, |r, c| m[(i0 + r, j0 + c)].clone()))
}

/// `n` elements of row `i` starting at column `j`.
pub fn sub_row<T: Clone>(m: &Matrix<T>, i: i64, j: i64, n: i64) -> Result<Vec<T>> {
    segment(&row(m, i)?, j, n)
}

/// `n` elements of column `j` starting at row `i`.
pub fn sub_col<T: Clone>(m: &Matrix<T>, i: i64, j: i64, n: i64) -> Result<Vec<T>> {
    segment(&col(m, j)?, i, n)
}

/// The diagonal.
pub fn diagonal<T: Clone>(m: &Matrix<T>) -> Vec<T> {
    (0..m.nrows().min(m.ncols()))
        .map(|i| m[(i, i)].clone())
        .collect()
}

/// The elements in column-major order.
pub fn to_vector<T: Clone>(m: &Matrix<T>) -> Vec<T> {
    m.as_slice().to_vec()
}

/// The square matrix with `d` on the diagonal and `zero` elsewhere.
pub fn diag_matrix<T: Clone>(d: &[T], zero: T) -> Matrix<T> {
    let n = d.len();
    let mut m = crate::stan::mat_filled(n, n, zero);
    for (i, x) in d.iter().enumerate() {
        m[(i, i)] = x.clone();
    }
    m
}

/// The symmetric matrix whose lower triangle is that of `m`.
pub fn symmetrize_from_lower_tri<T: Clone>(m: &Matrix<T>) -> Result<Matrix<T>> {
    if m.nrows() != m.ncols() {
        return Err(ModelError::new(
            ErrorKind::Size,
            "symmetrize_from_lower_tri: the matrix is not square",
        ));
    }
    let mut out = m.clone();
    for j in 0..m.ncols() {
        for i in 0..j {
            out[(i, j)] = m[(j, i)].clone();
        }
    }
    Ok(out)
}

/// Real parts.
pub fn get_real<T: Real>(z: &Complex<T>) -> T {
    z.re
}

/// Imaginary parts.
pub fn get_imag<T: Real>(z: &Complex<T>) -> T {
    z.im
}

/// The complex conjugate.
pub fn conj<T: Real>(z: &Complex<T>) -> Complex<T> {
    Complex::new(z.re, -z.im)
}
