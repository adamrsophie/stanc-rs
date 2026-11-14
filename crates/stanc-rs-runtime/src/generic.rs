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

/// The absolute value (modulus) of a complex number.
pub fn cabs<T: Real>(z: &Complex<T>) -> T {
    z.re.hypot(z.im)
}

/// The sum of the elements, starting from `zero`.
pub fn sum_from<T: Clone + std::ops::Add<Output = T>>(v: &[T], zero: T) -> T {
    v.iter().cloned().fold(zero, |a, b| a + b)
}

/// The product of the elements, starting from `one`.
pub fn prod_from<T: Clone + std::ops::Mul<Output = T>>(v: &[T], one: T) -> T {
    v.iter().cloned().fold(one, |a, b| a * b)
}

/// The matrix with `n` columns equal to `v` (Stan's `rep_matrix(vector, n)`).
pub fn rep_cols<T: Clone>(v: &[T], n: i64) -> Result<Matrix<T>> {
    let n = usize_of(n)?;
    Ok(Matrix::from_fn(v.len(), n, |i, _| v[i].clone()))
}

/// The matrix with `m` rows equal to `v` (Stan's `rep_matrix(row_vector, m)`).
pub fn rep_rows<T: Clone>(v: &[T], m: i64) -> Result<Matrix<T>> {
    let m = usize_of(m)?;
    Ok(Matrix::from_fn(m, v.len(), |_, j| v[j].clone()))
}

/// `B * inverse(A)` for complex `B` (with `rows` rows, stored column major in `b`) and square
/// `A`, by Gaussian elimination with partial pivoting on `A` transposed.
fn cdivide_right<T: Real>(
    b: &[Complex<T>],
    rows: usize,
    a: &Matrix<Complex<T>>,
) -> Result<Vec<Complex<T>>> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != rows * n {
        return Err(ModelError::new(
            ErrorKind::Size,
            "mdivide_right: the matrix must be square with as many columns as the left side",
        ));
    }
    // Solve A^T X^T = B^T: the augmented rows are the rows of A^T.
    let mut m: Vec<Vec<Complex<T>>> = (0..n)
        .map(|i| {
            let mut row: Vec<Complex<T>> = (0..n).map(|j| a[(j, i)]).collect();
            row.extend((0..rows).map(|r| b[r + i * rows]));
            row
        })
        .collect();
    for k in 0..n {
        let p = (k..n)
            .max_by(|x, y| by(&cabs(&m[*x][k]).value(), &cabs(&m[*y][k]).value()))
            .unwrap_or(k);
        if cabs(&m[p][k]).value() == 0.0 {
            return Err(ModelError::new(
                ErrorKind::Domain,
                "mdivide_right: the matrix is singular",
            ));
        }
        m.swap(k, p);
        let (top, below) = m.split_at_mut(k + 1);
        let pivot = &top[k];
        for row in below {
            let f = row[k] / pivot[k];
            for (x, p) in row[k..].iter_mut().zip(&pivot[k..]) {
                *x -= f * *p;
            }
        }
    }
    let mut x = vec![vec![Complex::new(T::from_f64(0.0), T::from_f64(0.0)); rows]; n];
    for i in (0..n).rev() {
        for r in 0..rows {
            let mut acc = m[i][n + r];
            for j in i + 1..n {
                acc -= m[i][j] * x[j][r];
            }
            x[i][r] = acc / m[i][i];
        }
    }
    // X^T has rows `rows` and columns `n`; return it column major.
    Ok((0..n)
        .flat_map(|j| (0..rows).map(move |r| (j, r)))
        .map(|(j, r)| x[j][r])
        .collect())
}

/// `b * inverse(a)` for a complex row vector `b`.
pub fn cmdivide_right_vec<T: Real>(
    b: &[Complex<T>],
    a: &Matrix<Complex<T>>,
) -> Result<Vec<Complex<T>>> {
    cdivide_right(b, 1, a)
}

/// `b * inverse(a)` for a complex matrix `b`.
pub fn cmdivide_right_mat<T: Real>(
    b: &Matrix<Complex<T>>,
    a: &Matrix<Complex<T>>,
) -> Result<Matrix<Complex<T>>> {
    let v = cdivide_right(b.as_slice(), b.nrows(), a)?;
    Ok(Matrix::from_column_major(b.nrows(), a.ncols(), v))
}

/// A row vector as a one-row matrix.
pub fn as_row<T: Clone>(v: &[T]) -> Matrix<T> {
    Matrix::from_column_major(1, v.len(), v.to_vec())
}

/// A vector as a one-column matrix.
pub fn as_col<T: Clone>(v: &[T]) -> Matrix<T> {
    Matrix::from_column_major(v.len(), 1, v.to_vec())
}

/// `a` on top of `b`.
pub fn stack_rows<T: Clone>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    if a.ncols() != b.ncols() {
        return Err(ModelError::new(
            ErrorKind::Size,
            format!(
                "append_row: {} columns and {} columns",
                a.ncols(),
                b.ncols()
            ),
        ));
    }
    let r = a.nrows();
    Ok(Matrix::from_fn(r + b.nrows(), a.ncols(), |i, j| {
        if i < r {
            a[(i, j)].clone()
        } else {
            b[(i - r, j)].clone()
        }
    }))
}

/// `a` left of `b`.
pub fn stack_cols<T: Clone>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    if a.nrows() != b.nrows() {
        return Err(ModelError::new(
            ErrorKind::Size,
            format!("append_col: {} rows and {} rows", a.nrows(), b.nrows()),
        ));
    }
    let mut v = a.as_slice().to_vec();
    v.extend_from_slice(b.as_slice());
    Ok(Matrix::from_column_major(
        a.nrows(),
        a.ncols() + b.ncols(),
        v,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_divide_right_inverts_multiplication() {
        let c = |re: f64, im: f64| Complex::new(re, im);
        let a = Matrix::from_column_major(
            2,
            2,
            vec![c(2.0, 1.0), c(0.5, 0.0), c(-1.0, 0.3), c(1.0, -2.0)],
        );
        let x = vec![c(0.7, -0.1), c(1.2, 0.4)];
        // b = x * a
        let b: Vec<Complex<f64>> = (0..2)
            .map(|j| x[0] * a[(0, j)] + x[1] * a[(1, j)])
            .collect();
        let got = cmdivide_right_vec(&b, &a).unwrap();
        for (g, w) in got.iter().zip(&x) {
            assert!((g.re - w.re).abs() < 1e-14 && (g.im - w.im).abs() < 1e-14);
        }
        let bm = Matrix::from_column_major(1, 2, b);
        let gm = cmdivide_right_mat(&bm, &a).unwrap();
        assert!((gm[(0, 1)].re - x[1].re).abs() < 1e-14);
    }
}
