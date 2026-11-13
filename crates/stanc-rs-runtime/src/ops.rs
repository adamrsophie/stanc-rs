//! Stan's arithmetic operators on vectors and matrices.
//!
//! Generated code lifts both operands to one scalar type before calling these, so every function
//! is homogeneous in its scalar `S` (a real scalar or a complex number). Vectors and row vectors
//! are both `Vec<S>`; the code generator picks the function from the static Stan types.

use std::ops::{Add, Div, Mul, Neg, Sub};

use stanmath_rs::Matrix;

use crate::{ErrorKind, ModelError, Result};

/// Scalars the operators work on.
pub trait Num:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
}

impl<S> Num for S where
    S: Copy
        + Add<Output = S>
        + Sub<Output = S>
        + Mul<Output = S>
        + Div<Output = S>
        + Neg<Output = S>
{
}

fn mismatch(op: &str, a: String, b: String) -> ModelError {
    ModelError::new(
        ErrorKind::Size,
        format!("{op}: operands have sizes {a} and {b}"),
    )
}

fn zip_vec<S: Num>(op: &str, a: &[S], b: &[S], f: impl Fn(S, S) -> S) -> Result<Vec<S>> {
    if a.len() != b.len() {
        return Err(mismatch(op, a.len().to_string(), b.len().to_string()));
    }
    Ok(a.iter().zip(b).map(|(x, y)| f(*x, *y)).collect())
}

fn zip_mat<S: Num>(
    op: &str,
    a: &Matrix<S>,
    b: &Matrix<S>,
    f: impl Fn(S, S) -> S,
) -> Result<Matrix<S>> {
    if a.shape() != b.shape() {
        return Err(mismatch(
            op,
            format!("{:?}", a.shape()),
            format!("{:?}", b.shape()),
        ));
    }
    let data = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| f(*x, *y))
        .collect();
    Ok(Matrix::from_column_major(a.nrows(), a.ncols(), data))
}

fn map_mat<S: Num>(a: &Matrix<S>, f: impl Fn(S) -> S) -> Matrix<S> {
    Matrix::from_column_major(
        a.nrows(),
        a.ncols(),
        a.as_slice().iter().map(|x| f(*x)).collect(),
    )
}

macro_rules! elementwise {
    ($vv:ident, $mm:ident, $sv:ident, $vs:ident, $sm:ident, $ms:ident, $op:tt, $name:literal) => {
        #[doc = concat!("`vector ", $name, " vector`")]
        pub fn $vv<S: Num>(a: &[S], b: &[S]) -> Result<Vec<S>> {
            zip_vec($name, a, b, |x, y| x $op y)
        }
        #[doc = concat!("`matrix ", $name, " matrix`")]
        pub fn $mm<S: Num>(a: &Matrix<S>, b: &Matrix<S>) -> Result<Matrix<S>> {
            zip_mat($name, a, b, |x, y| x $op y)
        }
        #[doc = concat!("`real ", $name, " vector`")]
        pub fn $sv<S: Num>(a: S, b: &[S]) -> Vec<S> {
            b.iter().map(|y| a $op *y).collect()
        }
        #[doc = concat!("`vector ", $name, " real`")]
        pub fn $vs<S: Num>(a: &[S], b: S) -> Vec<S> {
            a.iter().map(|x| *x $op b).collect()
        }
        #[doc = concat!("`real ", $name, " matrix`")]
        pub fn $sm<S: Num>(a: S, b: &Matrix<S>) -> Matrix<S> {
            map_mat(b, |y| a $op y)
        }
        #[doc = concat!("`matrix ", $name, " real`")]
        pub fn $ms<S: Num>(a: &Matrix<S>, b: S) -> Matrix<S> {
            map_mat(a, |x| x $op b)
        }
    };
}

elementwise!(add_vv, add_mm, add_sv, add_vs, add_sm, add_ms, +, "+");
elementwise!(sub_vv, sub_mm, sub_sv, sub_vs, sub_sm, sub_ms, -, "-");
elementwise!(emul_vv, emul_mm, mul_sv, mul_vs, mul_sm, mul_ms, *, ".*");
elementwise!(ediv_vv, ediv_mm, ediv_sv, div_vs, ediv_sm, div_ms, /, "./");

/// `-vector`
pub fn neg_v<S: Num>(a: &[S]) -> Vec<S> {
    a.iter().map(|x| -*x).collect()
}

/// `-matrix`
pub fn neg_m<S: Num>(a: &Matrix<S>) -> Matrix<S> {
    map_mat(a, |x| -x)
}

/// `matrix'`
pub fn transpose<S: Clone>(a: &Matrix<S>) -> Matrix<S> {
    a.transpose()
}

fn dot<S: Num>(a: impl Iterator<Item = S>, b: impl Iterator<Item = S>) -> Option<S> {
    a.zip(b).map(|(x, y)| x * y).reduce(|s, t| s + t)
}

/// `matrix * matrix`
pub fn mul_mm<S: Num>(a: &Matrix<S>, b: &Matrix<S>, zero: S) -> Result<Matrix<S>> {
    if a.ncols() != b.nrows() {
        return Err(mismatch(
            "matrix multiply",
            format!("{:?}", a.shape()),
            format!("{:?}", b.shape()),
        ));
    }
    let (n, m) = (a.nrows(), b.ncols());
    let mut out = Vec::with_capacity(n * m);
    for j in 0..m {
        for i in 0..n {
            out.push(
                dot(
                    (0..a.ncols()).map(|k| a[(i, k)]),
                    b.column(j).iter().copied(),
                )
                .unwrap_or(zero),
            );
        }
    }
    Ok(Matrix::from_column_major(n, m, out))
}

/// `matrix * vector`
pub fn mul_mv<S: Num>(a: &Matrix<S>, v: &[S], zero: S) -> Result<Vec<S>> {
    if a.ncols() != v.len() {
        return Err(mismatch(
            "matrix multiply",
            format!("{:?}", a.shape()),
            v.len().to_string(),
        ));
    }
    Ok((0..a.nrows())
        .map(|i| dot((0..a.ncols()).map(|k| a[(i, k)]), v.iter().copied()).unwrap_or(zero))
        .collect())
}

/// `row_vector * matrix`
pub fn mul_rm<S: Num>(v: &[S], a: &Matrix<S>, zero: S) -> Result<Vec<S>> {
    if a.nrows() != v.len() {
        return Err(mismatch(
            "matrix multiply",
            v.len().to_string(),
            format!("{:?}", a.shape()),
        ));
    }
    Ok((0..a.ncols())
        .map(|j| dot(v.iter().copied(), a.column(j).iter().copied()).unwrap_or(zero))
        .collect())
}

/// `row_vector * vector`
pub fn mul_rv<S: Num>(a: &[S], b: &[S], zero: S) -> Result<S> {
    if a.len() != b.len() {
        return Err(mismatch(
            "multiply",
            a.len().to_string(),
            b.len().to_string(),
        ));
    }
    Ok(dot(a.iter().copied(), b.iter().copied()).unwrap_or(zero))
}

/// `vector * row_vector` (outer product)
pub fn mul_vr<S: Num>(a: &[S], b: &[S]) -> Matrix<S> {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for y in b {
        for x in a {
            out.push(*x * *y);
        }
    }
    Matrix::from_column_major(a.len(), b.len(), out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn products() {
        let a = Matrix::from_column_major(2, 2, vec![1.0, 3.0, 2.0, 4.0]);
        assert_eq!(mul_mv(&a, &[1.0, 1.0], 0.0).unwrap(), vec![3.0, 7.0]);
        assert_eq!(mul_rm(&[1.0, 1.0], &a, 0.0).unwrap(), vec![4.0, 6.0]);
        let p = mul_mm(&a, &a, 0.0).unwrap();
        assert_eq!(p.as_slice(), &[7.0, 15.0, 10.0, 22.0]);
        assert_eq!(add_vs(&[1.0, 2.0], 1.0), vec![2.0, 3.0]);
        assert!(add_vv(&[1.0], &[1.0, 2.0]).is_err());
    }
}
