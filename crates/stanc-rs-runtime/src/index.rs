//! Stan's one-based indexing.
//!
//! Every Stan index goes through these functions, which check the range and convert to Rust's
//! zero-based positions. Generated code never adjusts index arithmetic itself.

use stanmath_rs::Matrix;

use crate::{ErrorKind, ModelError, Result};

/// One position of a multi-index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Idx {
    /// `:`
    All,
    /// A single integer.
    Single(i64),
    /// An array of integers.
    Multi(Vec<i64>),
    /// `a:`
    Upfrom(i64),
    /// `:b`
    Downfrom(i64),
    /// `a:b` (empty when `b < a`).
    Between(i64, i64),
}

fn out_of_range(i: i64, len: usize) -> ModelError {
    ModelError::new(
        ErrorKind::Index,
        format!("index {i} out of range; expecting index to be between 1 and {len}"),
    )
}

/// The zero-based position of one-based index `i` in a container of length `len`.
pub fn pos(i: i64, len: usize) -> Result<usize> {
    if i >= 1 && (i as u64) <= len as u64 { Ok((i - 1) as usize) } else { Err(out_of_range(i, len)) }
}

/// The zero-based positions selected by a multi-index.
pub fn positions(idx: &Idx, len: usize) -> Result<Vec<usize>> {
    let range = |a: i64, b: i64| -> Result<Vec<usize>> {
        if b < a {
            return Ok(vec![]);
        }
        let a0 = pos(a, len)?;
        let b0 = pos(b, len)?;
        Ok((a0..=b0).collect())
    };
    match idx {
        Idx::All => Ok((0..len).collect()),
        Idx::Single(i) => Ok(vec![pos(*i, len)?]),
        Idx::Multi(is) => is.iter().map(|i| pos(*i, len)).collect(),
        Idx::Upfrom(a) => {
            if len == 0 || *a > len as i64 {
                if *a >= 1 { Ok(vec![]) } else { Err(out_of_range(*a, len)) }
            } else {
                range(*a, len as i64)
            }
        }
        Idx::Downfrom(b) => {
            if *b < 1 {
                Ok(vec![])
            } else {
                range(1, *b)
            }
        }
        Idx::Between(a, b) => range(*a, *b),
    }
}

/// `v[i]` by reference.
pub fn at<T>(v: &[T], i: i64) -> Result<&T> {
    Ok(&v[pos(i, v.len())?])
}

/// `v[i]` for assignment.
pub fn at_mut<T>(v: &mut [T], i: i64) -> Result<&mut T> {
    let p = pos(i, v.len())?;
    Ok(&mut v[p])
}

/// The elements of `v` selected by a multi-index.
pub fn multi<T: Clone>(v: &[T], idx: &Idx) -> Result<Vec<T>> {
    Ok(positions(idx, v.len())?.into_iter().map(|p| v[p].clone()).collect())
}

/// `m[i, j]` by reference.
pub fn mat_at<T>(m: &Matrix<T>, i: i64, j: i64) -> Result<&T> {
    let r = pos(i, m.nrows())?;
    let c = pos(j, m.ncols())?;
    Ok(&m[(r, c)])
}

/// `m[i, j]` for assignment.
pub fn mat_at_mut<T>(m: &mut Matrix<T>, i: i64, j: i64) -> Result<&mut T> {
    let r = pos(i, m.nrows())?;
    let c = pos(j, m.ncols())?;
    Ok(&mut m[(r, c)])
}

/// Row `i` of a matrix.
pub fn row<T: Clone>(m: &Matrix<T>, i: i64) -> Result<Vec<T>> {
    let r = pos(i, m.nrows())?;
    Ok((0..m.ncols()).map(|c| m[(r, c)].clone()).collect())
}

/// Column `j` of a matrix.
pub fn col<T: Clone>(m: &Matrix<T>, j: i64) -> Result<Vec<T>> {
    let c = pos(j, m.ncols())?;
    Ok(m.column(c).to_vec())
}

/// The submatrix with the selected rows and columns.
pub fn submatrix<T: Clone>(m: &Matrix<T>, rows: &Idx, cols: &Idx) -> Result<Matrix<T>> {
    let rs = positions(rows, m.nrows())?;
    let cs = positions(cols, m.ncols())?;
    let mut data = Vec::with_capacity(rs.len() * cs.len());
    for &c in &cs {
        for &r in &rs {
            data.push(m[(r, c)].clone());
        }
    }
    Ok(Matrix::from_column_major(rs.len(), cs.len(), data))
}

/// Elements `rows` of column `j`.
pub fn col_multi<T: Clone>(m: &Matrix<T>, rows: &Idx, j: i64) -> Result<Vec<T>> {
    let c = pos(j, m.ncols())?;
    Ok(positions(rows, m.nrows())?.into_iter().map(|r| m[(r, c)].clone()).collect())
}

/// Elements `cols` of row `i`.
pub fn row_multi<T: Clone>(m: &Matrix<T>, i: i64, cols: &Idx) -> Result<Vec<T>> {
    let r = pos(i, m.nrows())?;
    Ok(positions(cols, m.ncols())?.into_iter().map(|c| m[(r, c)].clone()).collect())
}

fn size_mismatch(what: &str, want: usize, got: usize) -> ModelError {
    ModelError::new(ErrorKind::Size, format!("{what}: assigning {got} values to {want} positions"))
}

/// `v[idx] = values`.
pub fn assign_multi<T>(v: &mut [T], idx: &Idx, values: Vec<T>) -> Result<()> {
    let ps = positions(idx, v.len())?;
    if ps.len() != values.len() {
        return Err(size_mismatch("multi-index assignment", ps.len(), values.len()));
    }
    for (p, x) in ps.into_iter().zip(values) {
        v[p] = x;
    }
    Ok(())
}

/// `m[i] = row`.
pub fn assign_row<T>(m: &mut Matrix<T>, i: i64, values: Vec<T>) -> Result<()> {
    let r = pos(i, m.nrows())?;
    if values.len() != m.ncols() {
        return Err(size_mismatch("row assignment", m.ncols(), values.len()));
    }
    for (c, x) in values.into_iter().enumerate() {
        m[(r, c)] = x;
    }
    Ok(())
}

/// `m[rows, cols] = values`.
pub fn assign_submatrix<T>(m: &mut Matrix<T>, rows: &Idx, cols: &Idx, values: Matrix<T>) -> Result<()> {
    let rs = positions(rows, m.nrows())?;
    let cs = positions(cols, m.ncols())?;
    if values.nrows() != rs.len() || values.ncols() != cs.len() {
        return Err(size_mismatch("matrix assignment", rs.len() * cs.len(), values.len()));
    }
    let data = values.into_vec();
    let mut it = data.into_iter();
    for &c in &cs {
        for &r in &rs {
            m[(r, c)] = it.next().expect("sizes checked");
        }
    }
    Ok(())
}

/// `m[rows, j] = values`.
pub fn assign_col_multi<T>(m: &mut Matrix<T>, rows: &Idx, j: i64, values: Vec<T>) -> Result<()> {
    let c = pos(j, m.ncols())?;
    let rs = positions(rows, m.nrows())?;
    if rs.len() != values.len() {
        return Err(size_mismatch("column assignment", rs.len(), values.len()));
    }
    for (r, x) in rs.into_iter().zip(values) {
        m[(r, c)] = x;
    }
    Ok(())
}

/// `m[i, cols] = values`.
pub fn assign_row_multi<T>(m: &mut Matrix<T>, i: i64, cols: &Idx, values: Vec<T>) -> Result<()> {
    let r = pos(i, m.nrows())?;
    let cs = positions(cols, m.ncols())?;
    if cs.len() != values.len() {
        return Err(size_mismatch("row assignment", cs.len(), values.len()));
    }
    for (c, x) in cs.into_iter().zip(values) {
        m[(r, c)] = x;
    }
    Ok(())
}

/// Rows selected by a multi-index, as a matrix.
pub fn rows_multi<T: Clone>(m: &Matrix<T>, rows: &Idx) -> Result<Matrix<T>> {
    submatrix(m, rows, &Idx::All)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_based() {
        let v = vec![10, 20, 30];
        assert_eq!(*at(&v, 1).unwrap(), 10);
        assert_eq!(*at(&v, 3).unwrap(), 30);
        assert!(at(&v, 0).is_err());
        assert!(at(&v, 4).is_err());
        assert_eq!(multi(&v, &Idx::Between(2, 3)).unwrap(), vec![20, 30]);
        assert_eq!(multi(&v, &Idx::Between(3, 2)).unwrap(), Vec::<i32>::new());
        assert_eq!(multi(&v, &Idx::Upfrom(2)).unwrap(), vec![20, 30]);
        assert_eq!(multi(&v, &Idx::Downfrom(2)).unwrap(), vec![10, 20]);
        assert_eq!(multi(&v, &Idx::Multi(vec![3, 1, 3])).unwrap(), vec![30, 10, 30]);
        assert!(multi(&v, &Idx::Multi(vec![4])).is_err());
        let m = Matrix::from_column_major(2, 2, vec![1, 2, 3, 4]);
        assert_eq!(*mat_at(&m, 2, 1).unwrap(), 2);
        assert_eq!(row(&m, 1).unwrap(), vec![1, 3]);
        assert_eq!(col(&m, 2).unwrap(), vec![3, 4]);
    }
}
