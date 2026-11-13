//! Flattened output names and the column-major output order of CmdStan.

use crate::{VarBlock, VarInfo};

fn included(v: &VarInfo, include_tp: bool, include_gq: bool) -> bool {
    match v.block {
        VarBlock::Parameters => true,
        VarBlock::TransformedParameters => include_tp,
        VarBlock::GeneratedQuantities => include_gq,
    }
}

/// Every index tuple of `dims` (one-based) with the first index varying fastest.
fn column_major_indices(dims: &[usize]) -> Vec<Vec<usize>> {
    let n: usize = dims.iter().product();
    let mut out = Vec::with_capacity(n);
    if dims.contains(&0) {
        return out;
    }
    let mut idx = vec![1; dims.len()];
    for _ in 0..n {
        out.push(idx.clone());
        for (k, d) in dims.iter().enumerate() {
            if idx[k] < *d {
                idx[k] += 1;
                break;
            }
            idx[k] = 1;
        }
    }
    out
}

/// Flattened names (`beta.1.2`, `z.real`) of the included variables.
pub fn flat_names(vars: &[VarInfo], include_tp: bool, include_gq: bool) -> Vec<String> {
    let mut out = Vec::new();
    for v in vars.iter().filter(|v| included(v, include_tp, include_gq)) {
        for idx in column_major_indices(&v.dims) {
            let mut name = v.name.clone();
            for i in idx {
                name.push('.');
                name.push_str(&i.to_string());
            }
            if v.complex {
                out.push(format!("{name}.real"));
                out.push(format!("{name}.imag"));
            } else {
                out.push(name);
            }
        }
    }
    out
}

/// Reorders values from storage order (arrays row-major, matrices column-major, complex parts
/// adjacent) to CmdStan's output order (column-major over all indices, complex parts adjacent).
pub fn to_output_order(values: &[f64], info: &VarInfo) -> Vec<f64> {
    let unit = if info.complex { 2 } else { 1 };
    let dims = &info.dims;
    if dims.len() <= 1 {
        return values.to_vec();
    }
    // Storage strides: row-major over the array dimensions, then column-major for a matrix tail.
    let k = dims.len();
    let mut order: Vec<usize> = (0..k).collect();
    if info.matrix && k >= 2 {
        order.swap(k - 2, k - 1);
    }
    // `order` lists the dimensions from slowest to fastest in storage.
    let mut stride = vec![0; k];
    let mut s = unit;
    for &d in order.iter().rev() {
        stride[d] = s;
        s *= dims[d];
    }
    let mut out = Vec::with_capacity(values.len());
    for idx in column_major_indices(dims) {
        let p: usize = idx.iter().zip(&stride).map(|(i, st)| (i - 1) * st).sum();
        out.extend_from_slice(&values[p..p + unit]);
    }
    out
}

/// Appends the values of one variable to `out` in CmdStan's output order.
pub fn push_output(out: &mut Vec<f64>, x: &impl crate::check::Values, info: &VarInfo) {
    out.extend(to_output_order(&x.to_values(), info));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(dims: Vec<usize>, matrix: bool) -> VarInfo {
        VarInfo {
            name: "x".into(),
            dims,
            block: VarBlock::Parameters,
            complex: false,
            matrix,
        }
    }

    #[test]
    fn names_and_order() {
        let v = info(vec![2, 3], false);
        let names = flat_names(std::slice::from_ref(&v), false, false);
        assert_eq!(names[..3], ["x.1.1", "x.2.1", "x.1.2"]);
        // Array of arrays stored row-major: x[1][1], x[1][2], x[1][3], x[2][1], ...
        let vals = [11.0, 12.0, 13.0, 21.0, 22.0, 23.0];
        assert_eq!(
            to_output_order(&vals, &v),
            vec![11.0, 21.0, 12.0, 22.0, 13.0, 23.0]
        );
        // A matrix is stored column-major already.
        let m = info(vec![2, 3], true);
        let vals = [11.0, 21.0, 12.0, 22.0, 13.0, 23.0];
        assert_eq!(to_output_order(&vals, &m), vals.to_vec());
    }
}
