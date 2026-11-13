//! Reading model data in CmdStan's JSON format.
//!
//! Scalars are JSON numbers (reals may also be the strings `"NaN"`, `"Inf"`, `"-Inf"`,
//! `"Infinity"`, `"-Infinity"`), vectors and row vectors are lists, matrices are lists of rows,
//! arrays are nested lists, complex numbers are two-element lists `[re, im]`, and tuples are
//! objects with keys `"1"`, `"2"`, .... Integers must be JSON integers in the 32-bit range used by
//! Stan; a real-valued JSON number is not accepted for an `int`.

use serde_json::{Map, Value};
use stanmath_rs::Matrix;
use stanmath_rs::autodiff::Complex;

use crate::{ModelError, Result};

/// A set of named values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Data {
    root: Map<String, Value>,
}

impl Data {
    /// No variables (for models without data).
    pub fn empty() -> Data {
        Data::default()
    }

    /// Parses a JSON object.
    pub fn from_json_str(s: &str) -> Result<Data> {
        match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(root)) => Ok(Data { root }),
            Ok(_) => Err(ModelError::data("the JSON data must be an object")),
            Err(e) => Err(ModelError::data(format!("invalid JSON: {e}"))),
        }
    }

    /// Wraps an already parsed JSON object.
    pub fn from_json_value(v: Value) -> Result<Data> {
        match v {
            Value::Object(root) => Ok(Data { root }),
            _ => Err(ModelError::data("the JSON data must be an object")),
        }
    }

    /// The value of variable `name`.
    pub fn get(&self, name: &str) -> Result<&Value> {
        self.root
            .get(name)
            .ok_or_else(|| ModelError::data(format!("variable '{name}' not found in the data")))
    }

    /// True if variable `name` is present.
    pub fn contains(&self, name: &str) -> bool {
        self.root.contains_key(name)
    }
}

fn shape_err(name: &str, what: &str, v: &Value) -> ModelError {
    let found = match v {
        Value::Null => "null".to_string(),
        Value::Bool(_) => "a boolean".to_string(),
        Value::Number(n) => format!("the number {n}"),
        Value::String(s) => format!("the string {s:?}"),
        Value::Array(a) => format!("a list of length {}", a.len()),
        Value::Object(_) => "an object".to_string(),
    };
    ModelError::data(format!("variable '{name}': expected {what}, found {found}"))
}

/// An `int`.
pub fn int(v: &Value, name: &str) -> Result<i64> {
    match v {
        Value::Number(n) => match n.as_i64() {
            Some(i) if i64::from(i32::MIN) <= i && i <= i64::from(i32::MAX) => Ok(i),
            Some(i) => Err(ModelError::data(format!(
                "variable '{name}': integer {i} is outside the range of Stan integers"
            ))),
            None => Err(shape_err(name, "an integer", v)),
        },
        _ => Err(shape_err(name, "an integer", v)),
    }
}

/// A `real`.
pub fn real(v: &Value, name: &str) -> Result<f64> {
    match v {
        Value::Number(n) => n.as_f64().ok_or_else(|| shape_err(name, "a real number", v)),
        Value::String(s) => match s.as_str() {
            "NaN" | "nan" => Ok(f64::NAN),
            "Inf" | "inf" | "+Inf" | "+inf" | "Infinity" | "+Infinity" => Ok(f64::INFINITY),
            "-Inf" | "-inf" | "-Infinity" => Ok(f64::NEG_INFINITY),
            _ => Err(shape_err(name, "a real number", v)),
        },
        _ => Err(shape_err(name, "a real number", v)),
    }
}

/// A `complex` given as `[re, im]` (a real number is also accepted).
pub fn complex(v: &Value, name: &str) -> Result<Complex<f64>> {
    match v {
        Value::Array(a) if a.len() == 2 => Ok(Complex::new(real(&a[0], name)?, real(&a[1], name)?)),
        Value::Number(_) | Value::String(_) => Ok(Complex::new(real(v, name)?, 0.0)),
        _ => Err(shape_err(name, "a complex number [re, im]", v)),
    }
}

fn list<'a>(v: &'a Value, n: usize, name: &str) -> Result<&'a [Value]> {
    match v {
        Value::Array(a) if a.len() == n => Ok(a),
        _ => Err(shape_err(name, &format!("a list of length {n}"), v)),
    }
}

/// An array of `n` elements read by `elem`.
pub fn array<T>(v: &Value, n: usize, name: &str, mut elem: impl FnMut(&Value) -> Result<T>) -> Result<Vec<T>> {
    list(v, n, name)?.iter().map(&mut elem).collect()
}

/// A `vector` or `row_vector` of length `n`.
pub fn vector(v: &Value, n: usize, name: &str) -> Result<Vec<f64>> {
    array(v, n, name, |e| real(e, name))
}

/// A `complex_vector` or `complex_row_vector` of length `n`.
pub fn complex_vector(v: &Value, n: usize, name: &str) -> Result<Vec<Complex<f64>>> {
    array(v, n, name, |e| complex(e, name))
}

fn rows_of<T: Clone>(
    v: &Value,
    r: usize,
    c: usize,
    name: &str,
    mut elem: impl FnMut(&Value) -> Result<T>,
) -> Result<Matrix<T>> {
    if r == 0 {
        if matches!(v, Value::Array(a) if a.is_empty()) {
            return Ok(Matrix::from_column_major(0, c, vec![]));
        }
        return Err(shape_err(name, "an empty list", v));
    }
    let rows = list(v, r, name)?;
    let mut cells: Vec<Vec<T>> = Vec::with_capacity(r);
    for row in rows {
        cells.push(array(row, c, name, &mut elem)?);
    }
    let mut data = Vec::with_capacity(r * c);
    for j in 0..c {
        for row in &cells {
            data.push(row[j].clone());
        }
    }
    Ok(Matrix::from_column_major(r, c, data))
}

/// A `matrix` with `r` rows and `c` columns, given as a list of rows.
pub fn matrix(v: &Value, r: usize, c: usize, name: &str) -> Result<Matrix<f64>> {
    rows_of(v, r, c, name, |e| real(e, name))
}

/// A `complex_matrix`, given as a list of rows.
pub fn complex_matrix(v: &Value, r: usize, c: usize, name: &str) -> Result<Matrix<Complex<f64>>> {
    rows_of(v, r, c, name, |e| complex(e, name))
}

/// Slot `k` (one-based) of a tuple given as an object with keys `"1"`, `"2"`, ....
pub fn tuple_slot<'a>(v: &'a Value, k: usize, name: &str) -> Result<&'a Value> {
    match v {
        Value::Object(m) => m
            .get(&k.to_string())
            .ok_or_else(|| ModelError::data(format!("variable '{name}': tuple slot {k} is missing"))),
        _ => Err(shape_err(name, "a tuple object", v)),
    }
}

/// Converts a declared size to `usize`, rejecting negative sizes.
pub fn size(n: i64, name: &str) -> Result<usize> {
    usize::try_from(n).map_err(|_| ModelError::domain(format!("variable '{name}': size {n} is negative")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_values() {
        let d = Data::from_json_str(
            r#"{"N": 3, "y": [1, 2.5, "Inf"], "m": [[1, 2, 3], [4, 5, 6]], "x": 1.5, "c": [1, -2],
                "t": {"1": 2, "2": [1.0, 2.0]}}"#,
        )
        .unwrap();
        assert_eq!(int(d.get("N").unwrap(), "N").unwrap(), 3);
        assert!(int(d.get("x").unwrap(), "x").is_err());
        let y = vector(d.get("y").unwrap(), 3, "y").unwrap();
        assert_eq!(y[2], f64::INFINITY);
        assert!(vector(d.get("y").unwrap(), 2, "y").is_err());
        let m = matrix(d.get("m").unwrap(), 2, 3, "m").unwrap();
        assert_eq!(m.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(complex(d.get("c").unwrap(), "c").unwrap(), Complex::new(1.0, -2.0));
        let t = d.get("t").unwrap();
        assert_eq!(int(tuple_slot(t, 1, "t").unwrap(), "t").unwrap(), 2);
        assert!(d.get("missing").is_err());
        assert!(matrix(&serde_json::json!([]), 0, 3, "e").is_ok());
    }
}
