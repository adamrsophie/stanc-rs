//! Calls to the Stan Math library, resolved against the signatures of the stanmath-rs functions.
//!
//! `data/stanmath_rs_signatures.txt` lists, for every Stan library function, the stanmath-rs free
//! functions that implement it (the plain name and overload variants such as `_slice`, `_int`, or
//! `_2d`). A call is matched to the variant whose parameters accept the Stan argument types; a
//! function of scalars applied to containers is vectorized elementwise.

use std::collections::HashMap;
use std::sync::OnceLock;

use stanc_rs_syntax::ast::UnsizedType;

/// One parameter of a stanmath-rs function.
#[derive(Clone, Debug, PartialEq)]
pub enum Param {
    /// A scalar of a generic type bounded by `Real` (`x: T`).
    RealScalar(String),
    /// A broadcast argument: a scalar by value or a slice by reference (`y: Y`).
    Broadcast { generic: String, int: bool },
    /// `&[E]`.
    Slice(Elem),
    /// `&Matrix<E>`.
    Matrix(Elem),
    /// `f64`.
    F64,
    /// `i64`.
    I64,
    /// `i32`.
    I32,
    /// `usize`.
    Usize,
    /// `Complex<T>`.
    Complex(String),
    /// `&mut R` with `R: Rng`.
    Rng,
    /// `&D` with `D: Dims` (any container).
    Dims,
    /// Anything else (functions, options, fixed-size arrays): not callable from generated code.
    Other(String),
}

/// Element type of a slice or matrix parameter.
#[derive(Clone, Debug, PartialEq)]
pub enum Elem {
    /// A real scalar: a generic (with its name) or `f64` (`None`).
    Real(Option<String>),
    /// `i64`.
    Int,
    /// `Complex<T>`.
    Complex(String),
    /// `Vec<E>` (arrays of vectors).
    Vec(Box<Elem>),
}

/// The return type of a stanmath-rs function.
#[derive(Clone, Debug, PartialEq)]
pub struct Ret {
    /// The function returns `Result<..>`.
    pub fallible: bool,
    /// The value type after removing `Result`.
    pub ty: String,
}

/// A stanmath-rs function.
#[derive(Clone, Debug)]
pub struct Fun {
    pub rust_name: String,
    pub generics: Vec<(String, Vec<String>)>,
    pub params: Vec<Param>,
    pub ret: Ret,
}

impl Fun {
    /// Generic parameters with a `Real` bound.
    pub fn is_real_generic(&self, g: &str) -> bool {
        self.generics
            .iter()
            .any(|(n, bs)| n == g && bs.iter().any(|b| b == "Real"))
    }

    /// Generic parameters that no parameter type mentions (they need a turbofish).
    pub fn undetermined_generics(&self) -> Vec<String> {
        let raw = RAW
            .get()
            .and_then(|m| m.get(&self.rust_name))
            .cloned()
            .unwrap_or_default();
        self.generics
            .iter()
            .filter(|(g, _)| !mentions(&raw, g))
            .map(|(g, _)| g.clone())
            .collect()
    }
}

fn mentions(params: &str, g: &str) -> bool {
    params
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .any(|w| w == g)
}

static TABLE: OnceLock<HashMap<String, Vec<Fun>>> = OnceLock::new();
static RAW: OnceLock<HashMap<String, String>> = OnceLock::new();

/// The stanmath-rs functions implementing Stan function `name`, plain name first.
pub fn functions(name: &str) -> &'static [Fun] {
    let table = TABLE.get_or_init(load);
    table.get(name).map_or(&[], Vec::as_slice)
}

/// True if stanmath-rs has a function for Stan name `name`.
pub fn exists(name: &str) -> bool {
    !functions(name).is_empty()
}

fn split_top(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            _ => {}
        }
        if ch == ',' && depth == 0 {
            out.push(cur.trim().to_string());
            cur.clear();
        } else {
            cur.push(ch);
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn elem(s: &str, generics: &[(String, Vec<String>)]) -> Option<Elem> {
    let s = s.trim();
    if s == "f64" {
        return Some(Elem::Real(None));
    }
    if s == "i64" {
        return Some(Elem::Int);
    }
    if let Some(inner) = s.strip_prefix("Complex<").and_then(|r| r.strip_suffix('>')) {
        return Some(Elem::Complex(inner.to_string()));
    }
    if let Some(inner) = s.strip_prefix("Vec<").and_then(|r| r.strip_suffix('>')) {
        return elem(inner, generics).map(|e| Elem::Vec(Box::new(e)));
    }
    if generics.iter().any(|(g, _)| g == s) {
        return Some(Elem::Real(Some(s.to_string())));
    }
    None
}

fn param(s: &str, generics: &[(String, Vec<String>)]) -> Param {
    let bounds = |g: &str| -> Vec<String> {
        generics
            .iter()
            .find(|(n, _)| n == g)
            .map(|(_, b)| b.clone())
            .unwrap_or_default()
    };
    match s {
        "f64" => return Param::F64,
        "i64" => return Param::I64,
        "i32" => return Param::I32,
        "usize" => return Param::Usize,
        _ => {}
    }
    if let Some(g) = s.strip_prefix("&mut ")
        && bounds(g).iter().any(|b| b == "Rng")
    {
        return Param::Rng;
    }
    if let Some(inner) = s.strip_prefix("&[").and_then(|r| r.strip_suffix(']')) {
        return elem(inner, generics).map_or_else(|| Param::Other(s.to_string()), Param::Slice);
    }
    if let Some(inner) = s.strip_prefix("&Matrix<").and_then(|r| r.strip_suffix('>')) {
        return elem(inner, generics).map_or_else(|| Param::Other(s.to_string()), Param::Matrix);
    }
    if let Some(inner) = s.strip_prefix("Complex<").and_then(|r| r.strip_suffix('>')) {
        return Param::Complex(inner.to_string());
    }
    if let Some(g) = s.strip_prefix('&')
        && bounds(g).iter().any(|b| b == "Dims")
    {
        return Param::Dims;
    }
    if generics.iter().any(|(g, _)| g == s) {
        let bs = bounds(s);
        if bs.iter().any(|b| b == "Real") {
            return Param::RealScalar(s.to_string());
        }
        return Param::Broadcast {
            generic: s.to_string(),
            int: bs.iter().any(|b| b == "IntArg"),
        };
    }
    Param::Other(s.to_string())
}

fn load() -> HashMap<String, Vec<Fun>> {
    let text = include_str!("../data/stanmath_rs_signatures.txt");
    let mut table: HashMap<String, Vec<Fun>> = HashMap::new();
    let mut raw = HashMap::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut parts = line.splitn(3, ' ');
        let (_krate, stan, rest) = (parts.next(), parts.next(), parts.next());
        let (Some(stan), Some(rest)) = (stan, rest) else {
            continue;
        };
        let lt = rest.find('<').unwrap_or(rest.len());
        let rust_name = rest[..lt].to_string();
        // Generic list: up to the matching '>' before '('.
        let open = rest.find('(').unwrap_or(rest.len());
        let gen_str = rest[lt..open]
            .trim()
            .trim_start_matches('<')
            .trim_end_matches('>');
        let generics: Vec<(String, Vec<String>)> = split_top(gen_str)
            .into_iter()
            .filter(|g| !g.is_empty())
            .map(|g| match g.split_once(':') {
                Some((n, b)) => (
                    n.trim().to_string(),
                    b.split('+').map(|x| x.trim().to_string()).collect(),
                ),
                None => (g.trim().to_string(), vec![]),
            })
            .collect();
        let Some(arrow) = rest.rfind(") -> ") else {
            continue;
        };
        let params_str = &rest[open + 1..arrow];
        let out = rest[arrow + 5..].trim();
        let params = split_top(params_str)
            .iter()
            .map(|p| param(p, &generics))
            .collect();
        let ret = match out
            .strip_prefix("Result<")
            .and_then(|r| r.strip_suffix('>'))
        {
            Some(inner) => Ret {
                fallible: true,
                ty: inner.to_string(),
            },
            None => Ret {
                fallible: false,
                ty: out.to_string(),
            },
        };
        raw.insert(rust_name.clone(), params_str.to_string());
        table.entry(stan.to_string()).or_default().push(Fun {
            rust_name,
            generics,
            params,
            ret,
        });
    }
    for (stan, funs) in table.iter_mut() {
        // The function with the Stan name itself comes first.
        funs.sort_by_key(|f| f.rust_name != *stan);
    }
    let _ = RAW.set(raw);
    table
}

/// How a Stan argument is classified for matching.
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Int,
    Real,
    Complex,
    /// A vector, row vector, or one-dimensional array of scalars, with the scalar kind.
    Vec(Box<Shape>),
    /// A matrix of reals or complex numbers.
    Mat(Box<Shape>),
    /// An array of non-scalars.
    Arr(Box<Shape>),
    Other,
}

impl Shape {
    pub fn of(t: &UnsizedType) -> Shape {
        use UnsizedType as U;
        match t {
            U::Int => Shape::Int,
            U::Real => Shape::Real,
            U::Complex => Shape::Complex,
            U::Vector | U::RowVector => Shape::Vec(Box::new(Shape::Real)),
            U::ComplexVector | U::ComplexRowVector => Shape::Vec(Box::new(Shape::Complex)),
            U::Matrix => Shape::Mat(Box::new(Shape::Real)),
            U::ComplexMatrix => Shape::Mat(Box::new(Shape::Complex)),
            U::Array(inner) => match Shape::of(inner) {
                s @ (Shape::Int | Shape::Real | Shape::Complex) => Shape::Vec(Box::new(s)),
                Shape::Other => Shape::Other,
                s => Shape::Arr(Box::new(s)),
            },
            _ => Shape::Other,
        }
    }

    pub fn is_scalar(&self) -> bool {
        matches!(self, Shape::Int | Shape::Real | Shape::Complex)
    }
}

/// How one argument is passed.
#[derive(Clone, Debug, PartialEq)]
pub enum Pass {
    /// By value (scalars).
    Value,
    /// Cast an `int` to `f64`.
    IntToReal,
    /// `as i32`.
    I32,
    /// Checked conversion to `usize`.
    Usize,
    /// `&x` (vectors, arrays, matrices into matching parameters).
    Ref,
    /// `x.as_slice()` (a matrix into a slice parameter).
    MatSlice,
    /// Primal values of an autodiff scalar (`.value()`) for an `f64` parameter.
    ScalarValue,
    /// Primal values of a vector for a `&[f64]` parameter.
    VecValues,
    /// Primal values of a matrix for a `&Matrix<f64>` parameter.
    MatValues,
    /// The argument is a container passed to a scalar parameter: map over its elements.
    Vectorize,
    /// A vector passed to an array-of-vectors parameter as a one-element array (the multivariate
    /// `_array` variants broadcast a single location).
    Singleton,
}

/// A resolved call.
#[derive(Clone, Debug)]
pub struct Resolved {
    pub fun: &'static Fun,
    pub passes: Vec<Pass>,
    /// Lift data arguments to the autodiff scalar (homogeneous `T: Real` parameters).
    pub lift: Vec<bool>,
}

/// Matches Stan arguments `(shape, autodiff)` against the variants of `name`.
pub fn resolve(name: &str, args: &[(Shape, bool)], result_autodiff: bool) -> Option<Resolved> {
    let funs = functions(name);
    let mut best: Option<(usize, Resolved)> = None;
    for vectorize in [false, true] {
        for f in funs {
            if vectorize && f.rust_name != name {
                continue;
            }
            if let Some((score, r)) = try_fun(f, args, vectorize, result_autodiff)
                && best.as_ref().is_none_or(|(s, _)| score < *s)
            {
                best = Some((score, r));
            }
        }
        if best.is_some() {
            break;
        }
    }
    best.map(|(_, r)| r)
}

fn try_fun(
    f: &'static Fun,
    args: &[(Shape, bool)],
    vectorize: bool,
    result_autodiff: bool,
) -> Option<(usize, Resolved)> {
    let params: Vec<&Param> = f.params.iter().filter(|p| **p != Param::Rng).collect();
    if params.len() != args.len() {
        return None;
    }
    let mut score = 0;
    let mut passes = Vec::new();
    let mut lift = Vec::new();
    let mut any_vectorized = false;
    for (p, (s, ad)) in params.iter().zip(args) {
        let (pass, cost) = fit(p, s, *ad, vectorize)?;
        if pass == Pass::Vectorize {
            any_vectorized = true;
        }
        score += cost;
        passes.push(pass);
        let g = match p {
            Param::RealScalar(g) | Param::Complex(g) => Some(g.as_str()),
            Param::Slice(Elem::Real(Some(g)) | Elem::Complex(g))
            | Param::Matrix(Elem::Real(Some(g)) | Elem::Complex(g)) => Some(g.as_str()),
            Param::Slice(Elem::Vec(e)) | Param::Matrix(Elem::Vec(e)) => match e.as_ref() {
                Elem::Real(Some(g)) => Some(g.as_str()),
                _ => None,
            },
            _ => None,
        };
        lift.push(result_autodiff && g.is_some_and(|g| f.is_real_generic(g)));
    }
    if vectorize && !any_vectorized {
        return None;
    }
    Some((
        score,
        Resolved {
            fun: f,
            passes,
            lift,
        },
    ))
}

fn real_like(s: &Shape) -> bool {
    matches!(s, Shape::Int | Shape::Real)
}

fn fit(p: &Param, s: &Shape, autodiff: bool, vectorize: bool) -> Option<(Pass, usize)> {
    let scalar_param = matches!(
        p,
        Param::RealScalar(_) | Param::F64 | Param::I64 | Param::I32 | Param::Usize
    );
    if vectorize && scalar_param && !s.is_scalar() && leaf_ok(p, s) {
        return Some((Pass::Vectorize, 0));
    }
    match (p, s) {
        (Param::RealScalar(_), Shape::Real) => Some((Pass::Value, 0)),
        (Param::RealScalar(_), Shape::Int) => Some((Pass::IntToReal, 1)),
        (Param::F64, Shape::Real) => Some((
            if autodiff {
                Pass::ScalarValue
            } else {
                Pass::Value
            },
            0,
        )),
        (Param::F64, Shape::Int) => Some((Pass::IntToReal, 1)),
        (Param::I64, Shape::Int) => Some((Pass::Value, 0)),
        (Param::I32, Shape::Int) => Some((Pass::I32, 0)),
        (Param::Usize, Shape::Int) => Some((Pass::Usize, 0)),
        (Param::Complex(_), Shape::Complex) => Some((Pass::Value, 0)),
        (Param::Broadcast { int: true, .. }, Shape::Int) => Some((Pass::Value, 0)),
        (Param::Broadcast { int: true, .. }, Shape::Vec(e)) if **e == Shape::Int => {
            Some((Pass::Ref, 0))
        }
        (Param::Broadcast { int: false, .. }, Shape::Real) => Some((Pass::Value, 0)),
        (Param::Broadcast { int: false, .. }, Shape::Int) => Some((Pass::IntToReal, 1)),
        (Param::Broadcast { int: false, .. }, Shape::Vec(e)) if **e == Shape::Real => {
            Some((Pass::Ref, 0))
        }
        (Param::Broadcast { int: false, .. }, Shape::Mat(e)) if **e == Shape::Real => {
            Some((Pass::MatSlice, 2))
        }
        (Param::Slice(Elem::Real(g)), Shape::Vec(e)) if **e == Shape::Real => Some((
            if g.is_none() && autodiff {
                Pass::VecValues
            } else {
                Pass::Ref
            },
            0,
        )),
        (Param::Slice(Elem::Real(Some(_))), Shape::Mat(e)) if **e == Shape::Real => {
            Some((Pass::MatSlice, 2))
        }
        (Param::Slice(Elem::Int), Shape::Vec(e)) if **e == Shape::Int => Some((Pass::Ref, 0)),
        (Param::Slice(Elem::Complex(_)), Shape::Vec(e)) if **e == Shape::Complex => {
            Some((Pass::Ref, 0))
        }
        (Param::Slice(Elem::Vec(pe)), Shape::Arr(inner)) => match (pe.as_ref(), inner.as_ref()) {
            (Elem::Real(_), Shape::Vec(e)) if **e == Shape::Real => Some((Pass::Ref, 0)),
            (Elem::Int, Shape::Vec(e)) if **e == Shape::Int => Some((Pass::Ref, 0)),
            _ => None,
        },
        (Param::Slice(Elem::Vec(pe)), Shape::Vec(e))
            if matches!(pe.as_ref(), Elem::Real(_)) && **e == Shape::Real =>
        {
            Some((Pass::Singleton, 3))
        }
        (Param::Matrix(Elem::Real(g)), Shape::Mat(e)) if **e == Shape::Real => Some((
            if g.is_none() && autodiff {
                Pass::MatValues
            } else {
                Pass::Ref
            },
            0,
        )),
        (Param::Matrix(Elem::Complex(_)), Shape::Mat(e)) if **e == Shape::Complex => {
            Some((Pass::Ref, 0))
        }
        (Param::Dims, _) => Some((Pass::Ref, 0)),
        _ => None,
    }
}

/// True if the leaves of container `s` fit scalar parameter `p`.
fn leaf_ok(p: &Param, s: &Shape) -> bool {
    match s {
        Shape::Vec(e) | Shape::Mat(e) | Shape::Arr(e) => leaf_ok(p, e),
        Shape::Int => true,
        Shape::Real => matches!(p, Param::RealScalar(_) | Param::F64),
        _ => false,
    }
}

/// True if `s` is real-valued (used to choose lifting).
pub fn is_real_like(s: &Shape) -> bool {
    real_like(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_loads() {
        assert!(exists("normal_lpdf"));
        let f = &functions("normal_lpdf")[0];
        assert_eq!(f.params.len(), 3);
        assert!(matches!(f.params[0], Param::Broadcast { int: false, .. }));
        let p = &functions("poisson_lpmf")[0];
        assert!(matches!(p.params[0], Param::Broadcast { int: true, .. }));
        let r = resolve(
            "log_sum_exp",
            &[(Shape::Vec(Box::new(Shape::Real)), true)],
            true,
        )
        .unwrap();
        assert_eq!(r.fun.rust_name, "log_sum_exp_slice");
        let r = resolve("exp", &[(Shape::Vec(Box::new(Shape::Real)), true)], true).unwrap();
        assert_eq!(r.fun.rust_name, "exp");
        assert_eq!(r.passes, vec![Pass::Vectorize]);
        let r = resolve(
            "rep_array",
            &[
                (Shape::Real, false),
                (Shape::Int, false),
                (Shape::Int, false),
            ],
            false,
        )
        .unwrap();
        assert_eq!(r.fun.rust_name, "rep_array_2d");
        let r = resolve(
            "normal_rng",
            &[(Shape::Real, false), (Shape::Real, false)],
            false,
        )
        .unwrap();
        assert_eq!(r.fun.rust_name, "normal_rng");
        let r = resolve("identity_matrix", &[(Shape::Int, false)], false).unwrap();
        assert_eq!(r.fun.undetermined_generics(), vec!["T".to_string()]);
    }
}
