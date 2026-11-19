//! Signatures of the Stan Math library functions.
//!
//! The table is the output of `stanc --dump-stan-math-signatures` from stanc3 v2.40.0 (BSD-3-Clause,
//! Stan Development Team), with top-level `data` qualifiers restored (see `data/README.md`), stored
//! in `data/` and parsed on first use. Variadic higher-order
//! functions (ODE solvers, algebraic solvers, `integrate_1d` variants, `reduce_sum`) are not in the
//! dump and are described separately in [`variadic`].

use std::collections::HashMap;
use std::sync::OnceLock;

use stanc_rs_syntax::ast::{AdLevel, FnSuffix, FunctionType, UnsizedType};

/// One signature of a library function.
#[derive(Clone, Debug, PartialEq)]
pub struct Signature {
    /// Arguments with their autodiff requirement.
    pub args: Vec<(AdLevel, UnsizedType)>,
    /// Return type; `None` for `void`.
    pub ret: Option<UnsizedType>,
}

const DUMP: &str = include_str!("../data/stan_math_signatures_2.40.0.txt");

/// All library signatures, by function name.
pub fn table() -> &'static HashMap<String, Vec<Signature>> {
    static TABLE: OnceLock<HashMap<String, Vec<Signature>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut m: HashMap<String, Vec<Signature>> = HashMap::new();
        for line in DUMP.lines().filter(|l| !l.trim().is_empty()) {
            let (name, sig) =
                parse_line(line).unwrap_or_else(|| panic!("bad signature line: {line}"));
            m.entry(name).or_default().push(sig);
        }
        m
    })
}

/// Signatures of `name` (after normalizing `_lupdf` to `_lpdf` and `_lupmf` to `_lpmf`).
pub fn lookup(name: &str) -> &'static [Signature] {
    table()
        .get(&normalized_name(name))
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

/// Is `prefix` a Stan Math distribution family (has an `_lpdf` or `_lpmf`)?
pub fn is_distribution_family(prefix: &str) -> bool {
    table().contains_key(&format!("{prefix}_lpdf"))
        || table().contains_key(&format!("{prefix}_lpmf"))
}

/// True if `name` is a Stan Math function (including variadic ones).
pub fn is_stan_math_function(name: &str) -> bool {
    let n = normalized_name(name);
    table().contains_key(&n) || variadic(&n).is_some() || is_reduce_sum(&n) || is_laplace(&n)
}

/// `_lupdf` to `_lpdf`, `_lupmf` to `_lpmf`; other names unchanged.
pub fn normalized_name(name: &str) -> String {
    if let Some(p) = name.strip_suffix("_lupdf") {
        format!("{p}_lpdf")
    } else if let Some(p) = name.strip_suffix("_lupmf") {
        format!("{p}_lpmf")
    } else {
        name.to_string()
    }
}

/// True for `_lupdf` and `_lupmf` names.
pub fn is_unnormalized(name: &str) -> bool {
    name.ends_with("_lupdf") || name.ends_with("_lupmf")
}

/// The Stan Math functions implementing an operator, tried in order.
pub fn operator_functions(op: &str) -> &'static [&'static str] {
    match op {
        "+" => &["add"],
        "u+" => &["plus"],
        "-" => &["subtract"],
        "u-" => &["minus"],
        "*" => &["multiply"],
        "/" => &["mdivide_right", "divide"],
        "%" => &["modulus"],
        "\\" => &["mdivide_left"],
        ".*" => &["elt_multiply"],
        "./" => &["elt_divide"],
        "^" | ".^" => &["pow"],
        "||" => &["logical_or"],
        "&&" => &["logical_and"],
        "==" => &["logical_eq"],
        "!=" => &["logical_neq"],
        "<" => &["logical_lt"],
        "<=" => &["logical_lte"],
        ">" => &["logical_gt"],
        ">=" => &["logical_gte"],
        "!" => &["logical_negation"],
        "'" => &["transpose"],
        _ => &[],
    }
}

/// A variadic higher-order library function.
#[derive(Clone, Debug)]
pub struct Variadic {
    /// Return type.
    pub ret: UnsizedType,
    /// Control arguments after the function argument.
    pub control_args: Vec<(AdLevel, UnsizedType)>,
    /// Required leading arguments of the function argument.
    pub fn_args: Vec<(AdLevel, UnsizedType)>,
    /// Required return type of the function argument.
    pub fn_ret: UnsizedType,
}

/// The variadic signature of `name`, if it is one.
pub fn variadic(name: &str) -> Option<Variadic> {
    use AdLevel::*;
    use UnsizedType::*;
    let arr = |t| Array(Box::new(t));
    let ode_mand = vec![
        (AutoDiffable, Vector),
        (AutoDiffable, Real),
        (AutoDiffable, arr(Real)),
    ];
    let ode_tol = vec![(DataOnly, Real), (DataOnly, Real), (DataOnly, Int)];
    let ode_fn = vec![(AutoDiffable, Real), (AutoDiffable, Vector)];
    let dae_mand = vec![
        (AutoDiffable, Vector),
        (AutoDiffable, Vector),
        (DataOnly, Real),
        (DataOnly, arr(Real)),
    ];
    let dae_fn = vec![
        (AutoDiffable, Real),
        (AutoDiffable, Vector),
        (AutoDiffable, Vector),
    ];
    let v = |ret, control_args, fn_args, fn_ret| {
        Some(Variadic {
            ret,
            control_args,
            fn_args,
            fn_ret,
        })
    };
    match name {
        "dae" => v(arr(Vector), dae_mand, dae_fn, Vector),
        "dae_tol" => v(arr(Vector), [dae_mand, ode_tol].concat(), dae_fn, Vector),
        "ode_bdf" | "ode_rk45" | "ode_adams" | "ode_ckrk" => {
            v(arr(Vector), ode_mand, ode_fn, Vector)
        }
        "ode_bdf_tol" | "ode_rk45_tol" | "ode_adams_tol" | "ode_ckrk_tol" => {
            v(arr(Vector), [ode_mand, ode_tol].concat(), ode_fn, Vector)
        }
        "ode_adjoint_tol_ctl" => {
            let ctl = vec![
                (DataOnly, Real),
                (DataOnly, Vector),
                (DataOnly, Real),
                (DataOnly, Vector),
                (DataOnly, Real),
                (DataOnly, Real),
                (DataOnly, Int),
                (DataOnly, Int),
                (DataOnly, Int),
                (DataOnly, Int),
                (DataOnly, Int),
            ];
            v(arr(Vector), [ode_mand, ctl].concat(), ode_fn, Vector)
        }
        "solve_newton" | "solve_powell" => v(
            Vector,
            vec![(AutoDiffable, Vector)],
            vec![(AutoDiffable, Vector)],
            Vector,
        ),
        "solve_newton_tol" | "solve_powell_tol" => v(
            Vector,
            vec![
                (AutoDiffable, Vector),
                (DataOnly, Real),
                (DataOnly, Real),
                (DataOnly, Int),
            ],
            vec![(AutoDiffable, Vector)],
            Vector,
        ),
        "integrate_1d_gauss_kronrod" | "integrate_1d_double_exponential" => v(
            Real,
            vec![(AutoDiffable, Real), (AutoDiffable, Real)],
            vec![(AutoDiffable, Real), (AutoDiffable, Real)],
            Real,
        ),
        "integrate_1d_gauss_kronrod_tol" | "integrate_1d_double_exponential_tol" => v(
            Real,
            vec![
                (AutoDiffable, Real),
                (AutoDiffable, Real),
                (DataOnly, Real),
                (DataOnly, Real),
                (DataOnly, Int),
            ],
            vec![(AutoDiffable, Real), (AutoDiffable, Real)],
            Real,
        ),
        _ => None,
    }
}

/// `reduce_sum` and `reduce_sum_static`.
pub fn is_reduce_sum(name: &str) -> bool {
    name == "reduce_sum" || name == "reduce_sum_static"
}

/// The embedded Laplace approximation functions.
pub fn is_laplace(name: &str) -> bool {
    const L: &[&str] = &[
        "laplace_marginal",
        "laplace_marginal_tol",
        "laplace_latent_rng",
        "laplace_latent_tol_rng",
        "laplace_latent_solve",
        "laplace_latent_solve_tol",
        "laplace_marginal_bernoulli_logit_lpmf",
        "laplace_marginal_tol_bernoulli_logit_lpmf",
        "laplace_marginal_neg_binomial_2_log_lpmf",
        "laplace_marginal_tol_neg_binomial_2_log_lpmf",
        "laplace_marginal_poisson_log_lpmf",
        "laplace_marginal_tol_poisson_log_lpmf",
        "laplace_latent_bernoulli_logit_rng",
        "laplace_latent_tol_bernoulli_logit_rng",
        "laplace_latent_neg_binomial_2_log_rng",
        "laplace_latent_tol_neg_binomial_2_log_rng",
        "laplace_latent_poisson_log_rng",
        "laplace_latent_tol_poisson_log_rng",
    ];
    L.contains(&normalized_name(name).as_str())
}

/// Likelihood argument types of the Laplace helper functions (empty for the general forms).
pub fn laplace_helper_param_types(name: &str) -> Vec<(AdLevel, UnsizedType)> {
    use AdLevel::AutoDiffable as A;
    let n = normalized_name(name);
    let mut v = n.strip_prefix("laplace_").unwrap_or(&n);
    for p in ["marginal_", "latent_", "tol_"] {
        v = v.strip_prefix(p).unwrap_or(v);
    }
    let v = v
        .rsplit_once('_')
        .filter(|(_, s)| matches!(*s, "lpdf" | "lpmf" | "rng"))
        .map(|(a, _)| a)
        .unwrap_or(v);
    let ints = UnsizedType::Array(Box::new(UnsizedType::Int));
    match v {
        "bernoulli_logit" | "poisson_log" => {
            vec![(A, ints.clone()), (A, ints), (A, UnsizedType::Vector)]
        }
        "neg_binomial_2_log" => vec![
            (A, ints.clone()),
            (A, ints),
            (A, UnsizedType::Vector),
            (A, UnsizedType::Vector),
        ],
        _ => vec![],
    }
}

/// Types of the tolerance tuple of the `_tol` Laplace functions.
pub fn laplace_tolerance_argument_types() -> Vec<(AdLevel, UnsizedType)> {
    use AdLevel::*;
    vec![
        (AutoDiffable, UnsizedType::Vector),
        (DataOnly, UnsizedType::Real),
        (DataOnly, UnsizedType::Int),
        (DataOnly, UnsizedType::Int),
        (DataOnly, UnsizedType::Int),
        (DataOnly, UnsizedType::Int),
    ]
}

/// Library functions without second-order derivatives (not callable from Laplace likelihoods).
pub fn lacks_higher_order_autodiff(name: &str) -> bool {
    const L: &[&str] = &[
        "algebra_solver",
        "algebra_solver_newton",
        "integrate_1d",
        "integrate_ode",
        "integrate_ode_adams",
        "integrate_ode_bdf",
        "integrate_ode_rk45",
        "map_rect",
        "hmm_marginal",
        "hmm_hidden_state_prob",
    ];
    L.contains(&name) || is_special_function_name(name)
}

/// Names that user-defined functions may not take (variadic, `reduce_sum`, Laplace).
pub fn is_special_function_name(name: &str) -> bool {
    variadic(name).is_some() || is_reduce_sum(name) || is_laplace(name)
}

const SOA: &str = include_str!("../data/soa_signatures_2.40.0.txt");

/// The library signatures that accept struct of arrays matrices, by function name.
fn soa_table() -> &'static HashMap<String, Vec<Signature>> {
    static TABLE: OnceLock<HashMap<String, Vec<Signature>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut m: HashMap<String, Vec<Signature>> = HashMap::new();
        for line in SOA.lines().filter(|l| !l.trim().is_empty()) {
            let (name, sig) =
                parse_line(line).unwrap_or_else(|| panic!("bad signature line: {line}"));
            m.entry(name).or_default().push(sig);
        }
        m
    })
}

/// Does some signature of the library function `name` (an operator by its MIR name, such as
/// `Plus__`) that these arguments fit accept struct of arrays matrices
/// (`query_stan_math_mem_pattern_support`)?
pub fn supports_soa(name: &str, args: &[(crate::types::Ad, UnsizedType)]) -> bool {
    if is_special_function_name(name) {
        return false;
    }
    let name = normalized_name(name);
    let name = match name.as_str() {
        "Plus__" => "add",
        "PPlus__" => "plus",
        "Minus__" => "subtract",
        "PMinus__" => "minus",
        "Times__" => "multiply",
        "Divide__" | "IntDivide__" => "divide",
        "Modulo__" => "modulus",
        "LDivide__" => "mdivide_left",
        "EltTimes__" => "elt_multiply",
        "EltDivide__" => "elt_divide",
        "Pow__" | "EltPow__" => "pow",
        "Or__" => "logical_or",
        "And__" => "logical_and",
        "Equals__" => "logical_eq",
        "NEquals__" => "logical_neq",
        "Less__" => "logical_lt",
        "Leq__" => "logical_lte",
        "Greater__" => "logical_gt",
        "Geq__" => "logical_gte",
        "PNot__" => "logical_negation",
        "Transpose__" => "transpose",
        n => n,
    };
    soa_table().get(name).is_some_and(|sigs| {
        sigs.iter()
            .any(|s| crate::types::check_compatible_args(0, &s.args, args).is_ok())
    })
}

// ----- Dump parser -----

fn parse_line(line: &str) -> Option<(String, Signature)> {
    let open = line.find('(')?;
    let name = line[..open].trim().to_string();
    let mut p = P {
        s: line.as_bytes(),
        i: open,
    };
    let args = p.arg_list()?;
    p.ws();
    p.lit("=>")?;
    p.ws();
    let ret = p.return_type()?;
    Some((name, Signature { args, ret }))
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i] == b' ' {
            self.i += 1;
        }
    }

    fn lit(&mut self, l: &str) -> Option<()> {
        if self.s[self.i..].starts_with(l.as_bytes()) {
            self.i += l.len();
            Some(())
        } else {
            None
        }
    }

    fn word(&mut self) -> String {
        let st = self.i;
        while self.i < self.s.len()
            && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_')
        {
            self.i += 1;
        }
        String::from_utf8_lossy(&self.s[st..self.i]).into_owned()
    }

    fn arg_list(&mut self) -> Option<Vec<(AdLevel, UnsizedType)>> {
        self.lit("(")?;
        let mut v = Vec::new();
        self.ws();
        if self.lit(")").is_some() {
            return Some(v);
        }
        loop {
            self.ws();
            let ad = if self.s[self.i..].starts_with(b"data ") {
                self.i += 5;
                AdLevel::DataOnly
            } else {
                AdLevel::AutoDiffable
            };
            v.push((ad, self.ty()?));
            self.ws();
            if self.lit(",").is_some() {
                continue;
            }
            self.lit(")")?;
            return Some(v);
        }
    }

    fn return_type(&mut self) -> Option<Option<UnsizedType>> {
        if self.s[self.i..].starts_with(b"void") {
            self.i += 4;
            return Some(None);
        }
        Some(Some(self.ty()?))
    }

    fn ty(&mut self) -> Option<UnsizedType> {
        self.ws();
        if self.s.get(self.i) == Some(&b'(') {
            let args = self.arg_list()?;
            self.ws();
            self.lit("=>")?;
            self.ws();
            let ret = self.return_type()?;
            return Some(UnsizedType::Function(Box::new(FunctionType {
                ret,
                args,
                suffix: FnSuffix::Plain,
            })));
        }
        let w = self.word();
        Some(match w.as_str() {
            "int" => UnsizedType::Int,
            "real" => UnsizedType::Real,
            "complex" => UnsizedType::Complex,
            "vector" => UnsizedType::Vector,
            "row_vector" => UnsizedType::RowVector,
            "matrix" => UnsizedType::Matrix,
            "complex_vector" => UnsizedType::ComplexVector,
            "complex_row_vector" => UnsizedType::ComplexRowVector,
            "complex_matrix" => UnsizedType::ComplexMatrix,
            "array" => {
                self.lit("[")?;
                let mut n = 1;
                while self.lit(",").is_some() {
                    n += 1;
                }
                self.lit("]")?;
                self.ws();
                self.ty()?.array_of(n)
            }
            "tuple" => {
                self.lit("(")?;
                let mut ts = Vec::new();
                loop {
                    ts.push(self.ty()?);
                    self.ws();
                    if self.lit(",").is_some() {
                        continue;
                    }
                    self.lit(")")?;
                    break;
                }
                UnsizedType::Tuple(ts)
            }
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_parses() {
        let t = table();
        assert_eq!(t.len(), 571);
        let n: usize = t.values().map(Vec::len).sum();
        assert!(n > 24_000);
        let normal = lookup("normal_lupdf");
        assert!(normal.iter().any(|s| s.args
            == vec![
                (AdLevel::AutoDiffable, UnsizedType::Real),
                (AdLevel::AutoDiffable, UnsizedType::Real),
                (AdLevel::AutoDiffable, UnsizedType::Real)
            ]
            && s.ret == Some(UnsizedType::Real)));
        let alg = lookup("algebra_solver");
        assert!(matches!(alg[0].args[0].1, UnsizedType::Function(_)));
    }
}

#[cfg(test)]
mod soa_tests {
    use super::supports_soa;
    use crate::types::Ad;
    use stanc_rs_syntax::ast::UnsizedType::*;

    #[test]
    fn soa_support_matches_stanc3() {
        let ad = |t| (Ad::AutoDiffable, t);
        assert!(supports_soa("Plus__", &[ad(Vector), ad(Vector)]));
        assert!(supports_soa(
            "normal_lupdf",
            &[(Ad::DataOnly, Vector), ad(Vector), ad(Real)]
        ));
        assert!(supports_soa("rep_vector", &[ad(Real), (Ad::DataOnly, Int)]));
        assert!(!supports_soa("eigendecompose", &[ad(Matrix)]));
        assert!(!supports_soa(
            "lower_bound_unconstrain",
            &[ad(Vector), ad(Real)]
        ));
        // Integer arguments promote to the real signatures.
        assert!(supports_soa("Times__", &[(Ad::DataOnly, Int), ad(Vector)]));
    }
}
