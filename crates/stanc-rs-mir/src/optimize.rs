//! MIR optimization passes: stanc3's partial evaluator (`Partial_evaluator`) and the constant
//! and copy propagation of `Optimize`.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use stanc_rs_semantic::check::{operator_return_type, stan_math_return_type};

use crate::dataflow::{Graph, Lattice, Powerset, mfp};
use crate::mir::*;

/// A statement the partial evaluator proves always fails, with its location and message.
pub struct Rejected(pub Loc, pub String);

/// OCaml's `valid_float_lexem`: a trailing `.` when the text looks like an integer.
fn valid_float_lexem(s: String) -> String {
    if s.chars().all(|c| c.is_ascii_digit() || c == '-') {
        s + "."
    } else {
        s
    }
}

/// C's `%.{p}g`.
fn fmt_g(f: f64, p: usize) -> String {
    if f.is_nan() {
        return if f.is_sign_negative() { "-nan" } else { "nan" }.to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    let sci = format!("{:.*e}", p - 1, f);
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let trim = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if exp < -4 || exp >= p as i32 {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mant), exp.abs())
    } else {
        trim(&format!("{:.*}", (p as i32 - 1 - exp) as usize, f))
    }
}

/// stanc3's `Float.to_string`: the shortest of `%.15g` and `%.17g` that reads back exactly.
pub fn float_to_string(f: f64) -> String {
    let c = fmt_g(f, 15);
    valid_float_lexem(if c.parse::<f64>().ok() == Some(f) {
        c
    } else {
        fmt_g(f, 17)
    })
}

fn parse_f(s: &str) -> f64 {
    s.parse().unwrap_or(f64::NAN)
}

fn parse_i(s: &str) -> i64 {
    s.parse().unwrap_or(0)
}

/// A copy of an expression with every location cleared.
fn strip(e: &Expr) -> Expr {
    let mut e = e.clone();
    fn go(e: &mut Expr) {
        e.meta.loc = Loc::EMPTY;
        for c in e.children_mut() {
            go(c);
        }
    }
    go(&mut e);
    e
}

/// `Expr.Typed.equal`: equality ignoring locations.
pub fn expr_eq(a: &Expr, b: &Expr) -> bool {
    strip(a) == strip(b)
}

/// Whether the expression is an integer or real literal equal to `q` (`is_int`).
pub fn is_int(q: i64, e: &Expr) -> bool {
    match &e.pattern {
        ExprP::Lit(LitType::Int | LitType::Real, s) => parse_f(s) == q as f64,
        ExprP::Promotion(x, ..) => is_int(q, x),
        _ => false,
    }
}

fn is_eigen(t: &UnsizedType) -> bool {
    use UnsizedType::*;
    matches!(
        t,
        Vector | RowVector | Matrix | ComplexVector | ComplexRowVector | ComplexMatrix
    )
}

/// The arguments of a call of the library function `name` without suffix.
fn call<'a>(e: &'a Expr, name: &str) -> Option<&'a [Expr]> {
    match &e.pattern {
        ExprP::FunApp(FunKind::StanLib(n, Suffix::Plain), a) if n == name => Some(a),
        _ => None,
    }
}

fn call1<'a>(e: &'a Expr, name: &str) -> Option<&'a Expr> {
    call(e, name).and_then(|a| if a.len() == 1 { Some(&a[0]) } else { None })
}

fn call2<'a>(e: &'a Expr, name: &str) -> Option<(&'a Expr, &'a Expr)> {
    call(e, name).and_then(|a| {
        if a.len() == 2 {
            Some((&a[0], &a[1]))
        } else {
            None
        }
    })
}

fn is_matrix(e: &Expr) -> bool {
    e.meta.ty == UnsizedType::Matrix
}

/// A linear predictor `alpha + x * beta`, `x * beta + alpha`, or `x * beta` with a matrix `x`;
/// a missing `alpha` is zero.
fn linpred(e: &Expr) -> Option<(Expr, Expr, Expr)> {
    if let Some((alpha, t)) = call2(e, "Plus__")
        && let Some((x, beta)) = call2(t, "Times__")
        && is_matrix(x)
    {
        return Some((alpha.clone(), x.clone(), beta.clone()));
    }
    if let Some((t, alpha)) = call2(e, "Plus__")
        && let Some((x, beta)) = call2(t, "Times__")
        && is_matrix(x)
    {
        return Some((alpha.clone(), x.clone(), beta.clone()));
    }
    if let Some((x, beta)) = call2(e, "Times__")
        && is_matrix(x)
    {
        return Some((Expr::int(0), x.clone(), beta.clone()));
    }
    None
}

fn is_lit_num(e: &Expr) -> bool {
    matches!(e.pattern, ExprP::Lit(LitType::Real | LitType::Int, _))
}

fn imaginary(e: &Expr) -> Option<&str> {
    match &e.pattern {
        ExprP::Lit(LitType::Imaginary, i) => Some(i),
        _ => None,
    }
}

/// A real number literal, possibly promoted to complex.
fn real_part(e: &Expr) -> Option<&Expr> {
    if is_lit_num(e) {
        return Some(e);
    }
    match &e.pattern {
        ExprP::Promotion(x, UnsizedType::Complex, _) if is_lit_num(x) => Some(x),
        _ => None,
    }
}

/// The library rewrites of `eval_expr`: a better call, or `None` to keep the call.
fn rewrite(f: &str, l: &[Expr], ps: bool) -> Option<(&'static str, Vec<Expr>)> {
    let c = |e: &Expr| e.clone();
    let cs = |es: &[Expr]| es.to_vec();
    match (f, l) {
        ("bernoulli_lpmf", [y, p]) => {
            if let Some(lp) = call1(p, "inv_logit") {
                if let Some((a, x, b)) = linpred(lp) {
                    return Some(("bernoulli_logit_glm_lpmf", vec![c(y), x, a, b]));
                }
                return Some(("bernoulli_logit_lpmf", vec![c(y), c(lp)]));
            }
            None
        }
        ("bernoulli_logit_lpmf", [y, lp]) => {
            linpred(lp).map(|(a, x, b)| ("bernoulli_logit_glm_lpmf", vec![c(y), x, a, b]))
        }
        ("bernoulli_rng", [p]) => {
            call1(p, "inv_logit").map(|a| ("bernoulli_logit_rng", vec![c(a)]))
        }
        ("binomial_lpmf", [y, n, p]) => {
            call1(p, "inv_logit").map(|a| ("binomial_logit_lpmf", vec![c(y), c(n), c(a)]))
        }
        ("categorical_lpmf", [y, p]) => {
            call1(p, "softmax").map(|a| ("categorical_logit_lpmf", vec![c(y), c(a)]))
        }
        ("categorical_rng", [p]) => {
            call1(p, "softmax").map(|a| ("categorical_logit_rng", vec![c(a)]))
        }
        ("columns_dot_product", [x, y]) if expr_eq(x, y) => Some(("columns_dot_self", vec![c(x)])),
        ("dot_product", [x, y]) if expr_eq(x, y) => Some(("dot_self", vec![c(x)])),
        ("inv", [a]) => {
            if let Some(l) = call(a, "sqrt") {
                return Some(("inv_sqrt", cs(l)));
            }
            call1(a, "square").map(|x| ("inv_square", vec![c(x)]))
        }
        ("log", [a]) => {
            if let Some((y, e)) = call2(a, "Minus__")
                && let Some(x) = call1(e, "exp")
                && is_int(1, y)
                && !ps
            {
                return Some(("log1m_exp", vec![c(x)]));
            }
            if let Some((y, e)) = call2(a, "Minus__")
                && let Some(x) = call1(e, "inv_logit")
                && is_int(1, y)
                && !ps
            {
                return Some(("log1m_inv_logit", vec![c(x)]));
            }
            if let Some((y, x)) = call2(a, "Minus__")
                && is_int(1, y)
                && !ps
            {
                return Some(("log1m", vec![c(x)]));
            }
            if let Some((y, e)) = call2(a, "Plus__")
                && let Some(x) = call1(e, "exp")
                && is_int(1, y)
                && !ps
            {
                return Some(("log1p_exp", vec![c(x)]));
            }
            if let Some((y, x)) = call2(a, "Plus__")
                && is_int(1, y)
                && !ps
            {
                return Some(("log1p", vec![c(x)]));
            }
            if let Some(d) = call1(a, "fabs").or_else(|| call1(a, "abs"))
                && let Some(x) = call1(d, "determinant")
            {
                return Some(("log_determinant", vec![c(x)]));
            }
            if let Some((e1, e2)) = call2(a, "Minus__")
                && let Some(x) = call1(e1, "exp")
                && let Some(y) = call1(e2, "exp")
            {
                return Some(("log_diff_exp", vec![c(x), c(y)]));
            }
            for (from, to) in [
                ("falling_factorial", "log_falling_factorial"),
                ("rising_factorial", "log_rising_factorial"),
                ("inv_logit", "log_inv_logit"),
                ("softmax", "log_softmax"),
            ] {
                if let Some(l) = call(a, from) {
                    return Some((to, cs(l)));
                }
            }
            if let Some(s) = call1(a, "sum")
                && let Some(l) = call(s, "exp")
            {
                return Some(("log_sum_exp", cs(l)));
            }
            if let Some((e1, e2)) = call2(a, "Plus__")
                && let Some(x) = call1(e1, "exp")
                && let Some(y) = call1(e2, "exp")
            {
                return Some(("log_sum_exp", vec![c(x), c(y)]));
            }
            None
        }
        ("multi_normal_lpdf", [y, mu, s]) => {
            call1(s, "inverse").map(|tau| ("multi_normal_prec_lpdf", vec![c(y), c(mu), c(tau)]))
        }
        ("neg_binomial_2_lpmf", [y, m, phi]) => {
            let eta = call1(m, "exp")?;
            if let Some((a, x, b)) = linpred(eta) {
                return Some(("neg_binomial_2_log_glm_lpmf", vec![c(y), x, a, b, c(phi)]));
            }
            Some(("neg_binomial_2_log_lpmf", vec![c(y), c(eta), c(phi)]))
        }
        ("neg_binomial_2_log_lpmf", [y, lp, phi]) => linpred(lp)
            .map(|(a, x, b)| ("neg_binomial_2_log_glm_lpmf", vec![c(y), x, a, b, c(phi)])),
        ("neg_binomial_2_rng", [m, phi]) => {
            call1(m, "exp").map(|eta| ("neg_binomial_2_log_rng", vec![c(eta), c(phi)]))
        }
        ("normal_lpdf", [y, lp, sigma]) => {
            linpred(lp).map(|(a, x, b)| ("normal_id_glm_lpdf", vec![c(y), x, a, b, c(sigma)]))
        }
        ("poisson_lpmf", [y, m]) => {
            let eta = call1(m, "exp")?;
            if let Some((a, x, b)) = linpred(eta) {
                return Some(("poisson_log_glm_lpmf", vec![c(y), x, a, b]));
            }
            Some(("poisson_log_lpmf", vec![c(y), c(eta)]))
        }
        ("poisson_log_lpmf", [y, lp]) => {
            linpred(lp).map(|(a, x, b)| ("poisson_log_glm_lpmf", vec![c(y), x, a, b]))
        }
        ("poisson_rng", [m]) => call1(m, "exp").map(|eta| ("poisson_log_rng", vec![c(eta)])),
        ("pow", [y, x]) if is_int(2, y) => Some(("exp2", vec![c(x)])),
        ("rows_dot_product", [x, y]) if expr_eq(x, y) => Some(("rows_dot_self", vec![c(x)])),
        ("pow", [x, e]) => {
            match &e.pattern {
                ExprP::Lit(LitType::Int, s) if s == "2" => return Some(("square", vec![c(x)])),
                ExprP::Lit(LitType::Real, s) if s == "0.5" => {
                    return Some(("sqrt", vec![c(x)]));
                }
                _ => {}
            }
            if let Some((y, z)) = call2(e, "Divide__")
                && is_int(1, y)
                && is_int(2, z)
                && !(y.meta.ty == UnsizedType::Int && z.meta.ty == UnsizedType::Int)
            {
                return Some(("sqrt", vec![c(x)]));
            }
            None
        }
        ("square", [a]) => call1(a, "sd").map(|x| ("variance", vec![c(x)])),
        ("sqrt", [x]) if is_int(2, x) => Some(("sqrt2", vec![])),
        ("sum", [a]) => {
            if let Some(s) = call1(a, "square")
                && let Some((x, y)) = call2(s, "Minus__")
            {
                return Some(("squared_distance", vec![c(x), c(y)]));
            }
            call(a, "diagonal").map(|l| ("trace", cs(l)))
        }
        ("trace", [m]) => {
            if let Some((t2, cc)) = call2(m, "Times__")
                && let Some((t3, a)) = call2(t2, "Times__")
                && let Some((d, tb)) = call2(t3, "Times__")
                && let Some(b) = call1(tb, "transpose")
                && expr_eq(b, cc)
            {
                return Some(("trace_gen_quad_form", vec![c(d), c(a), c(b)]));
            }
            call2(m, "quad_form").map(|(a, b)| ("trace_quad_form", vec![c(a), c(b)]))
        }
        ("Plus__" | "add", [p, q])
            if (imaginary(p).is_some() && real_part(q).is_some())
                || (real_part(p).is_some() && imaginary(q).is_some()) =>
        {
            let (im, r) = if imaginary(p).is_some() {
                (p, real_part(q).unwrap())
            } else {
                (q, real_part(p).unwrap())
            };
            let mut im_part = im.clone();
            im_part.pattern = ExprP::Lit(LitType::Real, imaginary(im).unwrap().to_string());
            im_part.meta.ty = UnsizedType::Real;
            Some(("to_complex", vec![c(r), im_part]))
        }
        ("Minus__", [x, e]) if is_int(1, x) && call(e, "erf").is_some() => {
            Some(("erfc", cs(call(e, "erf").unwrap())))
        }
        ("Minus__", [x, e]) if is_int(1, x) && call(e, "erfc").is_some() => {
            Some(("erf", cs(call(e, "erfc").unwrap())))
        }
        ("Minus__", [e, x]) if is_int(1, x) && !ps && call(e, "exp").is_some() => {
            Some(("expm1", cs(call(e, "exp").unwrap())))
        }
        ("Plus__", [t, z])
            if !ps
                && call2(t, "Times__")
                    .is_some_and(|(x, y)| !(is_eigen(&x.meta.ty) && is_eigen(&y.meta.ty))) =>
        {
            let (x, y) = call2(t, "Times__").unwrap();
            Some(("fma", vec![c(x), c(y), c(z)]))
        }
        ("Plus__", [z, t])
            if !ps
                && call2(t, "Times__")
                    .is_some_and(|(x, y)| !(is_eigen(&x.meta.ty) && is_eigen(&y.meta.ty))) =>
        {
            let (x, y) = call2(t, "Times__").unwrap();
            Some(("fma", vec![c(x), c(y), c(z)]))
        }
        ("Plus__", [t, z])
            if !ps
                && call2(t, "elt_multiply")
                    .or_else(|| call2(t, "EltTimes__"))
                    .is_some() =>
        {
            let (x, y) = call2(t, "elt_multiply")
                .or_else(|| call2(t, "EltTimes__"))
                .unwrap();
            Some(("fma", vec![c(x), c(y), c(z)]))
        }
        ("Plus__", [z, t])
            if !ps
                && call2(t, "elt_multiply")
                    .or_else(|| call2(t, "EltTimes__"))
                    .is_some() =>
        {
            let (x, y) = call2(t, "elt_multiply")
                .or_else(|| call2(t, "EltTimes__"))
                .unwrap();
            Some(("fma", vec![c(x), c(y), c(z)]))
        }
        ("Minus__", [x, e]) if is_int(1, x) && call(e, "gamma_p").is_some() => {
            Some(("gamma_q", cs(call(e, "gamma_p").unwrap())))
        }
        ("Minus__", [x, e]) if is_int(1, x) && call(e, "gamma_q").is_some() => {
            Some(("gamma_p", cs(call(e, "gamma_q").unwrap())))
        }
        ("Times__", [p, q]) => {
            let scalar = |t: &Expr| matches!(t.meta.ty, UnsizedType::Int | UnsizedType::Real);
            if let Some(m) = call1(p, "matrix_exp")
                && let Some((t, a)) = call2(m, "Times__")
                && scalar(t)
            {
                return Some(("scale_matrix_exp_multiply", vec![c(t), c(a), c(q)]));
            }
            if let Some(m) = call1(p, "matrix_exp")
                && let Some((a, t)) = call2(m, "Times__")
                && scalar(t)
            {
                return Some(("scale_matrix_exp_multiply", vec![c(t), c(a), c(q)]));
            }
            if let Some(a) = call1(p, "matrix_exp") {
                return Some(("matrix_exp_multiply", vec![c(a), c(q)]));
            }
            if !ps {
                if let Some(y) = call1(q, "log") {
                    return Some(("lmultiply", vec![c(p), c(y)]));
                }
                if let Some(y) = call1(p, "log") {
                    return Some(("lmultiply", vec![c(q), c(y)]));
                }
            }
            if let Some(v) = call1(p, "diag_matrix")
                && let Some((a, w)) = call2(q, "diag_post_multiply")
                && expr_eq(v, w)
            {
                return Some(("quad_form_diag", vec![c(a), c(v)]));
            }
            if let Some((v, a)) = call2(p, "diag_pre_multiply")
                && let Some(w) = call1(q, "diag_matrix")
                && expr_eq(v, w)
            {
                return Some(("quad_form_diag", vec![c(a), c(v)]));
            }
            if let Some(b) = call1(p, "transpose")
                && let Some((a, cc)) = call2(q, "Times__")
                && expr_eq(b, cc)
            {
                return Some(("quad_form", vec![c(a), c(b)]));
            }
            if let Some((tb, a)) = call2(p, "Times__")
                && let Some(b) = call1(tb, "transpose")
                && expr_eq(b, q)
            {
                return Some(("quad_form", vec![c(a), c(b)]));
            }
            if let Some(v) = call1(q, "diag_matrix") {
                return Some(("diag_post_multiply", vec![c(p), c(v)]));
            }
            if let Some(v) = call1(p, "diag_matrix") {
                return Some(("diag_pre_multiply", vec![c(v), c(q)]));
            }
            None
        }
        _ => None,
    }
}

fn lit(t: LitType, s: String) -> ExprP {
    ExprP::Lit(t, s)
}

/// Constant folding of operators on literals.
fn fold(f: &str, l: &[Expr], e: &Expr) -> Result<Option<ExprP>, Rejected> {
    let lit_of = |e: &Expr| match &e.pattern {
        ExprP::Lit(t @ (LitType::Int | LitType::Real), s) => Some((*t, s.clone())),
        _ => None,
    };
    let b = |x: bool| lit(LitType::Int, (x as i64).to_string());
    match l {
        [a] => Ok(match lit_of(a) {
            Some((LitType::Int, s)) => {
                let i = parse_i(&s);
                match f {
                    "PPlus__" => Some(lit(LitType::Int, i.to_string())),
                    "PMinus__" => Some(lit(LitType::Int, i.wrapping_neg().to_string())),
                    "PNot__" => Some(b(i == 0)),
                    _ => None,
                }
            }
            Some((_, s)) => {
                let r = parse_f(&s);
                match f {
                    "PPlus__" => Some(lit(LitType::Real, float_to_string(r))),
                    "PMinus__" => Some(lit(LitType::Real, float_to_string(-r))),
                    _ => None,
                }
            }
            None => None,
        }),
        [a, z] => {
            if matches!(f, "Divide__" | "IntDivide__" | "Modulo__")
                && a.meta.ty == UnsizedType::Int
                && matches!(&z.pattern, ExprP::Lit(LitType::Int, s) if parse_i(s) == 0)
            {
                return Err(Rejected(e.meta.loc, "Integer division by zero".to_string()));
            }
            Ok(match (lit_of(a), lit_of(z)) {
                (Some((LitType::Int, s1)), Some((LitType::Int, s2))) => {
                    let (i1, i2) = (parse_i(&s1), parse_i(&s2));
                    let i = |x: i64| Some(lit(LitType::Int, x.to_string()));
                    match f {
                        "Plus__" => i(i1.wrapping_add(i2)),
                        "Minus__" => i(i1.wrapping_sub(i2)),
                        "Times__" => i(i1.wrapping_mul(i2)),
                        "Divide__" | "IntDivide__" => i(i1.wrapping_div(i2)),
                        "Modulo__" => i(i1.wrapping_rem(i2)),
                        "Equals__" => Some(b(i1 == i2)),
                        "NEquals__" => Some(b(i1 != i2)),
                        "Less__" => Some(b(i1 < i2)),
                        "Leq__" => Some(b(i1 <= i2)),
                        "Greater__" => Some(b(i1 > i2)),
                        "Geq__" => Some(b(i1 >= i2)),
                        _ => None,
                    }
                }
                (Some((_, s1)), Some((_, s2))) => {
                    let (r1, r2) = (parse_f(&s1), parse_f(&s2));
                    let r = |x: f64| Some(lit(LitType::Real, float_to_string(x)));
                    match f {
                        "Plus__" => r(r1 + r2),
                        "Minus__" => r(r1 - r2),
                        "Times__" => r(r1 * r2),
                        "Divide__" => r(r1 / r2),
                        "Equals__" => Some(b(r1 == r2)),
                        "NEquals__" => Some(b(r1 != r2)),
                        "Less__" => Some(b(r1 < r2)),
                        "Leq__" => Some(b(r1 <= r2)),
                        "Greater__" => Some(b(r1 > r2)),
                        "Geq__" => Some(b(r1 >= r2)),
                        _ => None,
                    }
                }
                _ => None,
            })
        }
        _ => Ok(None),
    }
}

/// The return type of a library function or operator call, when exactly one signature fits.
fn return_type_ok(name: &str, args: &[Expr]) -> bool {
    let tys: Vec<_> = args
        .iter()
        .map(|a| (a.meta.ad.clone(), a.meta.ty.clone()))
        .collect();
    match Operator::of_name(name) {
        Some(op) => {
            let key = match op {
                Operator::PPlus => "u+",
                Operator::PMinus => "u-",
                o => o.symbol(),
            };
            operator_return_type(key, &tys).is_some()
        }
        None => stan_math_return_type(name, &tys).is_some(),
    }
}

/// Partially evaluates an expression (`eval_expr`).
pub fn eval_expr(e: &Expr, ps: bool) -> Result<Expr, Rejected> {
    let pattern = match &e.pattern {
        ExprP::Var(_) | ExprP::Lit(..) => e.pattern.clone(),
        ExprP::Promotion(x, ut, ad) => {
            ExprP::Promotion(Box::new(eval_expr(x, ps)?), ut.clone(), ad.clone())
        }
        ExprP::FunApp(kind, l) => {
            let l = l
                .iter()
                .map(|a| eval_expr(a, ps))
                .collect::<Result<Vec<_>, _>>()?;
            match kind {
                FunKind::StanLib(f, suffix) => {
                    if let Some((f2, l2)) = rewrite(f, &l, ps) {
                        if return_type_ok(f2, &l2) {
                            ExprP::FunApp(FunKind::StanLib(f2.to_string(), *suffix), l2)
                        } else {
                            e.pattern.clone()
                        }
                    } else if let Some(p) = fold(f, &l, e)? {
                        p
                    } else {
                        ExprP::FunApp(kind.clone(), l)
                    }
                }
                _ => ExprP::FunApp(kind.clone(), l),
            }
        }
        ExprP::TernaryIf(c, a, b) => {
            let c2 = eval_expr(c, ps)?;
            if is_int(0, &c2) {
                eval_expr(b, ps)?.pattern
            } else if matches!(c2.pattern, ExprP::Lit(LitType::Int, _)) {
                eval_expr(a, ps)?.pattern
            } else {
                ExprP::TernaryIf(
                    Box::new(c2),
                    Box::new(try_eval_expr(a, ps)),
                    Box::new(try_eval_expr(b, ps)),
                )
            }
        }
        ExprP::EAnd(a, b) | ExprP::EOr(a, b) => {
            let and = matches!(e.pattern, ExprP::EAnd(..));
            let (a, b) = (eval_expr(a, ps)?, try_eval_expr(b, ps));
            if is_int(if and { 0 } else { 1 }, &a) {
                a.pattern
            } else {
                match (&a.pattern, &b.pattern) {
                    (ExprP::Lit(LitType::Int, s1), ExprP::Lit(LitType::Int, s2)) => {
                        let (x, y) = (parse_i(s1) != 0, parse_i(s2) != 0);
                        lit(
                            LitType::Int,
                            (if and { x && y } else { x || y } as i64).to_string(),
                        )
                    }
                    (ExprP::Lit(_, s1), ExprP::Lit(_, s2)) => {
                        let (x, y) = (parse_f(s1) != 0.0, parse_f(s2) != 0.0);
                        lit(
                            LitType::Int,
                            (if and { x && y } else { x || y } as i64).to_string(),
                        )
                    }
                    _ if and => ExprP::EAnd(Box::new(a), Box::new(b)),
                    _ => ExprP::EOr(Box::new(a), Box::new(b)),
                }
            }
        }
        ExprP::TupleProjection(t, ix) => match &t.pattern {
            ExprP::FunApp(FunKind::CompilerInternal(Internal::MakeTuple), ts) => {
                ts[ix - 1].pattern.clone()
            }
            _ => ExprP::TupleProjection(Box::new(eval_expr(t, false)?), *ix),
        },
        ExprP::Indexed(x, idx) => {
            let x = eval_expr(x, false)?;
            let idx = idx
                .iter()
                .map(|i| map_index(i, &mut |e| eval_expr(e, false)))
                .collect::<Result<Vec<_>, _>>()?;
            ExprP::Indexed(Box::new(x), idx)
        }
    };
    Ok(Expr {
        pattern,
        meta: e.meta.clone(),
    })
}

/// `try_eval_expr`: the partially evaluated expression, or the expression itself when its
/// evaluation fails.
pub fn try_eval_expr(e: &Expr, ps: bool) -> Expr {
    eval_expr(e, ps).unwrap_or_else(|_| e.clone())
}

fn map_index<E>(
    i: &Index<Expr>,
    f: &mut impl FnMut(&Expr) -> Result<Expr, E>,
) -> Result<Index<Expr>, E> {
    Ok(match i {
        Index::All => Index::All,
        Index::Single(e) => Index::Single(f(e)?),
        Index::Upfrom(e) => Index::Upfrom(f(e)?),
        Index::MultiIndex(e) => Index::MultiIndex(f(e)?),
        Index::Between(a, b) => Index::Between(f(a)?, f(b)?),
    })
}

fn is_multi(i: &Index<Expr>) -> bool {
    !matches!(i, Index::Single(_))
}

/// Folds a single index into a multi-index before it (`simplify_index_expr`).
fn simplify_index_expr(p: ExprP) -> ExprP {
    if let ExprP::Indexed(obj, outer) = &p
        && let ExprP::Indexed(inner_obj, inner) = &obj.pattern
        && let Some(Index::Single(single_e)) = outer.first()
        && single_e.meta.ty == UnsizedType::Int
        && inner.iter().any(is_multi)
    {
        let k = inner.iter().position(is_multi).unwrap();
        let single = Index::Single(single_e.clone());
        let mut idx: Vec<Index<Expr>> = inner[..k].to_vec();
        match &inner[k] {
            Index::MultiIndex(first) => {
                let mut meta = obj.meta.clone();
                meta.ty = UnsizedType::Int;
                idx.push(Index::Single(Expr::new(
                    ExprP::Indexed(Box::new(first.clone()), vec![single]),
                    meta,
                )));
            }
            Index::All => idx.push(single),
            Index::Between(bot, _) | Index::Upfrom(bot) => idx.push(Index::Single(Expr::binop(
                Expr::binop(bot.clone(), Operator::Plus, single_e.clone()),
                Operator::Minus,
                Expr::int(1),
            ))),
            Index::Single(_) => unreachable!(),
        }
        idx.extend(inner[k + 1..].iter().cloned());
        let new_obj = Expr::new(ExprP::Indexed(inner_obj.clone(), idx), obj.meta.clone());
        return simplify_index_expr(ExprP::Indexed(Box::new(new_obj), outer[1..].to_vec()));
    }
    p
}

/// Known sizes of declared variables, shared like stanc3's mutable arrays.
type Sizes = Rc<RefCell<Vec<Option<Expr>>>>;

#[derive(Clone, Default)]
struct DeclSizes {
    known: HashMap<String, Sizes>,
    deps: HashMap<String, Vec<(String, usize)>>,
}

impl DeclSizes {
    fn copy(&self) -> DeclSizes {
        DeclSizes {
            known: self
                .known
                .iter()
                .map(|(k, v)| (k.clone(), Rc::new(RefCell::new(v.borrow().clone()))))
                .collect(),
            deps: self.deps.clone(),
        }
    }

    fn add(&mut self, name: &str, sizes: Vec<&Expr>) {
        let sizes: Vec<Option<Expr>> = sizes
            .into_iter()
            .map(|d| {
                if cannot_duplicate_expr(d) {
                    None
                } else {
                    Some(d.clone())
                }
            })
            .collect();
        if !sizes.iter().any(Option::is_some) {
            return;
        }
        for (i, d) in sizes.iter().enumerate() {
            if let Some(d) = d {
                for v in expr_var_names(d) {
                    self.deps
                        .entry(v)
                        .or_default()
                        .insert(0, (name.to_string(), i));
                }
            }
        }
        self.known
            .insert(name.to_string(), Rc::new(RefCell::new(sizes)));
    }

    fn erase_var(&mut self, v: &str) {
        if let Some(l) = self.deps.remove(v) {
            for (name, i) in l {
                if let Some(a) = self.known.get(&name) {
                    a.borrow_mut()[i] = None;
                }
            }
        }
    }

    fn erase_stmt(&mut self, s: &Stmt) {
        if let StmtP::Assignment(lv, ..) = &s.pattern {
            self.erase_var(lv.variable());
        } else {
            for c in s.children() {
                self.erase_stmt(c);
            }
        }
    }

    fn expand_indices(&self, name: &str, idx: Vec<Index<Expr>>) -> Vec<Index<Expr>> {
        let idx: Vec<Index<Expr>> = match self.known.get(name) {
            None => idx,
            Some(s) => {
                let s = s.borrow();
                idx.into_iter()
                    .enumerate()
                    .map(|(i, x)| match (s.get(i), x) {
                        (Some(Some(s)), Index::Between(l, u)) if expr_eq(s, &u) => Index::Upfrom(l),
                        (_, x) => x,
                    })
                    .collect()
            }
        };
        idx.into_iter()
            .map(|i| match i {
                Index::Upfrom(n) if expr_eq(&n, &Expr::int(1)) => Index::All,
                i => i,
            })
            .collect()
    }

    /// `simplify_indices_expr`.
    fn simplify(&self, e: &Expr) -> Expr {
        let mut p = e.pattern.clone();
        if let ExprP::Indexed(obj, idx) = p {
            let idx = match &obj.pattern {
                ExprP::Var(name) => self.expand_indices(name, idx),
                _ => idx,
            };
            let mut idx = idx;
            while matches!(idx.last(), Some(Index::All)) {
                idx.pop();
            }
            p = ExprP::Indexed(obj, idx);
        }
        let mut e = Expr::new(simplify_index_expr(p), e.meta.clone());
        for c in e.children_mut() {
            *c = self.simplify(c);
        }
        e
    }
}

/// The variable names of an expression (`expr_var_names_set`).
pub fn expr_var_names(e: &Expr) -> BTreeSet<String> {
    let mut v = Vec::new();
    crate::dataflow::expr_vars(e, &mut v);
    v.into_iter().map(|(_, n)| n).collect()
}

/// Whether evaluating the expression twice may differ from evaluating it once
/// (`cannot_duplicate_expr`).
pub fn cannot_duplicate_expr(e: &Expr) -> bool {
    e.any(&mut |e: &Expr| match &e.pattern {
        ExprP::FunApp(k, _) => match k {
            FunKind::UserDefined(_, Suffix::Target | Suffix::Jacobian | Suffix::Rng) => true,
            FunKind::StanLib(_, Suffix::Jacobian | Suffix::Rng | Suffix::Target) => true,
            FunKind::CompilerInternal(i) => i.can_side_effect(),
            _ => false,
        },
        _ => false,
    })
}

/// The dimensions of a sized type, outermost first (`SizedType.get_dims`).
fn get_dims(st: &SizedType) -> Vec<&Expr> {
    use SizedType::*;
    match st {
        Tuple(_) | Int | Real | Complex => vec![],
        Matrix(a, b) | ComplexMatrix(a, b) => vec![a, b],
        RowVector(d) | Vector(d) | ComplexRowVector(d) | ComplexVector(d) => vec![d],
        Array(t, d) => {
            let mut v = vec![d];
            v.extend(get_dims(t));
            v
        }
    }
}

fn eval_stmt(info: DeclSizes, s: &Stmt) -> (DeclSizes, Stmt) {
    let orig = info.clone();
    match eval_stmt_inner(info, s) {
        Ok(r) => r,
        Err(Rejected(loc, m)) => (
            orig,
            Stmt::new(
                StmtP::NRFunApp(
                    FunKind::CompilerInternal(Internal::Reject),
                    vec![Expr::str(&m)],
                ),
                loc,
            ),
        ),
    }
}

fn eval_stmt_inner(mut info: DeclSizes, s: &Stmt) -> Result<(DeclSizes, Stmt), Rejected> {
    let expr = |info: &DeclSizes, e: &Expr| eval_expr(&info.simplify(e), false);
    let w = |p: StmtP| Stmt::new(p, s.loc);
    let stmts = |mut info: DeclSizes, ss: &[Stmt]| {
        let mut out = Vec::new();
        for c in ss {
            let (i, c) = eval_stmt(info, c);
            info = i;
            out.push(c);
        }
        (info, out)
    };
    Ok(match &s.pattern {
        StmtP::Decl {
            ad,
            id,
            ty: DeclType::Sized(st),
            init,
        } => {
            let mut st = st.clone();
            for e in st.exprs_mut() {
                *e = expr(&info, e)?;
            }
            let init = match init {
                DeclInit::Assign(e) => DeclInit::Assign(expr(&info, e)?),
                i => i.clone(),
            };
            info.add(id, get_dims(&st));
            let p = StmtP::Decl {
                ad: ad.clone(),
                id: id.clone(),
                ty: DeclType::Sized(st),
                init,
            };
            (info, w(p))
        }
        StmtP::Assignment(lv, ty, value) => {
            let mut lv = lv.clone();
            if let LBase::Variable(name) = &lv.base {
                let idx = lv
                    .idx
                    .iter()
                    .map(|i| map_index(i, &mut |e| expr(&info, e)))
                    .collect::<Result<Vec<_>, _>>()?;
                lv.idx = info.expand_indices(name, idx);
            } else {
                for e in lv.exprs_mut() {
                    *e = expr(&info, e)?;
                }
            }
            let value = expr(&info, value)?;
            info.erase_var(lv.variable());
            (info, w(StmtP::Assignment(lv, ty.clone(), value)))
        }
        StmtP::SList(ss) => {
            let (info, ss) = stmts(info, ss);
            (info, w(StmtP::SList(ss)))
        }
        StmtP::Block(ss) => {
            let (_, ss) = stmts(info.clone(), ss);
            (info, w(StmtP::Block(ss)))
        }
        StmtP::Profile(n, ss) => {
            let (_, ss) = stmts(info.clone(), ss);
            (info, w(StmtP::Profile(n.clone(), ss)))
        }
        StmtP::For {
            loopvar,
            lower,
            upper,
            body,
        } => {
            let lower = expr(&info, lower)?;
            info.erase_stmt(body);
            let upper = expr(&info, upper)?;
            let (info, body) = eval_stmt(info, body);
            let p = StmtP::For {
                loopvar: loopvar.clone(),
                lower,
                upper,
                body: Box::new(body),
            };
            (info, w(p))
        }
        StmtP::While(cond, body) => {
            info.erase_stmt(body);
            let (info, body) = eval_stmt(info, body);
            let cond = expr(&info, cond)?;
            (info, w(StmtP::While(cond, Box::new(body))))
        }
        StmtP::IfElse(cond, thn, None) => {
            let cond = expr(&info, cond)?;
            let (info, thn) = eval_stmt(info, thn);
            (info, w(StmtP::IfElse(cond, Box::new(thn), None)))
        }
        StmtP::IfElse(cond, thn, Some(els)) => {
            let cond = expr(&info, cond)?;
            let (_, thn) = eval_stmt(info.copy(), thn);
            let (mut info, els) = eval_stmt(info, els);
            info.erase_stmt(&thn);
            let p = StmtP::IfElse(cond, Box::new(thn), Some(Box::new(els)));
            (info, w(p))
        }
        _ => {
            let mut s2 = s.clone();
            for e in s2.exprs_mut() {
                *e = expr(&info, e)?;
            }
            (info, s2)
        }
    })
}

/// stanc3's `transform_program`: the transformation applied to all blocks packed in one
/// statement list, and separately to each function body.
pub(crate) fn transform_program(p: &mut Program, f: &impl Fn(&Stmt) -> Stmt) {
    let slist = |ss: Vec<Stmt>| Stmt::new(StmtP::SList(ss), Loc::EMPTY);
    let packed = slist(vec![
        slist(std::mem::take(&mut p.prepare_data)),
        slist(std::mem::take(&mut p.transform_inits)),
        slist(std::mem::take(&mut p.log_prob)),
        slist(std::mem::take(&mut p.reverse_mode_log_prob)),
        slist(std::mem::take(&mut p.generate_quantities)),
    ]);
    let packed = f(&f(&packed));
    let StmtP::SList(mut blocks) = packed.pattern else {
        unreachable!()
    };
    let mut take = |i: usize| match std::mem::replace(&mut blocks[i].pattern, StmtP::Skip) {
        StmtP::SList(ss) => ss,
        _ => unreachable!(),
    };
    p.prepare_data = take(0);
    p.transform_inits = take(1);
    p.log_prob = take(2);
    p.reverse_mode_log_prob = take(3);
    p.generate_quantities = take(4);
    for fd in &mut p.functions {
        if let Some(b) = &fd.body {
            fd.body = Some(f(b));
        }
    }
}

/// `partial_evaluation`.
pub fn partial_evaluation(p: &mut Program) {
    transform_program(p, &|s| eval_stmt(DeclSizes::default(), s).1);
    for (_, _, st) in &mut p.input_vars {
        for e in st.exprs_mut() {
            *e = try_eval_expr(e, false);
        }
    }
    for (_, _, o) in &mut p.output_vars {
        for e in o.unconstrained.exprs_mut() {
            *e = try_eval_expr(e, false);
        }
        for e in o.constrained.exprs_mut() {
            *e = try_eval_expr(e, false);
        }
        for e in o.trans.exprs_mut() {
            *e = try_eval_expr(e, false);
        }
    }
}

/// Partial functions from variables to expressions, ordered by inverse inclusion, with a
/// bottom element (`dual_partial_function_lattice_with_bot`).
struct PartialFns;

type PMap = Option<BTreeMap<String, Expr>>;

impl Lattice for PartialFns {
    type P = PMap;
    fn bottom(&self) -> PMap {
        None
    }
    fn initial(&self) -> PMap {
        Some(BTreeMap::new())
    }
    fn lub(&self, a: &PMap, b: &PMap) -> PMap {
        match (a, b) {
            (Some(a), Some(b)) => Some(
                a.iter()
                    .filter(|(k, v)| b.get(*k) == Some(v))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            ),
            (Some(a), None) => Some(a.clone()),
            (None, b) => b.clone(),
        }
    }
    fn leq(&self, a: &PMap, b: &PMap) -> bool {
        match (a, b) {
            (Some(a), Some(b)) => b.iter().all(|(k, v)| a.get(k) == Some(v)),
            (Some(_), None) => false,
            (None, _) => true,
        }
    }
}

/// `subst_expr`: replaces variables by their values.
fn subst_expr(m: &BTreeMap<String, Expr>, e: &mut Expr) {
    if let ExprP::Var(v) = &e.pattern
        && let Some(x) = m.get(v)
    {
        *e = x.clone();
        return;
    }
    for c in e.children_mut() {
        subst_expr(m, c);
    }
}

/// `subst_stmt_base`: substitutes into the expressions directly in a statement; of an
/// assignment, only the outer indices and the value.
pub(crate) fn subst_stmt_base(m: &BTreeMap<String, Expr>, s: &mut Stmt) {
    if m.is_empty() {
        return;
    }
    if let StmtP::Assignment(lv, _, e) = &mut s.pattern {
        for i in &mut lv.idx {
            for x in i.exprs_mut() {
                subst_expr(m, x);
            }
        }
        subst_expr(m, e);
    } else {
        for x in s.exprs_mut() {
            subst_expr(m, x);
        }
    }
}

/// The variables declared at the top of a statement list (`label_top_decls`).
fn top_decls(s: &Stmt, out: &mut Vec<String>) {
    match &s.pattern {
        StmtP::Decl { id, .. } => out.push(id.clone()),
        StmtP::SList(ss) => ss.iter().for_each(|c| top_decls(c, out)),
        _ => {}
    }
}

/// `propagation`: substitutes into each statement the values the transfer function proves
/// for its entry.
fn propagation(
    s: &Stmt,
    transfer: &impl Fn(&Stmt, &BTreeMap<String, Expr>) -> BTreeMap<String, Expr>,
) -> Stmt {
    let g = Graph::build(s, false);
    let res = mfp(&g, &PartialFns, |l, p| {
        p.as_ref().map(|m| transfer(g.nodes[l], m))
    });
    let mut out = s.clone();
    let mut label = 0;
    fn walk(s: &mut Stmt, label: &mut usize, res: &[crate::dataflow::EntryExit<PMap>]) {
        let l = *label;
        *label += 1;
        for c in s.children_mut() {
            walk(c, label, res);
        }
        if let Some(m) = &res[l].entry {
            subst_stmt_base(m, s);
        }
    }
    walk(&mut out, &mut label, &res);
    out
}

/// `UnsizedType.is_autodiffable`.
pub fn is_autodiffable(t: &UnsizedType) -> bool {
    use UnsizedType::*;
    match t {
        Real | Vector | RowVector | Matrix | Complex | ComplexVector | ComplexRowVector
        | ComplexMatrix => true,
        Array(t) => is_autodiffable(t),
        Tuple(ts) => ts.iter().any(is_autodiffable),
        _ => false,
    }
}

/// `constant_propagation`; with `preserve_stability`, only integer values propagate.
pub fn constant_propagation(p: &mut Program, preserve_stability: bool) {
    let transfer = |s: &Stmt, m: &BTreeMap<String, Expr>| {
        let mut m = m.clone();
        match &s.pattern {
            StmtP::Assignment(lv, t, e)
                if lv.idx.is_empty() && matches!(lv.base, LBase::Variable(_)) =>
            {
                let mut e = e.clone();
                subst_expr(&m, &mut e);
                let e = try_eval_expr(&e, false);
                let is_lit = match &e.pattern {
                    ExprP::Lit(..) => true,
                    ExprP::Promotion(x, ..) => matches!(x.pattern, ExprP::Lit(..)),
                    _ => false,
                };
                if is_lit && !(preserve_stability && is_autodiffable(t)) {
                    m.insert(lv.variable().to_string(), e);
                } else {
                    m.remove(lv.variable());
                }
            }
            StmtP::Decl { id, .. } => {
                m.remove(id);
            }
            StmtP::Assignment(lv, ..) => {
                m.remove(lv.variable());
            }
            _ => {}
        }
        m
    };
    transform_program(p, &|s| propagation(s, &transfer));
}

fn program_var_declarations(ss: &[Stmt], out: &mut BTreeSet<String>) {
    for s in ss {
        match &s.pattern {
            StmtP::Decl { id, .. } => {
                out.insert(id.clone());
            }
            StmtP::IfElse(..)
            | StmtP::While(..)
            | StmtP::For { .. }
            | StmtP::Block(_)
            | StmtP::SList(_)
            | StmtP::Profile(..) => {
                let cs: Vec<Stmt> = s.children().into_iter().cloned().collect();
                program_var_declarations(&cs, out);
            }
            _ => {}
        }
    }
}

/// `copy_propagation`.
pub fn copy_propagation(p: &mut Program) {
    let mut globals: BTreeSet<String> = p.output_vars.iter().map(|(n, ..)| n.clone()).collect();
    globals.extend(p.input_vars.iter().map(|(n, ..)| n.clone()));
    program_var_declarations(&p.prepare_data, &mut globals);
    globals.insert("target".to_string());
    let transfer = |s: &Stmt, m: &BTreeMap<String, Expr>| {
        let kill = |m: &mut BTreeMap<String, Expr>, v: &str| {
            m.retain(|k, d| k != v && !matches!(&d.pattern, ExprP::Var(x) if x == v));
        };
        let mut m = m.clone();
        match &s.pattern {
            StmtP::Assignment(lv, _, e)
                if lv.idx.is_empty()
                    && matches!(lv.base, LBase::Variable(_))
                    && matches!(e.pattern, ExprP::Var(_)) =>
            {
                let a = lv.variable();
                if !globals.contains(a) {
                    kill(&mut m, a);
                    m.insert(a.to_string(), e.clone());
                }
            }
            StmtP::Decl { id, .. } => kill(&mut m, id),
            StmtP::Assignment(lv, ..) => kill(&mut m, lv.variable()),
            StmtP::Profile(_, ss) | StmtP::Block(ss) => {
                let mut ks = Vec::new();
                ss.iter().for_each(|c| top_decls(c, &mut ks));
                for k in ks {
                    kill(&mut m, &k);
                }
            }
            _ => {}
        }
        m
    };
    transform_program(p, &|s| propagation(s, &transfer));
}

/// `can_side_effect_top_expr`.
fn can_side_effect_top(e: &Expr) -> bool {
    match &e.pattern {
        ExprP::FunApp(FunKind::UserDefined(_, Suffix::Target | Suffix::Jacobian), _)
        | ExprP::FunApp(FunKind::StanLib(_, Suffix::Jacobian), _) => true,
        ExprP::FunApp(FunKind::CompilerInternal(f), _) => f.can_side_effect(),
        _ => false,
    }
}

/// `Mir_utils.expr_any`: the predicate is not applied to an indexing itself.
fn expr_any(e: &Expr, pred: &impl Fn(&Expr) -> bool) -> bool {
    if !matches!(e.pattern, ExprP::Indexed(..)) && pred(e) {
        return true;
    }
    e.children().into_iter().any(|c| expr_any(c, pred))
}

/// `cannot_remove_expr`.
fn cannot_remove_expr(e: &Expr) -> bool {
    expr_any(e, &can_side_effect_top)
}

/// `free_vars_expr`: the variables and called user-defined functions.
fn free_vars_expr(e: &Expr, out: &mut BTreeSet<String>) {
    match &e.pattern {
        ExprP::Var(x) | ExprP::FunApp(FunKind::UserDefined(x, _), _) => {
            out.insert(x.clone());
        }
        _ => {}
    }
    for c in e.children() {
        free_vars_expr(c, out);
    }
}

/// `top_free_vars_stmt`: the variables a statement reads, without its sub-statements.
fn top_free_vars(s: &Stmt) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    match &s.pattern {
        StmtP::Decl {
            init: DeclInit::Assign(e),
            ..
        }
        | StmtP::Return(Some(e))
        | StmtP::TargetPE(e)
        | StmtP::JacobianPE(e)
        | StmtP::IfElse(e, ..)
        | StmtP::While(e, _) => free_vars_expr(e, &mut out),
        StmtP::Assignment(lv, _, e) => {
            free_vars_expr(e, &mut out);
            for x in lv.exprs() {
                free_vars_expr(x, &mut out);
            }
        }
        StmtP::NRFunApp(k, args) => {
            if let FunKind::UserDefined(f, _) = k {
                out.insert(f.clone());
            }
            if let FunKind::CompilerInternal(f) = k {
                for x in f.exprs() {
                    free_vars_expr(x, &mut out);
                }
            }
            for x in args {
                free_vars_expr(x, &mut out);
            }
        }
        StmtP::For { lower, upper, .. } => {
            free_vars_expr(lower, &mut out);
            free_vars_expr(upper, &mut out);
        }
        _ => {}
    }
    out
}

/// The variables that are always observable (`Monotone_framework.globals`): output and input
/// variables, the variables declared in `prepare_data`, and `target`.
fn globals(p: &Program) -> BTreeSet<String> {
    let mut g: BTreeSet<String> = p.output_vars.iter().map(|(n, ..)| n.clone()).collect();
    g.extend(p.input_vars.iter().map(|(n, ..)| n.clone()));
    program_var_declarations(&p.prepare_data, &mut g);
    g.insert("target".to_string());
    g
}

/// Applies `f` bottom up with each statement's preorder label (`map_rec_stmt_loc_num`).
fn map_labeled(s: &mut Stmt, label: &mut usize, f: &mut impl FnMut(usize, &mut Stmt)) {
    let l = *label;
    *label += 1;
    for c in s.children_mut() {
        map_labeled(c, label, f);
    }
    f(l, s);
}

/// `dead_code_elimination`: removes assignments to variables that are not live afterwards,
/// and statements that do nothing.
pub fn dead_code_elimination(p: &mut Program) {
    let never_kill = globals(p);
    let transform = |s: &Stmt| -> Stmt {
        let g = Graph::build(s, false).reversed();
        let live = mfp(
            &g,
            &Powerset(never_kill.clone()),
            |l, p: &BTreeSet<String>| {
                let node = g.nodes[l];
                let kill: Option<&str> = match &node.pattern {
                    StmtP::Assignment(lv, ..)
                        if lv.idx.is_empty() && matches!(lv.base, LBase::Variable(_)) =>
                    {
                        Some(lv.variable())
                    }
                    StmtP::Decl { id, .. } => Some(id),
                    _ => None,
                };
                let mut out = top_free_vars(node);
                out.extend(
                    p.iter()
                        .filter(|v| Some(v.as_str()) != kill || never_kill.contains(*v))
                        .cloned(),
                );
                out
            },
        );
        let mut out = s.clone();
        let is_skip = |s: &Stmt| s.pattern == StmtP::Skip;
        map_labeled(&mut out, &mut 0, &mut |l, st| {
            let live = &live[l].entry;
            let p = std::mem::replace(&mut st.pattern, StmtP::Skip);
            st.pattern = match p {
                StmtP::Assignment(lv, t, rhs) => {
                    if live.contains(lv.variable())
                        || cannot_remove_expr(&rhs)
                        || lv
                            .idx
                            .iter()
                            .any(|i| i.exprs().into_iter().any(cannot_remove_expr))
                    {
                        StmtP::Assignment(lv, t, rhs)
                    } else {
                        StmtP::Skip
                    }
                }
                StmtP::Decl {
                    ad,
                    id,
                    ty,
                    init: DeclInit::Assign(e),
                } => {
                    let init = if live.contains(&id) || cannot_remove_expr(&e) {
                        DeclInit::Assign(e)
                    } else {
                        DeclInit::Uninit
                    };
                    StmtP::Decl { ad, id, ty, init }
                }
                StmtP::IfElse(e, b1, b2) => {
                    if !cannot_remove_expr(&e)
                        && is_skip(&b1)
                        && b2.as_ref().is_none_or(|b| is_skip(b))
                    {
                        StmtP::Skip
                    } else if is_int(0, &e) {
                        b2.map_or(StmtP::Skip, |b| b.pattern)
                    } else if matches!(e.pattern, ExprP::Lit(..)) {
                        b1.pattern
                    } else {
                        StmtP::IfElse(e, b1, b2)
                    }
                }
                StmtP::While(e, b) => {
                    if (!cannot_remove_expr(&e) && b.pattern == StmtP::Break) || is_int(0, &e) {
                        StmtP::Skip
                    } else {
                        StmtP::While(e, b)
                    }
                }
                StmtP::For {
                    loopvar,
                    lower,
                    upper,
                    body,
                } => {
                    if !cannot_remove_expr(&lower)
                        && !cannot_remove_expr(&upper)
                        && matches!(body.pattern, StmtP::Skip | StmtP::Break | StmtP::Continue)
                    {
                        StmtP::Skip
                    } else {
                        StmtP::For {
                            loopvar,
                            lower,
                            upper,
                            body,
                        }
                    }
                }
                StmtP::Profile(n, l) => {
                    let l: Vec<Stmt> = l.into_iter().filter(|x| !is_skip(x)).collect();
                    if l.is_empty() {
                        StmtP::Skip
                    } else {
                        StmtP::Profile(n, l)
                    }
                }
                StmtP::Block(l) => {
                    let l: Vec<Stmt> = l.into_iter().filter(|x| !is_skip(x)).collect();
                    if l.is_empty() {
                        StmtP::Skip
                    } else {
                        StmtP::Block(l)
                    }
                }
                StmtP::SList(l) => StmtP::SList(l.into_iter().filter(|x| !is_skip(x)).collect()),
                p => p,
            };
        });
        out
    };
    transform_program(p, &transform);
}

/// `transform_program_blockwise`: the transformation applied to each function body and to
/// each block (except `unconstrain_array`) on its own.
fn transform_program_blockwise(p: &mut Program, f: &impl Fn(Option<&FunDef>, Stmt) -> Stmt) {
    let functions = std::mem::take(&mut p.functions);
    p.functions = functions
        .into_iter()
        .map(|mut fd| {
            if let Some(b) = fd.body.take() {
                fd.body = Some(f(Some(&fd), b));
            }
            fd
        })
        .collect();
    for b in [
        &mut p.prepare_data,
        &mut p.transform_inits,
        &mut p.log_prob,
        &mut p.reverse_mode_log_prob,
        &mut p.generate_quantities,
    ] {
        let s = f(None, Stmt::new(StmtP::SList(std::mem::take(b)), Loc::EMPTY));
        let StmtP::SList(l) = s.pattern else {
            panic!("a transformed block is not a statement list")
        };
        *b = l;
    }
}

/// Applies `f` to every statement, bottom up.
fn map_rec(s: &mut Stmt, f: &impl Fn(StmtP) -> StmtP) {
    for c in s.children_mut() {
        map_rec(c, f);
    }
    let p = std::mem::replace(&mut s.pattern, StmtP::Skip);
    s.pattern = f(p);
}

/// `list_collapsing`: splices statement lists into the enclosing lists and blocks.
pub fn list_collapsing(p: &mut Program) {
    fn collapse(l: Vec<Stmt>) -> Vec<Stmt> {
        let mut out = Vec::new();
        for x in l {
            match x.pattern {
                StmtP::SList(inner) => out.extend(inner),
                _ => out.push(x),
            }
        }
        out
    }
    transform_program_blockwise(p, &|_, mut s| {
        map_rec(&mut s, &|p| match p {
            StmtP::Block(l) => StmtP::Block(collapse(l)),
            StmtP::SList(l) => StmtP::SList(collapse(l)),
            p => p,
        });
        s
    });
}

/// `block_fixing`: makes statement lists under conditionals and loops into blocks.
pub fn block_fixing(p: &mut Program) {
    let fix = |b: &mut Stmt| {
        if let StmtP::SList(l) = &mut b.pattern {
            b.pattern = StmtP::Block(std::mem::take(l));
        }
    };
    transform_program_blockwise(p, &|_, mut s| {
        map_rec(&mut s, &|mut p| {
            match &mut p {
                StmtP::IfElse(_, t, e) => {
                    fix(t);
                    if let Some(e) = e {
                        fix(e);
                    }
                }
                StmtP::While(_, b) | StmtP::For { body: b, .. } => fix(b),
                _ => {}
            }
            p
        });
        s
    });
}

/// `allow_uninitialized_decls`: declarations that the next statement assigns as a whole need
/// no initialization.
pub fn allow_uninitialized_decls(p: &mut Program) {
    fn find_assignment_idx<'a>(name: &str, s: &'a Stmt) -> Option<&'a [Index<Expr>]> {
        match &s.pattern {
            StmtP::Assignment(lv, ut, rhs)
                if lv.variable() == name
                    && !expr_var_names(rhs).contains(name)
                    && !(rhs.meta.ad == stanc_rs_semantic::types::Ad::DataOnly
                        && matches!(ut, UnsizedType::Array(_))) =>
            {
                Some(&lv.idx)
            }
            _ => None,
        }
    }
    fn patt(s: &mut Stmt, next: Option<&Stmt>) {
        match &mut s.pattern {
            StmtP::Decl {
                id,
                init: init @ DeclInit::Default,
                ..
            } => {
                if let Some(idx) = next.and_then(|n| find_assignment_idx(id, n))
                    && idx.iter().all(|i| matches!(i, Index::All))
                {
                    *init = DeclInit::Uninit;
                }
            }
            StmtP::Block(l) | StmtP::SList(l) | StmtP::Profile(_, l) => unenforce(l),
            StmtP::While(_, b) | StmtP::For { body: b, .. } => patt(b, None),
            StmtP::IfElse(_, t, e) => {
                if let Some(e) = e {
                    patt(e, None);
                }
                patt(t, None);
            }
            _ => {}
        }
    }
    fn unenforce(l: &mut [Stmt]) {
        for i in 0..l.len() {
            let (head, tail) = l.split_at_mut(i + 1);
            patt(&mut head[i], tail.first());
        }
    }
    for fd in &mut p.functions {
        if let Some(Stmt {
            pattern: StmtP::SList(l) | StmtP::Block(l),
            ..
        }) = &mut fd.body
        {
            unenforce(l);
        }
    }
    for b in [
        &mut p.prepare_data,
        &mut p.log_prob,
        &mut p.reverse_mode_log_prob,
        &mut p.generate_quantities,
        &mut p.transform_inits,
        &mut p.unconstrain_array,
    ] {
        unenforce(b);
    }
}

/// stanc3's `--O1` optimization suite, applied to the transformed MIR.
pub fn optimization_suite_o1(p: &mut Program) {
    crate::inline::function_inlining(p);
    constant_propagation(p, false);
    copy_propagation(p);
    constant_propagation(p, false);
    dead_code_elimination(p);
    partial_evaluation(p);
    copy_propagation(p);
    constant_propagation(p, false);
    dead_code_elimination(p);
    list_collapsing(p);
    crate::soa::optimize_soa(p);
    allow_uninitialized_decls(p);
    block_fixing(p);
}

/// The passes stanc3 runs before its pedantic analysis: constant and copy propagation and the
/// partial evaluator, in `optimization_suite` order, preserving numerical stability.
pub fn pedantic_suite(p: &mut Program) {
    constant_propagation(p, true);
    copy_propagation(p);
    constant_propagation(p, true);
    partial_evaluation(p);
    copy_propagation(p);
    constant_propagation(p, true);
}

#[cfg(test)]
mod tests {
    use super::float_to_string;

    #[test]
    fn floats_print_like_ocaml() {
        assert_eq!(float_to_string(1.0), "1.");
        assert_eq!(float_to_string(0.5), "0.5");
        assert_eq!(float_to_string(-2.0), "-2.");
        assert_eq!(float_to_string(0.1), "0.1");
        assert_eq!(float_to_string(1e20), "1e+20");
        assert_eq!(float_to_string(1.0 / 3.0), "0.33333333333333331");
        assert_eq!(float_to_string(f64::INFINITY), "inf");
        assert_eq!(float_to_string(1.5e-7), "1.5e-07");
    }
}
