//! Expressions: values, places, conversions, operators, indexing, and calls.

use stanc_rs_semantic::typed::{FunKind, TExpr, TExprKind, TIndex, TLValue, TLValueKind};
use stanc_rs_semantic::types::{Ad, Promotion};
use stanc_rs_syntax::ast::{BinOp, FnSuffix, PrefixOp, UnsizedType as U};

use crate::library::{self, Pass, Shape};
use crate::{Gen, R, mangle, unsupported};

/// The autodiff level of slot `i` of a tuple value.
pub fn slot_ad(ad: &Ad, i: usize) -> Ad {
    match ad {
        Ad::Tuple(v) => v.get(i).cloned().unwrap_or(Ad::DataOnly),
        a => a.clone(),
    }
}

/// Scalar type name for an autodiff level.
pub fn scalar(ad: &Ad) -> &'static str {
    if ad.has_autodiff() { "T" } else { "f64" }
}

/// The Rust type of a Stan value.
pub fn rust_type(ty: &U, ad: &Ad) -> String {
    let s = scalar(ad);
    match ty {
        U::Int => "i64".into(),
        U::Real => s.into(),
        U::Complex => format!("sm::Complex<{s}>"),
        U::Vector | U::RowVector => format!("Vec<{s}>"),
        U::Matrix => format!("sm::Matrix<{s}>"),
        U::ComplexVector | U::ComplexRowVector => format!("Vec<sm::Complex<{s}>>"),
        U::ComplexMatrix => format!("sm::Matrix<sm::Complex<{s}>>"),
        U::Array(t) => format!("Vec<{}>", rust_type(t, ad)),
        U::Tuple(ts) => {
            let parts: Vec<String> = ts
                .iter()
                .enumerate()
                .map(|(i, t)| rust_type(t, &slot_ad(ad, i)))
                .collect();
            format!("({},)", parts.join(", "))
        }
        U::Function(_) | U::MathLibraryFunction => "()".into(),
    }
}

/// True for types whose Rust representation is `Copy`.
pub fn is_copy(ty: &U) -> bool {
    matches!(ty, U::Int | U::Real | U::Complex)
}

fn needs_lift(from: &Ad, to: &Ad) -> bool {
    to.has_autodiff() && !from.has_autodiff()
}

/// Element type of a vector-like type.
fn elem_of(ty: &U) -> U {
    match ty {
        U::Vector | U::RowVector | U::Matrix => U::Real,
        U::ComplexVector | U::ComplexRowVector | U::ComplexMatrix => U::Complex,
        U::Array(t) => (**t).clone(),
        t => t.clone(),
    }
}

fn real_literal(s: &str) -> String {
    let v: f64 = s.parse().unwrap_or(f64::NAN);
    if v.is_infinite() {
        "f64::INFINITY".into()
    } else if v.is_nan() {
        "f64::NAN".into()
    } else {
        format!("{v:?}_f64")
    }
}

impl Gen {
    /// A fresh suffix for temporaries.
    pub fn fresh(&mut self, base: &str) -> String {
        self.tmp += 1;
        format!("{base}{}__", self.tmp)
    }

    /// A variable reference.
    pub fn var(&self, name: &str) -> String {
        if self.fields.contains(name) {
            format!("self.{}", mangle(name))
        } else {
            mangle(name)
        }
    }

    /// Converts a value between Stan types: integers to reals, reals to complex numbers, and data
    /// to autodiff values.
    pub fn convert(&mut self, code: String, fty: &U, fad: &Ad, tty: &U, tad: &Ad) -> String {
        let lift = needs_lift(fad, tad);
        match (fty, tty) {
            (U::Array(fi), U::Array(ti)) => {
                if fi == ti && !lift && !matches!(**fi, U::Tuple(_)) {
                    return code;
                }
                let v = self.fresh("e");
                let inner = self.convert(v.clone(), fi, fad, ti, tad);
                if inner == v {
                    return code;
                }
                format!("{code}.into_iter().map(|{v}| {inner}).collect::<Vec<_>>()")
            }
            (U::Tuple(fs), U::Tuple(ts)) => {
                let t = self.fresh("t");
                let mut parts = Vec::new();
                let mut changed = false;
                for (i, (f, tt)) in fs.iter().zip(ts).enumerate() {
                    let slot = format!("{t}.{i}");
                    let c = self.convert(slot.clone(), f, &slot_ad(fad, i), tt, &slot_ad(tad, i));
                    changed |= c != slot;
                    parts.push(c);
                }
                if !changed {
                    return code;
                }
                format!("{{ let {t} = {code}; ({},) }}", parts.join(", "))
            }
            (U::Int, U::Real) => {
                let c = format!("(({code}) as f64)");
                if tad.has_autodiff() {
                    format!("T::from_f64({c})")
                } else {
                    c
                }
            }
            (U::Int, U::Complex) => {
                let c = format!("(({code}) as f64)");
                let c = if tad.has_autodiff() {
                    format!("T::from_f64({c})")
                } else {
                    c
                };
                format!("rt::stan::to_complex({c})")
            }
            (U::Real, U::Real) if lift => format!("T::from_f64({code})"),
            (U::Real, U::Complex) => {
                let c = if lift {
                    format!("T::from_f64({code})")
                } else {
                    code
                };
                format!("rt::stan::to_complex({c})")
            }
            (U::Complex, U::Complex) if lift => format!("rt::stan::lift_complex::<T>({code})"),
            (U::Vector | U::RowVector, U::Vector | U::RowVector) if lift => {
                format!("rt::stan::lift_vec::<T>(&{code})")
            }
            (U::Matrix, U::Matrix) if lift => format!("rt::stan::lift_mat::<T>(&{code})"),
            (U::Vector | U::RowVector, U::ComplexVector | U::ComplexRowVector) => {
                let c = if lift {
                    format!("rt::stan::lift_vec::<T>(&{code})")
                } else {
                    code
                };
                format!("rt::stan::to_complex_vec({c})")
            }
            (U::Matrix, U::ComplexMatrix) => {
                let c = if lift {
                    format!("rt::stan::lift_mat::<T>(&{code})")
                } else {
                    code
                };
                format!("rt::stan::to_complex_mat({c})")
            }
            (U::ComplexVector | U::ComplexRowVector, U::ComplexVector | U::ComplexRowVector)
                if lift =>
            {
                format!("rt::stan::lift_cvec::<T>(&{code})")
            }
            (U::ComplexMatrix, U::ComplexMatrix) if lift => {
                format!("rt::stan::lift_cmat::<T>(&{code})")
            }
            _ => code,
        }
    }

    /// The primal (`f64`) value of `code`, of type `ty` at level `ad`.
    pub fn to_f64(&mut self, code: String, ty: &U, ad: &Ad) -> String {
        if !ad.has_autodiff() {
            return code;
        }
        let cval = |c: &str| format!("sm::Complex::new({c}.re.value(), {c}.im.value())");
        match ty {
            U::Real => format!("({code}).value()"),
            U::Complex => {
                let t = self.fresh("c");
                format!("{{ let {t} = {code}; {} }}", cval(&t))
            }
            U::Vector | U::RowVector => format!("rt::check::vals(&{code})"),
            U::Matrix => format!("rt::check::mat_vals(&{code})"),
            U::ComplexVector | U::ComplexRowVector => {
                format!(
                    "({code}).iter().map(|c__| {}).collect::<Vec<_>>()",
                    cval("c__")
                )
            }
            U::ComplexMatrix => format!("({code}).map(|c__| {})", cval("c__")),
            U::Array(t) => {
                let v = self.fresh("e");
                let inner = self.to_f64(v.clone(), t, ad);
                format!("({code}).into_iter().map(|{v}| {inner}).collect::<Vec<_>>()")
            }
            U::Tuple(ts) => {
                let t = self.fresh("t");
                let parts: Vec<String> = ts
                    .iter()
                    .enumerate()
                    .map(|(i, st)| self.to_f64(format!("{t}.{i}"), st, &slot_ad(ad, i)))
                    .collect();
                format!("{{ let {t} = {code}; ({},) }}", parts.join(", "))
            }
            _ => code,
        }
    }

    /// The value of `e` converted to type `ty` at level `ad`.
    pub fn expr_as(&mut self, e: &TExpr, ty: &U, ad: &Ad) -> R<String> {
        let c = self.expr(e)?;
        Ok(self.convert(c, &e.ty, &e.ad, ty, ad))
    }

    /// The truth value of a condition.
    pub fn truthy(&mut self, e: &TExpr) -> R<String> {
        let c = self.expr(e)?;
        Ok(match e.ty {
            U::Int => format!("({c} != 0)"),
            _ => format!("({c} != 0.0)"),
        })
    }

    /// A readable place for `e` (a variable or single-indexed element), if it is one.
    pub fn place(&mut self, e: &TExpr) -> R<Option<String>> {
        match &e.kind {
            TExprKind::Variable { name, .. } => {
                if matches!(e.ty, U::Function(_) | U::MathLibraryFunction) {
                    return Ok(None);
                }
                Ok(Some(self.var(name)))
            }
            TExprKind::Indexed(base, idxs) => {
                if idxs.iter().any(TIndex::is_multi) {
                    return Ok(None);
                }
                let Some(mut p) = self.place(base)? else {
                    return Ok(None);
                };
                let mut ty = base.ty.clone();
                let mut i = 0;
                while i < idxs.len() {
                    let TIndex::Single(ix) = &idxs[i] else {
                        return Ok(None);
                    };
                    let ic = self.expr(ix)?;
                    match ty {
                        U::Matrix | U::ComplexMatrix => {
                            let Some(TIndex::Single(jx)) = idxs.get(i + 1) else {
                                return Ok(None);
                            };
                            let jc = self.expr(jx)?;
                            p = format!("(*rt::index::mat_at(&{p}, {ic}, {jc})?)");
                            ty = elem_of(&ty);
                            i += 2;
                            continue;
                        }
                        U::Array(_)
                        | U::Vector
                        | U::RowVector
                        | U::ComplexVector
                        | U::ComplexRowVector => {
                            p = format!("(*rt::index::at(&{p}, {ic})?)");
                            ty = elem_of(&ty);
                        }
                        _ => return Ok(None),
                    }
                    i += 1;
                }
                Ok(Some(p))
            }
            TExprKind::TupleProjection(x, k) => {
                Ok(self.place(x)?.map(|p| format!("{p}.{}", k - 1)))
            }
            _ => Ok(None),
        }
    }

    /// `&e` without copying when `e` is a place.
    pub fn expr_ref(&mut self, e: &TExpr) -> R<String> {
        if let Some(p) = self.place(e)? {
            return Ok(format!("&{p}"));
        }
        Ok(format!("&({})", self.expr(e)?))
    }

    /// The value of an expression.
    pub fn expr(&mut self, e: &TExpr) -> R<String> {
        if let Some(p) = self.place(e)? {
            return Ok(if is_copy(&e.ty) {
                p
            } else {
                format!("{p}.clone()")
            });
        }
        match &e.kind {
            TExprKind::Variable { name, .. } => Err(unsupported(
                e.span,
                format!("function '{name}' used as a value is not supported by stanc-rs yet"),
            )),
            TExprKind::IntLit(i) => Ok(format!("{i}_i64")),
            TExprKind::RealLit(s) => Ok(real_literal(s)),
            TExprKind::ImagLit(s) => Ok(format!("sm::Complex::new(0.0_f64, {})", real_literal(s))),
            TExprKind::FunApp {
                kind, name, args, ..
            } => self.call(Some(e), kind, name, args),
            TExprKind::GetTarget => Ok("(*target__)".into()),
            TExprKind::TernaryIf(c, a, b) => {
                let cc = self.truthy(c)?;
                let ac = self.expr_as(a, &e.ty, &e.ad)?;
                let bc = self.expr_as(b, &e.ty, &e.ad)?;
                Ok(format!("(if {cc} {{ {ac} }} else {{ {bc} }})"))
            }
            TExprKind::BinOp(l, op, r) => self.binop(e, l, *op, r),
            TExprKind::PrefixOp(op, x) => self.prefix(e, *op, x),
            TExprKind::Transpose(x) => {
                let c = self.expr_ref(x)?;
                Ok(match x.ty {
                    U::Matrix | U::ComplexMatrix => format!("rt::ops::transpose({c})"),
                    _ => format!("({c}).clone()"),
                })
            }
            TExprKind::Indexed(base, idxs) => {
                let b = match self.place(base)? {
                    Some(p) => p,
                    None => format!("({})", self.expr(base)?),
                };
                self.index(b, &base.ty, idxs)
            }
            TExprKind::ArrayExpr(es) => {
                let et = elem_of(&e.ty);
                let parts = es
                    .iter()
                    .map(|x| self.expr_as(x, &et, &e.ad))
                    .collect::<R<Vec<_>>>()?;
                Ok(format!("vec![{}]", parts.join(", ")))
            }
            TExprKind::RowVectorExpr(es) => match e.ty {
                U::Matrix | U::ComplexMatrix => {
                    let rt = if e.ty == U::Matrix {
                        U::RowVector
                    } else {
                        U::ComplexRowVector
                    };
                    let parts = es
                        .iter()
                        .map(|x| self.expr_as(x, &rt, &e.ad))
                        .collect::<R<Vec<_>>>()?;
                    Ok(format!(
                        "rt::stan::matrix_from_rows(vec![{}])?",
                        parts.join(", ")
                    ))
                }
                _ => {
                    let et = elem_of(&e.ty);
                    let parts = es
                        .iter()
                        .map(|x| self.expr_as(x, &et, &e.ad))
                        .collect::<R<Vec<_>>>()?;
                    Ok(format!("vec![{}]", parts.join(", ")))
                }
            },
            TExprKind::TupleExpr(es) => {
                let U::Tuple(ts) = &e.ty else {
                    unreachable!("tuple type")
                };
                let parts = es
                    .iter()
                    .zip(ts)
                    .enumerate()
                    .map(|(i, (x, t))| self.expr_as(x, t, &slot_ad(&e.ad, i)))
                    .collect::<R<Vec<_>>>()?;
                Ok(format!("({},)", parts.join(", ")))
            }
            TExprKind::TupleProjection(x, k) => Ok(format!("({}).{}", self.expr(x)?, k - 1)),
            TExprKind::Promotion(x, _) => {
                let c = self.expr(x)?;
                Ok(self.convert(c, &x.ty, &x.ad, &e.ty, &e.ad))
            }
        }
    }

    /// `rt::index::Idx` for one index.
    pub fn idx(&mut self, i: &TIndex) -> R<String> {
        Ok(match i {
            TIndex::All => "rt::index::Idx::All".into(),
            TIndex::Single(e) if e.ty == U::Int => {
                format!("rt::index::Idx::Single({})", self.expr(e)?)
            }
            TIndex::Single(e) => format!("rt::index::Idx::Multi({})", self.expr(e)?),
            TIndex::Upfrom(e) => format!("rt::index::Idx::Upfrom({})", self.expr(e)?),
            TIndex::Downfrom(e) => format!("rt::index::Idx::Downfrom({})", self.expr(e)?),
            TIndex::Between(a, b) => format!(
                "rt::index::Idx::Between({}, {})",
                self.expr(a)?,
                self.expr(b)?
            ),
        })
    }

    fn single(&mut self, i: &TIndex) -> R<Option<String>> {
        match i {
            TIndex::Single(e) if e.ty == U::Int => Ok(Some(self.expr(e)?)),
            _ => Ok(None),
        }
    }

    /// Indexing `base` (an expression usable after `&`) of type `ty`.
    pub fn index(&mut self, base: String, ty: &U, idxs: &[TIndex]) -> R<String> {
        if idxs.is_empty() {
            return Ok(base);
        }
        match ty {
            U::Array(inner) => {
                let rest = &idxs[1..];
                if let Some(i) = self.single(&idxs[0])? {
                    let v = format!("rt::index::at(&{base}, {i})?.clone()");
                    return self.index(v, inner, rest);
                }
                let ix = self.idx(&idxs[0])?;
                let v = format!("rt::index::multi(&{base}, &{ix})?");
                if rest.is_empty() {
                    return Ok(v);
                }
                let e = self.fresh("e");
                let inner_code = self.index(e.clone(), inner, rest)?;
                Ok(format!(
                    "rt::stan::try_map_vec({v}, |{e}| Ok({inner_code}))?"
                ))
            }
            U::Vector | U::RowVector | U::ComplexVector | U::ComplexRowVector => {
                if idxs.len() != 1 {
                    return Err(self.internal("too many indexes for a vector"));
                }
                if let Some(i) = self.single(&idxs[0])? {
                    return Ok(format!("(*rt::index::at(&{base}, {i})?)"));
                }
                let ix = self.idx(&idxs[0])?;
                Ok(format!("rt::index::multi(&{base}, &{ix})?"))
            }
            U::Matrix | U::ComplexMatrix => {
                if idxs.len() == 1 {
                    if let Some(i) = self.single(&idxs[0])? {
                        return Ok(format!("rt::index::row(&{base}, {i})?"));
                    }
                    let ix = self.idx(&idxs[0])?;
                    return Ok(format!("rt::index::rows_multi(&{base}, &{ix})?"));
                }
                if idxs.len() != 2 {
                    return Err(self.internal("too many indexes for a matrix"));
                }
                match (self.single(&idxs[0])?, self.single(&idxs[1])?) {
                    (Some(i), Some(j)) => Ok(format!("(*rt::index::mat_at(&{base}, {i}, {j})?)")),
                    (Some(i), None) => {
                        let jx = self.idx(&idxs[1])?;
                        Ok(format!("rt::index::row_multi(&{base}, {i}, &{jx})?"))
                    }
                    (None, Some(j)) => {
                        let ix = self.idx(&idxs[0])?;
                        Ok(format!("rt::index::col_multi(&{base}, &{ix}, {j})?"))
                    }
                    (None, None) => {
                        let ix = self.idx(&idxs[0])?;
                        let jx = self.idx(&idxs[1])?;
                        Ok(format!("rt::index::submatrix(&{base}, &{ix}, &{jx})?"))
                    }
                }
            }
            _ => Err(self.internal("indexing a value that is not a container")),
        }
    }

    fn prefix(&mut self, e: &TExpr, op: PrefixOp, x: &TExpr) -> R<String> {
        match op {
            PrefixOp::Plus => self.expr(x),
            PrefixOp::Not => {
                let c = self.expr(x)?;
                Ok(match x.ty {
                    U::Int => format!("(({c} == 0) as i64)"),
                    _ => format!("(({c} == 0.0) as i64)"),
                })
            }
            PrefixOp::Minus => {
                let c = self.expr_as(x, &x.ty, &e.ad)?;
                Ok(match x.ty {
                    U::Int | U::Real | U::Complex => format!("(-{c})"),
                    U::Vector | U::RowVector | U::ComplexVector | U::ComplexRowVector => {
                        format!("rt::ops::neg_v(&{c})")
                    }
                    U::Matrix | U::ComplexMatrix => format!("rt::ops::neg_m(&{c})"),
                    _ => return self.lib_call(Some(e), "minus", std::slice::from_ref(x)),
                })
            }
        }
    }

    fn zero(&self, ty: &U, ad: &Ad) -> String {
        let z = if ad.has_autodiff() {
            "T::from_f64(0.0)"
        } else {
            "0.0_f64"
        };
        match ty {
            U::Complex | U::ComplexVector | U::ComplexRowVector | U::ComplexMatrix => {
                format!("rt::stan::to_complex({z})")
            }
            _ => z.into(),
        }
    }

    fn binop(&mut self, e: &TExpr, l: &TExpr, op: BinOp, r: &TExpr) -> R<String> {
        use BinOp::*;
        let scalar_ty = |t: &U| matches!(t, U::Int | U::Real | U::Complex);
        if matches!(op, And | Or) {
            let a = self.truthy(l)?;
            let b = self.truthy(r)?;
            let o = if op == And { "&&" } else { "||" };
            return Ok(format!("(({a} {o} {b}) as i64)"));
        }
        if scalar_ty(&l.ty) && scalar_ty(&r.ty) {
            if matches!(op, Equals | NEquals | Less | Leq | Greater | Geq) {
                // Compare in the common type of the operands.
                let ty = if l.ty == U::Complex || r.ty == U::Complex {
                    U::Complex
                } else if l.ty == U::Real || r.ty == U::Real {
                    U::Real
                } else {
                    U::Int
                };
                let ad = Ad::lub_all([&l.ad, &r.ad]).unwrap_or(Ad::AutoDiffable);
                let a = self.expr_as(l, &ty, &ad)?;
                let b = self.expr_as(r, &ty, &ad)?;
                let sym = op.symbol();
                return Ok(format!("(({a} {sym} {b}) as i64)"));
            }
            if l.ty == U::Int && r.ty == U::Int && e.ty == U::Int {
                let a = self.expr(l)?;
                let b = self.expr(r)?;
                return Ok(match op {
                    Plus => format!("({a} + {b})"),
                    Minus => format!("({a} - {b})"),
                    Times | EltTimes => format!("({a} * {b})"),
                    Divide | IntDivide | EltDivide => format!("rt::stan::int_divide({a}, {b})?"),
                    Modulo => format!("rt::stan::modulus({a}, {b})?"),
                    _ => {
                        return Err(unsupported(
                            e.span,
                            format!("operator '{}' on integers", op.symbol()),
                        ));
                    }
                });
            }
            let a = self.expr_as(l, &e.ty, &e.ad)?;
            let b = self.expr_as(r, &e.ty, &e.ad)?;
            return Ok(match op {
                Plus => format!("({a} + {b})"),
                Minus => format!("({a} - {b})"),
                Times | EltTimes => format!("({a} * {b})"),
                Divide | EltDivide => format!("({a} / {b})"),
                Pow | EltPow if e.ty == U::Real => format!("sm::pow({a}, {b})"),
                _ => return self.lib_call(Some(e), op.function_name(), &[l.clone(), r.clone()]),
            });
        }
        let kind = |t: &U| match t {
            U::Int | U::Real | U::Complex => 's',
            U::Vector | U::ComplexVector => 'v',
            U::RowVector | U::ComplexRowVector => 'r',
            U::Matrix | U::ComplexMatrix => 'm',
            _ => '?',
        };
        let (lk, rk) = (kind(&l.ty), kind(&r.ty));
        let vec_like = |k: char| k == 'v' || k == 'r';
        let name: Option<&str> = match (op, lk, rk) {
            (Plus, a, b) if vec_like(a) && vec_like(b) => Some("add_vv"),
            (Plus, 'm', 'm') => Some("add_mm"),
            (Plus, 's', b) if vec_like(b) => Some("add_sv"),
            (Plus, a, 's') if vec_like(a) => Some("add_vs"),
            (Plus, 's', 'm') => Some("add_sm"),
            (Plus, 'm', 's') => Some("add_ms"),
            (Minus, a, b) if vec_like(a) && vec_like(b) => Some("sub_vv"),
            (Minus, 'm', 'm') => Some("sub_mm"),
            (Minus, 's', b) if vec_like(b) => Some("sub_sv"),
            (Minus, a, 's') if vec_like(a) => Some("sub_vs"),
            (Minus, 's', 'm') => Some("sub_sm"),
            (Minus, 'm', 's') => Some("sub_ms"),
            (Times | EltTimes, 's', b) if vec_like(b) => Some("mul_sv"),
            (Times | EltTimes, a, 's') if vec_like(a) => Some("mul_vs"),
            (Times | EltTimes, 's', 'm') => Some("mul_sm"),
            (Times | EltTimes, 'm', 's') => Some("mul_ms"),
            (Times, 'm', 'm') => Some("mul_mm+"),
            (Times, 'm', 'v') => Some("mul_mv+"),
            (Times, 'r', 'm') => Some("mul_rm+"),
            (Times, 'r', 'v') => Some("mul_rv+"),
            (Times, 'v', 'r') => Some("mul_vr"),
            (EltTimes, a, b) if vec_like(a) && vec_like(b) => Some("emul_vv"),
            (EltTimes, 'm', 'm') => Some("emul_mm"),
            (Divide | EltDivide, a, 's') if vec_like(a) => Some("div_vs"),
            (Divide | EltDivide, 'm', 's') => Some("div_ms"),
            (EltDivide, a, b) if vec_like(a) && vec_like(b) => Some("ediv_vv"),
            (EltDivide, 'm', 'm') => Some("ediv_mm"),
            (EltDivide, 's', b) if vec_like(b) => Some("ediv_sv"),
            (EltDivide, 's', 'm') => Some("ediv_sm"),
            _ => None,
        };
        let Some(name) = name else {
            return self.lib_call(Some(e), op.function_name(), &[l.clone(), r.clone()]);
        };
        // Lift both operands to the result's scalar (and complexness).
        let complex = matches!(
            e.ty,
            U::Complex | U::ComplexVector | U::ComplexRowVector | U::ComplexMatrix
        );
        let to_complex = |t: &U| -> U {
            if !complex {
                return t.clone();
            }
            match t {
                U::Int | U::Real => U::Complex,
                U::Vector => U::ComplexVector,
                U::RowVector => U::ComplexRowVector,
                U::Matrix => U::ComplexMatrix,
                t => t.clone(),
            }
        };
        let lt = if l.ty == U::Int {
            if complex { U::Complex } else { U::Real }
        } else {
            to_complex(&l.ty)
        };
        let rt_ = if r.ty == U::Int {
            if complex { U::Complex } else { U::Real }
        } else {
            to_complex(&r.ty)
        };
        let a = self.expr_as(l, &lt, &e.ad)?;
        let b = self.expr_as(r, &rt_, &e.ad)?;
        let a = if lk == 's' { a } else { format!("&{a}") };
        let b = if rk == 's' { b } else { format!("&{b}") };
        Ok(match name.strip_suffix('+') {
            Some(n) => {
                let z = self.zero(&e.ty, &e.ad);
                format!("rt::ops::{n}({a}, {b}, {z})?")
            }
            None if name.ends_with("_vv") || name.ends_with("_mm") => {
                format!("rt::ops::{name}({a}, {b})?")
            }
            None => format!("rt::ops::{name}({a}, {b})"),
        })
    }

    /// A function call; `e` is the call expression when it returns a value.
    pub fn call(
        &mut self,
        e: Option<&TExpr>,
        kind: &FunKind,
        name: &str,
        args: &[TExpr],
    ) -> R<String> {
        match kind {
            FunKind::UserDefined => self.udf_call(e, name, args),
            FunKind::StanLib => self.lib_call(e, name, args),
        }
    }

    fn udf_call(&mut self, e: Option<&TExpr>, name: &str, args: &[TExpr]) -> R<String> {
        let canonical = name.replace("_lupdf", "_lpdf").replace("_lupmf", "_lpmf");
        let suffix = FnSuffix::from_name(&canonical);
        let arg_types: Vec<U> = args.iter().map(|a| a.ty.clone()).collect();
        let Some(def) = self
            .udfs
            .get(&canonical)
            .and_then(|defs| {
                defs.iter()
                    .find(|d| d.args.iter().map(|a| &a.ty).eq(arg_types.iter()))
            })
            .cloned()
        else {
            let span = e
                .map(|e| e.span)
                .or_else(|| args.first().map(|a| a.span))
                .unwrap_or_default();
            return Err(unsupported(
                span,
                format!("no definition of function '{name}' matches its arguments"),
            ));
        };
        let autodiff = args.iter().any(|a| a.ad.has_autodiff())
            || e.is_some_and(|e| e.ad.has_autodiff())
            || matches!(suffix, FnSuffix::Lp | FnSuffix::Jacobian);
        let s = if autodiff { "T" } else { "f64" };
        let mut parts = Vec::new();
        for (a, p) in args.iter().zip(&def.args) {
            if matches!(a.ty, U::Function(_) | U::MathLibraryFunction) {
                return Err(unsupported(
                    a.span,
                    "functions passed as arguments are not supported by stanc-rs yet",
                ));
            }
            let pad: Ad = if autodiff { p.ad.into() } else { Ad::DataOnly };
            let pad = pad.fill_for(&p.ty);
            parts.push(self.expr_as(a, &p.ty, &pad)?);
        }
        match suffix {
            FnSuffix::Rng => parts.push("&mut *rng__".into()),
            FnSuffix::Lp => parts.push("&mut *target__".into()),
            FnSuffix::Jacobian => {
                parts.push("&mut *target__".into());
                parts.push("jacobian__".into());
            }
            FnSuffix::Lpdf | FnSuffix::Lpmf => {
                let propto = name.ends_with("_lupdf") || name.ends_with("_lupmf");
                parts.push(if propto {
                    "propto__".into()
                } else {
                    "false".into()
                });
            }
            FnSuffix::Plain => {}
        }
        Ok(format!("{}::<{s}>({})?", def.rust_name, parts.join(", ")))
    }

    /// Higher-order functions that lower to direct calls of the user-defined function argument.
    fn higher_order(&mut self, e: Option<&TExpr>, name: &str, args: &[TExpr]) -> R<Option<String>> {
        let fn_name = |a: &TExpr| match &a.kind {
            TExprKind::Variable { name, .. } => Some(name.clone()),
            _ => None,
        };
        match name {
            // The sum of the partial sums over any partition equals one call on the whole slice.
            "reduce_sum" | "reduce_sum_static" if args.len() >= 3 => {
                let Some(f) = fn_name(&args[0]) else {
                    return Ok(None);
                };
                let x = &args[1];
                let int = |kind: TExprKind| TExpr {
                    kind,
                    ty: U::Int,
                    ad: Ad::DataOnly,
                    span: x.span,
                };
                let size = int(TExprKind::FunApp {
                    kind: FunKind::StanLib,
                    name: "size".into(),
                    args: vec![x.clone()],
                    cond: false,
                });
                let mut cargs = vec![x.clone(), int(TExprKind::IntLit(1)), size];
                cargs.extend(args[3..].iter().cloned());
                Ok(Some(self.udf_call(e, &f, &cargs)?))
            }
            "map_rect" if args.len() == 5 => {
                let Some(f) = fn_name(&args[0]) else {
                    return Ok(None);
                };
                let Some(def) = self.udfs.get(&f).and_then(|d| d.first()).cloned() else {
                    return Ok(None);
                };
                let ad = e.map_or(Ad::DataOnly, |e| e.ad.clone());
                let s = scalar(&ad);
                let phi = self.expr_as(&args[1], &U::Vector, &ad)?;
                let thetas = self.expr_as(&args[2], &U::Array(Box::new(U::Vector)), &ad)?;
                let xr = self.expr(&args[3])?;
                let xi = self.expr(&args[4])?;
                Ok(Some(format!(
                    "rt::stan::map_rect({phi}, {thetas}, {xr}, {xi}, |p__, t__, r__, i__| {}::<{s}>(p__, t__, r__, i__))?",
                    def.rust_name
                )))
            }
            _ => self.solver(e, name, args),
        }
    }

    /// A call to a Stan Math library function.
    pub fn lib_call(&mut self, e: Option<&TExpr>, name: &str, args: &[TExpr]) -> R<String> {
        let span = e
            .map(|e| e.span)
            .or_else(|| args.first().map(|a| a.span))
            .unwrap_or_default();
        if let Some(code) = self.higher_order(e, name, args)? {
            return Ok(code);
        }
        if args
            .iter()
            .any(|a| matches!(a.ty, U::Function(_) | U::MathLibraryFunction))
        {
            return Err(unsupported(
                span,
                format!("the higher-order function '{name}' is not supported by stanc-rs yet"),
            ));
        }
        // Library functions take data and autodiff arguments as they are; drop the checker's
        // data-to-autodiff promotions.
        let args: Vec<TExpr> = args
            .iter()
            .map(|a| match &a.kind {
                TExprKind::Promotion(x, Promotion::ToVar) if x.ty == a.ty => (**x).clone(),
                _ => a.clone(),
            })
            .collect();
        if let Some(code) = self.special_call(e, name, &args)? {
            return Ok(code);
        }
        let result_ad = e.map_or(Ad::DataOnly, |e| e.ad.clone());
        // A call on data alone computes in f64; lift the result where autodiff is expected.
        if let Some(e) = e
            && result_ad.has_autodiff()
            && args.iter().all(|a| !a.ad.has_autodiff())
        {
            let data_e = TExpr {
                ad: Ad::DataOnly.fill_for(&e.ty),
                ..e.clone()
            };
            let code = self.lib_call(Some(&data_e), name, &args)?;
            return Ok(self.convert(code, &e.ty, &data_e.ad, &e.ty, &e.ad));
        }
        let lupdf = name.ends_with("_lupdf") || name.ends_with("_lupmf");
        if lupdf {
            let full = name.replace("_lupdf", "_lpdf").replace("_lupmf", "_lpmf");
            let a = self.lib_call_named(span, &full, &args, &result_ad)?;
            let b = self.lib_call_named(span, name, &args, &result_ad)?;
            return Ok(format!("(if propto__ {{ {b} }} else {{ {a} }})"));
        }
        self.lib_call_named(span, name, &args, &result_ad)
    }

    fn lib_call_named(
        &mut self,
        span: stanc_rs_syntax::Span,
        name: &str,
        args: &[TExpr],
        result_ad: &Ad,
    ) -> R<String> {
        let shapes: Vec<(Shape, bool)> = args
            .iter()
            .map(|a| (Shape::of(&a.ty), a.ad.has_autodiff()))
            .collect();
        let result_autodiff = result_ad.has_autodiff();
        let Some(res) = library::resolve(name, &shapes, result_autodiff) else {
            let types: Vec<String> = args.iter().map(|a| a.ty.to_string()).collect();
            return Err(unsupported(
                span,
                format!(
                    "Stan Math function '{name}({})' is not available in stanmath-rs or not supported by stanc-rs yet",
                    types.join(", ")
                ),
            ));
        };
        let f = res.fun;
        if res.passes.contains(&Pass::Vectorize) {
            return self.vectorized(name, args, &res, result_ad);
        }
        let mut codes = Vec::new();
        for (i, a) in args.iter().enumerate() {
            codes.push(self.pass_arg(a, &res.passes[i], res.lift[i])?);
        }
        let mut parts = Vec::new();
        let mut it = codes.into_iter();
        for p in &f.params {
            if *p == library::Param::Rng {
                parts.push("&mut *rng__".to_string());
            } else {
                parts.push(it.next().expect("argument count checked"));
            }
        }
        let turbo = self.turbofish(f, result_ad);
        let call = format!("sm::{}{turbo}({})", f.rust_name, parts.join(", "));
        let call = if f.ret.fallible {
            format!("{call}?")
        } else {
            call
        };
        Ok(convert_return(call, &f.ret.ty))
    }

    fn turbofish(&self, f: &library::Fun, result_ad: &Ad) -> String {
        let undetermined = f.undetermined_generics();
        if undetermined.is_empty() {
            return String::new();
        }
        let s = scalar(result_ad);
        let parts: Vec<&str> = f
            .generics
            .iter()
            .map(|(g, _)| if undetermined.contains(g) { s } else { "_" })
            .collect();
        format!("::<{}>", parts.join(", "))
    }

    fn pass_arg(&mut self, a: &TExpr, pass: &Pass, lift: bool) -> R<String> {
        let to_ad = if lift { Ad::AutoDiffable } else { a.ad.clone() };
        Ok(match pass {
            Pass::Value => self.expr_as(a, &a.ty, &to_ad)?,
            Pass::IntToReal => {
                let real = if a.ty == U::Int {
                    U::Real
                } else {
                    int_to_real(&a.ty)
                };
                self.expr_as(a, &real, &to_ad)?
            }
            Pass::I32 => format!("(({}) as i32)", self.expr(a)?),
            Pass::Usize => format!("rt::stan::usize_of({})?", self.expr(a)?),
            Pass::Ref | Pass::MatSlice => {
                let r = if lift && !a.ad.has_autodiff() && a.ty != U::Int && !is_int_array(&a.ty) {
                    let c = self.expr_as(a, &a.ty, &Ad::AutoDiffable)?;
                    format!("&({c})")
                } else {
                    self.expr_ref(a)?
                };
                if *pass == Pass::MatSlice {
                    format!("({r}).as_slice()")
                } else {
                    r
                }
            }
            Pass::ScalarValue => format!("({}).value()", self.expr(a)?),
            Pass::VecValues => format!("&rt::check::vals({})", self.expr_ref(a)?),
            Pass::MatValues => format!("&rt::check::mat_vals({})", self.expr_ref(a)?),
            Pass::Singleton => {
                let c = if lift {
                    self.expr_as(a, &a.ty, &Ad::AutoDiffable)?
                } else {
                    self.expr(a)?
                };
                format!("std::slice::from_ref(&{c})")
            }
            Pass::Vectorize => unreachable!("handled by vectorized"),
        })
    }

    /// A scalar function applied elementwise to container arguments.
    fn vectorized(
        &mut self,
        name: &str,
        args: &[TExpr],
        res: &library::Resolved,
        result_ad: &Ad,
    ) -> R<String> {
        let f = res.fun;
        let containers: Vec<usize> = (0..args.len())
            .filter(|i| res.passes[*i] == Pass::Vectorize)
            .collect();
        let mut pre = Vec::new();
        let mut leaf_codes: Vec<String> = Vec::new();
        let mut leaf_types: Vec<(U, Ad)> = Vec::new();
        for (i, a) in args.iter().enumerate() {
            let t = self.fresh("a");
            let c = if containers.contains(&i) {
                self.expr(a)?
            } else {
                self.pass_arg(a, &res.passes[i], res.lift[i])?
            };
            pre.push(format!("let {t} = {c};"));
            leaf_codes.push(t);
            leaf_types.push((a.ty.clone(), a.ad.clone()));
        }
        let scalar_param = |i: usize| -> &library::Param {
            f.params
                .iter()
                .filter(|p| **p != library::Param::Rng)
                .nth(i)
                .expect("arity")
        };
        // The scalar call on leaf values `leaves`.
        let leaf_call = |g: &mut Gen, leaves: &[String]| -> String {
            let mut parts = Vec::new();
            let mut k = 0;
            for p in &f.params {
                if *p == library::Param::Rng {
                    parts.push("&mut *rng__".to_string());
                    continue;
                }
                let i = k;
                k += 1;
                let code = leaves[i].clone();
                if !containers.contains(&i) {
                    parts.push(code);
                    continue;
                }
                let leaf_ty = leaf_scalar(&leaf_types[i].0);
                let lad = leaf_types[i].1.clone();
                let target_ad = if res.lift[i] {
                    Ad::AutoDiffable
                } else {
                    lad.clone()
                };
                parts.push(match scalar_param(i) {
                    library::Param::I32 => format!("({code} as i32)"),
                    library::Param::Usize => format!("rt::stan::usize_of({code})?"),
                    library::Param::I64 => code,
                    library::Param::F64 if lad.has_autodiff() => format!("({code}).value()"),
                    library::Param::F64 => g.convert(code, &leaf_ty, &lad, &U::Real, &Ad::DataOnly),
                    _ => g.convert(code, &leaf_ty, &lad, &U::Real, &target_ad),
                });
            }
            let turbo = g.turbofish(f, result_ad);
            let call = format!("sm::{}{turbo}({})", f.rust_name, parts.join(", "));
            let call = if f.ret.fallible {
                format!("{call}?")
            } else {
                call
            };
            convert_return(call, &f.ret.ty)
        };
        let body = if containers.len() == 1 {
            let ci = containers[0];
            let shape = Shape::of(&args[ci].ty);
            let mut leaves = leaf_codes.clone();
            self.map_shape(&leaf_codes[ci].clone(), &shape, &mut |g, v| {
                leaves[ci] = v.to_string();
                leaf_call(g, &leaves)
            })?
        } else {
            // Several containers: one-dimensional ones of equal length.
            for &i in &containers {
                if !matches!(Shape::of(&args[i].ty), Shape::Vec(_)) {
                    return Err(unsupported(
                        args[i].span,
                        format!(
                            "vectorized '{name}' over several multi-dimensional containers is not supported by stanc-rs yet"
                        ),
                    ));
                }
            }
            let n = self.fresh("n");
            let i = self.fresh("i");
            let out = self.fresh("out");
            let lens: Vec<String> = containers
                .iter()
                .map(|c| format!("{}.len()", leaf_codes[*c]))
                .collect();
            let mut leaves = leaf_codes.clone();
            for &c in &containers {
                leaves[c] = format!("{}[{i}]", leaf_codes[c]);
            }
            let call = leaf_call(self, &leaves);
            format!(
                "{{ let {n} = rt::stan::common_len({name:?}, &[{}])?; let mut {out} = Vec::with_capacity({n}); for {i} in 0..{n} {{ {out}.push({call}); }} {out} }}",
                lens.join(", ")
            )
        };
        Ok(format!("{{ {} {body} }}", pre.join(" ")))
    }

    /// Maps `leaf` over every scalar of `code` (a container of shape `shape`).
    fn map_shape(
        &mut self,
        code: &str,
        shape: &Shape,
        leaf: &mut dyn FnMut(&mut Gen, &str) -> String,
    ) -> R<String> {
        let v = self.fresh("v");
        let inner = match shape {
            Shape::Vec(_) | Shape::Mat(_) => leaf(self, &v),
            Shape::Arr(s) => self.map_shape(&v, s, leaf)?,
            _ => return Ok(leaf(self, code)),
        };
        let f = if matches!(shape, Shape::Mat(_)) {
            "try_map_mat"
        } else {
            "try_map_vec"
        };
        Ok(format!("rt::stan::{f}({code}, |{v}| Ok({inner}))?"))
    }

    /// Calls whose Rust form differs from the stanmath-rs signatures.
    fn special_call(&mut self, e: Option<&TExpr>, name: &str, args: &[TExpr]) -> R<Option<String>> {
        let tys: Vec<&U> = args.iter().map(|a| &a.ty).collect();
        let int_arr = |t: &U| matches!(t, U::Array(i) if **i == U::Int);
        Ok(Some(match (name, tys.as_slice()) {
            ("sum", [t]) if int_arr(t) => {
                format!("rt::stan::sum_int({})", self.expr_ref(&args[0])?)
            }
            ("max", [U::Int, U::Int]) => format!(
                "std::cmp::max({}, {})",
                self.expr(&args[0])?,
                self.expr(&args[1])?
            ),
            ("min", [U::Int, U::Int]) => format!(
                "std::cmp::min({}, {})",
                self.expr(&args[0])?,
                self.expr(&args[1])?
            ),
            ("max" | "min", [a, b])
                if matches!(a, U::Int | U::Real) && matches!(b, U::Int | U::Real) =>
            {
                let ad = e.map_or(Ad::DataOnly, |e| e.ad.clone());
                let x = self.expr_as(&args[0], &U::Real, &ad)?;
                let y = self.expr_as(&args[1], &U::Real, &ad)?;
                let f = if name == "max" { "fmax" } else { "fmin" };
                format!("sm::{f}({x}, {y})")
            }
            ("abs", [U::Int]) => format!("({}).abs()", self.expr(&args[0])?),
            ("size", [t]) if !matches!(t, U::Int | U::Real) => {
                format!("(({}).len() as i64)", self.expr_ref(&args[0])?)
            }
            ("num_elements", [_]) => {
                let a = &args[0];
                let c = if a.ad.has_autodiff() {
                    let v = self.expr(a)?;
                    format!("&({})", self.to_f64(v, &a.ty, &a.ad))
                } else {
                    self.expr_ref(a)?
                };
                let per = if matches!(leaf_scalar(&a.ty), U::Complex) {
                    2
                } else {
                    1
                };
                format!("((rt::check::Values::to_values({c}).len() / {per}) as i64)")
            }
            ("rows", [U::Vector | U::ComplexVector]) => {
                format!("(({}).len() as i64)", self.expr_ref(&args[0])?)
            }
            ("rows", [U::RowVector | U::ComplexRowVector])
            | ("cols", [U::Vector | U::ComplexVector]) => {
                let _ = self.expr(&args[0])?;
                "1_i64".into()
            }
            ("cols", [U::RowVector | U::ComplexRowVector]) => {
                format!("(({}).len() as i64)", self.expr_ref(&args[0])?)
            }
            ("rows", [U::Matrix | U::ComplexMatrix]) => {
                format!("(({}).nrows() as i64)", self.expr_ref(&args[0])?)
            }
            ("cols", [U::Matrix | U::ComplexMatrix]) => {
                format!("(({}).ncols() as i64)", self.expr_ref(&args[0])?)
            }
            ("to_int", [U::Real]) => format!(
                "rt::stan::to_int({})?",
                self.expr_as(&args[0], &U::Real, &Ad::DataOnly)?
            ),
            ("to_int", [U::Int]) => self.expr(&args[0])?,
            _ => return Ok(None),
        }))
    }

    /// Converts a TLValue to the expression that reads it.
    pub fn lvalue_expr(lv: &TLValue) -> TExpr {
        let kind = match &lv.kind {
            TLValueKind::Variable { name, origin } => TExprKind::Variable {
                name: name.clone(),
                origin: *origin,
                global: false,
            },
            TLValueKind::Indexed(b, idxs) => {
                TExprKind::Indexed(Box::new(Self::lvalue_expr(b)), idxs.clone())
            }
            TLValueKind::TupleProjection(b, k) => {
                TExprKind::TupleProjection(Box::new(Self::lvalue_expr(b)), *k)
            }
            TLValueKind::Tuple(lvs) => {
                TExprKind::TupleExpr(lvs.iter().map(Self::lvalue_expr).collect())
            }
        };
        TExpr {
            kind,
            ty: lv.ty.clone(),
            ad: lv.ad.clone(),
            span: lv.span,
        }
    }
}

fn is_int_array(t: &U) -> bool {
    match t {
        U::Array(i) => **i == U::Int || is_int_array(i),
        _ => false,
    }
}

fn int_to_real(t: &U) -> U {
    match t {
        U::Int => U::Real,
        U::Array(i) => U::Array(Box::new(int_to_real(i))),
        t => t.clone(),
    }
}

/// The scalar type at the leaves of a container type.
fn leaf_scalar(t: &U) -> U {
    match t {
        U::Array(i) => leaf_scalar(i),
        U::Vector | U::RowVector | U::Matrix => U::Real,
        U::ComplexVector | U::ComplexRowVector | U::ComplexMatrix => U::Complex,
        t => t.clone(),
    }
}

/// Converts a library return value to the Stan representation.
fn convert_return(call: String, ty: &str) -> String {
    match ty {
        "bool" | "i32" | "usize" => format!("(({call}) as i64)"),
        "Vec<usize>" => format!("rt::stan::usizes_to_ints({call})"),
        _ => call,
    }
}
