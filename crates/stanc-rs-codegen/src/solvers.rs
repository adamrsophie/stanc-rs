//! Higher-order library functions whose function argument is a user-defined function: ODE
//! solvers, algebraic solvers, and one-dimensional integration.
//!
//! stanmath-rs takes the function argument as a value implementing a trait with a method generic
//! in the scalar type (`OdeSystem`, `EquationSystem`, `Integrand`, ...). For each call the
//! generated code defines a local struct that holds the data arguments and implements that
//! trait by calling the user-defined function. Arguments that can carry derivatives are
//! flattened into the solver's parameter vector `theta` and rebuilt inside the method. Errors
//! raised by the user-defined function are stored in the struct and returned after the solver.

use stanc_rs_semantic::typed::{TExpr, TExprKind, TFunArg};
use stanc_rs_semantic::types::{Ad, Promotion};
use stanc_rs_syntax::ast::UnsizedType as U;

use crate::expr::{rust_type, scalar};
use crate::{Gen, R, unsupported};

/// The solver families.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// `ode_*(f, y0, t0, ts, [rtol, atol, max_steps,] ...)`, `f(t, y, ...)`.
    Ode { tol: bool },
    /// `integrate_ode_*(f, y0, t0, ts, theta, x_r, x_i[, rtol, atol, max_steps])`.
    IntegrateOde,
    /// `algebra_solver[_newton](f, guess, theta, x_r, x_i[, rel_tol, f_tol, max_steps])`.
    Algebra,
    /// `solve_newton|solve_powell(f, guess, ...)` and the `_tol` forms with three tolerances.
    Solve { tol: bool },
    /// `integrate_1d(f, a, b, theta, x_r, x_i[, rel_tol])`.
    Integrate1d,
}

fn kind(name: &str) -> Option<Kind> {
    Some(match name {
        "ode_rk45" | "ode_ckrk" | "ode_adams" | "ode_bdf" => Kind::Ode { tol: false },
        "ode_rk45_tol" | "ode_ckrk_tol" | "ode_adams_tol" | "ode_bdf_tol" => {
            Kind::Ode { tol: true }
        }
        "integrate_ode" | "integrate_ode_rk45" | "integrate_ode_bdf" | "integrate_ode_adams" => {
            Kind::IntegrateOde
        }
        "algebra_solver" | "algebra_solver_newton" => Kind::Algebra,
        "solve_newton" | "solve_powell" => Kind::Solve { tol: false },
        "solve_newton_tol" | "solve_powell_tol" => Kind::Solve { tol: true },
        "integrate_1d" => Kind::Integrate1d,
        _ => return None,
    })
}

/// Drops the checker's data-to-autodiff promotions so that data stays data.
fn strip(a: &TExpr) -> TExpr {
    match &a.kind {
        TExprKind::Promotion(x, Promotion::ToVar) if x.ty == a.ty => (**x).clone(),
        _ => a.clone(),
    }
}

/// Code for the variadic arguments of one call.
#[derive(Default)]
struct Extras {
    /// Struct fields: name, type, initializer.
    fields: Vec<(String, String, String)>,
    /// Statements that push the autodiff arguments into `th__`.
    flatten: Vec<String>,
    /// The arguments as passed to the user-defined function inside the trait method.
    in_eval: Vec<String>,
}

impl Gen {
    /// Lowers a call to a solver; `None` when `name` is not one.
    pub(crate) fn solver(
        &mut self,
        e: Option<&TExpr>,
        name: &str,
        args: &[TExpr],
    ) -> R<Option<String>> {
        let Some(k) = kind(name) else { return Ok(None) };
        let span = e.map_or(args[0].span, |e| e.span);
        let TExprKind::Variable { name: fname, .. } = &args[0].kind else {
            return Err(unsupported(
                span,
                format!("'{name}' needs the name of a function"),
            ));
        };
        let canonical = fname.clone();
        let Some(defs) = self.udfs.get(&canonical) else {
            return Err(unsupported(
                span,
                format!("function '{fname}' has no definition"),
            ));
        };
        let def = defs[0].clone();
        let ad = e.map_or(Ad::DataOnly, |e| e.ad.clone());
        let s = scalar(&ad).to_string();
        let args: Vec<TExpr> = args.iter().map(strip).collect();
        for (i, a) in args.iter().enumerate().skip(1) {
            // Times and tolerances are data in stanmath-rs.
            let time_like = match k {
                Kind::Ode { .. } | Kind::IntegrateOde => i == 2 || i == 3,
                _ => false,
            };
            if time_like && a.ad.has_autodiff() {
                return Err(unsupported(
                    a.span,
                    format!(
                        "'{name}' with parameter-dependent times is not supported by stanc-rs yet"
                    ),
                ));
            }
        }
        let st = self.fresh("Sys");
        let a_ = self.fresh("sys");
        let data = |g: &mut Gen, a: &TExpr| -> R<String> {
            g.expr_as(a, &a.ty, &Ad::DataOnly.fill_for(&a.ty))
        };
        let data_real =
            |g: &mut Gen, a: &TExpr| -> R<String> { g.expr_as(a, &U::Real, &Ad::DataOnly) };
        let ode_opts = |g: &mut Gen, r: &TExpr, a: &TExpr, m: &TExpr| -> R<String> {
            Ok(format!(
                "sm::OdeOptions {{ relative_tolerance: {}, absolute_tolerance: {}, max_num_steps: rt::stan::usize_of({})? }}",
                data_real(g, r)?,
                data_real(g, a)?,
                g.expr(m)?
            ))
        };
        let sys_opts = |g: &mut Gen, r: &TExpr, f: &TExpr, m: &TExpr| -> R<String> {
            Ok(format!(
                "sm::SystemOptions {{ tolerance: {}, function_tolerance: {}, max_iterations: rt::stan::usize_of({})? }}",
                data_real(g, r)?,
                data_real(g, f)?,
                g.expr(m)?
            ))
        };
        let lifted = |g: &mut Gen, a: &TExpr| -> R<String> {
            g.expr_as(a, &a.ty, &ad.clone().fill_for(&a.ty))
        };
        // (trait, method signature, fallback value, fixed arguments, extras start, call)
        let (trait_, sig, fallback, fixed, first_extra): (
            &str,
            &str,
            &str,
            Vec<(&str, &str)>,
            usize,
        ) = match k {
            Kind::Ode { tol } => (
                "sm::OdeSystem",
                "fn eval<T: sm::Real>(&self, t: T, y: &[T], theta: &[T]) -> Vec<T>",
                "vec![T::from_f64(f64::NAN); y.len()]",
                vec![("t", "r"), ("y.to_vec()", "v")],
                if tol { 7 } else { 4 },
            ),
            Kind::IntegrateOde => (
                "sm::OdeSystemData",
                "fn eval<T: sm::Real>(&self, t: T, y: &[T], theta: &[T], x_r: &[f64], x_i: &[i64]) -> Vec<T>",
                "vec![T::from_f64(f64::NAN); y.len()]",
                vec![
                    ("t", "r"),
                    ("y.to_vec()", "a"),
                    ("theta.to_vec()", "a"),
                    ("x_r.to_vec()", "d"),
                    ("x_i.to_vec()", "i"),
                ],
                args.len(),
            ),
            Kind::Algebra => (
                "sm::AlgebraSystemData",
                "fn eval<T: sm::Real>(&self, y: &[T], theta: &[T], x_r: &[f64], x_i: &[i64]) -> Vec<T>",
                "vec![T::from_f64(f64::NAN); y.len()]",
                vec![
                    ("y.to_vec()", "v"),
                    ("theta.to_vec()", "v"),
                    ("x_r.to_vec()", "d"),
                    ("x_i.to_vec()", "i"),
                ],
                args.len(),
            ),
            Kind::Solve { tol } => (
                "sm::EquationSystem",
                "fn eval<T: sm::Real>(&self, y: &[T], theta: &[T]) -> Vec<T>",
                "vec![T::from_f64(f64::NAN); y.len()]",
                vec![("y.to_vec()", "v")],
                if tol { 5 } else { 2 },
            ),
            Kind::Integrate1d => (
                "sm::Integrand",
                "fn eval<T: sm::Real>(&self, x: T, theta: &[T]) -> T",
                "T::from_f64(f64::NAN)",
                // stanmath-rs does not compute the complement `xc`; Stan passes NaN when it is
                // not needed, and so does stanc-rs.
                vec![
                    ("x", "r"),
                    ("T::from_f64(f64::NAN)", "r"),
                    ("theta.to_vec()", "a"),
                    ("self.x_r__.clone()", "d"),
                    ("self.x_i__.clone()", "i"),
                ],
                args.len(),
            ),
        };
        let params: Vec<TFunArg> = def.args.clone();
        if params.len() != fixed.len() + args.len().saturating_sub(first_extra) {
            return Err(unsupported(
                span,
                format!("'{name}': the arguments do not match function '{fname}'"),
            ));
        }
        let ex = self.extras(&args[first_extra..], &params[fixed.len()..])?;
        let mut fields = ex.fields;
        if k == Kind::Integrate1d {
            fields.push(("x_r__".into(), "Vec<f64>".into(), data(self, &args[4])?));
            fields.push(("x_i__".into(), "Vec<i64>".into(), self.expr(&args[5])?));
        }
        let mut call_args = Vec::new();
        for ((code, what), p) in fixed.iter().zip(&params) {
            let reals = U::Array(Box::new(U::Real));
            let (ty, ad) = match *what {
                "r" => (U::Real, Ad::AutoDiffable),
                "v" => (U::Vector, Ad::AutoDiffable),
                "a" => (reals, Ad::AutoDiffable),
                "d" => (reals, Ad::DataOnly),
                _ => (U::Array(Box::new(U::Int)), Ad::DataOnly),
            };
            let pad = Ad::from(p.ad).fill_for(&p.ty);
            call_args.push(self.convert(code.to_string(), &ty, &ad, &p.ty, &pad));
        }
        call_args.extend(ex.in_eval);
        let field_decls: String = fields
            .iter()
            .map(|(n, t, _)| format!("{n}: {t}, "))
            .collect();
        let field_inits: String = fields
            .iter()
            .map(|(n, _, v)| format!("{n}: {v}, "))
            .collect();
        let theta_flat = if ex.flatten.is_empty() {
            format!("let th__: Vec<{s}> = Vec::new();")
        } else {
            format!(
                "let mut th__: Vec<{s}> = Vec::new(); {}",
                ex.flatten.join(" ")
            )
        };
        let solve = match k {
            Kind::Ode { tol } => {
                let y0 = lifted(self, &args[1])?;
                let t0 = data_real(self, &args[2])?;
                let ts = data(self, &args[3])?;
                if tol {
                    let o = ode_opts(self, &args[4], &args[5], &args[6])?;
                    format!("sm::{name}(&{a_}, &{y0}, {t0}, &{ts}, &th__, &{o})")
                } else {
                    format!("sm::{name}(&{a_}, &{y0}, {t0}, &{ts}, &th__)")
                }
            }
            Kind::IntegrateOde => {
                let f = if name == "integrate_ode" {
                    "integrate_ode_rk45"
                } else {
                    name
                };
                let y0 = lifted(self, &args[1])?;
                let t0 = data_real(self, &args[2])?;
                let ts = data(self, &args[3])?;
                let theta = lifted(self, &args[4])?;
                let xr = data(self, &args[5])?;
                let xi = self.expr(&args[6])?;
                let o = if args.len() == 10 {
                    format!("Some(&{})", ode_opts(self, &args[7], &args[8], &args[9])?)
                } else {
                    "None".into()
                };
                format!("sm::{f}(&{a_}, &{y0}, {t0}, &{ts}, &{theta}, &{xr}, &{xi}, {o})")
            }
            Kind::Algebra => {
                let guess = if args[1].ad.has_autodiff() {
                    let c = self.expr(&args[1])?;
                    self.to_f64(c, &args[1].ty, &args[1].ad)
                } else {
                    self.expr(&args[1])?
                };
                let theta = lifted(self, &args[2])?;
                let xr = data(self, &args[3])?;
                let xi = self.expr(&args[4])?;
                let o = if args.len() == 8 {
                    format!("Some(&{})", sys_opts(self, &args[5], &args[6], &args[7])?)
                } else {
                    "None".into()
                };
                format!("sm::{name}(&{a_}, &{guess}, &{theta}, &{xr}, &{xi}, {o})")
            }
            Kind::Solve { tol } => {
                let guess = if args[1].ad.has_autodiff() {
                    let c = self.expr(&args[1])?;
                    self.to_f64(c, &args[1].ty, &args[1].ad)
                } else {
                    self.expr(&args[1])?
                };
                if tol {
                    let o = sys_opts(self, &args[2], &args[3], &args[4])?;
                    format!("sm::{name}(&{a_}, &{guess}, &th__, &{o})")
                } else {
                    format!("sm::{name}(&{a_}, &{guess}, &th__)")
                }
            }
            Kind::Integrate1d => {
                let a = self.expr_as(&args[1], &U::Real, &ad)?;
                let b = self.expr_as(&args[2], &U::Real, &ad)?;
                let theta = lifted(self, &args[3])?;
                let tol = if args.len() == 7 {
                    data_real(self, &args[6])?
                } else {
                    "1.4901161193847656e-8".into()
                };
                format!("sm::integrate_1d(&{a_}, {a}, {b}, &{theta}, {tol})")
            }
        };
        let udf = &def.rust_name;
        let call = format!("{udf}::<T>({})", call_args.join(", "));
        Ok(Some(format!(
            "{{ struct {st} {{ {field_decls}err__: std::cell::RefCell<Option<rt::ModelError>> }} \
             impl {trait_} for {st} {{ {sig} {{ let k__ = std::cell::Cell::new(0_usize); let _ = &k__; \
             match {call} {{ Ok(v__) => v__, Err(e__) => {{ *self.err__.borrow_mut() = Some(e__); {fallback} }} }} }} }} \
             {theta_flat} let {a_} = {st} {{ {field_inits}err__: std::cell::RefCell::new(None) }}; \
             let r__ = {solve}; if let Some(e__) = {a_}.err__.take() {{ return Err(e__); }} r__? }}"
        )))
    }

    /// Fields, flattening, and rebuilt arguments for the variadic arguments `args`, passed to
    /// parameters `params` of the user-defined function.
    fn extras(&mut self, args: &[TExpr], params: &[TFunArg]) -> R<Extras> {
        let mut ex = Extras::default();
        for (i, (a, p)) in args.iter().zip(params).enumerate() {
            let pad = Ad::from(p.ad).fill_for(&p.ty);
            let f = format!("a{i}__");
            let data_ty = rust_type(&a.ty, &Ad::DataOnly);
            if a.ad.has_autodiff() {
                if !flattenable(&a.ty) {
                    return Err(unsupported(
                        a.span,
                        "this argument type of a solver is not supported by stanc-rs yet",
                    ));
                }
                let v = self.expr(a)?;
                let t = self.fresh("x");
                let push = flatten(&t, &a.ty);
                let tmpl = self.to_f64(t.clone(), &a.ty, &a.ad);
                ex.flatten.push(format!("let {t} = {v}; {push}"));
                ex.fields.push((f.clone(), data_ty, tmpl));
                let rebuilt = unflatten(&format!("self.{f}"), &a.ty);
                let rebuilt = self.convert(
                    rebuilt,
                    &a.ty,
                    &Ad::AutoDiffable.fill_for(&a.ty),
                    &p.ty,
                    &pad,
                );
                ex.in_eval.push(rebuilt);
            } else {
                let v = self.expr(a)?;
                ex.fields.push((f.clone(), data_ty, v));
                let c = self.convert(
                    format!("self.{f}.clone()"),
                    &a.ty,
                    &Ad::DataOnly.fill_for(&a.ty),
                    &p.ty,
                    &pad,
                );
                ex.in_eval.push(c);
            }
        }
        Ok(ex)
    }
}

fn flattenable(t: &U) -> bool {
    match t {
        U::Real | U::Vector | U::RowVector | U::Matrix => true,
        U::Array(i) => flattenable(i),
        _ => false,
    }
}

/// Statements pushing the scalars of `v` (an owned value of type `t`) into `th__`, in the order
/// [`unflatten`] reads them.
fn flatten(v: &str, t: &U) -> String {
    match t {
        U::Real => format!("th__.push({v}.clone());"),
        U::Vector | U::RowVector => format!("th__.extend({v}.iter().cloned());"),
        U::Matrix => format!("th__.extend({v}.as_slice().iter().cloned());"),
        U::Array(i) => {
            let e = format!("{v}_e");
            format!("for {e} in {v}.iter() {{ {} }}", flatten(&e, i))
        }
        _ => String::new(),
    }
}

/// An expression rebuilding a value of type `t` from `theta` (cursor `k__`), shaped like the
/// `f64` template `tmpl`.
fn unflatten(tmpl: &str, t: &U) -> String {
    match t {
        U::Real => "rt::stan::next_theta(theta, &k__)".into(),
        U::Vector | U::RowVector => {
            format!("{tmpl}.iter().map(|_| rt::stan::next_theta(theta, &k__)).collect::<Vec<T>>()")
        }
        U::Matrix => format!("rt::stan::unflatten_mat(&{tmpl}, theta, &k__)"),
        U::Array(i) => {
            let e = format!("{}_e", tmpl.replace(['.', '&', '(', ')'], "_"));
            format!(
                "{tmpl}.iter().map(|{e}| {}).collect::<Vec<_>>()",
                unflatten(&e, i)
            )
        }
        _ => String::new(),
    }
}
