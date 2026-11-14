//! The Laplace approximation functions: `laplace_marginal`, `laplace_latent_rng`,
//! `laplace_latent_solve`, their `_tol` forms, and the built in likelihoods
//! `laplace_marginal[_tol]_{poisson_log,neg_binomial_2_log,bernoulli_logit}_lpmf`.
//!
//! As for the solvers, each user-defined function argument becomes a local struct implementing
//! the stanmath-rs trait (`LaplaceLikelihood`, `LaplaceCovariance`). The autodiff arguments of
//! the likelihood tuple are flattened into `eta__` and those of the covariance tuple into
//! `phi__`.

use stanc_rs_semantic::typed::{TExpr, TExprKind};
use stanc_rs_semantic::types::Ad;
use stanc_rs_syntax::ast::UnsizedType as U;

use crate::expr::scalar;
use crate::solvers::strip;
use crate::{Gen, R, unsupported};

/// The elements of a tuple-valued argument.
fn tuple_elems(a: &TExpr) -> Option<Vec<TExpr>> {
    let a = strip(a);
    if let TExprKind::TupleExpr(es) = &a.kind {
        return Some(es.iter().map(strip).collect());
    }
    let U::Tuple(ts) = &a.ty else { return None };
    Some(
        ts.iter()
            .enumerate()
            .map(|(i, t)| TExpr {
                kind: TExprKind::TupleProjection(Box::new(a.clone()), i + 1),
                ty: t.clone(),
                ad: match &a.ad {
                    Ad::Tuple(v) => v[i].clone(),
                    ad => ad.clone(),
                },
                span: a.span,
            })
            .collect(),
    )
}

/// One user-defined function argument lowered to a struct.
struct Functor {
    /// Statements defining the struct, its impl, and the value `name`.
    code: String,
    /// The local holding the struct value.
    name: String,
}

impl Gen {
    /// Lowers a Laplace function call; `None` when `name` is not one.
    pub(crate) fn laplace(
        &mut self,
        e: Option<&TExpr>,
        name: &str,
        args: &[TExpr],
    ) -> R<Option<String>> {
        let user = matches!(
            name,
            "laplace_marginal"
                | "laplace_marginal_tol"
                | "laplace_latent_rng"
                | "laplace_latent_tol_rng"
                | "laplace_latent_solve"
                | "laplace_latent_solve_tol"
        );
        let builtin = ["poisson_log", "neg_binomial_2_log", "bernoulli_logit"]
            .into_iter()
            .find(|l| {
                let after = |prefix: &str| {
                    name.strip_prefix(prefix)
                        .map(|r| r.strip_prefix("tol_").unwrap_or(r))
                        .and_then(|r| r.strip_prefix(l))
                };
                after("laplace_marginal_").is_some_and(|r| r == "_lpmf" || r == "_lupmf")
                    || after("laplace_latent_") == Some("_rng")
            });
        if !user && builtin.is_none() {
            return Ok(None);
        }
        let span = e.map_or(args[0].span, |e| e.span);
        let ad = e.map_or(Ad::DataOnly, |e| e.ad.clone());
        let s = scalar(&ad).to_string();
        let tol = name.contains("_tol");
        let unsupported_args = || {
            unsupported(
                span,
                format!("'{name}' with these arguments is not supported by stanc-rs yet"),
            )
        };
        // Position of the Hessian block size; the covariance function and its tuple follow.
        let hbs_at = match builtin {
            None => 2,
            Some("neg_binomial_2_log") => 4,
            Some(_) => 3,
        };
        if args.len() != hbs_at + 3 + usize::from(tol) {
            return Err(unsupported_args());
        }
        let hbs = self.expr(&args[hbs_at])?;
        let opts = if tol {
            let Some(o) = tuple_elems(&args[hbs_at + 3]) else {
                return Err(unsupported_args());
            };
            if o.len() != 6 {
                return Err(unsupported_args());
            }
            let init = self.expr_as(&o[0], &U::Vector, &Ad::DataOnly)?;
            format!(
                "sm::LaplaceOptions {{ theta_init: Some({init}), tolerance: {}, max_num_steps: rt::stan::usize_of({})?, hessian_block_size: rt::stan::usize_of({hbs})?, max_steps_line_search: rt::stan::usize_of({})?, solver: ({}) as u8, allow_fallback: ({}) != 0 }}",
                self.expr_as(&o[1], &U::Real, &Ad::DataOnly)?,
                self.expr(&o[2])?,
                self.expr(&o[4])?,
                self.expr(&o[3])?,
                self.expr(&o[5])?
            )
        } else {
            format!(
                "sm::LaplaceOptions {{ hessian_block_size: rt::stan::usize_of({hbs})?, ..Default::default() }}"
            )
        };
        let cov = self.functor(
            &args[hbs_at + 1],
            &args[hbs_at + 2],
            "phi__",
            &s,
            "sm::LaplaceCovariance",
            "fn covariance<T: sm::Real>(&self, theta: &[T]) -> sm::Result<sm::Matrix<T>>",
            &[],
            "Ok(v__)",
            "Ok(sm::Matrix::from_column_major(0, 0, Vec::new()))",
        )?;
        let mut pre = cov.code.clone();
        let mut errs = vec![cov.name.clone()];
        if builtin == Some("neg_binomial_2_log") && !matches!(args[2].ty, U::Int | U::Real) {
            return Err(unsupported(
                args[2].span,
                format!(
                    "'{name}' with a non-scalar dispersion is not supported by stanmath-rs yet"
                ),
            ));
        }
        let call = if let Some(l) = builtin {
            let y = self.expr(&args[0])?;
            let idx = self.expr(&args[1])?;
            let idx = format!(
                "{idx}.iter().map(|i__| rt::stan::usize_of(*i__ - 1)).collect::<rt::Result<Vec<usize>>>()?"
            );
            let mut rest = Vec::new();
            for a in &args[2..hbs_at] {
                let c = self.expr_as(a, &a.ty, &ad.clone().fill_for(&a.ty))?;
                rest.push(if a.ty == U::Real { c } else { format!("&{c}") });
            }
            let mut f = name.to_string();
            if l == "neg_binomial_2_log" {
                // stanmath-rs has no `_lupmf` form; the full density differs by a constant.
                f = f.replace("_lupmf", "_lpmf");
            }
            let mut o = if tol {
                format!(", &{opts}")
            } else {
                String::new()
            };
            // The latent draws run on data (generated quantities) and take the generator.
            if name.ends_with("_rng") {
                o.push_str(", &mut *rng__");
            }
            format!(
                "sm::{f}(&{y}, &{idx}, {}, &{}, &phi__{o})",
                rest.join(", "),
                cov.name
            )
        } else {
            let lik = self.functor(
                &args[0],
                &args[1],
                "eta__",
                &s,
                "sm::LaplaceLikelihood",
                "fn log_likelihood<T: sm::Real>(&self, lat: &[T], theta: &[T]) -> T",
                &["lat.to_vec()"],
                "v__",
                "T::from_f64(f64::NAN)",
            )?;
            pre.push_str(&lik.code);
            errs.push(lik.name.clone());
            let (l, c) = (&lik.name, &cov.name);
            match name {
                "laplace_marginal" | "laplace_marginal_tol" => {
                    format!("sm::laplace_marginal_tol(&{l}, &eta__, &{c}, &phi__, &{opts})")
                }
                _ => {
                    // The latent draws and solves take data.
                    pre.push_str(
                        "let eta__: Vec<f64> = eta__.iter().map(|x__| x__.value()).collect(); \
                         let phi__: Vec<f64> = phi__.iter().map(|x__| x__.value()).collect(); ",
                    );
                    if name.ends_with("_rng") {
                        format!(
                            "sm::laplace_latent_tol_rng(&{l}, &eta__, &{c}, &phi__, &{opts}, &mut *rng__)"
                        )
                    } else {
                        format!("sm::laplace_latent_solve_tol(&{l}, &eta__, &{c}, &phi__, &{opts})")
                    }
                }
            }
        };
        let checks: String = errs
            .iter()
            .map(|n| format!("if let Some(e__) = {n}.err__.take() {{ return Err(e__); }} "))
            .collect();
        Ok(Some(format!("{{ {pre}let r__ = {call}; {checks}r__? }}")))
    }

    /// The struct for function argument `f` with its arguments `tuple`, flattened into `vec`.
    /// `fixed` are the method arguments passed before the tuple's elements; `ok` and `fallback`
    /// are the method results when the function returns `v__` and when it fails.
    #[allow(clippy::too_many_arguments)]
    fn functor(
        &mut self,
        f: &TExpr,
        tuple: &TExpr,
        vec: &str,
        s: &str,
        trait_: &str,
        sig: &str,
        fixed: &[&str],
        ok: &str,
        fallback: &str,
    ) -> R<Functor> {
        let TExprKind::Variable { name: fname, .. } = &f.kind else {
            return Err(unsupported(
                f.span,
                "a Laplace function needs the name of a function",
            ));
        };
        let Some(elems) = tuple_elems(tuple) else {
            return Err(unsupported(
                tuple.span,
                "the arguments of a Laplace function must be a tuple",
            ));
        };
        let Some(defs) = self.udfs.get(fname) else {
            return Err(unsupported(
                f.span,
                format!("function '{fname}' has no definition"),
            ));
        };
        let defs = defs.clone();
        let Some(def) = defs
            .iter()
            .find(|d| d.args.len() == fixed.len() + elems.len())
        else {
            return Err(unsupported(
                f.span,
                format!("the Laplace arguments do not match function '{fname}'"),
            ));
        };
        let ex = self.extras(&elems, &def.args[fixed.len()..], vec)?;
        let mut call_args = Vec::new();
        for (code, p) in fixed.iter().zip(&def.args) {
            let pad = Ad::from(p.ad).fill_for(&p.ty);
            call_args.push(self.convert(
                code.to_string(),
                &U::Vector,
                &Ad::AutoDiffable,
                &p.ty,
                &pad,
            ));
        }
        call_args.extend(ex.in_eval);
        let st = self.fresh("Fn");
        let name = self.fresh("fn");
        let decls: String = ex
            .fields
            .iter()
            .map(|(n, t, _)| format!("{n}: {t}, "))
            .collect();
        let inits: String = ex
            .fields
            .iter()
            .map(|(n, _, v)| format!("{n}: {v}, "))
            .collect();
        let flat = if ex.flatten.is_empty() {
            format!("let {vec}: Vec<{s}> = Vec::new(); ")
        } else {
            format!(
                "let mut {vec}: Vec<{s}> = Vec::new(); {} ",
                ex.flatten.join(" ")
            )
        };
        let call = format!("{}::<T>({})", def.rust_name, call_args.join(", "));
        let code = format!(
            "struct {st} {{ {decls}err__: std::cell::RefCell<Option<rt::ModelError>> }} \
             impl {trait_} for {st} {{ {sig} {{ let k__ = std::cell::Cell::new(0_usize); let _ = &k__; \
             match {call} {{ Ok(v__) => {ok}, Err(e__) => {{ *self.err__.borrow_mut() = Some(e__); {fallback} }} }} }} }} \
             {flat}let {name} = {st} {{ {inits}err__: std::cell::RefCell::new(None) }}; "
        );
        Ok(Functor { code, name })
    }
}
