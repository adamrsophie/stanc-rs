//! The type checker. It follows the rules of stanc3 v2.40.0's `Typechecker` module: scoping,
//! overload resolution with promotion costs, block restrictions (where `target`, `_rng`, `_lp`,
//! `_jacobian`, and `_lupdf` functions may appear), read-only variables, and return analysis of
//! function bodies.

use std::collections::HashMap;

use stanc_rs_syntax::ast::*;
use stanc_rs_syntax::{Diagnostic, DiagnosticKind, Span};

use crate::mismatch::{self, Sig};
use crate::signatures::{self, Signature};
use crate::typed::*;
use crate::types::*;

type R<T> = Result<T, Diagnostic>;

fn err<T>(span: Span, msg: impl Into<String>) -> R<T> {
    Err(Diagnostic::new(DiagnosticKind::Semantic, span, msg))
}

const RESERVED: &[&str] = &[
    "generated",
    "quantities",
    "transformed",
    "repeat",
    "until",
    "then",
    "true",
    "false",
    "typedef",
    "struct",
    "var",
    "export",
    "extern",
    "static",
    "auto",
];

#[derive(Clone, Debug)]
enum EntryKind {
    UserDeclared,
    UserDefined,
    Variable {
        origin: Origin,
        global: bool,
        readonly: bool,
        span: Span,
    },
}

#[derive(Clone, Debug)]
struct Entry {
    kind: EntryKind,
    ty: UnsizedType,
}

#[derive(Default)]
struct Env {
    scopes: Vec<HashMap<String, Vec<Entry>>>,
}

impl Env {
    fn find(&self, name: &str) -> Vec<&Entry> {
        let mut v = Vec::new();
        for s in self.scopes.iter().rev() {
            if let Some(es) = s.get(name) {
                v.extend(es.iter().rev());
            }
        }
        v
    }

    fn add(&mut self, name: &str, e: Entry) {
        self.scopes
            .last_mut()
            .expect("scope")
            .entry(name.to_string())
            .or_default()
            .push(e);
    }

    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop(&mut self) {
        self.scopes.pop();
    }

    fn all_names(&self) -> impl Iterator<Item = &String> {
        self.scopes.iter().flat_map(|s| s.keys())
    }
}

#[derive(Clone, Debug, PartialEq)]
enum FnCtx {
    None,
    NonReturning(FnSuffix),
    Returning(FnSuffix, UnsizedType),
}

#[derive(Clone, Debug)]
struct Ctx {
    block: Origin,
    in_toplevel_decl: bool,
    func: FnCtx,
    loop_depth: usize,
}

impl Ctx {
    fn new(block: Origin) -> Self {
        Ctx {
            block,
            in_toplevel_decl: false,
            func: FnCtx::None,
            loop_depth: 0,
        }
    }
    fn suffix(&self) -> Option<FnSuffix> {
        match &self.func {
            FnCtx::None => None,
            FnCtx::NonReturning(s) | FnCtx::Returning(s, _) => Some(*s),
        }
    }
    fn in_lp(&self) -> bool {
        self.suffix() == Some(FnSuffix::Lp)
    }
    fn in_jacobian(&self) -> bool {
        self.suffix() == Some(FnSuffix::Jacobian)
    }
    fn in_rng(&self) -> bool {
        self.suffix() == Some(FnSuffix::Rng)
    }
    fn in_udf_dist(&self) -> bool {
        matches!(self.suffix(), Some(FnSuffix::Lpdf | FnSuffix::Lpmf))
    }
}

/// Return-flow classification of statements, used to check that functions return.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Rt {
    Incomplete,
    Complete,
    NonlocalControlFlow,
}

/// Options for type checking.
#[derive(Clone, Debug, Default)]
pub struct CheckOptions {
    /// Allow functions that are declared but never defined.
    pub allow_undefined: bool,
    /// Model name: identifiers may not equal it (stanc3 uses `<file stem>_model`).
    pub model_name: String,
}

/// Result of type checking: the typed program and warnings.
pub struct Checked {
    /// Typed program.
    pub program: TProgram,
    /// Warnings, including the deprecation warnings.
    pub warnings: Vec<Diagnostic>,
    /// How many of the last `warnings` are deprecation warnings, which stanc3 leaves out when it
    /// canonicalizes deprecations away.
    pub deprecations: usize,
}

/// Type checks a program.
pub fn check_program(p: &Program, opts: &CheckOptions) -> Result<Checked, Diagnostic> {
    let mut c = Checker {
        env: Env::default(),
        warnings: Vec::new(),
        opts: opts.clone(),
        higher_order: Vec::new(),
    };
    c.env.push();
    let mut out = TProgram {
        name: opts.model_name.clone(),
        ..TProgram::default()
    };
    // Functions: signatures first (allows mutual recursion), then bodies.
    if let Some(b) = &p.functions {
        for s in &b.stmts {
            c.add_function_signature(s)?;
        }
        let ctx = Ctx::new(Origin::Functions);
        for s in &b.stmts {
            out.functions.push(c.check_stmt(&ctx, s)?.0);
        }
        if !opts.allow_undefined {
            for s in &b.stmts {
                if let StmtKind::FunDef {
                    name, args, ret, ..
                } = &s.kind
                {
                    let ty = fun_type(
                        args.iter().map(|a| (a.ad, a.ty.clone())).collect(),
                        ret.clone(),
                        FnSuffix::from_name(&name.name),
                    );
                    let declared_only = c
                        .env
                        .find(&name.name)
                        .iter()
                        .any(|e| matches!(e.kind, EntryKind::UserDeclared) && e.ty == ty);
                    if declared_only {
                        return err(
                            s.span,
                            format!("Function '{}' is declared but not defined.", name.name),
                        );
                    }
                }
            }
        }
    }
    let blocks: [(Origin, &Option<Block>); 6] = [
        (Origin::Data, &p.data),
        (Origin::TData, &p.transformed_data),
        (Origin::Param, &p.parameters),
        (Origin::TParam, &p.transformed_parameters),
        (Origin::Model, &p.model),
        (Origin::GQuant, &p.generated_quantities),
    ];
    for (origin, b) in blocks {
        let Some(b) = b else { continue };
        let ctx = Ctx::new(origin);
        // The model block's variables are local to it; generated quantities sees everything else.
        if origin == Origin::Model {
            c.env.push();
        }
        let mut stmts = Vec::new();
        for s in &b.stmts {
            stmts.push(c.check_stmt(&ctx, s)?.0);
        }
        c.warn_unreachable(&stmts);
        if origin == Origin::Model {
            c.env.pop();
        }
        let dst = match origin {
            Origin::Data => &mut out.data,
            Origin::TData => &mut out.transformed_data,
            Origin::Param => &mut out.parameters,
            Origin::TParam => &mut out.transformed_parameters,
            Origin::Model => &mut out.model,
            _ => &mut out.generated_quantities,
        };
        *dst = stmts;
    }
    c.verify_second_order(&out.functions)?;
    let deprecations = crate::deprecation::warnings(p);
    Ok(Checked {
        program: out,
        deprecations: deprecations.len(),
        warnings: {
            let mut w = c.warnings;
            w.extend(deprecations);
            w
        },
    })
}

struct Checker {
    env: Env,
    warnings: Vec<Diagnostic>,
    opts: CheckOptions,
    /// Functions used as Laplace likelihoods, with the use site.
    higher_order: Vec<(String, Span)>,
}

enum Match {
    Unique {
        ret: Option<UnsizedType>,
        kind: FunKind,
        promotions: Vec<Promotion>,
    },
    Ambiguous,
    Errors(Vec<(Sig, Mismatch)>),
}

fn calc_ad(ctx: &Ctx, origin: Origin, ty: &UnsizedType) -> Ad {
    let (_, inner) = ty.array_dims();
    if let UnsizedType::Tuple(ts) = inner {
        return Ad::Tuple(ts.iter().map(|t| calc_ad(ctx, origin, t)).collect());
    }
    let diff = matches!(
        origin,
        Origin::Param | Origin::TParam | Origin::Model | Origin::Functions
    );
    if diff && !is_discrete(inner) && ctx.block != Origin::GQuant {
        Ad::AutoDiffable
    } else {
        Ad::DataOnly
    }
}

fn texpr(kind: TExprKind, ty: UnsizedType, ad: Ad, span: Span) -> TExpr {
    TExpr { kind, ty, ad, span }
}

/// Wraps `e` in a promotion node when `p` does something.
fn promote(e: TExpr, p: &Promotion) -> TExpr {
    if p.is_none() {
        return e;
    }
    // Promote array and row-vector literals element by element.
    match e.kind {
        TExprKind::ArrayExpr(es)
            if !matches!(p, Promotion::Tuple(_))
                || matches!(e.ty.array_dims().1, UnsizedType::Tuple(_)) =>
        {
            let pes: Vec<TExpr> = es.into_iter().map(|x| promote(x, p)).collect();
            let ty = UnsizedType::Array(Box::new(pes[0].ty.clone()));
            let ad = pes[0].ad.clone();
            texpr(TExprKind::ArrayExpr(pes), ty, ad, e.span)
        }
        TExprKind::TupleExpr(es) => {
            let Promotion::Tuple(ps) = p else {
                return texpr(TExprKind::TupleExpr(es), e.ty, e.ad, e.span);
            };
            let pes: Vec<TExpr> = es.into_iter().zip(ps).map(|(x, p)| promote(x, p)).collect();
            let ty = UnsizedType::Tuple(pes.iter().map(|x| x.ty.clone()).collect());
            let ad = Ad::Tuple(pes.iter().map(|x| x.ad.clone()).collect());
            texpr(TExprKind::TupleExpr(pes), ty, ad, e.span)
        }
        kind => {
            let e = TExpr { kind, ..e };
            let ty = promoted_type(&e.ty, p);
            let ad = match p {
                Promotion::ToVar | Promotion::ToComplexVar => Ad::AutoDiffable.fill_for(&ty),
                Promotion::Tuple(_) => promoted_tuple_ad(&e.ad, p),
                _ => e.ad.clone(),
            };
            let span = e.span;
            texpr(TExprKind::Promotion(Box::new(e), p.clone()), ty, ad, span)
        }
    }
}

fn promoted_tuple_ad(ad: &Ad, p: &Promotion) -> Ad {
    match (ad, p) {
        (Ad::Tuple(ads), Promotion::Tuple(ps)) => Ad::Tuple(
            ads.iter()
                .zip(ps)
                .map(|(a, p)| promoted_tuple_ad(a, p))
                .collect(),
        ),
        (_, Promotion::ToVar | Promotion::ToComplexVar) => Ad::AutoDiffable,
        (a, _) => a.clone(),
    }
}

fn arg_types(es: &[TExpr]) -> Vec<(Ad, UnsizedType)> {
    es.iter().map(|e| (e.ad.clone(), e.ty.clone())).collect()
}

fn fun_app_ad(ty: &UnsizedType, args: &[TExpr]) -> Ad {
    let ad = if is_discrete(ty) {
        Ad::DataOnly
    } else if args.iter().any(|a| a.ad.has_autodiff()) {
        Ad::AutoDiffable
    } else {
        Ad::DataOnly
    };
    ad.fill_for(ty)
}

impl Checker {
    // ----- Environment helpers -----

    fn verify_identifier(&self, id: &Ident) -> R<()> {
        if !self.opts.model_name.is_empty() && id.name == self.opts.model_name {
            return err(
                id.span,
                format!("Identifier '{}' clashes with the model name.", id.name),
            );
        }
        if id.name.ends_with("__") || RESERVED.contains(&id.name.as_str()) {
            return err(
                id.span,
                format!("Identifier '{}' clashes with a reserved keyword.", id.name),
            );
        }
        Ok(())
    }

    fn verify_name_fresh_var(&self, id: &Ident) -> R<()> {
        if signatures::is_unnormalized(&id.name) {
            return err(
                id.span,
                format!(
                    "Identifier '{}' has a _lupdf/_lupmf suffix, which is only allowed for functions.",
                    id.name
                ),
            );
        }
        if let Some(prev) = self.env.find(&id.name).iter().find_map(|e| match &e.kind {
            EntryKind::Variable { span, .. } => Some(*span),
            _ => None,
        }) {
            let _ = prev;
            return err(
                id.span,
                format!("Identifier '{}' is already in use.", id.name),
            );
        }
        Ok(())
    }

    fn nearest(&self, name: &str) -> String {
        let mut best: Option<(usize, String)> = None;
        let cands = self
            .env
            .all_names()
            .cloned()
            .chain(signatures::table().keys().cloned());
        for c in cands {
            let d = edit_distance(name, &c);
            if d <= 3 && best.as_ref().is_none_or(|(bd, _)| d < *bd) {
                best = Some((d, c));
            }
        }
        best.map(|(_, c)| format!(" A similar known identifier is '{c}'."))
            .unwrap_or_default()
    }

    fn user_signatures(&self, name: &str) -> Vec<Signature> {
        let n = signatures::normalized_name(name);
        self.env
            .find(&n)
            .into_iter()
            .filter_map(|e| match (&e.kind, &e.ty) {
                (EntryKind::UserDeclared | EntryKind::UserDefined, UnsizedType::Function(ft)) => {
                    Some(Signature {
                        args: ft.args.clone(),
                        ret: ft.ret.clone(),
                    })
                }
                _ => None,
            })
            .collect()
    }

    fn function_candidates(&self, name: &str) -> Vec<(Signature, FunKind)> {
        let mut v: Vec<(Signature, FunKind)> = self
            .user_signatures(name)
            .into_iter()
            .map(|s| (s, FunKind::UserDefined))
            .collect();
        v.extend(
            signatures::lookup(name)
                .iter()
                .map(|s| (s.clone(), FunKind::StanLib)),
        );
        v
    }

    fn matching_function(&self, name: &str, args: &[(Ad, UnsizedType)]) -> Match {
        let user = self.user_signatures(name);
        let cands = user.iter().map(|s| (s, FunKind::UserDefined)).chain(
            signatures::lookup(name)
                .iter()
                .map(|s| (s, FunKind::StanLib)),
        );
        resolve(cands, args)
    }

    fn signature_error(
        &self,
        span: Span,
        name: &str,
        args: &[TExpr],
        errs: Vec<(Sig, Mismatch)>,
    ) -> Diagnostic {
        let (sigs, omitted) = mismatch::select(errs);
        let tys: Vec<UnsizedType> = args.iter().map(|e| e.ty.clone()).collect();
        Diagnostic::new(
            DiagnosticKind::Semantic,
            span,
            mismatch::signature_mismatch(name, &tys, &sigs, omitted),
        )
    }

    /// A higher-order call's error (stanc3 `illtyped_variadic`), at the argument it concerns.
    fn variadic_error(
        &self,
        span: Span,
        name: &str,
        args: &[TExpr],
        (expected, e): VariadicError,
        ret: UnsizedType,
    ) -> Diagnostic {
        let tys: Vec<UnsizedType> = args.iter().map(|e| e.ty.clone()).collect();
        let span = specialize_span(span, &e, args);
        Diagnostic::new(
            DiagnosticKind::Semantic,
            span,
            mismatch::signature_mismatch(name, &tys, &[((Some(ret), expected), e)], false),
        )
    }

    // ----- Expressions -----

    fn check_expr(&mut self, ctx: &Ctx, e: &Expr) -> R<TExpr> {
        let span = e.span;
        match &e.kind {
            ExprKind::Paren(inner) => {
                let t = self.check_expr(ctx, inner)?;
                Ok(TExpr { span, ..t })
            }
            ExprKind::IntNumeral(s) => {
                let digits: String = s.chars().filter(|c| *c != '_').collect();
                match digits.parse::<f64>() {
                    Ok(v) if v < 2_147_483_648.0 => Ok(texpr(
                        TExprKind::IntLit(v as i64),
                        UnsizedType::Int,
                        Ad::DataOnly,
                        span,
                    )),
                    _ => err(span, "Integer literal cannot be larger than 2_147_483_647."),
                }
            }
            ExprKind::RealNumeral(s) => Ok(texpr(
                TExprKind::RealLit(s.clone()),
                UnsizedType::Real,
                Ad::DataOnly,
                span,
            )),
            ExprKind::ImagNumeral(s) => Ok(texpr(
                TExprKind::ImagLit(s.clone()),
                UnsizedType::Complex,
                Ad::DataOnly,
                span,
            )),
            ExprKind::Variable(id) => {
                self.verify_identifier(id)?;
                let (ty, ad, origin, global) = self.check_id(ctx, id)?;
                Ok(texpr(
                    TExprKind::Variable {
                        name: id.name.clone(),
                        origin,
                        global,
                    },
                    ty,
                    ad,
                    span,
                ))
            }
            ExprKind::GetTarget => {
                if !(ctx.in_lp() || ctx.block == Origin::Model || ctx.block == Origin::TParam) {
                    return err(
                        span,
                        "Target can only be accessed in the model block or in definitions of functions with the suffix _lp.",
                    );
                }
                let ad = calc_ad(ctx, ctx.block, &UnsizedType::Real);
                Ok(texpr(TExprKind::GetTarget, UnsizedType::Real, ad, span))
            }
            ExprKind::TernaryIf(c, a, b) => {
                let pc = self.check_expr(ctx, c)?;
                let ta = self.check_expr(ctx, a)?;
                let tb = self.check_expr(ctx, b)?;
                let common = common_type(&ta.ty, &tb.ty);
                let ad = Ad::lub_all([&pc.ad, &ta.ad, &tb.ad]);
                match (pc.ty == UnsizedType::Int, common, ad) {
                    (true, Some(ty), Some(ad)) if !is_fun_type(&ty) => {
                        let pa = type_promotion(&ad.clone().fill_for(&ty), &ty, &ta.ad, &ta.ty);
                        let pb = type_promotion(&ad.clone().fill_for(&ty), &ty, &tb.ad, &tb.ty);
                        let ta = promote(ta, &pa);
                        let tb = promote(tb, &pb);
                        let ad = ad.fill_for(&ty);
                        Ok(texpr(
                            TExprKind::TernaryIf(Box::new(pc), Box::new(ta), Box::new(tb)),
                            ty,
                            ad,
                            span,
                        ))
                    }
                    _ => err(
                        span,
                        format!(
                            "Ill-typed arguments supplied to ? : operator. Available signatures: (int, T, T) => T. Instead supplied arguments of incompatible type: {}, {}, {}.",
                            pc.ty, ta.ty, tb.ty
                        ),
                    ),
                }
            }
            ExprKind::BinOp(l, op, r) => {
                let tl = self.check_expr(ctx, l)?;
                let tr = self.check_expr(ctx, r)?;
                self.binop_warnings(e, (l, &tl), *op, (r, &tr));
                self.check_binop(span, *op, tl, tr)
            }
            ExprKind::PrefixOp(op, a) => {
                let ta = self.check_expr(ctx, a)?;
                let key = match op {
                    PrefixOp::Not => "!",
                    PrefixOp::Minus => "u-",
                    PrefixOp::Plus => "u+",
                };
                match self.operator_return(key, &[(ta.ad.clone(), ta.ty.clone())]) {
                    Some((Some(ty), _)) => {
                        let ad = ta.ad.clone();
                        Ok(texpr(TExprKind::PrefixOp(*op, Box::new(ta)), ty, ad, span))
                    }
                    _ => err(
                        span,
                        format!(
                            "Ill-typed arguments supplied to prefix operator {}. Available signatures are not compatible with argument type {}.",
                            op.symbol(),
                            ta.ty
                        ),
                    ),
                }
            }
            ExprKind::Transpose(a) => {
                let ta = self.check_expr(ctx, a)?;
                match self.operator_return("'", &[(ta.ad.clone(), ta.ty.clone())]) {
                    Some((Some(ty), _)) => {
                        let ad = ta.ad.clone();
                        Ok(texpr(TExprKind::Transpose(Box::new(ta)), ty, ad, span))
                    }
                    _ => err(
                        span,
                        format!(
                            "Ill-typed arguments supplied to postfix operator '. Available signatures are not compatible with argument type {}.",
                            ta.ty
                        ),
                    ),
                }
            }
            ExprKind::ArrayExpr(es) => {
                let tes = es
                    .iter()
                    .map(|x| self.check_expr(ctx, x))
                    .collect::<R<Vec<_>>>()?;
                let first_ty = tes[0].ty.clone();
                let (ad, ty, proms) = self.consistent_types(first_ty, &tes).map_err(|(ty, bad)| {
                    Diagnostic::new(
                        DiagnosticKind::Semantic,
                        bad.span,
                        format!("Array expression must have entries of consistent type. Expected {} but found {}.", ty, bad.ty),
                    )
                })?;
                let pes: Vec<TExpr> = tes
                    .into_iter()
                    .zip(&proms)
                    .map(|(x, p)| promote(x, p))
                    .collect();
                let aty = UnsizedType::Array(Box::new(ty));
                Ok(texpr(TExprKind::ArrayExpr(pes), aty, ad, span))
            }
            ExprKind::RowVectorExpr(es) => {
                let tes = es
                    .iter()
                    .map(|x| self.check_expr(ctx, x))
                    .collect::<R<Vec<_>>>()?;
                self.check_rowvector(span, tes)
            }
            ExprKind::TupleExpr(es) => {
                let tes = es
                    .iter()
                    .map(|x| self.check_expr(ctx, x))
                    .collect::<R<Vec<_>>>()?;
                let ty = UnsizedType::Tuple(tes.iter().map(|t| t.ty.clone()).collect());
                let ad = Ad::Tuple(tes.iter().map(|t| t.ad.clone()).collect());
                Ok(texpr(TExprKind::TupleExpr(tes), ty, ad, span))
            }
            ExprKind::TupleProjection(a, k) => {
                let ta = self.check_expr(ctx, a)?;
                match (&ta.ty, &ta.ad) {
                    (UnsizedType::Tuple(ts), Ad::Tuple(ads)) => {
                        if *k == 0 || *k > ts.len() {
                            return err(
                                span,
                                format!(
                                    "Tried to access index {} for a tuple of length {}.\nOnly indices indices between 1 and {} are valid.",
                                    k,
                                    ts.len(),
                                    ts.len()
                                ),
                            );
                        }
                        let (ty, ad) = (ts[k - 1].clone(), ads[k - 1].clone());
                        Ok(texpr(
                            TExprKind::TupleProjection(Box::new(ta), *k),
                            ty,
                            ad,
                            span,
                        ))
                    }
                    _ => err(
                        span,
                        format!(
                            "Attempting to index a tuple, but the expression has type {}.",
                            ta.ty
                        ),
                    ),
                }
            }
            ExprKind::Indexed(a, idx) => {
                let tidx = idx
                    .iter()
                    .map(|i| self.check_index(ctx, i))
                    .collect::<R<Vec<_>>>()?;
                let ta = self.check_expr(ctx, a)?;
                let ty = indexed_type(span, &ta.ty, &tidx)?;
                let ad = indexed_ad(&ta.ad, &ty, &tidx);
                Ok(texpr(TExprKind::Indexed(Box::new(ta), tidx), ty, ad, span))
            }
            ExprKind::FunApp(id, args) | ExprKind::CondDistApp(id, args) => {
                let cond = matches!(e.kind, ExprKind::CondDistApp(..));
                let tes = args
                    .iter()
                    .map(|x| self.check_expr(ctx, x))
                    .collect::<R<Vec<_>>>()?;
                self.check_funapp(ctx, span, id, tes, cond)
            }
        }
    }

    fn check_rowvector(&mut self, span: Span, tes: Vec<TExpr>) -> R<TExpr> {
        let first = tes.first().map(|t| t.ty.clone());
        let (base, kind) = match first {
            Some(UnsizedType::RowVector) => (UnsizedType::RowVector, 0),
            Some(UnsizedType::ComplexRowVector) => (UnsizedType::ComplexRowVector, 1),
            _ => (UnsizedType::Real, 2),
        };
        let res = self.consistent_types(base, &tes);
        let (ad, typ, proms) = match res {
            Ok(x) => x,
            Err((_, bad)) => {
                return err(
                    bad.span,
                    if kind == 2 {
                        format!(
                            "Row_vector expression must have all int or real entries. Found type {}.",
                            bad.ty
                        )
                    } else {
                        format!(
                            "Matrix expression must have all row_vector entries. Found type {}.",
                            bad.ty
                        )
                    },
                );
            }
        };
        let ty = match (kind, &typ) {
            (0, UnsizedType::ComplexRowVector) | (1, _) => UnsizedType::ComplexMatrix,
            (0, _) => UnsizedType::Matrix,
            (_, UnsizedType::Complex) => UnsizedType::ComplexRowVector,
            _ => UnsizedType::RowVector,
        };
        if kind == 2 && !matches!(typ, UnsizedType::Real | UnsizedType::Complex) && !tes.is_empty()
        {
            return err(
                span,
                format!(
                    "Row_vector expression must have all int or real entries. Found type {}.",
                    typ
                ),
            );
        }
        let pes: Vec<TExpr> = tes
            .into_iter()
            .zip(&proms)
            .map(|(x, p)| promote(x, p))
            .collect();
        Ok(texpr(TExprKind::RowVectorExpr(pes), ty, ad, span))
    }

    #[allow(clippy::result_large_err)]
    fn consistent_types(
        &self,
        start: UnsizedType,
        es: &[TExpr],
    ) -> Result<(Ad, UnsizedType, Vec<Promotion>), (UnsizedType, TExpr)> {
        let mut ty = start;
        for e in es {
            match common_type(&ty, &e.ty) {
                Some(t) => ty = t,
                None => return Err((ty, e.clone())),
            }
        }
        let ad = Ad::lub_all(es.iter().map(|e| &e.ad)).unwrap_or(Ad::DataOnly);
        let proms = es
            .iter()
            .map(|e| type_promotion(&ad, &ty, &e.ad, &e.ty))
            .collect();
        Ok((ad, ty, proms))
    }

    fn check_index(&mut self, ctx: &Ctx, i: &Index) -> R<TIndex> {
        let int = |c: &mut Self, e: &Expr| -> R<TExpr> {
            let t = c.check_expr(ctx, e)?;
            if t.ty != UnsizedType::Int {
                return err(
                    t.span,
                    format!(
                        "Range bound must be of type int. Instead found type {}.",
                        t.ty
                    ),
                );
            }
            Ok(t)
        };
        Ok(match i {
            Index::All => TIndex::All,
            Index::Single(e) => {
                let t = self.check_expr(ctx, e)?;
                if t.ty != UnsizedType::Int
                    && t.ty != UnsizedType::Array(Box::new(UnsizedType::Int))
                {
                    return err(
                        t.span,
                        format!(
                            "Index must be of type int or int[] or must be a range. Instead found type {}.",
                            t.ty
                        ),
                    );
                }
                TIndex::Single(t)
            }
            Index::Upfrom(e) => TIndex::Upfrom(int(self, e)?),
            Index::Downfrom(e) => TIndex::Downfrom(int(self, e)?),
            Index::Between(a, b) => TIndex::Between(int(self, a)?, int(self, b)?),
        })
    }

    /// Looks up a variable or function name used as a value.
    fn check_id(&mut self, ctx: &Ctx, id: &Ident) -> R<(UnsizedType, Ad, Origin, bool)> {
        let lookup = signatures::normalized_name(&id.name);
        let entries = self.env.find(&lookup);
        if signatures::is_unnormalized(&id.name)
            && !(ctx.in_udf_dist() || ctx.block == Origin::Model)
        {
            return err(
                id.span,
                format!(
                    "Functions with names ending in _lupdf and _lupmf can only be used in the model block or user-defined functions with names ending in _lpdf or _lpmf. Found '{}'.",
                    id.name
                ),
            );
        }
        match entries.first() {
            Some(Entry {
                kind:
                    EntryKind::Variable {
                        origin,
                        global,
                        span,
                        ..
                    },
                ty,
            }) => {
                let _ = span;
                if ctx.in_toplevel_decl
                    && matches!(origin, Origin::Param | Origin::TParam | Origin::GQuant)
                {
                    return err(
                        id.span,
                        format!(
                            "Non-data variables are not allowed in top level size declarations; found variable '{}' declared in the {} block.",
                            id.name,
                            origin.block_name()
                        ),
                    );
                }
                Ok((ty.clone(), calc_ad(ctx, *origin, ty), *origin, *global))
            }
            Some(Entry {
                ty: UnsizedType::Function(ft),
                ..
            }) => {
                let ty = UnsizedType::Function(Box::new(FunctionType {
                    suffix: FnSuffix::from_name(&id.name),
                    ..(**ft).clone()
                }));
                let ad = calc_ad(ctx, Origin::Functions, &ty);
                Ok((ty, ad, Origin::Functions, true))
            }
            Some(_) => Ok((
                UnsizedType::MathLibraryFunction,
                Ad::DataOnly,
                Origin::MathLibrary,
                true,
            )),
            None if signatures::is_stan_math_function(&lookup) => Ok((
                UnsizedType::MathLibraryFunction,
                Ad::DataOnly,
                Origin::MathLibrary,
                true,
            )),
            None => err(
                id.span,
                format!(
                    "Identifier '{}' not in scope.{}",
                    id.name,
                    self.nearest(&id.name)
                ),
            ),
        }
    }

    fn operator_return(
        &self,
        key: &str,
        args: &[(Ad, UnsizedType)],
    ) -> Option<(Option<UnsizedType>, Vec<Promotion>)> {
        if key == "%/%" {
            return match args {
                [(_, UnsizedType::Int), (_, UnsizedType::Int)] => Some((
                    Some(UnsizedType::Int),
                    vec![Promotion::None, Promotion::None],
                )),
                _ => None,
            };
        }
        for name in signatures::operator_functions(key) {
            if let Match::Unique {
                ret, promotions, ..
            } = self.match_stanlib(name, args)
            {
                return Some((ret, promotions));
            }
        }
        None
    }

    fn match_stanlib(&self, name: &str, args: &[(Ad, UnsizedType)]) -> Match {
        resolve(
            signatures::lookup(name)
                .iter()
                .map(|s| (s, FunKind::StanLib)),
            args,
        )
    }

    fn check_binop(&mut self, span: Span, op: BinOp, l: TExpr, r: TExpr) -> R<TExpr> {
        let key = op.symbol();
        let args = [(l.ad.clone(), l.ty.clone()), (r.ad.clone(), r.ty.clone())];
        let ad = Ad::lub_all([&l.ad, &r.ad]);
        match (self.operator_return(key, &args), ad) {
            (Some((Some(ty), ps)), Some(ad)) => {
                let l = promote(l, &ps[0]);
                let r = promote(r, &ps[1]);
                let ad = ad.fill_for(&ty);
                Ok(texpr(
                    TExprKind::BinOp(Box::new(l), op, Box::new(r)),
                    ty,
                    ad,
                    span,
                ))
            }
            _ => err(
                span,
                format!(
                    "Ill-typed arguments supplied to infix operator {}. Available signatures do not accept argument types {}, {}.",
                    key, l.ty, r.ty
                ),
            ),
        }
    }

    /// stanc3's `binop_type_warnings` and its chained comparison warning.
    fn binop_warnings(
        &mut self,
        whole: &Expr,
        (l, tl): (&Expr, &TExpr),
        op: BinOp,
        (r, tr): (&Expr, &TExpr),
    ) {
        use UnsizedType as U;
        use stanc_rs_syntax::autoformat::Message;
        let is_cmp = |op: BinOp| {
            use BinOp::*;
            matches!(op, Equals | NEquals | Less | Leq | Greater | Geq)
        };
        let mut m = Message::new();
        let span = match (&tl.ty, &tr.ty, op) {
            (U::Int, U::Int, BinOp::Divide) => {
                m.text("Found int division:")
                    .indented(|m| {
                        m.expr(whole);
                    })
                    .text(
                        "Values will be rounded towards zero. If rounding is not desired you \
                         can write the division as",
                    )
                    .indented(|m| match (&l.kind, &r.kind) {
                        (ExprKind::IntNumeral(x), _) => {
                            m.text(&format!("{x}.0 / ")).expr(r);
                        }
                        (_, ExprKind::IntNumeral(y)) => {
                            m.expr(l).text(&format!(" / {y}.0"));
                        }
                        _ => {
                            m.expr(l).text(" * 1.0 / ").expr(r);
                        }
                    })
                    .text("If rounding is intended please use the integer division operator %/%.");
                l.span
            }
            (U::Matrix, U::Int | U::Real, BinOp::Pow) => {
                self.matrix_pow(&mut m, whole, l, r);
                l.span
            }
            (U::Array(t), U::Int | U::Real, BinOp::Pow) if **t == U::Matrix => {
                self.matrix_pow(&mut m, whole, l, r);
                l.span
            }
            _ if is_cmp(op) => match &l.kind {
                ExprKind::BinOp(e1, op2, e2) if is_cmp(*op2) => {
                    let (s, s2) = (op.symbol(), op2.symbol());
                    m.text("Found chained comparison")
                        .indented(|m| {
                            m.expr(whole);
                        })
                        .text("This is interpreted as")
                        .indented(|m| {
                            m.text("(").expr(l).text(&format!(") {s} ")).expr(r);
                        })
                        .text("Consider if the intended meaning was")
                        .indented(|m| {
                            m.expr(e1)
                                .text(&format!(" {s2} "))
                                .expr(e2)
                                .text(" && ")
                                .expr(e2)
                                .text(&format!(" {s} "))
                                .expr(r);
                        })
                        .text(
                            "instead. You can silence this warning by adding explicit \
                             parentheses. This can be automatically changed using the \
                             canonicalize flag for stanc",
                        );
                    whole.span
                }
                _ => return,
            },
            _ => return,
        };
        self.warnings
            .push(Diagnostic::new(DiagnosticKind::Warning, span, m.finish()));
    }

    fn matrix_pow(
        &self,
        m: &mut stanc_rs_syntax::autoformat::Message,
        whole: &Expr,
        l: &Expr,
        r: &Expr,
    ) {
        m.text("Found matrix^scalar:")
            .indented(|m| {
                m.expr(whole);
            })
            .text(
                "matrix ^ number is interpreted as element-wise exponentiation. If this is \
                 intended, you can silence this warning by using elementwise operator .^",
            )
            .newline()
            .text("If you intended matrix exponentiation, use")
            .indented(|m| {
                m.text("matrix_power(").expr(l).text(", ").expr(r).text(")");
            })
            .text("instead.");
    }

    fn check_funapp(
        &mut self,
        ctx: &Ctx,
        span: Span,
        id: &Ident,
        tes: Vec<TExpr>,
        cond: bool,
    ) -> R<TExpr> {
        let res = self.check_fn(ctx, span, id, tes, cond)?;
        self.verify_identifier(id)?;
        let has_cond_suffix = CONDITIONING_SUFFIXES.iter().any(|s| id.name.ends_with(s));
        if cond && !has_cond_suffix {
            return err(
                span,
                "Only functions with names ending in _lpdf, _lupdf, _lpmf, _lupmf, _cdf, _lcdf, _lccdf can make use of conditional notation.",
            );
        }
        if !cond && has_cond_suffix {
            return err(
                span,
                "Probability functions with suffixes _lpdf, _lupdf, _lpmf, _lupmf, _cdf, _lcdf and _lccdf, require a vertical bar (|) between the first two arguments.",
            );
        }
        let args = match &res.kind {
            TExprKind::FunApp { args, .. } => args.clone(),
            _ => vec![],
        };
        self.verify_fn_target(ctx, span, &id.name)?;
        self.verify_fn_jacobian(ctx, span, &id.name, &args)?;
        self.verify_fn_rng(ctx, span, &id.name)?;
        if signatures::is_unnormalized(&id.name)
            && !(ctx.in_udf_dist() || ctx.block == Origin::Model)
        {
            return err(
                span,
                format!(
                    "Functions with names ending in _lupdf and _lupmf can only be used in the model block or user-defined functions with names ending in _lpdf or _lpmf. Found '{}'.",
                    id.name
                ),
            );
        }
        Ok(res)
    }

    fn verify_fn_target(&mut self, ctx: &Ctx, span: Span, name: &str) -> R<()> {
        if name.ends_with("_lp") {
            if ctx.block == Origin::TParam {
                self.warnings.push(Diagnostic::new(DiagnosticKind::Warning, span, "Using _lp functions in transformed parameters is deprecated and will be disallowed in Stan 2.40. Use an _jacobian function instead, as this allows change of variable adjustments which are conditionally enabled by the algorithms."));
            } else if !(ctx.in_lp() || ctx.block == Origin::Model) {
                return err(
                    span,
                    "Target can only be accessed in the model block or in definitions of functions with the suffix _lp.",
                );
            }
        }
        Ok(())
    }

    fn verify_fn_jacobian(&mut self, ctx: &Ctx, span: Span, name: &str, args: &[TExpr]) -> R<()> {
        if name.ends_with("_jacobian") {
            if !(ctx.in_jacobian() || ctx.block == Origin::TParam) {
                return err(
                    span,
                    "The jacobian adjustment can only be applied in the transformed parameters block or in functions ending with _jacobian.",
                );
            }
            if !args.iter().any(|a| a.ad.has_autodiff()) {
                let mut msg = String::from(
                    "Calling a _jacobian function without any parameter arguments still applies the Jacobian adjustments, ensure this is intentional!",
                );
                let alt = format!("{}_constrain", name.trim_end_matches("_jacobian"));
                if !self.function_candidates(&alt).is_empty() || !self.env.find(&alt).is_empty() {
                    msg.push_str(&format!(" Consider using {alt} instead."));
                }
                self.warnings
                    .push(Diagnostic::new(DiagnosticKind::Warning, span, msg));
            }
        }
        Ok(())
    }

    fn verify_fn_rng(&self, ctx: &Ctx, span: Span, name: &str) -> R<()> {
        if name.ends_with("_rng") {
            if ctx.in_toplevel_decl {
                return err(
                    span,
                    "Random number generators are not allowed in top level size declarations.",
                );
            }
            if !(ctx.in_rng() || ctx.block == Origin::GQuant || ctx.block == Origin::TData) {
                return err(
                    span,
                    "Random number generators are only allowed in transformed data block, generated quantities block or user-defined functions with names ending in _rng.",
                );
            }
        }
        Ok(())
    }

    fn check_fn(
        &mut self,
        ctx: &Ctx,
        span: Span,
        id: &Ident,
        tes: Vec<TExpr>,
        cond: bool,
    ) -> R<TExpr> {
        let name = signatures::normalized_name(&id.name);
        if let Some(v) = signatures::variadic(&name) {
            return self.check_variadic(ctx, span, id, tes, cond, v);
        }
        if signatures::is_reduce_sum(&name) {
            return self.check_reduce_sum(ctx, span, id, tes, cond);
        }
        if signatures::is_laplace(&name) {
            return self.check_laplace(span, id, tes, cond);
        }
        self.check_normal_fn(span, id, tes, cond)
    }

    fn check_normal_fn(&mut self, span: Span, id: &Ident, tes: Vec<TExpr>, cond: bool) -> R<TExpr> {
        let lookup = signatures::normalized_name(&id.name);
        let entries = self.env.find(&lookup);
        if let Some(Entry {
            kind: EntryKind::Variable { .. },
            ..
        }) = entries.first()
            && !signatures::is_stan_math_function(&lookup)
        {
            return err(
                span,
                format!(
                    "A returning function was expected but a non-function value '{}' was supplied.",
                    id.name
                ),
            );
        }
        if entries.is_empty() && signatures::lookup(&lookup).is_empty() {
            let mut msg = format!(
                "A returning function was expected but an undeclared identifier '{}' was supplied.",
                id.name
            );
            if let Some((prefix, suffix)) = id.name.rsplit_once('_') {
                let has = |s: &str| {
                    !self
                        .function_candidates(&format!("{prefix}_{s}"))
                        .is_empty()
                };
                if matches!(suffix, "lpmf" | "lupmf") && has("lpdf") {
                    msg = format!(
                        "Function '{}_{}' is not defined. Did you mean '{}_lpdf'?",
                        prefix, suffix, prefix
                    );
                } else if matches!(suffix, "lpdf" | "lupdf") && has("lpmf") {
                    msg = format!(
                        "Function '{}_{}' is not defined. Did you mean '{}_lpmf'?",
                        prefix, suffix, prefix
                    );
                }
            }
            msg.push_str(&self.nearest(&id.name));
            return err(span, msg);
        }
        match self.matching_function(&id.name, &arg_types(&tes)) {
            Match::Unique { ret: None, .. } => err(
                span,
                format!(
                    "A returning function was expected but a non-returning function '{}' was supplied.",
                    id.name
                ),
            ),
            Match::Unique {
                ret: Some(ty),
                kind,
                promotions,
            } => {
                let args: Vec<TExpr> = tes
                    .into_iter()
                    .zip(&promotions)
                    .map(|(e, p)| promote(e, p))
                    .collect();
                // Library random draws are data, whatever their arguments.
                let ad = if kind == FunKind::StanLib && id.name.ends_with("_rng") {
                    Ad::DataOnly.fill_for(&ty)
                } else if kind == FunKind::UserDefined
                    && (id.name.ends_with("_lp") || id.name.ends_with("_jacobian"))
                    && !is_discrete(&ty)
                {
                    // These read the target, so their values are on its level, as in stanc3's
                    // C++, whose return type includes the target's type.
                    Ad::AutoDiffable.fill_for(&ty)
                } else {
                    fun_app_ad(&ty, &args)
                };
                Ok(texpr(
                    TExprKind::FunApp {
                        kind,
                        name: id.name.clone(),
                        args,
                        cond,
                    },
                    ty,
                    ad,
                    span,
                ))
            }
            Match::Ambiguous => err(
                span,
                format!(
                    "No unique minimum promotion found for function '{}'. Overloaded functions must not have multiple equally valid promotion paths for a given set of arguments.",
                    id.name
                ),
            ),
            Match::Errors(errs) => Err(self.signature_error(span, &id.name, &tes, errs)),
        }
    }

    /// The function entries usable as a function argument named `fname`.
    fn function_values(&self, fname: &Ident) -> Vec<FunctionType> {
        let n = signatures::normalized_name(&fname.name);
        let mut v = Vec::new();
        for e in self.env.find(&n) {
            if let (EntryKind::UserDeclared | EntryKind::UserDefined, UnsizedType::Function(ft)) =
                (&e.kind, &e.ty)
            {
                let mut ft = (**ft).clone();
                ft.suffix = FnSuffix::from_name(&fname.name);
                v.push(ft);
            }
        }
        for s in signatures::lookup(&n) {
            v.push(FunctionType {
                ret: s.ret.clone(),
                args: s.args.clone(),
                suffix: FnSuffix::from_name(&fname.name),
            });
        }
        v
    }

    fn check_variadic(
        &mut self,
        ctx: &Ctx,
        span: Span,
        id: &Ident,
        tes: Vec<TExpr>,
        cond: bool,
        v: signatures::Variadic,
    ) -> R<TExpr> {
        let fid = match tes.first() {
            Some(TExpr {
                kind: TExprKind::Variable { name, .. },
                span,
                ..
            }) => Ident {
                name: name.clone(),
                span: *span,
            },
            _ => {
                let e = check_variadic_args(
                    false,
                    &v.control_args,
                    &v.fn_args,
                    &v.fn_ret,
                    &arg_types(&tes),
                )
                .expect_err("not a function");
                return Err(self.variadic_error(span, &id.name, &tes, e, v.fn_ret));
            }
        };
        let rest = arg_types(&tes[1..]);
        let mut matches = Vec::new();
        let mut errs = Vec::new();
        for ft in self.function_values(&fid) {
            let fty = UnsizedType::Function(Box::new(ft.clone()));
            let mut args = vec![(calc_ad(ctx, Origin::Functions, &fty), fty)];
            args.extend(rest.iter().cloned());
            match check_variadic_args(false, &v.control_args, &v.fn_args, &v.fn_ret, &args) {
                Ok(ps) => matches.push((ps.iter().map(Promotion::cost).sum::<usize>(), ft, ps)),
                Err(e) => errs.push(e),
            }
        }
        self.finish_higher_order(span, id, &fid, tes, cond, matches, errs, v.ret, v.fn_ret)
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_higher_order(
        &mut self,
        span: Span,
        id: &Ident,
        fid: &Ident,
        tes: Vec<TExpr>,
        cond: bool,
        matches: Vec<(usize, FunctionType, Vec<Promotion>)>,
        errs: Vec<VariadicError>,
        ret: UnsizedType,
        fn_ret: UnsizedType,
    ) -> R<TExpr> {
        let Some(min) = matches.iter().map(|m| m.0).min() else {
            let Some(e) = errs.into_iter().next() else {
                return err(
                    fid.span,
                    format!("Identifier '{}' not in scope or not a function.", fid.name),
                );
            };
            return Err(self.variadic_error(span, &id.name, &tes, e, fn_ret));
        };
        let mut best: Vec<_> = matches.into_iter().filter(|m| m.0 == min).collect();
        if best.len() > 1 {
            return err(
                span,
                format!(
                    "No unique minimum promotion found for function '{}'.",
                    fid.name
                ),
            );
        }
        let (_, ft, ps) = best.pop().expect("one");
        let mut it = tes.into_iter();
        let f = it.next().expect("function argument");
        let fty = UnsizedType::Function(Box::new(ft));
        let f = TExpr {
            ty: fty,
            ad: Ad::DataOnly,
            ..f
        };
        let mut args = vec![f];
        args.extend(it.zip(ps.iter().skip(1)).map(|(e, p)| promote(e, p)));
        let ad = fun_app_ad(&ret, &args);
        Ok(texpr(
            TExprKind::FunApp {
                kind: FunKind::StanLib,
                name: id.name.clone(),
                args,
                cond,
            },
            ret,
            ad,
            span,
        ))
    }

    fn check_reduce_sum(
        &mut self,
        ctx: &Ctx,
        span: Span,
        id: &Ident,
        tes: Vec<TExpr>,
        cond: bool,
    ) -> R<TExpr> {
        use UnsizedType::{Array, Int, Real};
        let basic = |tes: &[TExpr]| {
            check_variadic_args(
                true,
                &[
                    (AdLevel::AutoDiffable, Array(Box::new(Real))),
                    (AdLevel::AutoDiffable, Int),
                ],
                &[
                    (AdLevel::AutoDiffable, Array(Box::new(Real))),
                    (AdLevel::DataOnly, Int),
                    (AdLevel::DataOnly, Int),
                ],
                &Real,
                &arg_types(tes),
            )
            .expect_err("reduce_sum without a sliced argument")
        };
        let (fid, slice) = match (tes.first(), tes.get(1)) {
            (
                Some(TExpr {
                    kind: TExprKind::Variable { name, .. },
                    span: fspan,
                    ..
                }),
                Some(slice),
            ) => (
                Ident {
                    name: name.clone(),
                    span: *fspan,
                },
                slice,
            ),
            _ => {
                let e = basic(&tes);
                return Err(self.variadic_error(span, &id.name, &tes, e, Real));
            }
        };
        let (n, inner) = slice.ty.array_dims();
        if n == 0 {
            return err(
                slice.span,
                format!(
                    "The first argument to reduce_sum must be an array, found {}.",
                    slice.ty
                ),
            );
        }
        if !matches!(
            inner,
            UnsizedType::Real
                | UnsizedType::Int
                | UnsizedType::Matrix
                | UnsizedType::Vector
                | UnsizedType::RowVector
        ) {
            return err(
                slice.span,
                format!(
                    "The first argument to reduce_sum must be an array of int, real, vector, row_vector, or matrix, found {}.",
                    slice.ty
                ),
            );
        }
        let rest = arg_types(&tes[1..]);
        let mut matches = Vec::new();
        let mut errs = Vec::new();
        for ft in self.function_values(&fid) {
            let fty = UnsizedType::Function(Box::new(ft.clone()));
            let Some(sliced) = ft.args.first().cloned() else {
                errs.push(basic(&tes));
                continue;
            };
            let mut args = vec![(calc_ad(ctx, Origin::Functions, &fty), fty)];
            args.extend(rest.iter().cloned());
            let mandatory = [sliced.clone(), (AdLevel::AutoDiffable, Int)];
            let fn_args = [sliced, (AdLevel::DataOnly, Int), (AdLevel::DataOnly, Int)];
            match check_variadic_args(true, &mandatory, &fn_args, &Real, &args) {
                Ok(ps) => matches.push((ps.iter().map(Promotion::cost).sum::<usize>(), ft, ps)),
                Err(e) => errs.push(e),
            }
        }
        self.finish_higher_order(span, id, &fid, tes, cond, matches, errs, Real, Real)
    }

    fn tuple_arg_types(&self, te: &TExpr, what: &str) -> R<Vec<(Ad, UnsizedType)>> {
        match (&te.ty, &te.ad) {
            (UnsizedType::Tuple(ts), Ad::Tuple(ads)) => {
                Ok(ads.iter().cloned().zip(ts.iter().cloned()).collect())
            }
            _ => err(
                te.span,
                format!("{what} must be a tuple. Instead found type {}.", te.ty),
            ),
        }
    }

    /// Checks that the function named `fname` can be called with the required arguments followed by
    /// the contents of the tuple `tupl` (stanc3 `check_function_callable_with_tuple`).
    fn check_callable_with_tuple(
        &mut self,
        caller: &Ident,
        fname: &Ident,
        required: &[(AdLevel, UnsizedType)],
        tupl: TExpr,
        ret: UnsizedType,
    ) -> R<(TExpr, TExpr)> {
        let what = format!(
            "Forwarded arguments to '{}' in call to '{}'",
            fname.name, caller.name
        );
        let tys = self.tuple_arg_types(&tupl, &what)?;
        let req_ad: Vec<(Ad, UnsizedType)> = required
            .iter()
            .map(|(a, t)| ((*a).into(), t.clone()))
            .collect();
        let all: Vec<(Ad, UnsizedType)> = req_ad.iter().cloned().chain(tys).collect();
        let cands = self.function_values(fname);
        if cands.is_empty() {
            return match self.env.find(&fname.name).first() {
                Some(_) => err(
                    fname.span,
                    format!(
                        "A returning function was expected but a non-function value '{}' was supplied.",
                        fname.name
                    ),
                ),
                None => err(
                    fname.span,
                    format!(
                        "Identifier '{}' not in scope.{}",
                        fname.name,
                        self.nearest(&fname.name)
                    ),
                ),
            };
        }
        let mut best: Vec<(usize, FunctionType, Vec<Promotion>)> = Vec::new();
        let mut fn_err = None;
        let mut arg_err = None;
        for ft in cands {
            if ft.ret.as_ref() != Some(&ret) {
                fn_err.get_or_insert(format!("its return type must be {ret}"));
                continue;
            }
            if ft.suffix != FnSuffix::Plain {
                fn_err.get_or_insert(
                    "it must be a plain function (no _lpdf, _rng, _lp, or _jacobian suffix)"
                        .to_string(),
                );
                continue;
            }
            let n = required.len().min(ft.args.len());
            let head: Vec<(Ad, UnsizedType)> = ft.args[..n]
                .iter()
                .map(|(a, t)| ((*a).into(), t.clone()))
                .collect();
            if check_compatible_args(1, required, &head).is_err()
                || check_compatible_args(1, &ft.args[..n], &req_ad).is_err()
            {
                fn_err.get_or_insert(format!(
                    "its leading arguments must be ({})",
                    required
                        .iter()
                        .map(|(_, t)| t.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                continue;
            }
            match check_compatible_args(0, &ft.args, &all) {
                Ok(ps) => {
                    let c = ps.iter().map(Promotion::cost).sum();
                    best.push((c, ft, ps));
                }
                Err(e) => {
                    arg_err.get_or_insert(e);
                }
            }
        }
        let Some(min) = best.iter().map(|b| b.0).min() else {
            if let Some(e) = arg_err {
                return err(
                    tupl.span,
                    format!(
                        "Function '{}' cannot be called with the arguments forwarded by '{}': {}",
                        fname.name,
                        caller.name,
                        mismatch::input_mismatch(&e)
                    ),
                );
            }
            return err(
                fname.span,
                format!(
                    "Function '{}' does not have a valid signature for use in '{}': {}.",
                    fname.name,
                    caller.name,
                    fn_err.unwrap_or_default()
                ),
            );
        };
        let mut best: Vec<_> = best.into_iter().filter(|b| b.0 == min).collect();
        if best.len() > 1 {
            return err(
                fname.span,
                format!(
                    "No unique minimum promotion found for function '{}'.",
                    fname.name
                ),
            );
        }
        let (_, ft, ps) = best.pop().expect("one");
        let f = texpr(
            TExprKind::Variable {
                name: fname.name.clone(),
                origin: Origin::Functions,
                global: true,
            },
            UnsizedType::Function(Box::new(ft)),
            Ad::DataOnly,
            fname.span,
        );
        let tupl = promote(tupl, &Promotion::Tuple(ps[required.len()..].to_vec()));
        Ok((f, tupl))
    }

    fn check_laplace(&mut self, span: Span, id: &Ident, tes: Vec<TExpr>, cond: bool) -> R<TExpr> {
        let name = signatures::normalized_name(&id.name);
        let latent_solve = name.contains("_solve");
        let generic = |span: Span| -> R<TExpr> {
            err(
                span,
                format!(
                    "Ill-typed arguments supplied to function '{}'. See the Stan documentation for its signatures.",
                    id.name
                ),
            )
        };
        let latent_solve_err = |e: &TExpr| -> R<TExpr> {
            let loc = match &e.kind {
                TExprKind::TupleExpr(es) => es
                    .iter()
                    .find(|x| x.ad.has_autodiff())
                    .map(|x| x.span)
                    .unwrap_or(e.span),
                _ => e.span,
            };
            err(
                loc,
                format!(
                    "The arguments forwarded by '{}' must be data only.",
                    id.name
                ),
            )
        };
        let helper = signatures::laplace_helper_param_types(&name);
        let mut it: std::collections::VecDeque<TExpr> = tes.into();
        let mut args: Vec<TExpr> = Vec::new();
        if !helper.is_empty() {
            let n = helper.len().min(it.len());
            let for_prob: Vec<TExpr> = it.drain(..n).collect();
            match check_compatible_args(0, &helper, &arg_types(&for_prob)) {
                Ok(ps) => args.extend(for_prob.into_iter().zip(&ps).map(|(e, p)| promote(e, p))),
                Err(e) => {
                    let loc = match &e {
                        Mismatch::Arg(i, _) => for_prob.get(i - 1).map(|x| x.span).unwrap_or(span),
                        _ => for_prob.last().map(|x| x.span).unwrap_or(span),
                    };
                    return err(
                        loc,
                        format!(
                            "Ill-typed arguments supplied to the likelihood of '{}': {}",
                            id.name,
                            mismatch::input_mismatch(&e)
                        ),
                    );
                }
            }
        } else {
            if latent_solve && it.len() >= 2 && it[1].ad.has_autodiff() {
                return latent_solve_err(&it[1]);
            }
            match (it.pop_front(), it.pop_front()) {
                (
                    Some(TExpr {
                        kind: TExprKind::Variable { name: f, .. },
                        span: fspan,
                        ..
                    }),
                    Some(tupl),
                ) => {
                    let fid = Ident {
                        name: f,
                        span: fspan,
                    };
                    self.higher_order.push((fid.name.clone(), fid.span));
                    let (f, t) = self.check_callable_with_tuple(
                        id,
                        &fid,
                        &[(AdLevel::AutoDiffable, UnsizedType::Vector)],
                        tupl,
                        UnsizedType::Real,
                    )?;
                    args.push(f);
                    args.push(t);
                }
                _ => return generic(span),
            }
        }
        // Hessian block size.
        match it.pop_front() {
            Some(h) if h.ty == UnsizedType::Int && !h.ad.has_autodiff() => args.push(h),
            Some(h) => {
                return err(
                    h.span,
                    format!(
                        "The hessian block size argument of '{}' must be a data-only int. Instead found {}.",
                        id.name, h.ty
                    ),
                );
            }
            None => {
                let loc = args
                    .last()
                    .map(|a| Span {
                        start: a.span.end,
                        ..a.span
                    })
                    .unwrap_or(span);
                return err(
                    loc,
                    format!(
                        "'{}' requires a hessian block size argument of type int.",
                        id.name
                    ),
                );
            }
        }
        if latent_solve && it.len() >= 2 && it[1].ad.has_autodiff() {
            return latent_solve_err(&it[1]);
        }
        match (it.pop_front(), it.pop_front()) {
            (
                Some(TExpr {
                    kind: TExprKind::Variable { name: f, .. },
                    span: fspan,
                    ..
                }),
                Some(tupl),
            ) => {
                let fid = Ident {
                    name: f,
                    span: fspan,
                };
                let (f, t) =
                    self.check_callable_with_tuple(id, &fid, &[], tupl, UnsizedType::Matrix)?;
                args.push(f);
                args.push(t);
            }
            _ => return generic(span),
        }
        let control: Vec<TExpr> = it.into_iter().collect();
        let tol = id.name.contains("_tol");
        match (tol, control.as_slice()) {
            (false, []) => {}
            (true, [t]) => {
                let tys =
                    self.tuple_arg_types(t, &format!("Control arguments for '{}'", id.name))?;
                if let Err(e) =
                    check_compatible_args(0, &signatures::laplace_tolerance_argument_types(), &tys)
                {
                    let loc = match (&e, &t.kind) {
                        (Mismatch::Arg(i, _), TExprKind::TupleExpr(es)) => {
                            es.get(i - 1).map(|x| x.span).unwrap_or(t.span)
                        }
                        _ => t.span,
                    };
                    return err(
                        loc,
                        format!(
                            "Ill-typed tolerance arguments supplied to '{}': {}",
                            id.name,
                            mismatch::input_mismatch(&e)
                        ),
                    );
                }
            }
            (true, []) => {
                return err(
                    span,
                    format!(
                        "'{}' requires a tuple of tolerance arguments (vector, real, int, int, int, int).",
                        id.name
                    ),
                );
            }
            (true, [_, a, ..]) => {
                return err(
                    a.span,
                    format!(
                        "'{}' received {} unexpected extra arguments.",
                        id.name,
                        control.len() - 1
                    ),
                );
            }
            (false, [a, ..]) => {
                return err(
                    a.span,
                    format!(
                        "'{}' received {} unexpected extra arguments.",
                        id.name,
                        control.len()
                    ),
                );
            }
        }
        args.extend(control);
        let ty = if name.ends_with("_rng") {
            UnsizedType::Vector
        } else if latent_solve {
            UnsizedType::Tuple(vec![UnsizedType::Vector, UnsizedType::Matrix])
        } else {
            UnsizedType::Real
        };
        let ad = fun_app_ad(&ty, &args);
        Ok(texpr(
            TExprKind::FunApp {
                kind: FunKind::StanLib,
                name: id.name.clone(),
                args,
                cond,
            },
            ty,
            ad,
            span,
        ))
    }

    /// Functions passed as Laplace likelihoods must not (transitively) call library functions
    /// without second-order derivatives.
    fn verify_second_order(&self, functions: &[TStmt]) -> R<()> {
        fn body<'a>(fs: &'a [TStmt], name: &str) -> Vec<&'a TStmt> {
            fs.iter()
                .filter_map(|s| match &s.kind {
                    TStmtKind::FunDef {
                        name: n,
                        body: Some(b),
                        ..
                    } if n == name => Some(&**b),
                    _ => None,
                })
                .collect()
        }
        fn walk(fs: &[TStmt], name: &str, use_span: Span, seen: &mut Vec<String>) -> R<()> {
            if seen.iter().any(|s| s == name) {
                return Ok(());
            }
            seen.push(name.to_string());
            let mut calls: Vec<(String, bool)> = Vec::new();
            for b in body(fs, name) {
                stmt_calls(b, &mut calls);
            }
            for (c, lib) in calls {
                if lib {
                    if signatures::lacks_higher_order_autodiff(&c) {
                        return err(
                            use_span,
                            format!(
                                "The function '{name}' is used in an embedded Laplace approximation, but it calls '{c}', which does not support higher-order derivatives."
                            ),
                        );
                    }
                } else {
                    walk(fs, &c, use_span, seen)?;
                }
            }
            Ok(())
        }
        let mut seen = Vec::new();
        for (name, span) in &self.higher_order {
            walk(functions, name, *span, &mut seen)?;
        }
        Ok(())
    }

    // ----- Statements -----

    fn add_function_signature(&mut self, s: &Stmt) -> R<()> {
        let StmtKind::FunDef {
            ret,
            name,
            args,
            body,
        } = &s.kind
        else {
            return Ok(());
        };
        let arg_tys: Vec<(AdLevel, UnsizedType)> =
            args.iter().map(|a| (a.ad, a.ty.clone())).collect();
        let suffix = FnSuffix::from_name(&name.name);
        let ty = fun_type(arg_tys.clone(), ret.clone(), suffix);
        let existing = self
            .env
            .find(&name.name)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let declared = existing
            .iter()
            .any(|e| matches!(e.kind, EntryKind::UserDeclared) && e.ty == ty);
        let is_decl = matches!(body.kind, StmtKind::Skip);
        if !declared {
            // A new overload must have unique argument types.
            let same_args = |sig_args: &[(AdLevel, UnsizedType)]| {
                sig_args
                    .iter()
                    .map(|(_, t)| t)
                    .eq(arg_tys.iter().map(|(_, t)| t))
            };
            for e in &existing {
                if let UnsizedType::Function(ft) = &e.ty
                    && same_args(&ft.args)
                {
                    if ft.ret != *ret {
                        return err(
                            s.span,
                            format!(
                                "Function '{}' has already been declared with the same arguments but a different return type.",
                                name.name
                            ),
                        );
                    }
                    return err(
                        s.span,
                        format!(
                            "Function '{}' has already been declared for signature {}.",
                            name.name, ty
                        ),
                    );
                }
            }
            if signatures::lookup(&name.name)
                .iter()
                .any(|sig| same_args(&sig.args))
            {
                return err(
                    s.span,
                    format!(
                        "Function '{}' is already declared in the Stan Math library with signature {}.",
                        name.name, ty
                    ),
                );
            }
        } else if is_decl {
            return err(
                s.span,
                format!(
                    "Function '{}' has already been declared to have this signature.",
                    name.name
                ),
            );
        }
        if signatures::is_special_function_name(&name.name) {
            return err(
                name.span,
                format!(
                    "Identifier '{}' clashes with a Stan Math library function.",
                    name.name
                ),
            );
        }
        if signatures::is_unnormalized(&name.name) {
            return err(
                name.span,
                format!(
                    "Identifier '{}' has a _lupdf/_lupmf suffix, which is only allowed for functions.",
                    name.name
                ),
            );
        }
        if existing
            .iter()
            .any(|e| matches!(e.kind, EntryKind::Variable { .. }))
        {
            return err(
                name.span,
                format!("Identifier '{}' is already in use.", name.name),
            );
        }
        let kind = if is_decl {
            EntryKind::UserDeclared
        } else {
            EntryKind::UserDefined
        };
        if !is_decl {
            // Replace a matching forward declaration.
            for scope in &mut self.env.scopes {
                if let Some(es) = scope.get_mut(&name.name) {
                    es.retain(|e| !(matches!(e.kind, EntryKind::UserDeclared) && e.ty == ty));
                }
            }
        }
        self.env.add(&name.name, Entry { kind, ty });
        Ok(())
    }

    fn check_block_stmts(&mut self, ctx: &Ctx, stmts: &[Stmt]) -> R<(Vec<TStmt>, Rt)> {
        self.env.push();
        let mut out = Vec::new();
        let r = (|| {
            for s in stmts {
                out.push(self.check_stmt(ctx, s)?);
            }
            Ok::<_, Diagnostic>(())
        })();
        self.env.pop();
        r?;
        let mut rt = Rt::Incomplete;
        let mut seen_escape = false;
        for (s, srt) in &out {
            if seen_escape {
                self.warnings.push(Diagnostic::new(DiagnosticKind::Warning, s.span, "Unreachable statement (following a reject, fatal_error, break, continue, or return) found, is this intended?"));
                break;
            }
            rt = match (rt, *srt) {
                (Rt::Complete, Rt::Complete) | (Rt::Incomplete, Rt::Complete) => Rt::Complete,
                (Rt::NonlocalControlFlow, _) | (_, Rt::NonlocalControlFlow) => {
                    Rt::NonlocalControlFlow
                }
                _ => Rt::Incomplete,
            };
            seen_escape = is_escape(s);
        }
        Ok((out.into_iter().map(|(s, _)| s).collect(), rt))
    }

    fn warn_unreachable(&mut self, stmts: &[TStmt]) {
        for w in stmts.windows(2) {
            if is_escape(&w[0]) {
                self.warnings.push(Diagnostic::new(DiagnosticKind::Warning, w[1].span, "Unreachable statement (following a reject, fatal_error, break, continue, or return) found, is this intended?"));
                return;
            }
        }
    }

    fn check_int_expr(&mut self, ctx: &Ctx, e: &Expr, what: &str) -> R<TExpr> {
        let t = self.check_expr(ctx, e)?;
        if t.ty != UnsizedType::Int {
            return err(
                t.span,
                format!("{what} must be of type int. Instead found type {}.", t.ty),
            );
        }
        Ok(t)
    }

    fn check_stmt(&mut self, ctx: &Ctx, s: &Stmt) -> R<(TStmt, Rt)> {
        let span = s.span;
        let st = |kind| TStmt { kind, span };
        Ok(match &s.kind {
            StmtKind::Skip => (st(TStmtKind::Skip), Rt::Incomplete),
            StmtKind::Break => {
                if ctx.loop_depth == 0 {
                    return err(span, "Break statements may only be used in loops.");
                }
                (st(TStmtKind::Break), Rt::NonlocalControlFlow)
            }
            StmtKind::Continue => {
                if ctx.loop_depth == 0 {
                    return err(span, "Continue statements may only be used in loops.");
                }
                (st(TStmtKind::Continue), Rt::Incomplete)
            }
            StmtKind::Return(None) => match ctx.func {
                FnCtx::NonReturning(_) => (st(TStmtKind::Return(None)), Rt::Complete),
                _ => {
                    return err(
                        span,
                        "Void return statements may only be used inside non-returning function definitions.",
                    );
                }
            },
            StmtKind::Return(Some(e)) => match ctx.func.clone() {
                FnCtx::Returning(_, expected) => {
                    let te = self.check_expr(ctx, e)?;
                    if check_same_type(0, &expected, &te.ty).is_err() {
                        return err(
                            span,
                            format!(
                                "Invalid return statement. Expected type {} but found {}.",
                                expected, te.ty
                            ),
                        );
                    }
                    let p = type_promotion(
                        &Ad::AutoDiffable.fill_for(&expected),
                        &expected,
                        &te.ad,
                        &te.ty,
                    );
                    (st(TStmtKind::Return(Some(promote(te, &p)))), Rt::Complete)
                }
                _ => {
                    return err(
                        span,
                        "Expression return statements may only be used inside returning function definitions.",
                    );
                }
            },
            StmtKind::Print(ps) => (
                st(TStmtKind::Print(self.check_printables(ctx, ps)?)),
                Rt::Incomplete,
            ),
            StmtKind::Reject(ps) => (
                st(TStmtKind::Reject(self.check_printables(ctx, ps)?)),
                Rt::Complete,
            ),
            StmtKind::FatalError(ps) => (
                st(TStmtKind::FatalError(self.check_printables(ctx, ps)?)),
                Rt::Complete,
            ),
            StmtKind::TargetPlusAssign(e) => {
                let te = self.check_expr(ctx, e)?;
                if !(ctx.in_lp() || ctx.block == Origin::Model) {
                    return err(
                        span,
                        "Target can only be accessed in the model block or in definitions of functions with the suffix _lp.",
                    );
                }
                if is_fun_type(&te.ty)
                    || matches!(te.ty.array_dims().1, UnsizedType::Tuple(_))
                    || is_complex_type(&te.ty)
                {
                    return err(
                        span,
                        format!(
                            "A real, int, or container of those was expected. Instead found type {}.",
                            te.ty
                        ),
                    );
                }
                (st(TStmtKind::TargetPlusAssign(te)), Rt::Incomplete)
            }
            StmtKind::JacobianPlusAssign(e) => {
                let te = self.check_expr(ctx, e)?;
                if !(ctx.in_jacobian() || ctx.block == Origin::TParam) {
                    return err(
                        span,
                        "The jacobian adjustment can only be applied in the transformed parameters block or in functions ending with _jacobian.",
                    );
                }
                if is_fun_type(&te.ty)
                    || matches!(te.ty.array_dims().1, UnsizedType::Tuple(_))
                    || is_complex_type(&te.ty)
                {
                    return err(
                        span,
                        format!(
                            "A real, int, or container of those was expected. Instead found type {}.",
                            te.ty
                        ),
                    );
                }
                (st(TStmtKind::JacobianPlusAssign(te)), Rt::Incomplete)
            }
            StmtKind::NRFunApp(id, args) => {
                let tes = args
                    .iter()
                    .map(|a| self.check_expr(ctx, a))
                    .collect::<R<Vec<_>>>()?;
                self.verify_identifier(id)?;
                self.verify_fn_target(ctx, span, &id.name)?;
                self.verify_fn_jacobian(ctx, span, &id.name, &tes)?;
                self.verify_fn_rng(ctx, span, &id.name)?;
                let lookup = signatures::normalized_name(&id.name);
                let entries = self.env.find(&lookup);
                if let Some(Entry {
                    kind: EntryKind::Variable { .. },
                    ..
                }) = entries.first()
                    && !signatures::is_stan_math_function(&lookup)
                {
                    return err(
                        span,
                        format!(
                            "A non-returning function was expected but a non-function value '{}' was supplied.",
                            id.name
                        ),
                    );
                }
                if entries.is_empty() && !signatures::is_stan_math_function(&lookup) {
                    return err(
                        span,
                        format!(
                            "A non-returning function was expected but an undeclared identifier '{}' was supplied.{}",
                            id.name,
                            self.nearest(&id.name)
                        ),
                    );
                }
                match self.matching_function(&id.name, &arg_types(&tes)) {
                    Match::Unique {
                        ret: None,
                        kind,
                        promotions,
                    } => {
                        let args = tes
                            .into_iter()
                            .zip(&promotions)
                            .map(|(e, p)| promote(e, p))
                            .collect();
                        (
                            st(TStmtKind::NRFunApp {
                                kind,
                                name: id.name.clone(),
                                args,
                            }),
                            Rt::Incomplete,
                        )
                    }
                    Match::Unique { .. } => {
                        return err(
                            span,
                            format!(
                                "A non-returning function was expected but a returning function '{}' was supplied.",
                                id.name
                            ),
                        );
                    }
                    Match::Ambiguous => {
                        return err(
                            span,
                            format!(
                                "No unique minimum promotion found for function '{}'.",
                                id.name
                            ),
                        );
                    }
                    Match::Errors(errs) => {
                        return Err(self.signature_error(span, &id.name, &tes, errs));
                    }
                }
            }
            StmtKind::Assign { lhs, op, rhs } => (
                self.check_assignment(ctx, span, lhs, *op, rhs)?,
                Rt::Incomplete,
            ),
            StmtKind::Tilde {
                arg,
                dist,
                args,
                truncation,
            } => (
                self.check_tilde(ctx, span, arg, dist, args, truncation)?,
                Rt::Incomplete,
            ),
            StmtKind::IfThenElse(c, a, b) => {
                let tc = self.check_int_expr(ctx, c, "Condition in conditional")?;
                let (ta, ra) = self.check_scoped(ctx, a)?;
                let (tb, rb) = match b {
                    Some(b) => {
                        let (t, r) = self.check_scoped(ctx, b)?;
                        (Some(Box::new(t)), r)
                    }
                    None => (None, Rt::Incomplete),
                };
                let rt = match (ra, rb) {
                    (Rt::Complete, Rt::Complete) => Rt::Complete,
                    (Rt::NonlocalControlFlow, _) | (_, Rt::NonlocalControlFlow) => {
                        Rt::NonlocalControlFlow
                    }
                    _ => Rt::Incomplete,
                };
                (st(TStmtKind::IfThenElse(tc, Box::new(ta), tb)), rt)
            }
            StmtKind::While(c, body) => {
                let tc = self.check_int_expr(ctx, c, "Condition in while-loop")?;
                let lctx = Ctx {
                    loop_depth: ctx.loop_depth + 1,
                    ..ctx.clone()
                };
                let (tb, rb) = self.check_scoped(&lctx, body)?;
                let always = matches!(&c.kind, ExprKind::IntNumeral(s) if s.chars().any(|ch| ('1'..='9').contains(&ch)));
                let rt = match rb {
                    Rt::Complete => Rt::Complete,
                    Rt::Incomplete if always => Rt::Complete,
                    _ => Rt::Incomplete,
                };
                (st(TStmtKind::While(tc, Box::new(tb))), rt)
            }
            StmtKind::For {
                var,
                lower,
                upper,
                body,
            } => {
                let tl = self.check_int_expr(ctx, lower, "Lower bound of for-loop")?;
                let tu = self.check_int_expr(ctx, upper, "Upper bound of for-loop")?;
                self.verify_identifier(var)?;
                let (tb, rb) = self.check_loop_body(ctx, var, UnsizedType::Int, body)?;
                let rt = if rb == Rt::Complete {
                    Rt::Complete
                } else {
                    Rt::Incomplete
                };
                (
                    st(TStmtKind::For {
                        var: var.name.clone(),
                        lower: tl,
                        upper: tu,
                        body: Box::new(tb),
                    }),
                    rt,
                )
            }
            StmtKind::ForEach { var, iter, body } => {
                let te = self.check_expr(ctx, iter)?;
                self.verify_identifier(var)?;
                let elem = match &te.ty {
                    UnsizedType::Array(t) => (**t).clone(),
                    UnsizedType::Vector | UnsizedType::RowVector | UnsizedType::Matrix => {
                        UnsizedType::Real
                    }
                    t => {
                        return err(
                            te.span,
                            format!(
                                "Foreach-loop must be over array, vector, row_vector or matrix. Instead found expression of type {t}."
                            ),
                        );
                    }
                };
                let (tb, rb) = self.check_loop_body(ctx, var, elem.clone(), body)?;
                let rt = if rb == Rt::Complete {
                    Rt::Complete
                } else {
                    Rt::Incomplete
                };
                (
                    st(TStmtKind::ForEach {
                        var: var.name.clone(),
                        iter: te,
                        elem_ty: elem,
                        body: Box::new(tb),
                    }),
                    rt,
                )
            }
            StmtKind::Block(stmts) => {
                let (ts, rt) = self.check_block_stmts(ctx, stmts)?;
                (st(TStmtKind::Block(ts)), rt)
            }
            StmtKind::Profile(name, stmts) => {
                let (ts, rt) = self.check_block_stmts(ctx, stmts)?;
                (st(TStmtKind::Profile(name.clone(), ts)), rt)
            }
            StmtKind::VarDecl {
                ty,
                trans,
                vars,
                is_global,
            } => (
                self.check_var_decl(ctx, span, ty, trans, vars, *is_global)?,
                Rt::Incomplete,
            ),
            StmtKind::FunDef {
                ret,
                name,
                args,
                body,
            } => (
                self.check_fundef(ctx, span, ret, name, args, body)?,
                Rt::Incomplete,
            ),
        })
    }

    /// Checks a nested statement; a declaration directly under `if` or a loop gets its own scope.
    fn check_scoped(&mut self, ctx: &Ctx, s: &Stmt) -> R<(TStmt, Rt)> {
        self.env.push();
        let r = self.check_stmt(ctx, s);
        self.env.pop();
        r
    }

    fn check_loop_body(
        &mut self,
        ctx: &Ctx,
        var: &Ident,
        ty: UnsizedType,
        body: &Stmt,
    ) -> R<(TStmt, Rt)> {
        self.verify_name_fresh_var(var)?;
        self.env.push();
        self.env.add(
            &var.name,
            Entry {
                kind: EntryKind::Variable {
                    origin: ctx.block,
                    global: false,
                    readonly: true,
                    span: var.span,
                },
                ty,
            },
        );
        let lctx = Ctx {
            loop_depth: ctx.loop_depth + 1,
            ..ctx.clone()
        };
        let r = self.check_scoped(&lctx, body);
        self.env.pop();
        r
    }

    fn check_printables(&mut self, ctx: &Ctx, ps: &[Printable]) -> R<Vec<TPrintable>> {
        ps.iter()
            .map(|p| match p {
                Printable::Str(s) => Ok(TPrintable::Str(s.clone())),
                Printable::Expr(e) => {
                    let te = self.check_expr(ctx, e)?;
                    if is_fun_type(&te.ty) {
                        return err(te.span, "Functions cannot be printed.");
                    }
                    Ok(TPrintable::Expr(te))
                }
            })
            .collect()
    }

    fn check_lvalue(&mut self, ctx: &Ctx, l: &LValue) -> R<TLValue> {
        let span = l.span;
        match &l.kind {
            LValueKind::Variable(id) => {
                self.verify_identifier(id)?;
                let entry = self.env.find(&id.name).first().map(|e| (*e).clone());
                match entry {
                    Some(Entry {
                        kind:
                            EntryKind::Variable {
                                origin,
                                global,
                                readonly,
                                ..
                            },
                        ty,
                    }) => {
                        if global && origin != ctx.block {
                            return err(
                                id.span,
                                format!(
                                    "Cannot assign to global variable '{}' declared in previous blocks.",
                                    id.name
                                ),
                            );
                        }
                        if readonly {
                            return err(
                                id.span,
                                format!(
                                    "Cannot assign to function argument or loop identifier '{}'.",
                                    id.name
                                ),
                            );
                        }
                        if is_fun_type(&ty) {
                            return err(
                                id.span,
                                format!("Cannot assign a function type to variable '{}'.", id.name),
                            );
                        }
                        let ad = calc_ad(ctx, origin, &ty);
                        Ok(TLValue {
                            kind: TLValueKind::Variable {
                                name: id.name.clone(),
                                origin,
                            },
                            ty,
                            ad,
                            span,
                        })
                    }
                    Some(_) => err(id.span, format!("Cannot assign to function '{}'.", id.name)),
                    None if signatures::is_stan_math_function(&id.name) => err(
                        id.span,
                        format!(
                            "Cannot assign to global variable '{}' declared in previous blocks.",
                            id.name
                        ),
                    ),
                    None => err(
                        id.span,
                        format!(
                            "Identifier '{}' not in scope.{}",
                            id.name,
                            self.nearest(&id.name)
                        ),
                    ),
                }
            }
            LValueKind::TupleProjection(inner, k) => {
                let t = self.check_lvalue(ctx, inner)?;
                match (&t.ty, &t.ad) {
                    (UnsizedType::Tuple(ts), Ad::Tuple(ads)) if *k >= 1 && *k <= ts.len() => {
                        let (ty, ad) = (ts[k - 1].clone(), ads[k - 1].clone());
                        Ok(TLValue {
                            kind: TLValueKind::TupleProjection(Box::new(t), *k),
                            ty,
                            ad,
                            span,
                        })
                    }
                    (UnsizedType::Tuple(ts), _) => err(
                        span,
                        format!(
                            "Tried to access index {} for a tuple of length {}.",
                            k,
                            ts.len()
                        ),
                    ),
                    _ => err(
                        span,
                        format!(
                            "Attempting to index a tuple, but the expression has type {}.",
                            t.ty
                        ),
                    ),
                }
            }
            LValueKind::Indexed(inner, idx) => {
                // Multi-indexing is only allowed at the outermost level.
                let t = self.check_lvalue(ctx, inner)?;
                if has_multi_index(&t) {
                    return err(
                        span,
                        "Left hand side of an assignment cannot have nested multi-indexing.",
                    );
                }
                let tidx = idx
                    .iter()
                    .map(|i| self.check_index(ctx, i))
                    .collect::<R<Vec<_>>>()?;
                let ty = indexed_type(span, &t.ty, &tidx)?;
                let ad = indexed_ad(&t.ad, &ty, &tidx);
                Ok(TLValue {
                    kind: TLValueKind::Indexed(Box::new(t), tidx),
                    ty,
                    ad,
                    span,
                })
            }
            LValueKind::Tuple(ls) => {
                let ts = ls
                    .iter()
                    .map(|l| self.check_lvalue(ctx, l))
                    .collect::<R<Vec<_>>>()?;
                let ty = UnsizedType::Tuple(ts.iter().map(|t| t.ty.clone()).collect());
                let ad = Ad::Tuple(ts.iter().map(|t| t.ad.clone()).collect());
                Ok(TLValue {
                    kind: TLValueKind::Tuple(ts),
                    ty,
                    ad,
                    span,
                })
            }
        }
    }

    fn check_assignment(
        &mut self,
        ctx: &Ctx,
        span: Span,
        lhs: &LValue,
        op: AssignOp,
        rhs: &Expr,
    ) -> R<TStmt> {
        let tl = self.check_lvalue(ctx, lhs)?;
        verify_lvalue_unique(&tl)?;
        let tr = self.check_expr(ctx, rhs)?;
        let rhs = match op {
            AssignOp::Assign => {
                // stanc3 warns when the right side, without outer parentheses, is the left side.
                let mut bare = rhs;
                while let ExprKind::Paren(e) = &bare.kind {
                    bare = e;
                }
                if stanc_rs_syntax::pretty::lvalue_str(lhs) == stanc_rs_syntax::pretty::expr(bare) {
                    self.warnings.push(Diagnostic::new(
                        DiagnosticKind::Warning,
                        span,
                        "Assignment of variable to itself.",
                    ));
                }
                let p = assign_promotion(&tl, &tr).ok_or_else(|| {
                    Diagnostic::new(
                        DiagnosticKind::Semantic,
                        tr.span,
                        format!("Ill-typed arguments supplied to assignment operator =: lhs has type {} and rhs has type {}.", tl.ty, tr.ty),
                    )
                })?;
                promote(tr, &p)
            }
            AssignOp::Compound(bop) => {
                let ok = self.compound_assign_ok(bop, &tl, &tr);
                if !ok {
                    return err(
                        tr.span,
                        format!(
                            "Ill-typed arguments supplied to assignment operator {}=: lhs has type {} and rhs has type {}.",
                            bop.symbol(),
                            tl.ty,
                            tr.ty
                        ),
                    );
                }
                tr
            }
        };
        Ok(TStmt {
            kind: TStmtKind::Assign { lhs: tl, op, rhs },
            span,
        })
    }

    fn compound_assign_ok(&self, op: BinOp, l: &TLValue, r: &TExpr) -> bool {
        let args = [
            (Ad::AutoDiffable, l.ty.clone()),
            (r.ad.clone(), r.ty.clone()),
        ];
        let ret = match op {
            BinOp::Divide => match self.match_stanlib("divide", &args) {
                Match::Unique { ret, .. } => ret,
                _ => return false,
            },
            BinOp::Plus | BinOp::Minus | BinOp::Times | BinOp::EltTimes | BinOp::EltDivide => {
                match self.operator_return(op.symbol(), &args) {
                    Some((ret, _)) => ret,
                    None => return false,
                }
            }
            _ => return false,
        };
        match ret {
            Some(t) => {
                t == l.ty && !(matches!(op, BinOp::EltTimes | BinOp::EltDivide) && is_scalar(&t))
            }
            None => false,
        }
    }

    fn check_tilde(
        &mut self,
        ctx: &Ctx,
        span: Span,
        arg: &Expr,
        dist: &Ident,
        args: &[Expr],
        trunc: &Truncation,
    ) -> R<TStmt> {
        let te = self.check_expr(ctx, arg)?;
        let tes = args
            .iter()
            .map(|a| self.check_expr(ctx, a))
            .collect::<R<Vec<_>>>()?;
        let ttrunc = match trunc {
            Truncation::None => TTruncation::None,
            Truncation::UpFrom(e) => TTruncation::UpFrom(self.check_trunc_bound(ctx, e)?),
            Truncation::DownFrom(e) => TTruncation::DownFrom(self.check_trunc_bound(ctx, e)?),
            Truncation::Between(a, b) => TTruncation::Between(
                self.check_trunc_bound(ctx, a)?,
                self.check_trunc_bound(ctx, b)?,
            ),
        };
        self.verify_identifier(dist)?;
        let n = &dist.name;
        if n.ends_with("_lpdf")
            || n.ends_with("_lpmf")
            || n.ends_with("_lupdf")
            || n.ends_with("_lupmf")
        {
            return err(
                dist.span,
                "~ statement should refer to a distribution without its \"_lpdf/_lupdf\" or \"_lpmf/_lupmf\" suffix.\nFor example, \"target += normal_lpdf(y | 0, 1)\" should become \"y ~ normal(0, 1).\"",
            );
        }
        if !(ctx.in_lp() || ctx.block == Origin::Model) {
            return err(
                span,
                "Target can only be accessed in the model block or in definitions of functions with the suffix _lp.",
            );
        }
        if n.ends_with("_cdf") || n.ends_with("_ccdf") {
            return err(
                span,
                format!(
                    "~ statement should refer to a distribution without its \"_cdf\" or \"_ccdf\" suffix. Found '{n}'."
                ),
            );
        }
        let mut all = vec![te];
        all.extend(tes);
        for suf in ["_lpmf", "_lpdf"] {
            let fname = format!("{n}{suf}");
            if signatures::is_laplace(&fname) {
                let fid = Ident {
                    name: fname.clone(),
                    span: dist.span,
                };
                let app = self.check_laplace(span, &fid, all, true)?;
                let TExprKind::FunApp { mut args, .. } = app.kind else {
                    unreachable!("laplace is an application")
                };
                let targ = args.remove(0);
                return Ok(TStmt {
                    kind: TStmtKind::Tilde {
                        arg: targ,
                        dist: fname,
                        dist_kind: FunKind::StanLib,
                        args,
                        truncation: ttrunc,
                    },
                    span,
                });
            }
        }
        let at = arg_types(&all);
        // Prefer whichever of _lpmf and _lpdf matches best (unique beats ambiguous beats errors).
        let mut results: Vec<(String, Match)> = Vec::new();
        for suf in ["_lpmf", "_lpdf"] {
            let name = format!("{n}{suf}");
            results.push((name.clone(), self.matching_function(&name, &at)));
        }
        let rank = |m: &Match| match m {
            Match::Unique { .. } => 0,
            Match::Ambiguous => 1,
            Match::Errors(e) if !e.is_empty() => 2,
            Match::Errors(_) => 3,
        };
        results.sort_by_key(|(_, m)| rank(m));
        let (fname, m) = results.into_iter().next().expect("two");
        let (kind, promotions) = match m {
            Match::Unique {
                kind, promotions, ..
            } => (kind, promotions),
            Match::Ambiguous => {
                return err(
                    span,
                    format!("No unique minimum promotion found for function '{n}'."),
                );
            }
            Match::Errors(errs) if errs.is_empty() => {
                let int = is_int_type(&all[0].ty);
                return err(
                    dist.span,
                    format!(
                        "Ill-typed arguments to '~' statement. No distribution '{n}' was found{}.",
                        if int {
                            " with an integer first argument"
                        } else {
                            ""
                        }
                    ),
                );
            }
            Match::Errors(errs) => return Err(self.signature_error(span, n, &all, errs)),
        };
        let mut all: Vec<TExpr> = all
            .into_iter()
            .zip(&promotions)
            .map(|(e, p)| promote(e, p))
            .collect();
        let targ = all.remove(0);
        // Truncation requires both CDF variants.
        let bounds: Vec<&TExpr> = match &ttrunc {
            TTruncation::None => vec![],
            TTruncation::UpFrom(e) | TTruncation::DownFrom(e) => vec![e],
            TTruncation::Between(a, b) => vec![a, b],
        };
        for b in bounds {
            let mut bt = vec![(b.ad.clone(), b.ty.clone())];
            bt.extend(arg_types(&all));
            for suf in ["_lcdf", "_lccdf"] {
                match self.matching_function(&format!("{n}{suf}"), &bt) {
                    Match::Unique {
                        ret: Some(UnsizedType::Real),
                        ..
                    } => {}
                    _ => {
                        return err(
                            span,
                            format!(
                                "Truncation is only defined if distribution has _lcdf and _lccdf functions implemented with appropriate signature. No matching signature for '{n}'."
                            ),
                        );
                    }
                }
            }
        }
        Ok(TStmt {
            kind: TStmtKind::Tilde {
                arg: targ,
                dist: fname,
                dist_kind: kind,
                args: all,
                truncation: ttrunc,
            },
            span,
        })
    }

    fn check_trunc_bound(&mut self, ctx: &Ctx, e: &Expr) -> R<TExpr> {
        let t = self.check_expr(ctx, e)?;
        if !matches!(t.ty, UnsizedType::Int | UnsizedType::Real) {
            return err(
                t.span,
                format!(
                    "Truncation bound must be of type int or real. Instead found type {}.",
                    t.ty
                ),
            );
        }
        Ok(t)
    }

    fn check_sized(&mut self, ctx: &Ctx, t: &SizedType) -> R<TSizedType> {
        let size = |c: &mut Self, e: &Expr, what: &str| c.check_int_expr(ctx, e, what);
        Ok(match t {
            SizedType::Int => TSizedType::Int,
            SizedType::Real => TSizedType::Real,
            SizedType::Complex => TSizedType::Complex,
            SizedType::Vector(e) => TSizedType::Vector(size(self, e, "Vector sizes")?),
            SizedType::RowVector(e) => TSizedType::RowVector(size(self, e, "Row vector sizes")?),
            SizedType::Matrix(a, b) => TSizedType::Matrix(
                size(self, a, "Matrix row size")?,
                size(self, b, "Matrix column size")?,
            ),
            SizedType::ComplexVector(e) => {
                TSizedType::ComplexVector(size(self, e, "complex vector sizes")?)
            }
            SizedType::ComplexRowVector(e) => {
                TSizedType::ComplexRowVector(size(self, e, "complex row vector sizes")?)
            }
            SizedType::ComplexMatrix(a, b) => TSizedType::ComplexMatrix(
                size(self, a, "Complex matrix row size")?,
                size(self, b, "Complex matrix column size")?,
            ),
            SizedType::Array(inner, e) => {
                let ti = self.check_sized(ctx, inner)?;
                TSizedType::Array(Box::new(ti), size(self, e, "Array sizes")?)
            }
            SizedType::Tuple(ts) => TSizedType::Tuple(
                ts.iter()
                    .map(|t| self.check_sized(ctx, t))
                    .collect::<R<_>>()?,
            ),
        })
    }

    fn check_transformation(
        &mut self,
        ctx: &Ctx,
        ut: &UnsizedType,
        t: &Transformation,
    ) -> R<TTransformation> {
        let bound = |c: &mut Self, e: &Expr, what: &str| -> R<TExpr> {
            let te = c.check_expr(ctx, e)?;
            if is_scalar(&te.ty) || te.ty == *ut {
                Ok(te)
            } else {
                err(
                    te.span,
                    format!(
                        "{what} must be a scalar or of type {ut}. Instead found type {}.",
                        te.ty
                    ),
                )
            }
        };
        use TTransformation as TT;
        use Transformation as T;
        Ok(match t {
            T::Identity => TT::Identity,
            T::Lower(e) => TT::Lower(bound(self, e, "Lower bound")?),
            T::Upper(e) => TT::Upper(bound(self, e, "Upper bound")?),
            T::LowerUpper(a, b) => TT::LowerUpper(
                bound(self, a, "Lower bound")?,
                bound(self, b, "Upper bound")?,
            ),
            T::Offset(e) => TT::Offset(bound(self, e, "Offset")?),
            T::Multiplier(e) => TT::Multiplier(bound(self, e, "Multiplier")?),
            T::OffsetMultiplier(a, b) => {
                TT::OffsetMultiplier(bound(self, a, "Offset")?, bound(self, b, "Multiplier")?)
            }
            T::Ordered => TT::Ordered,
            T::PositiveOrdered => TT::PositiveOrdered,
            T::Simplex => TT::Simplex,
            T::UnitVector => TT::UnitVector,
            T::SumToZero => TT::SumToZero,
            T::CholeskyCorr => TT::CholeskyCorr,
            T::CholeskyCov => TT::CholeskyCov,
            T::Correlation => TT::Correlation,
            T::Covariance => TT::Covariance,
            T::StochasticColumn => TT::StochasticColumn,
            T::StochasticRow => TT::StochasticRow,
            T::Tuple(ts) => {
                let UnsizedType::Tuple(uts) = ut.array_dims().1.clone() else {
                    return err(
                        Span::default(),
                        "Internal: tuple transformation on non-tuple type.",
                    );
                };
                TT::Tuple(
                    uts.iter()
                        .zip(ts)
                        .map(|(u, t)| self.check_transformation(ctx, u, t))
                        .collect::<R<_>>()?,
                )
            }
        })
    }

    fn check_var_decl(
        &mut self,
        ctx: &Ctx,
        span: Span,
        ty: &SizedType,
        trans: &Transformation,
        vars: &[DeclVar],
        is_global: bool,
    ) -> R<TStmt> {
        let sctx = Ctx {
            in_toplevel_decl: is_global,
            ..ctx.clone()
        };
        let tty = self.check_sized(&sctx, ty)?;
        let ut = tty.unsized_type();
        let ttrans = self.check_transformation(ctx, &ut, trans)?;
        let mut tvars = Vec::new();
        for v in vars {
            self.verify_identifier(&v.ident)?;
            self.verify_name_fresh_var(&v.ident)?;
            self.env.add(
                &v.ident.name,
                Entry {
                    kind: EntryKind::Variable {
                        origin: ctx.block,
                        global: is_global,
                        readonly: false,
                        span: v.ident.span,
                    },
                    ty: ut.clone(),
                },
            );
            let init = match &v.init {
                None => None,
                Some(e) => {
                    if expr_mentions(e, &v.ident.name) {
                        self.warnings.push(Diagnostic::new(DiagnosticKind::Warning, span, "Assignment of variable to itself during declaration. This is almost certainly a bug."));
                    }
                    let lad = calc_ad(ctx, ctx.block, &ut);
                    let tr = self.check_expr(ctx, e)?;
                    match check_same_type(0, &ut, &tr.ty) {
                        Ok(_) => {
                            let p = type_promotion(&lad, &ut, &tr.ad, &tr.ty);
                            Some(promote(tr, &p))
                        }
                        Err(_) => {
                            return err(
                                span,
                                format!(
                                    "Ill-typed arguments supplied to assignment operator =: lhs has type {} and rhs has type {}.",
                                    ut, tr.ty
                                ),
                            );
                        }
                    }
                }
            };
            tvars.push(TDeclVar {
                name: v.ident.name.clone(),
                span: v.ident.span,
                init,
            });
        }
        // Transformation validity.
        if is_global {
            let real_bound = |e: &TExpr| e.ty == UnsizedType::Real;
            let realt = match &ttrans {
                TTransformation::Lower(e) | TTransformation::Upper(e) => real_bound(e),
                TTransformation::LowerUpper(a, b) => real_bound(a) || real_bound(b),
                _ => false,
            };
            if tty == TSizedType::Int && realt {
                return err(
                    span,
                    "Bounds of integer variable must be of type int. Found type real.",
                );
            }
            if is_complex_type(&ut) && ttrans != TTransformation::Identity {
                return err(span, "Complex types do not support transformations.");
            }
            if matches!(ctx.block, Origin::Param | Origin::TParam) && contains_int(&ut) {
                return err(span, "(Transformed) Parameters cannot be integers.");
            }
        }
        let ad = calc_ad(ctx, ctx.block, &ut);
        Ok(TStmt {
            kind: TStmtKind::VarDecl {
                ty: tty,
                trans: ttrans,
                vars: tvars,
                is_global,
                ad,
            },
            span,
        })
    }

    fn check_fundef(
        &mut self,
        ctx: &Ctx,
        span: Span,
        ret: &Option<UnsizedType>,
        name: &Ident,
        args: &[FunArg],
        body: &Stmt,
    ) -> R<TStmt> {
        for a in args {
            self.verify_identifier(&a.ident)?;
        }
        self.verify_identifier(name)?;
        let is_dist = CONDITIONING_SUFFIXES.iter().any(|s| name.name.ends_with(s));
        if is_dist && *ret != Some(UnsizedType::Real) {
            return err(
                span,
                format!(
                    "Real return type required for probability functions ending in _log, _lpdf, _lupdf, _lpmf, _lupmf, _cdf, _lcdf or _lccdf. Found {}.",
                    ret.as_ref().map(|t| t.to_string()).unwrap_or("void".into())
                ),
            );
        }
        if name.name.ends_with("_lpdf") {
            match args.first() {
                Some(a) if !is_discrete(&a.ty) => {}
                a => {
                    return err(
                        span,
                        format!(
                            "Probability density functions require real variates (first argument). Instead found type {}.",
                            a.map(|a| a.ty.to_string()).unwrap_or_default()
                        ),
                    );
                }
            }
        }
        if name.name.ends_with("_lpmf") {
            match args.first() {
                Some(a) if is_discrete(&a.ty) => {}
                a => {
                    return err(
                        span,
                        format!(
                            "Probability mass functions require integer variates (first argument). Instead found type {}.",
                            a.map(|a| a.ty.to_string()).unwrap_or_default()
                        ),
                    );
                }
            }
        }
        for a in args {
            self.verify_name_fresh_var(&a.ident)?;
        }
        for (i, a) in args.iter().enumerate() {
            if args[..i].iter().any(|b| b.ident.name == a.ident.name) {
                return err(
                    span,
                    format!(
                        "All function arguments must have distinct identifiers. Found duplicate '{}'.",
                        a.ident.name
                    ),
                );
            }
        }
        if matches!(body.kind, StmtKind::Skip) {
            let targs = args
                .iter()
                .map(|a| TFunArg {
                    ad: a.ad,
                    ty: a.ty.clone(),
                    name: a.ident.name.clone(),
                })
                .collect();
            return Ok(TStmt {
                kind: TStmtKind::FunDef {
                    ret: ret.clone(),
                    name: name.name.clone(),
                    args: targs,
                    body: None,
                },
                span,
            });
        }
        self.env.push();
        for a in args {
            let origin = if a.ad == AdLevel::DataOnly {
                Origin::Data
            } else {
                Origin::Param
            };
            self.env.add(
                &a.ident.name,
                Entry {
                    kind: EntryKind::Variable {
                        origin,
                        global: false,
                        readonly: true,
                        span: a.ident.span,
                    },
                    ty: a.ty.clone(),
                },
            );
        }
        let suffix = FnSuffix::from_name(&name.name);
        let fctx = Ctx {
            func: match ret {
                Some(t) => FnCtx::Returning(suffix, t.clone()),
                None => FnCtx::NonReturning(suffix),
            },
            ..ctx.clone()
        };
        let r = self.check_stmt(&fctx, body);
        self.env.pop();
        let (tbody, rt) = r?;
        if ret.is_some() && rt != Rt::Complete {
            return err(
                span,
                "Function bodies must contain a return statement of correct type in every branch.",
            );
        }
        let targs = args
            .iter()
            .map(|a| TFunArg {
                ad: a.ad,
                ty: a.ty.clone(),
                name: a.ident.name.clone(),
            })
            .collect();
        Ok(TStmt {
            kind: TStmtKind::FunDef {
                ret: ret.clone(),
                name: name.name.clone(),
                args: targs,
                body: Some(Box::new(tbody)),
            },
            span,
        })
    }
}

/// Picks the unique cheapest compatible signature.
fn resolve<'a>(
    cands: impl Iterator<Item = (&'a Signature, FunKind)>,
    args: &[(Ad, UnsizedType)],
) -> Match {
    let mut best: Vec<(usize, &'a Signature, FunKind, Vec<Promotion>)> = Vec::new();
    let mut errs = Vec::new();
    for (sig, kind) in cands {
        match check_compatible_args(0, &sig.args, args) {
            Ok(ps) => {
                let cost: usize = ps.iter().map(Promotion::cost).sum();
                match best.first().map(|b| b.0) {
                    Some(c) if cost > c => {}
                    Some(c) if cost < c => best = vec![(cost, sig, kind, ps)],
                    _ => {
                        // Identical signatures (stanc3 lists some twice) count once.
                        if !best.iter().any(|b| b.1.ret == sig.ret && b.3 == ps) {
                            best.push((cost, sig, kind, ps));
                        }
                    }
                }
            }
            Err(e) => errs.push(((sig.ret.clone(), sig.args.clone()), e)),
        }
    }
    match best.len() {
        0 => Match::Errors(errs),
        1 => {
            let (_, sig, kind, promotions) = best.pop().expect("one");
            Match::Unique {
                ret: sig.ret.clone(),
                kind,
                promotions,
            }
        }
        _ => Match::Ambiguous,
    }
}

/// Every function called in a statement: (name, is library function). Function values passed as
/// arguments count as user calls.
fn stmt_calls(s: &TStmt, out: &mut Vec<(String, bool)>) {
    let mut es: Vec<&TExpr> = Vec::new();
    let mut ss: Vec<&TStmt> = Vec::new();
    match &s.kind {
        TStmtKind::Assign { lhs, rhs, .. } => {
            es.push(rhs);
            lvalue_exprs(lhs, &mut es);
        }
        TStmtKind::NRFunApp { kind, name, args } => {
            out.push((name.clone(), *kind == FunKind::StanLib));
            es.extend(args);
        }
        TStmtKind::TargetPlusAssign(e) | TStmtKind::JacobianPlusAssign(e) => es.push(e),
        // As in stanc3, the distribution of a tilde statement is not followed.
        TStmtKind::Tilde { arg, args, .. } => {
            es.push(arg);
            es.extend(args);
        }
        TStmtKind::Return(Some(e)) => es.push(e),
        TStmtKind::Print(ps) | TStmtKind::Reject(ps) | TStmtKind::FatalError(ps) => {
            es.extend(ps.iter().filter_map(|p| match p {
                TPrintable::Expr(e) => Some(e),
                TPrintable::Str(_) => None,
            }))
        }
        TStmtKind::IfThenElse(c, a, b) => {
            es.push(c);
            ss.push(a);
            if let Some(b) = b {
                ss.push(b);
            }
        }
        TStmtKind::While(c, b) => {
            es.push(c);
            ss.push(b);
        }
        TStmtKind::For {
            lower, upper, body, ..
        } => {
            es.push(lower);
            es.push(upper);
            ss.push(body);
        }
        TStmtKind::ForEach { iter, body, .. } => {
            es.push(iter);
            ss.push(body);
        }
        TStmtKind::Profile(_, b) | TStmtKind::Block(b) => ss.extend(b),
        TStmtKind::VarDecl { vars, .. } => es.extend(vars.iter().filter_map(|v| v.init.as_ref())),
        _ => {}
    }
    for e in es {
        expr_calls(e, out);
    }
    for s in ss {
        stmt_calls(s, out);
    }
}

fn lvalue_exprs<'a>(l: &'a TLValue, out: &mut Vec<&'a TExpr>) {
    match &l.kind {
        TLValueKind::Variable { .. } => {}
        TLValueKind::Indexed(i, idx) => {
            lvalue_exprs(i, out);
            out.extend(idx.iter().flat_map(index_exprs));
        }
        TLValueKind::TupleProjection(i, _) => lvalue_exprs(i, out),
        TLValueKind::Tuple(ls) => ls.iter().for_each(|x| lvalue_exprs(x, out)),
    }
}

fn index_exprs(i: &TIndex) -> Vec<&TExpr> {
    match i {
        TIndex::All => vec![],
        TIndex::Single(e) | TIndex::Upfrom(e) | TIndex::Downfrom(e) => vec![e],
        TIndex::Between(a, b) => vec![a, b],
    }
}

fn expr_calls(e: &TExpr, out: &mut Vec<(String, bool)>) {
    match &e.kind {
        TExprKind::FunApp {
            kind, name, args, ..
        } => {
            out.push((signatures::normalized_name(name), *kind == FunKind::StanLib));
            args.iter().for_each(|a| expr_calls(a, out));
        }
        TExprKind::Variable { name, .. } if matches!(e.ty, UnsizedType::Function(_)) => {
            out.push((name.clone(), false))
        }
        TExprKind::ArrayExpr(a) | TExprKind::RowVectorExpr(a) | TExprKind::TupleExpr(a) => {
            a.iter().for_each(|x| expr_calls(x, out))
        }
        TExprKind::TernaryIf(a, b, c) => {
            expr_calls(a, out);
            expr_calls(b, out);
            expr_calls(c, out);
        }
        TExprKind::BinOp(a, _, b) => {
            expr_calls(a, out);
            expr_calls(b, out);
        }
        TExprKind::PrefixOp(_, a)
        | TExprKind::Transpose(a)
        | TExprKind::TupleProjection(a, _)
        | TExprKind::Promotion(a, _) => expr_calls(a, out),
        TExprKind::Indexed(a, idx) => {
            expr_calls(a, out);
            idx.iter()
                .flat_map(index_exprs)
                .for_each(|x| expr_calls(x, out));
        }
        _ => {}
    }
}

fn is_escape(s: &TStmt) -> bool {
    matches!(
        s.kind,
        TStmtKind::Break
            | TStmtKind::Continue
            | TStmtKind::Reject(_)
            | TStmtKind::FatalError(_)
            | TStmtKind::Return(_)
    )
}

fn has_multi_index(l: &TLValue) -> bool {
    match &l.kind {
        TLValueKind::Indexed(inner, idx) => {
            idx.iter().any(TIndex::is_multi) || has_multi_index(inner)
        }
        TLValueKind::TupleProjection(inner, _) => has_multi_index(inner),
        _ => false,
    }
}

/// The type of indexing a value of type `ut`.
fn indexed_type(span: Span, ut: &UnsizedType, idx: &[TIndex]) -> R<UnsizedType> {
    use UnsizedType::*;
    fn aux(
        span: Span,
        orig: &UnsizedType,
        n: usize,
        t: &UnsizedType,
        idx: &[bool],
    ) -> R<UnsizedType> {
        let (vec, rowvec, scalar) = if is_complex_type(t) {
            (ComplexVector, ComplexRowVector, Complex)
        } else {
            (Vector, RowVector, Real)
        };
        let not_indexable = || {
            err(
                span,
                format!(
                    "Too many indexes, expression dimensions={}, indexes found={}.",
                    dims_of(orig),
                    n
                ),
            )
        };
        Ok(match (t, idx) {
            (_, []) => t.clone(),
            (Array(inner), [false, rest @ ..]) => aux(span, orig, n, inner, rest)?,
            (Array(inner), [true, rest @ ..]) => Array(Box::new(aux(span, orig, n, inner, rest)?)),
            (Vector | RowVector | ComplexVector | ComplexRowVector, [false])
            | (Matrix | ComplexMatrix, [false, false]) => scalar,
            (
                Vector | RowVector | Matrix | ComplexVector | ComplexRowVector | ComplexMatrix,
                [true],
            )
            | (Matrix | ComplexMatrix, [true, true]) => t.clone(),
            (Matrix | ComplexMatrix, [false] | [false, true]) => rowvec,
            (Matrix | ComplexMatrix, [true, false]) => vec,
            _ => return not_indexable(),
        })
    }
    let flags: Vec<bool> = idx.iter().map(TIndex::is_multi).collect();
    aux(span, ut, idx.len(), ut, &flags)
}

fn dims_of(t: &UnsizedType) -> usize {
    match t {
        UnsizedType::Array(i) => 1 + dims_of(i),
        UnsizedType::Vector
        | UnsizedType::RowVector
        | UnsizedType::ComplexVector
        | UnsizedType::ComplexRowVector => 1,
        UnsizedType::Matrix | UnsizedType::ComplexMatrix => 2,
        _ => 0,
    }
}

fn indexed_ad(base: &Ad, ty: &UnsizedType, idx: &[TIndex]) -> Ad {
    let mut ads = vec![base.clone()];
    for i in idx {
        match i {
            TIndex::All => {}
            TIndex::Single(e) | TIndex::Upfrom(e) | TIndex::Downfrom(e) => ads.push(e.ad.clone()),
            TIndex::Between(a, b) => {
                ads.push(a.ad.clone());
                ads.push(b.ad.clone());
            }
        }
    }
    let lub = Ad::lub_all(ads.iter()).unwrap_or(Ad::DataOnly);
    // Projection of tuple autodiff levels through indexing keeps the base's structure.
    match (&lub, ty.array_dims().1) {
        (Ad::Tuple(_), UnsizedType::Tuple(_)) => lub,
        (Ad::Tuple(xs), _) => {
            if xs.iter().any(Ad::has_autodiff) {
                Ad::AutoDiffable
            } else {
                Ad::DataOnly
            }
        }
        _ => lub.fill_for(ty),
    }
}

fn assign_promotion(l: &TLValue, r: &TExpr) -> Option<Promotion> {
    match (&l.kind, &r.ty, &r.ad) {
        (TLValueKind::Tuple(ls), UnsizedType::Tuple(ts), Ad::Tuple(ads)) => {
            if ls.len() != ts.len() {
                return None;
            }
            let mut ps = Vec::new();
            for ((l, t), a) in ls.iter().zip(ts).zip(ads) {
                let fake = TExpr {
                    kind: TExprKind::IntLit(0),
                    ty: t.clone(),
                    ad: a.clone(),
                    span: r.span,
                };
                ps.push(assign_promotion(l, &fake)?);
            }
            Some(Promotion::Tuple(ps))
        }
        (TLValueKind::Tuple(_), _, _) => None,
        _ => {
            if is_fun_type(&r.ty) {
                return None;
            }
            check_same_type(0, &l.ty, &r.ty).ok()?;
            Some(type_promotion(&l.ad, &l.ty, &r.ad, &r.ty))
        }
    }
}

fn lvalue_base_name(l: &TLValue) -> Option<&str> {
    match &l.kind {
        TLValueKind::Variable { name, .. } => Some(name),
        TLValueKind::Indexed(i, _) | TLValueKind::TupleProjection(i, _) => lvalue_base_name(i),
        TLValueKind::Tuple(_) => None,
    }
}

fn verify_lvalue_unique(l: &TLValue) -> R<()> {
    fn flat<'a>(l: &'a TLValue, out: &mut Vec<&'a TLValue>) {
        match &l.kind {
            TLValueKind::Tuple(ls) => ls.iter().for_each(|x| flat(x, out)),
            _ => out.push(l),
        }
    }
    fn path(l: &TLValue) -> (String, Vec<usize>) {
        match &l.kind {
            TLValueKind::Variable { name, .. } => (name.clone(), vec![]),
            TLValueKind::Indexed(i, _) => path(i),
            TLValueKind::TupleProjection(i, k) => {
                let (n, mut p) = path(i);
                p.push(*k);
                (n, p)
            }
            TLValueKind::Tuple(_) => (String::new(), vec![]),
        }
    }
    let mut all = Vec::new();
    flat(l, &mut all);
    let paths: Vec<(String, Vec<usize>)> = all.iter().map(|x| path(x)).collect();
    for i in 0..paths.len() {
        for j in 0..i {
            let (a, pa) = &paths[i];
            let (b, pb) = &paths[j];
            if a == b && (pa.starts_with(pb) || pb.starts_with(pa)) {
                return err(
                    l.span,
                    format!(
                        "The same value cannot be assigned to multiple times in one assignment: {a}"
                    ),
                );
            }
        }
    }
    // Variables read in indices of the lvalue may not be assigned in the same statement.
    let names: Vec<&str> = all.iter().filter_map(|x| lvalue_base_name(x)).collect();
    fn idx_vars(l: &TLValue, out: &mut Vec<String>) {
        match &l.kind {
            TLValueKind::Indexed(i, idx) => {
                idx_vars(i, out);
                for ix in idx {
                    let es: Vec<&TExpr> = match ix {
                        TIndex::All => vec![],
                        TIndex::Single(e) | TIndex::Upfrom(e) | TIndex::Downfrom(e) => vec![e],
                        TIndex::Between(a, b) => vec![a, b],
                    };
                    for e in es {
                        texpr_vars(e, out);
                    }
                }
            }
            TLValueKind::TupleProjection(i, _) => idx_vars(i, out),
            TLValueKind::Tuple(ls) => ls.iter().for_each(|x| idx_vars(x, out)),
            TLValueKind::Variable { .. } => {}
        }
    }
    let mut read = Vec::new();
    idx_vars(l, &mut read);
    if let Some(n) = read.iter().find(|r| names.contains(&r.as_str())) {
        return err(
            l.span,
            format!(
                "The same variable cannot be both assigned to and read from on the left hand side of an assignment: {n}"
            ),
        );
    }
    Ok(())
}

/// Every variable name an expression mentions.
pub fn texpr_vars(e: &TExpr, out: &mut Vec<String>) {
    match &e.kind {
        TExprKind::Variable { name, .. } => out.push(name.clone()),
        TExprKind::FunApp { args, .. }
        | TExprKind::ArrayExpr(args)
        | TExprKind::RowVectorExpr(args)
        | TExprKind::TupleExpr(args) => args.iter().for_each(|a| texpr_vars(a, out)),
        TExprKind::TernaryIf(a, b, c) => {
            texpr_vars(a, out);
            texpr_vars(b, out);
            texpr_vars(c, out);
        }
        TExprKind::BinOp(a, _, b) => {
            texpr_vars(a, out);
            texpr_vars(b, out);
        }
        TExprKind::PrefixOp(_, a)
        | TExprKind::Transpose(a)
        | TExprKind::TupleProjection(a, _)
        | TExprKind::Promotion(a, _) => texpr_vars(a, out),
        TExprKind::Indexed(a, idx) => {
            texpr_vars(a, out);
            for ix in idx {
                match ix {
                    TIndex::All => {}
                    TIndex::Single(e) | TIndex::Upfrom(e) | TIndex::Downfrom(e) => {
                        texpr_vars(e, out)
                    }
                    TIndex::Between(a, b) => {
                        texpr_vars(a, out);
                        texpr_vars(b, out);
                    }
                }
            }
        }
        _ => {}
    }
}

fn expr_mentions(e: &Expr, name: &str) -> bool {
    let rec = |x: &Expr| expr_mentions(x, name);
    match &e.kind {
        ExprKind::Variable(id) => id.name == name,
        ExprKind::FunApp(_, a)
        | ExprKind::CondDistApp(_, a)
        | ExprKind::ArrayExpr(a)
        | ExprKind::RowVectorExpr(a)
        | ExprKind::TupleExpr(a) => a.iter().any(rec),
        ExprKind::TernaryIf(a, b, c) => rec(a) || rec(b) || rec(c),
        ExprKind::BinOp(a, _, b) => rec(a) || rec(b),
        ExprKind::PrefixOp(_, a)
        | ExprKind::Transpose(a)
        | ExprKind::Paren(a)
        | ExprKind::TupleProjection(a, _) => rec(a),
        ExprKind::Indexed(a, idx) => {
            rec(a)
                || idx.iter().any(|i| match i {
                    Index::All => false,
                    Index::Single(x) | Index::Upfrom(x) | Index::Downfrom(x) => rec(x),
                    Index::Between(x, y) => rec(x) || rec(y),
                })
        }
        _ => false,
    }
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let c = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + c);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// The argument list a higher-order call was checked against, and why it does not match.
type VariadicError = (Vec<(AdLevel, UnsizedType)>, Mismatch);

/// stanc3's `check_variadic_args`: `args` starts with the function argument. On failure, returns
/// the argument list the call was checked against and the mismatch.
fn check_variadic_args(
    allow_lpdf: bool,
    mandatory_args: &[(AdLevel, UnsizedType)],
    mandatory_fn_args: &[(AdLevel, UnsizedType)],
    fn_ret: &UnsizedType,
    args: &[(Ad, UnsizedType)],
) -> Result<Vec<Promotion>, VariadicError> {
    let minimal = UnsizedType::Function(Box::new(FunctionType {
        ret: Some(fn_ret.clone()),
        args: mandatory_fn_args.to_vec(),
        suffix: FnSuffix::Plain,
    }));
    let mut minimal_args = vec![(AdLevel::AutoDiffable, minimal.clone())];
    minimal_args.extend(mandatory_args.iter().cloned());
    let wrap = |x| Err((minimal_args.clone(), Mismatch::Arg(1, Box::new(x))));
    match args.first() {
        Some((_, fty @ UnsizedType::Function(ft))) if ft.ret.is_some() => {
            let wrap_fn = |d| {
                wrap(Mismatch::Type(
                    minimal.clone(),
                    fty.clone(),
                    Some(Box::new(d)),
                ))
            };
            let (mandatory, extra) = ft.args.split_at(mandatory_fn_args.len().min(ft.args.len()));
            let ok_suffix = ft.suffix == FnSuffix::Plain
                || (allow_lpdf && matches!(ft.suffix, FnSuffix::Lpdf | FnSuffix::Lpmf));
            if !ok_suffix {
                return wrap_fn(Details::Suffix(FnSuffix::Plain, ft.suffix));
            }
            let want: Vec<(Ad, UnsizedType)> = mandatory_fn_args
                .iter()
                .map(|(a, t)| ((*a).into(), t.clone()))
                .collect();
            if let Err(x) = check_compatible_args(1, mandatory, &want) {
                return wrap_fn(Details::Input(Box::new(x)));
            }
            let ret = ft.ret.as_ref().expect("returning");
            if check_same_type(1, ret, fn_ret).is_err() {
                return wrap_fn(Details::ReturnType(Some(fn_ret.clone()), ft.ret.clone()));
            }
            let mut expected = vec![(AdLevel::AutoDiffable, fty.clone())];
            expected.extend(mandatory_args.iter().cloned());
            expected.extend(extra.iter().cloned());
            check_compatible_args(0, &expected, args).map_err(|x| (expected, x))
        }
        Some((_, x)) => wrap(Mismatch::Type(minimal.clone(), x.clone(), None)),
        None => Err((Vec::new(), Mismatch::ArgCount(mandatory_args.len(), 0))),
    }
}

/// stanc3's `specialize_loc`: the argument an error concerns.
fn specialize_span(span: Span, e: &Mismatch, args: &[TExpr]) -> Span {
    let which = match e {
        Mismatch::Arg(i, _) => *i as isize - 1,
        Mismatch::ArgCount(expected, found) => {
            if expected > found {
                *found as isize - 1
            } else {
                *expected as isize
            }
        }
        _ => 0,
    };
    args.get(which.max(0) as usize)
        .map(|a| a.span)
        .unwrap_or(span)
}
