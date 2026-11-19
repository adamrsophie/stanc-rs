//! Lowering of the typed program to the MIR, a port of stanc3 v2.40.0's `Ast_to_Mir`.

use std::cell::Cell;
use std::collections::HashMap;

use stanc_rs_semantic::signatures;
use stanc_rs_semantic::typed::{
    FunKind as TFunKind, Origin, TExpr, TExprKind, TIndex, TLValue, TLValueKind, TPrintable,
    TProgram, TSizedType, TStmt, TStmtKind, TTransformation, TTruncation,
};
use stanc_rs_semantic::types::Ad;
use stanc_rs_syntax::Span;
use stanc_rs_syntax::ast::{self, AssignOp, BinOp, PrefixOp, UnsizedType};

use crate::mir::*;
use crate::pretty;

/// stanc3's `Gensym`: a counter of generated names, reset for every program.
#[derive(Default)]
struct Gensym(Cell<usize>);

impl Gensym {
    fn generate(&self) -> String {
        self.0.set(self.0.get() + 1);
        format!("sym{}__", self.0.get())
    }

    /// A new name and the counter value to restore once its scope ends.
    fn enter(&self) -> (String, usize) {
        let old = self.0.get();
        (self.generate(), old)
    }

    fn reset(&self, old: usize) {
        self.0.set(old);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Action {
    Check,
    Constrain,
    Ignore,
}

#[derive(Clone, Copy)]
struct DeclCtx {
    action: Action,
    ad: Ad2,
}

/// `DataOnly` or `AutoDiffable`, the levels a declaration context sets.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ad2 {
    Data,
    Var,
}

impl Ad2 {
    fn ad(self) -> Ad {
        match self {
            Ad2::Data => Ad::DataOnly,
            Ad2::Var => Ad::AutoDiffable,
        }
    }
}

struct Lower<'a> {
    sym: Gensym,
    /// User-defined functions and the types of their first arguments.
    ud_dists: Vec<(String, UnsizedType)>,
    /// The untyped sized types of declarations, by statement span, for printing sizes as
    /// written.
    decl_types: HashMap<Span, &'a ast::SizedType>,
}

fn loc(s: Span) -> Loc {
    Loc::of_span(s)
}

fn without_underscores(s: &str) -> String {
    s.replace('_', "")
}

fn drop_leading_zeros(s: &str) -> String {
    match s.find(|c| c != '0') {
        Some(p) if p > 0 => match s.as_bytes()[p] {
            b'e' | b'E' | b'.' => s[p - 1..].to_string(),
            _ => s[p..].to_string(),
        },
        Some(_) => s.to_string(),
        None => "0".to_string(),
    }
}

fn format_number(s: &str) -> String {
    drop_leading_zeros(&without_underscores(s))
}

fn binop_operator(op: BinOp) -> Operator {
    match op {
        BinOp::Plus => Operator::Plus,
        BinOp::Minus => Operator::Minus,
        BinOp::Times => Operator::Times,
        BinOp::Divide => Operator::Divide,
        BinOp::IntDivide => Operator::IntDivide,
        BinOp::Modulo => Operator::Modulo,
        BinOp::LDivide => Operator::LDivide,
        BinOp::EltTimes => Operator::EltTimes,
        BinOp::EltDivide => Operator::EltDivide,
        BinOp::Pow => Operator::Pow,
        BinOp::EltPow => Operator::EltPow,
        BinOp::Or => Operator::Or,
        BinOp::And => Operator::And,
        BinOp::Equals => Operator::Equals,
        BinOp::NEquals => Operator::NEquals,
        BinOp::Less => Operator::Less,
        BinOp::Leq => Operator::Leq,
        BinOp::Greater => Operator::Greater,
        BinOp::Geq => Operator::Geq,
    }
}

fn prefix_operator(op: PrefixOp) -> Operator {
    match op {
        PrefixOp::Not => Operator::PNot,
        PrefixOp::Minus => Operator::PMinus,
        PrefixOp::Plus => Operator::PPlus,
    }
}

/// `Utils.stdlib_distribution_name`: `_lupdf` to `_lpdf`, `_lupmf` to `_lpmf`.
fn stdlib_distribution_name(name: &str) -> String {
    signatures::normalized_name(name)
}

fn trans_fn_kind(kind: &TFunKind, name: &str, suffix: Suffix) -> FunKind {
    let fname = stdlib_distribution_name(name);
    match kind {
        TFunKind::StanLib => FunKind::StanLib(fname, suffix),
        TFunKind::UserDefined => FunKind::UserDefined(fname, suffix),
    }
}

fn lub_ad<'a>(xs: impl IntoIterator<Item = &'a Ad>) -> Ad {
    xs.into_iter()
        .try_fold(Ad::DataOnly, |a, b| a.lub(b))
        .unwrap_or(Ad::AutoDiffable)
}

fn is_container(t: &UnsizedType) -> bool {
    use UnsizedType::*;
    match t {
        Vector | RowVector | Matrix | Array(_) | Tuple(_) | ComplexRowVector | ComplexVector
        | ComplexMatrix => true,
        Function(ft) => ft.ret.as_ref().is_some_and(is_container),
        _ => false,
    }
}

/// `Promotion.scalarize`: promotions record the scalar type they promote to.
fn scalarize(t: &UnsizedType) -> UnsizedType {
    use UnsizedType::*;
    match t.array_dims().1 {
        Tuple(ts) => Tuple(ts.iter().map(scalarize).collect()),
        Int => Int,
        Real | Vector | RowVector | Matrix => Real,
        Complex | ComplexVector | ComplexRowVector | ComplexMatrix => Complex,
        t => t.clone(),
    }
}

fn is_scalar_type(t: &UnsizedType) -> bool {
    matches!(t, UnsizedType::Int | UnsizedType::Real)
}

fn is_int_type(t: &UnsizedType) -> bool {
    match t {
        UnsizedType::Int => true,
        UnsizedType::Array(t) => is_int_type(t),
        _ => false,
    }
}

/// `UnsizedType.fill_adtype_for_type`.
fn fill_ad(ad: &Ad, ut: &UnsizedType) -> Ad {
    match (ad, ut) {
        (_, UnsizedType::Array(t)) => fill_ad(ad, t),
        (Ad::Tuple(ads), UnsizedType::Tuple(ts)) => {
            Ad::Tuple(ads.iter().zip(ts).map(|(a, t)| fill_ad(a, t)).collect())
        }
        (_, UnsizedType::Tuple(ts)) => Ad::Tuple(ts.iter().map(|t| fill_ad(ad, t)).collect()),
        _ => ad.clone(),
    }
}

fn meta_of(e: &TExpr) -> Meta {
    Meta {
        ty: e.ty.clone(),
        loc: loc(e.span),
        ad: e.ad.clone(),
    }
}

/// `Expr.Helpers.add_int_index`.
pub(crate) fn add_int_index(e: Expr, i: Index<Expr>) -> Expr {
    let ty = infer_type_of_indexed(&e.meta.ty, std::slice::from_ref(&i));
    let meta = Meta {
        ty,
        ..e.meta.clone()
    };
    let pattern = match e.pattern {
        ExprP::Indexed(obj, mut idx) => {
            idx.push(i);
            ExprP::Indexed(obj, idx)
        }
        p => ExprP::Indexed(
            Box::new(Expr {
                pattern: p,
                meta: e.meta,
            }),
            vec![i],
        ),
    };
    Expr { pattern, meta }
}

/// `Expr.Helpers.add_tuple_index`.
fn add_tuple_index(e: Expr, i: usize) -> Expr {
    let ty = match &e.meta.ty {
        UnsizedType::Tuple(ts) => ts[i - 1].clone(),
        t => t.clone(),
    };
    let meta = Meta {
        ty,
        ..e.meta.clone()
    };
    Expr {
        pattern: ExprP::TupleProjection(Box::new(e), i),
        meta,
    }
}

fn op_to_funapp(l: &Lower, op: Operator, args: &[&TExpr], ty: &UnsizedType) -> Expr {
    let lc = match args {
        [] => Loc::EMPTY,
        [a] => loc(a.span),
        [a, rest @ ..] => rest
            .iter()
            .fold(loc(a.span), |acc, e| acc.merge(loc(e.span))),
    };
    let ad = lub_ad(args.iter().map(|a| &a.ad));
    Expr::new(
        ExprP::FunApp(FunKind::op(op), args.iter().map(|a| l.expr(a)).collect()),
        Meta {
            ty: ty.clone(),
            loc: lc,
            ad,
        },
    )
}

impl<'a> Lower<'a> {
    fn expr(&self, e: &TExpr) -> Expr {
        let w = |p: ExprP| Expr::new(p, meta_of(e));
        let b = |x: &TExpr| Box::new(self.expr(x));
        match &e.kind {
            TExprKind::Variable { name, .. } => w(ExprP::Var(name.clone())),
            TExprKind::IntLit(i) => w(ExprP::Lit(LitType::Int, i.to_string())),
            TExprKind::RealLit(s) => w(ExprP::Lit(LitType::Real, format_number(s))),
            TExprKind::ImagLit(s) => w(ExprP::Lit(LitType::Imaginary, format_number(s))),
            TExprKind::FunApp {
                kind, name, args, ..
            } => w(ExprP::FunApp(
                trans_fn_kind(kind, name, Suffix::from_name(name)),
                self.exprs(args),
            )),
            TExprKind::GetTarget => w(ExprP::FunApp(
                FunKind::StanLib("target".to_string(), Suffix::Target),
                vec![],
            )),
            TExprKind::TernaryIf(c, t, f) => w(ExprP::TernaryIf(b(c), b(t), b(f))),
            TExprKind::BinOp(x, BinOp::And, y) => w(ExprP::EAnd(b(x), b(y))),
            TExprKind::BinOp(x, BinOp::Or, y) => w(ExprP::EOr(b(x), b(y))),
            TExprKind::BinOp(x, op, y) => op_to_funapp(self, binop_operator(*op), &[x, y], &e.ty),
            TExprKind::PrefixOp(op, x) => op_to_funapp(self, prefix_operator(*op), &[x], &e.ty),
            TExprKind::Transpose(x) => op_to_funapp(self, Operator::Transpose, &[x], &e.ty),
            TExprKind::Indexed(x, idx) => w(ExprP::Indexed(
                b(x),
                idx.iter().map(|i| self.index(i)).collect(),
            )),
            TExprKind::ArrayExpr(xs) => w(ExprP::FunApp(
                FunKind::CompilerInternal(Internal::MakeArray),
                self.exprs(xs),
            )),
            TExprKind::RowVectorExpr(xs) => w(ExprP::FunApp(
                FunKind::CompilerInternal(Internal::MakeRowVec),
                self.exprs(xs),
            )),
            TExprKind::TupleExpr(xs) => w(ExprP::FunApp(
                FunKind::CompilerInternal(Internal::MakeTuple),
                self.exprs(xs),
            )),
            TExprKind::TupleProjection(x, i) => w(ExprP::TupleProjection(b(x), *i)),
            TExprKind::Promotion(x, _) => w(ExprP::Promotion(b(x), scalarize(&e.ty), e.ad.clone())),
        }
    }

    fn exprs(&self, es: &[TExpr]) -> Vec<Expr> {
        es.iter().map(|e| self.expr(e)).collect()
    }

    fn index(&self, i: &TIndex) -> Index<Expr> {
        match i {
            TIndex::All => Index::All,
            TIndex::Upfrom(e) => Index::Upfrom(self.expr(e)),
            TIndex::Downfrom(e) => Index::Between(Expr::int(1), self.expr(e)),
            TIndex::Between(a, b) => Index::Between(self.expr(a), self.expr(b)),
            TIndex::Single(e) => {
                if e.ty == UnsizedType::Int {
                    Index::Single(self.expr(e))
                } else {
                    Index::MultiIndex(self.expr(e))
                }
            }
        }
    }

    fn sized_type(&self, st: &TSizedType) -> SizedType {
        match st {
            TSizedType::Int => SizedType::Int,
            TSizedType::Real => SizedType::Real,
            TSizedType::Complex => SizedType::Complex,
            TSizedType::Vector(e) => SizedType::Vector(self.expr(e)),
            TSizedType::RowVector(e) => SizedType::RowVector(self.expr(e)),
            TSizedType::Matrix(a, b) => SizedType::Matrix(self.expr(a), self.expr(b)),
            TSizedType::ComplexVector(e) => SizedType::ComplexVector(self.expr(e)),
            TSizedType::ComplexRowVector(e) => SizedType::ComplexRowVector(self.expr(e)),
            TSizedType::ComplexMatrix(a, b) => SizedType::ComplexMatrix(self.expr(a), self.expr(b)),
            TSizedType::Array(t, e) => SizedType::Array(Box::new(self.sized_type(t)), self.expr(e)),
            TSizedType::Tuple(ts) => {
                SizedType::Tuple(ts.iter().map(|t| self.sized_type(t)).collect())
            }
        }
    }

    fn transformation(&self, t: &TTransformation) -> Transformation {
        use TTransformation as T;
        match t {
            T::Identity => Transformation::Identity,
            T::Lower(e) => Transformation::Lower(self.expr(e)),
            T::Upper(e) => Transformation::Upper(self.expr(e)),
            T::LowerUpper(a, b) => Transformation::LowerUpper(self.expr(a), self.expr(b)),
            T::Offset(e) => Transformation::Offset(self.expr(e)),
            T::Multiplier(e) => Transformation::Multiplier(self.expr(e)),
            T::OffsetMultiplier(a, b) => {
                Transformation::OffsetMultiplier(self.expr(a), self.expr(b))
            }
            T::Ordered => Transformation::Ordered,
            T::PositiveOrdered => Transformation::PositiveOrdered,
            T::Simplex => Transformation::Simplex,
            T::UnitVector => Transformation::UnitVector,
            T::SumToZero => Transformation::SumToZero,
            T::CholeskyCorr => Transformation::CholeskyCorr,
            T::CholeskyCov => Transformation::CholeskyCov,
            T::Correlation => Transformation::Correlation,
            T::Covariance => Transformation::Covariance,
            T::StochasticColumn => Transformation::StochasticColumn,
            T::StochasticRow => Transformation::StochasticRow,
            T::Tuple(ts) => {
                Transformation::Tuple(ts.iter().map(|t| self.transformation(t)).collect())
            }
        }
    }

    /// `Stmt.Helpers.temp_vars`: declares temporaries for the expressions that are not
    /// variables. Returns the setup statements, the replaced expressions, and the counter value
    /// to restore.
    fn temp_vars(&self, es: Vec<Expr>) -> (Vec<Stmt>, Vec<Expr>, usize) {
        let (mut sym, old) = self.sym.enter();
        let mut inits: Vec<Stmt> = Vec::new();
        let mut vars: Vec<Expr> = Vec::new();
        for e in es.into_iter().rev() {
            if matches!(e.pattern, ExprP::Var(_)) {
                vars.insert(0, e);
                continue;
            }
            let decl = Stmt::new(
                StmtP::Decl {
                    ad: e.meta.ad.clone(),
                    id: sym.clone(),
                    ty: DeclType::Unsized(e.meta.ty.clone()),
                    init: DeclInit::Default,
                },
                e.meta.loc,
            );
            let assign = Stmt::new(
                StmtP::Assignment(LValue::var(&sym), e.meta.ty.clone(), e.clone()),
                e.meta.loc,
            );
            let var = Expr {
                pattern: ExprP::Var(sym.clone()),
                meta: e.meta,
            };
            inits.splice(0..0, [decl, assign]);
            vars.insert(0, var);
            sym = self.sym.generate();
        }
        (inits, vars, old)
    }

    /// `Stmt.Helpers.mk_for`: a loop from 1 to `upper` with `body` applied to the loop
    /// variable.
    fn mk_for(&self, upper: Expr, body: &mut dyn FnMut(&Self, Expr) -> Stmt, l: Loc) -> Stmt {
        let (loopvar, old) = self.sym.enter();
        let lv = Expr::new(
            ExprP::Var(loopvar.clone()),
            Meta {
                ty: UnsizedType::Int,
                loc: l,
                ad: Ad::DataOnly,
            },
        );
        let inner = body(self, lv);
        let body = Stmt::new(StmtP::Block(vec![inner]), l);
        self.sym.reset(old);
        Stmt::new(
            StmtP::For {
                loopvar,
                lower: Expr::int(1),
                upper,
                body: Box::new(body),
            },
            l,
        )
    }

    /// `Stmt.Helpers.mk_nested_for`.
    fn mk_nested_for(
        &self,
        uppers: &[Expr],
        body: &mut dyn FnMut(&Self, Vec<Expr>) -> Stmt,
        l: Loc,
    ) -> Stmt {
        match uppers {
            [] => body(self, vec![]),
            [upper, rest @ ..] => self.mk_for(
                upper.clone(),
                &mut |me: &Self, lv: Expr| {
                    me.mk_nested_for(
                        rest,
                        &mut |me: &Self, mut lvs: Vec<Expr>| {
                            lvs.insert(0, lv.clone());
                            body(me, lvs)
                        },
                        Loc::EMPTY,
                    )
                },
                l,
            ),
        }
    }

    fn length(e: &Expr) -> Expr {
        Expr::internal(
            Internal::Length,
            vec![e.clone()],
            Meta {
                ty: UnsizedType::Int,
                ..e.meta.clone()
            },
        )
    }

    /// `Stmt.Helpers.for_each`.
    fn for_each(&self, body: &mut dyn FnMut(&Self, Expr) -> Stmt, it: Expr, l: Loc) -> Stmt {
        use UnsizedType::*;
        match it.meta.ty.clone() {
            Int | Real | Complex => body(self, it),
            Vector | RowVector | ComplexVector | ComplexRowVector | Array(_) => {
                let upper = Self::length(&it);
                self.mk_for(
                    upper,
                    &mut |me: &Self, lv: Expr| {
                        body(me, add_int_index(it.clone(), Index::Single(lv)))
                    },
                    l,
                )
            }
            Matrix | ComplexMatrix => {
                let rows = Expr::new(
                    ExprP::FunApp(
                        FunKind::StanLib("rows".to_string(), Suffix::Plain),
                        vec![it.clone()],
                    ),
                    Meta {
                        ty: Int,
                        ..it.meta.clone()
                    },
                );
                self.mk_for(
                    rows,
                    &mut |me: &Self, lv: Expr| {
                        me.for_each(body, add_int_index(it.clone(), Index::Single(lv)), l)
                    },
                    l,
                )
            }
            _ => body(self, it),
        }
    }

    // -- Truncation --

    fn truncate_dist(&self, dist: &str, obs: &TExpr, args: &[TExpr], t: &TTruncation) -> Vec<Stmt> {
        let find_function_info = |sfx: &str| {
            let name = format!("{dist}{sfx}");
            match self.ud_dists.iter().find(|(n, _)| *n == name) {
                Some((n, tp)) => (TFunKind::UserDefined, n.clone(), tp.clone()),
                None => (
                    TFunKind::StanLib,
                    name,
                    if signatures::is_stan_math_function(&format!("{dist}_lpmf")) {
                        UnsizedType::Int
                    } else {
                        UnsizedType::Real
                    },
                ),
            }
        };
        let targetme =
            |l: Loc, e: Expr| Stmt::new(StmtP::TargetPE(Expr::unary_op(Operator::PMinus, e)), l);
        let neg_inf = || {
            Expr::internal(
                Internal::NegInf,
                vec![],
                Meta {
                    ty: UnsizedType::Real,
                    loc: Loc::EMPTY,
                    ad: Ad::DataOnly,
                },
            )
        };
        let trunc = |cond: Operator, extrema: &str, x: &Expr, y: Stmt| {
            let smeta = x.meta.loc;
            let obs_e = if is_container(&obs.ty) {
                Expr::new(
                    ExprP::FunApp(
                        FunKind::StanLib(extrema.to_string(), Suffix::Plain),
                        vec![self.expr(obs)],
                    ),
                    Meta {
                        ty: UnsizedType::Real,
                        loc: smeta,
                        ad: obs.ad.clone(),
                    },
                )
            } else {
                self.expr(obs)
            };
            Stmt::new(
                StmtP::IfElse(
                    Expr::binop(obs_e, cond, x.clone()),
                    Box::new(Stmt::new(StmtP::TargetPE(neg_inf()), smeta)),
                    Some(Box::new(y)),
                ),
                smeta,
            )
        };
        let funapp = |meta: &Meta, kind: &TFunKind, name: &str, args: Vec<Expr>| {
            Expr::new(
                ExprP::FunApp(trans_fn_kind(kind, name, Suffix::Plain), args),
                meta.clone(),
            )
        };
        let maybe_promote = |tp: &UnsizedType, lb: &Expr| -> Expr {
            if *tp == UnsizedType::Int {
                lb.clone()
            } else if lb.meta.ty == UnsizedType::Int {
                Expr::new(
                    ExprP::Promotion(Box::new(lb.clone()), UnsizedType::Real, lb.meta.ad.clone()),
                    Meta {
                        ty: UnsizedType::Real,
                        ..lb.meta.clone()
                    },
                )
            } else {
                lb.clone()
            }
        };
        let inclusive_bound = |tp: &UnsizedType, lb: &Expr| {
            if is_int_type(tp) {
                Expr::binop(lb.clone(), Operator::Minus, Expr::int(1))
            } else {
                maybe_promote(tp, lb)
            }
        };
        let size_adjust = |e: Expr| {
            if !is_container(&obs.ty) || args.iter().any(|a| is_container(&a.ty)) {
                e
            } else {
                let o = self.expr(obs);
                let m = Meta {
                    ty: UnsizedType::Real,
                    ..o.meta.clone()
                };
                Expr::binop(
                    e,
                    Operator::Times,
                    Expr::internal(Internal::Length, vec![o], m),
                )
            }
        };
        let with = |first: Expr, rest: Vec<Expr>| {
            let mut v = vec![first];
            v.extend(rest);
            v
        };
        match t {
            TTruncation::None => vec![],
            TTruncation::UpFrom(lb) => {
                let (fk, fname, tp) = find_function_info("_lccdf");
                let lb = self.expr(lb);
                let call = funapp(
                    &lb.meta,
                    &fk,
                    &fname,
                    with(inclusive_bound(&tp, &lb), self.exprs(args)),
                );
                vec![trunc(
                    Operator::Less,
                    "min",
                    &lb,
                    targetme(lb.meta.loc, size_adjust(call)),
                )]
            }
            TTruncation::DownFrom(ub) => {
                let (fk, fname, tp) = find_function_info("_lcdf");
                let ub = self.expr(ub);
                let call = funapp(
                    &ub.meta,
                    &fk,
                    &fname,
                    with(maybe_promote(&tp, &ub), self.exprs(args)),
                );
                vec![trunc(
                    Operator::Greater,
                    "max",
                    &ub,
                    targetme(ub.meta.loc, size_adjust(call)),
                )]
            }
            TTruncation::Between(lb, ub) => {
                let (fk, fname, tp) = find_function_info("_lcdf");
                let lb = self.expr(lb);
                let ub = self.expr(ub);
                let expr = |args: Vec<Expr>| {
                    funapp(
                        &ub.meta,
                        &TFunKind::StanLib,
                        "log_diff_exp",
                        vec![
                            funapp(
                                &ub.meta,
                                &fk,
                                &fname,
                                with(maybe_promote(&tp, &ub), args.clone()),
                            ),
                            funapp(&ub.meta, &fk, &fname, with(inclusive_bound(&tp, &lb), args)),
                        ],
                    )
                };
                let statement = match args.iter().position(|a| is_container(&a.ty)) {
                    Some(i) => {
                        let (temp_decls, targs, old) = self.temp_vars(self.exprs(args));
                        let bound = {
                            let e = &targs[i];
                            Self::length(e)
                        };
                        let lp = self.mk_for(
                            bound,
                            &mut |_: &Self, idx: Expr| {
                                let a = targs
                                    .iter()
                                    .map(|e| {
                                        if is_container(&e.meta.ty) {
                                            add_int_index(e.clone(), Index::Single(idx.clone()))
                                        } else {
                                            e.clone()
                                        }
                                    })
                                    .collect();
                                targetme(ub.meta.loc, size_adjust(expr(a)))
                            },
                            ub.meta.loc,
                        );
                        self.sym.reset(old);
                        let l = lp.loc;
                        let mut v = temp_decls;
                        v.push(lp);
                        Stmt::new(StmtP::Block(v), l)
                    }
                    None => targetme(ub.meta.loc, size_adjust(expr(self.exprs(args)))),
                };
                vec![trunc(
                    Operator::Less,
                    "min",
                    &lb,
                    trunc(Operator::Greater, "max", &ub, statement),
                )]
            }
        }
    }

    fn printables(&self, l: Loc, ps: &[TPrintable]) -> Vec<Expr> {
        ps.iter()
            .map(|p| match p {
                TPrintable::Str(s) => {
                    let s = s
                        .strip_prefix('"')
                        .and_then(|s| s.strip_suffix('"'))
                        .unwrap_or(s);
                    Expr::new(
                        ExprP::Lit(LitType::Str, s.to_string()),
                        Meta {
                            ty: UnsizedType::Real,
                            loc: l,
                            ad: Ad::DataOnly,
                        },
                    )
                }
                TPrintable::Expr(e) => self.expr(e),
            })
            .collect()
    }

    // -- Declarations --

    fn same_shape(decl_id: &str, decl_var: &Expr, id: &str, var: &Expr, l: Loc) -> Vec<Stmt> {
        if is_scalar_type(&var.meta.ty) {
            vec![]
        } else {
            vec![Stmt::new(
                StmtP::NRFunApp(
                    FunKind::StanLib("check_matching_dims".to_string(), Suffix::Plain),
                    vec![
                        Expr::str("constraint"),
                        Expr::str(decl_id),
                        decl_var.clone(),
                        Expr::str(id),
                        var.clone(),
                    ],
                ),
                l,
            )]
        }
    }

    fn check_transform_shape(decl_id: &str, v: &Expr, l: Loc, t: &Transformation) -> Vec<Stmt> {
        use Transformation as T;
        match t {
            T::Offset(e) => Self::same_shape(decl_id, v, "offset", e, l),
            T::Multiplier(e) => Self::same_shape(decl_id, v, "multiplier", e, l),
            T::Lower(e) => Self::same_shape(decl_id, v, "lower", e, l),
            T::Upper(e) => Self::same_shape(decl_id, v, "upper", e, l),
            T::OffsetMultiplier(a, b) => {
                let mut s = Self::same_shape(decl_id, v, "offset", a, l);
                s.extend(Self::same_shape(decl_id, v, "multiplier", b, l));
                s
            }
            T::LowerUpper(a, b) => {
                let mut s = Self::same_shape(decl_id, v, "lower", a, l);
                s.extend(Self::same_shape(decl_id, v, "upper", b, l));
                s
            }
            _ => vec![],
        }
    }

    fn copy_indices(indexed: &Expr, var: &Expr) -> Expr {
        if is_scalar_type(&var.meta.ty) {
            return var.clone();
        }
        let indices = indexed.collect_indices();
        if indices.is_empty() {
            var.clone()
        } else {
            let ty = infer_type_of_indexed(&var.meta.ty, &indices);
            Expr::new(
                ExprP::Indexed(Box::new(var.clone()), indices),
                Meta {
                    ty,
                    ..var.meta.clone()
                },
            )
        }
    }

    fn extract_transform_args(var: &Expr, t: &Transformation) -> Vec<Expr> {
        use Transformation as T;
        let lit = |a: &Expr, v: &str| Expr {
            pattern: ExprP::Lit(LitType::Int, v.to_string()),
            meta: a.meta.clone(),
        };
        match t {
            T::Lower(a) | T::Upper(a) => vec![Self::copy_indices(var, a)],
            T::Offset(a) => vec![Self::copy_indices(var, a), lit(a, "1")],
            T::Multiplier(a) => vec![lit(a, "0"), Self::copy_indices(var, a)],
            T::LowerUpper(a, b) | T::OffsetMultiplier(a, b) => {
                vec![Self::copy_indices(var, a), Self::copy_indices(var, b)]
            }
            _ => vec![],
        }
    }

    fn check_decl(&self, var: &Expr, st: &SizedType, t: &Transformation, l: Loc) -> Vec<Stmt> {
        match t {
            Transformation::LowerUpper(lb, ub) => {
                let mut v = self.check_decl(var, st, &Transformation::Lower(lb.clone()), l);
                v.extend(self.check_decl(var, st, &Transformation::Upper(ub.clone()), l));
                v
            }
            Transformation::Tuple(ts) if t.has_check() => {
                let (inner, dims) = st.array_dims();
                let SizedType::Tuple(subs) = inner else {
                    return vec![];
                };
                let pairs: Vec<(SizedType, Transformation)> =
                    subs.iter().cloned().zip(ts.iter().cloned()).collect();
                let check_tuple = |me: &Self, var: &Expr| -> Vec<Stmt> {
                    pairs
                        .iter()
                        .enumerate()
                        .flat_map(|(i, (st, tr))| {
                            me.check_decl(&add_tuple_index(var.clone(), i + 1), st, tr, l)
                        })
                        .collect()
                };
                if dims.is_empty() {
                    check_tuple(self, var)
                } else {
                    let rev: Vec<Expr> = dims.into_iter().rev().cloned().collect();
                    vec![self.mk_nested_for(
                        &rev,
                        &mut |me: &Self, loopvars: Vec<Expr>| {
                            let v = loopvars
                                .into_iter()
                                .rev()
                                .fold(var.clone(), |v, i| add_int_index(v, Index::Single(i)));
                            Stmt::new(StmtP::Block(check_tuple(me, &v)), l)
                        },
                        l,
                    )]
                }
            }
            _ if t.has_check() => {
                let var_name = pretty::expr_string(var);
                let args = Self::extract_transform_args(var, t);
                vec![Stmt::new(
                    StmtP::NRFunApp(
                        FunKind::CompilerInternal(Internal::Check {
                            trans: Box::new(t.clone()),
                            var_name,
                            var: Box::new(var.clone()),
                        }),
                        args,
                    ),
                    l,
                )]
            }
            _ => vec![],
        }
    }

    /// `check_transformed_sizedtype`: size validation statements and the sized type, with
    /// complex size expressions moved into declared variables.
    fn check_transformed_sizedtype(
        &self,
        action: Action,
        tr: &Transformation,
        name: &str,
        st: &TSizedType,
        untyped: Option<&ast::SizedType>,
    ) -> (Vec<Stmt>, SizedType) {
        self.go_size(action, tr, name, st, untyped, 1)
    }

    fn validate_dim(
        &self,
        f: Internal,
        name: &str,
        n: usize,
        e: &TExpr,
        u: Option<&ast::Expr>,
    ) -> (Vec<Stmt>, Expr) {
        let te = self.expr(e);
        let printed = u
            .map(|u| {
                let mut m = stanc_rs_syntax::autoformat::Message::new();
                m.expr(u);
                m.finish()
            })
            .unwrap_or_default();
        let check = |te: &Expr| {
            vec![Stmt::new(
                StmtP::NRFunApp(
                    FunKind::CompilerInternal(f.clone()),
                    vec![Expr::str(name), Expr::str(&printed), te.clone()],
                ),
                te.meta.loc,
            )]
        };
        let int_value = match u.map(|u| &u.kind) {
            Some(ast::ExprKind::IntNumeral(i)) => {
                Some(i.replace('_', "").parse::<f64>().unwrap_or(0.0))
            }
            _ => None,
        };
        let is_var = matches!(u.map(|u| &u.kind), Some(ast::ExprKind::Variable(_)));
        match (&f, int_value) {
            (Internal::ValidateSize, Some(i)) if i >= 0.0 => return (vec![], te),
            (Internal::ValidateSizePositive, Some(i)) if i >= 1.0 => return (vec![], te),
            (Internal::ValidateSizeUnitVector, Some(i)) if i >= 2.0 => return (vec![], te),
            _ => {}
        }
        if int_value.is_some() || is_var {
            let c = check(&te);
            return (c, te);
        }
        let decl_name = name.replace("[]", "_brack").replace('.', "_dot");
        let decl_id = format!("{decl_name}_{n}dim__");
        let decl = Stmt::new(
            StmtP::Decl {
                ad: Ad::DataOnly,
                id: decl_id.clone(),
                ty: DeclType::Sized(SizedType::Int),
                init: DeclInit::Assign(te.clone()),
            },
            te.meta.loc,
        );
        let var = Expr::new(ExprP::Var(decl_id), meta_of(e));
        let mut v = vec![decl];
        v.extend(check(&var));
        (v, var)
    }

    fn go_size(
        &self,
        action: Action,
        tr: &Transformation,
        name: &str,
        st: &TSizedType,
        u: Option<&ast::SizedType>,
        n: usize,
    ) -> (Vec<Stmt>, SizedType) {
        use TSizedType as T;
        let ue = |k: usize| -> Option<&ast::Expr> {
            match (u, k) {
                (Some(ast::SizedType::Vector(e)), 0)
                | (Some(ast::SizedType::RowVector(e)), 0)
                | (Some(ast::SizedType::ComplexVector(e)), 0)
                | (Some(ast::SizedType::ComplexRowVector(e)), 0)
                | (Some(ast::SizedType::Matrix(e, _)), 0)
                | (Some(ast::SizedType::ComplexMatrix(e, _)), 0)
                | (Some(ast::SizedType::Matrix(_, e)), 1)
                | (Some(ast::SizedType::ComplexMatrix(_, e)), 1)
                | (Some(ast::SizedType::Array(_, e)), 0) => Some(e),
                _ => None,
            }
        };
        match st {
            T::Int => (vec![], SizedType::Int),
            T::Real => (vec![], SizedType::Real),
            T::Complex => (vec![], SizedType::Complex),
            T::Vector(s) => {
                let f = match (action, tr) {
                    (Action::Constrain, Transformation::Simplex | Transformation::SumToZero) => {
                        Internal::ValidateSizePositive
                    }
                    (Action::Constrain, Transformation::UnitVector) => {
                        Internal::ValidateSizeUnitVector
                    }
                    _ => Internal::ValidateSize,
                };
                let (l, s) = self.validate_dim(f, name, n, s, ue(0));
                (l, SizedType::Vector(s))
            }
            T::RowVector(s) => {
                let (l, s) = self.validate_dim(Internal::ValidateSize, name, n, s, ue(0));
                (l, SizedType::RowVector(s))
            }
            T::ComplexRowVector(s) => {
                let (l, s) = self.validate_dim(Internal::ValidateSize, name, n, s, ue(0));
                (l, SizedType::ComplexRowVector(s))
            }
            T::ComplexVector(s) => {
                let (l, s) = self.validate_dim(Internal::ValidateSize, name, n, s, ue(0));
                (l, SizedType::ComplexVector(s))
            }
            T::Matrix(r, c) => {
                let (f1, f2) = match (action, tr) {
                    (Action::Constrain, Transformation::SumToZero) => (
                        Internal::ValidateSizePositive,
                        Internal::ValidateSizePositive,
                    ),
                    (Action::Constrain, Transformation::StochasticColumn) => {
                        (Internal::ValidateSizePositive, Internal::ValidateSize)
                    }
                    (Action::Constrain, Transformation::StochasticRow) => {
                        (Internal::ValidateSize, Internal::ValidateSizePositive)
                    }
                    _ => (Internal::ValidateSize, Internal::ValidateSize),
                };
                let (mut l1, r) = self.validate_dim(f1, name, n, r, ue(0));
                let (l2, c) = self.validate_dim(f2, name, n + 1, c, ue(1));
                l1.extend(l2);
                if action == Action::Constrain && *tr == Transformation::CholeskyCov {
                    let l = r.meta.loc;
                    l1.push(Stmt::new(
                        StmtP::NRFunApp(
                            FunKind::StanLib("check_greater_or_equal".to_string(), Suffix::Plain),
                            vec![
                                Expr::str(&format!("cholesky_factor_cov {name}")),
                                Expr::str("num rows (must be greater or equal to num cols)"),
                                r.clone(),
                                c.clone(),
                            ],
                        ),
                        l,
                    ));
                }
                (l1, SizedType::Matrix(r, c))
            }
            T::ComplexMatrix(r, c) => {
                let (mut l1, r) = self.validate_dim(Internal::ValidateSize, name, n, r, ue(0));
                let (l2, c) = self.validate_dim(Internal::ValidateSize, name, n + 1, c, ue(1));
                l1.extend(l2);
                (l1, SizedType::ComplexMatrix(r, c))
            }
            T::Array(t, s) => {
                let (mut l, s) = self.validate_dim(Internal::ValidateSize, name, n, s, ue(0));
                let inner_u = match u {
                    Some(ast::SizedType::Array(t, _)) => Some(&**t),
                    _ => None,
                };
                let (ll, t) = self.go_size(action, tr, name, t, inner_u, n + 1);
                l.extend(ll);
                (l, SizedType::Array(Box::new(t), s))
            }
            T::Tuple(subs) => {
                let former = "[]".repeat(n - 1);
                let trs: Vec<Transformation> = match tr {
                    Transformation::Tuple(ts) => ts.clone(),
                    _ => vec![Transformation::Identity; subs.len()],
                };
                let us: Vec<Option<&ast::SizedType>> = match u {
                    Some(ast::SizedType::Tuple(us)) => us.iter().map(Some).collect(),
                    _ => vec![None; subs.len()],
                };
                let mut stmts = Vec::new();
                let mut out = Vec::new();
                for (ix, ((st, tr), u)) in subs.iter().zip(trs.iter()).zip(us).enumerate() {
                    let (s, t) = self.check_transformed_sizedtype(
                        action,
                        tr,
                        &format!("{name}{former}.{}", ix + 1),
                        st,
                        u,
                    );
                    stmts.extend(s);
                    out.push(t);
                }
                (stmts, SizedType::Tuple(out))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_decl_with_assign(
        &self,
        decl_id: &str,
        ctx: DeclCtx,
        decl_type: SizedType,
        init: Option<&TExpr>,
        transform: &Transformation,
        l: Loc,
    ) -> Vec<Stmt> {
        let rhs = init.map(|e| self.expr(e));
        let ut = decl_type.to_unsized();
        let ad = fill_ad(&ctx.ad.ad(), &ut);
        let decl_var = Expr::new(
            ExprP::Var(decl_id.to_string()),
            Meta {
                ty: ut.clone(),
                loc: l,
                ad: ad.clone(),
            },
        );
        let mut v = vec![Stmt::new(
            StmtP::Decl {
                ad: ad.clone(),
                id: decl_id.to_string(),
                ty: DeclType::Sized(decl_type.clone()),
                init: DeclInit::Default,
            },
            l,
        )];
        if let Some(e) = rhs {
            v.push(Stmt::new(
                StmtP::Assignment(LValue::var(decl_id), e.meta.ty.clone(), e),
                l,
            ));
        }
        if !decl_id.ends_with("__") {
            match ctx.action {
                Action::Constrain => v.extend(Self::check_transform_shape(
                    decl_id, &decl_var, l, transform,
                )),
                Action::Check => {
                    v.extend(Self::check_transform_shape(
                        decl_id, &decl_var, l, transform,
                    ));
                    v.extend(self.check_decl(&decl_var, &decl_type, transform, l));
                }
                Action::Ignore => {}
            }
        }
        v
    }

    // -- Statements --

    fn single_stmt(&self, ctx: DeclCtx, s: &TStmt) -> Stmt {
        let mut v = self.stmt(ctx, s);
        if v.len() == 1 {
            v.pop().expect("one statement")
        } else {
            Stmt::new(StmtP::SList(v), loc(s.span))
        }
    }

    fn stmt(&self, ctx: DeclCtx, s: &TStmt) -> Vec<Stmt> {
        let l = loc(s.span);
        let inner = DeclCtx {
            action: Action::Ignore,
            ..ctx
        };
        let wrap = |p: StmtP| vec![Stmt::new(p, l)];
        match &s.kind {
            TStmtKind::Assign { lhs, op, rhs } => {
                if let TLValueKind::Tuple(lvs) = &lhs.kind {
                    self.packed_assign(inner, l, lvs, rhs, *op)
                } else {
                    vec![self.single_assignment(l, lhs, rhs.clone(), *op)]
                }
            }
            TStmtKind::NRFunApp { kind, name, args } => wrap(StmtP::NRFunApp(
                trans_fn_kind(kind, name, Suffix::from_name(name)),
                self.exprs(args),
            )),
            TStmtKind::TargetPlusAssign(e) => wrap(StmtP::TargetPE(self.expr(e))),
            TStmtKind::JacobianPlusAssign(e) => wrap(StmtP::JacobianPE(self.expr(e))),
            TStmtKind::Tilde {
                arg,
                dist,
                dist_kind,
                args,
                truncation,
            } => {
                let suffix = if dist.ends_with("_lpmf") || dist.ends_with("_lupmf") {
                    Suffix::Lpmf(true)
                } else {
                    Suffix::Lpdf(true)
                };
                let base = dist
                    .strip_suffix("_lpdf")
                    .or_else(|| dist.strip_suffix("_lpmf"))
                    .or_else(|| dist.strip_suffix("_lupdf"))
                    .or_else(|| dist.strip_suffix("_lupmf"))
                    .unwrap_or(dist);
                let name = format!(
                    "{base}{}",
                    if matches!(suffix, Suffix::Lpmf(_)) {
                        "_lpmf"
                    } else {
                        "_lpdf"
                    }
                );
                let ad = if std::iter::once(arg)
                    .chain(args.iter())
                    .any(|e| e.ad.has_autodiff())
                {
                    Ad::AutoDiffable
                } else {
                    Ad::DataOnly
                };
                let mut all = vec![self.expr(arg)];
                all.extend(self.exprs(args));
                let mut v = wrap(StmtP::TargetPE(Expr::new(
                    ExprP::FunApp(trans_fn_kind(dist_kind, &name, suffix), all),
                    Meta {
                        ty: UnsizedType::Real,
                        loc: l,
                        ad,
                    },
                )));
                v.extend(self.truncate_dist(base, arg, args, truncation));
                v
            }
            TStmtKind::Print(ps) => wrap(StmtP::NRFunApp(
                FunKind::CompilerInternal(Internal::Print),
                self.printables(l, ps),
            )),
            TStmtKind::Reject(ps) => wrap(StmtP::NRFunApp(
                FunKind::CompilerInternal(Internal::Reject),
                self.printables(l, ps),
            )),
            TStmtKind::FatalError(ps) => wrap(StmtP::NRFunApp(
                FunKind::CompilerInternal(Internal::FatalError),
                self.printables(l, ps),
            )),
            TStmtKind::IfThenElse(c, t, e) => wrap(StmtP::IfElse(
                self.expr(c),
                Box::new(self.single_stmt(inner, t)),
                e.as_ref().map(|e| Box::new(self.single_stmt(inner, e))),
            )),
            TStmtKind::While(c, b) => wrap(StmtP::While(
                self.expr(c),
                Box::new(self.single_stmt(inner, b)),
            )),
            TStmtKind::For {
                var,
                lower,
                upper,
                body,
            } => {
                let b = self.single_stmt(inner, body);
                let b = match b.pattern {
                    StmtP::Block(_) => b,
                    _ => {
                        let l = b.loc;
                        Stmt::new(StmtP::Block(vec![b]), l)
                    }
                };
                wrap(StmtP::For {
                    loopvar: var.clone(),
                    lower: self.expr(lower),
                    upper: self.expr(upper),
                    body: Box::new(b),
                })
            }
            TStmtKind::ForEach {
                var, iter, body, ..
            } => {
                let it = self.expr(iter);
                let body_stmts = match self.single_stmt(inner, body) {
                    Stmt {
                        pattern: StmtP::Block(ss),
                        ..
                    } => ss,
                    b => vec![b],
                };
                let decl_type = match &it.meta.ty {
                    UnsizedType::Matrix => UnsizedType::Real,
                    t => infer_type_of_indexed(t, &[Index::Single(Expr::int(1))]),
                };
                let decl_loopvar = Stmt::new(
                    StmtP::Decl {
                        ad: it.meta.ad.clone(),
                        id: var.clone(),
                        ty: DeclType::Unsized(decl_type.clone()),
                        init: DeclInit::Default,
                    },
                    l,
                );
                let mut bodyfn = |_: &Self, v: Expr| {
                    let mut ss = vec![
                        decl_loopvar.clone(),
                        Stmt::new(StmtP::Assignment(LValue::var(var), decl_type.clone(), v), l),
                    ];
                    ss.extend(body_stmts.clone());
                    Stmt::new(StmtP::Block(ss), l)
                };
                vec![self.ensure_var(&mut bodyfn, it, l)]
            }
            TStmtKind::VarDecl {
                ty, trans, vars, ..
            } => {
                let untyped = self.decl_types.get(&s.span).copied();
                let transform = self.transformation(trans);
                let mut out = Vec::new();
                for v in vars {
                    let (checks, dt) = self.check_transformed_sizedtype(
                        Action::Ignore,
                        &Transformation::Identity,
                        &v.name,
                        ty,
                        untyped,
                    );
                    out.extend(checks);
                    out.extend(self.create_decl_with_assign(
                        &v.name,
                        ctx,
                        dt,
                        v.init.as_ref(),
                        &transform,
                        l,
                    ));
                }
                out
            }
            TStmtKind::Block(ss) => wrap(StmtP::Block(
                ss.iter().flat_map(|s| self.stmt(inner, s)).collect(),
            )),
            TStmtKind::Profile(name, ss) => wrap(StmtP::Profile(
                name.clone(),
                ss.iter().flat_map(|s| self.stmt(inner, s)).collect(),
            )),
            TStmtKind::Return(e) => wrap(StmtP::Return(e.as_ref().map(|e| self.expr(e)))),
            TStmtKind::Break => wrap(StmtP::Break),
            TStmtKind::Continue => wrap(StmtP::Continue),
            TStmtKind::Skip => wrap(StmtP::Skip),
            TStmtKind::FunDef { .. } => vec![],
        }
    }

    /// `Stmt.Helpers.ensure_var`.
    fn ensure_var(&self, body: &mut dyn FnMut(&Self, Expr) -> Stmt, e: Expr, l: Loc) -> Stmt {
        if matches!(e.pattern, ExprP::Var(_)) {
            return self.for_each(body, e, l);
        }
        let (mut pre, temps, old) = self.temp_vars(vec![e]);
        let b = self.for_each(body, temps.into_iter().next().expect("one temporary"), l);
        self.sym.reset(old);
        let bl = b.loc;
        pre.push(b);
        Stmt::new(StmtP::Block(pre), bl)
    }

    fn packed_assign(
        &self,
        ctx: DeclCtx,
        l: Loc,
        lvs: &[TLValue],
        rhs: &TExpr,
        op: AssignOp,
    ) -> Vec<Stmt> {
        let (sym, old) = self.sym.enter();
        let rl = loc(rhs.span);
        let temp = Stmt::new(
            StmtP::Decl {
                ad: rhs.ad.clone(),
                id: sym.clone(),
                ty: DeclType::Unsized(rhs.ty.clone()),
                init: DeclInit::Uninit,
            },
            rl,
        );
        let assign = Stmt::new(
            StmtP::Assignment(LValue::var(&sym), rhs.ty.clone(), self.expr(rhs)),
            rl,
        );
        let temp_expr = TExpr {
            kind: TExprKind::Variable {
                name: sym,
                origin: Origin::Functions,
                global: false,
            },
            ..rhs.clone()
        };
        let mut v = vec![temp, assign];
        for (i, lv) in lvs.iter().enumerate() {
            let (ty, ad) = match (&rhs.ty, &rhs.ad) {
                (UnsizedType::Tuple(ts), Ad::Tuple(ads)) => (ts[i].clone(), ads[i].clone()),
                (UnsizedType::Tuple(ts), a) => (ts[i].clone(), a.clone()),
                (t, a) => (t.clone(), a.clone()),
            };
            let proj = TExpr {
                kind: TExprKind::TupleProjection(Box::new(temp_expr.clone()), i + 1),
                ty,
                ad,
                span: rhs.span,
            };
            let st = TStmt {
                kind: TStmtKind::Assign {
                    lhs: lv.clone(),
                    op,
                    rhs: proj,
                },
                span: s_of(l),
            };
            v.extend(self.stmt(ctx, &st));
        }
        self.sym.reset(old);
        vec![Stmt::new(StmtP::Block(v), l)]
    }

    fn single_assignment(&self, l: Loc, lhs: &TLValue, rhs: TExpr, op: AssignOp) -> Stmt {
        let grouped = group_lvalue(vec![], lhs);
        let lv = self.lvalue(&grouped);
        let unindexed_type = match &grouped.kind {
            TLValueKind::Indexed(inner, _) => inner.ty.clone(),
            _ => grouped.ty.clone(),
        };
        let rhs = match op {
            AssignOp::Assign => self.expr(&rhs),
            AssignOp::Compound(bop) => {
                let assignee = expr_of_lvalue(&grouped);
                op_to_funapp(self, binop_operator(bop), &[&assignee, &rhs], &assignee.ty)
            }
        };
        Stmt::new(StmtP::Assignment(lv, unindexed_type, rhs), l)
    }

    fn lvalue(&self, lv: &TLValue) -> LValue {
        match &lv.kind {
            TLValueKind::Variable { name, .. } => LValue::var(name),
            TLValueKind::TupleProjection(inner, i) => LValue {
                base: LBase::TupleProjection(Box::new(self.lvalue(inner)), *i),
                idx: vec![],
            },
            TLValueKind::Indexed(inner, idx) => {
                let mut l = self.lvalue(inner);
                l.idx.extend(idx.iter().map(|i| self.index(i)));
                l
            }
            TLValueKind::Tuple(_) => LValue::var(""),
        }
    }

    fn fun_def(&self, s: &TStmt) -> Option<FunDef> {
        let TStmtKind::FunDef {
            ret,
            name,
            args,
            body,
        } = &s.kind
        else {
            return None;
        };
        let ctx = DeclCtx {
            action: Action::Ignore,
            ad: Ad2::Var,
        };
        let body = body.as_ref().and_then(|b| {
            let mut v = self.stmt(ctx, b);
            match v.as_slice() {
                [
                    Stmt {
                        pattern: StmtP::Block(_),
                        ..
                    },
                ] => v.pop(),
                _ => None,
            }
        });
        Some(FunDef {
            rt: ret.clone(),
            name: name.clone(),
            suffix: Suffix::from_name(name).without_propto(),
            args: args
                .iter()
                .map(|a| (Ad::from(a.ad), a.name.clone(), a.ty.clone()))
                .collect(),
            body,
            loc: loc(s.span),
        })
    }

    /// `trans_block`: output variables, size checks, and statements of a block.
    fn block(
        &self,
        ctx: DeclCtx,
        block: IoBlock,
        stmts: &[TStmt],
    ) -> (OutputVars, Vec<Stmt>, Vec<Stmt>) {
        let (mut outs, mut sizes, mut body) = (Vec::new(), Vec::new(), Vec::new());
        for s in stmts {
            match &s.kind {
                TStmtKind::VarDecl {
                    ty,
                    trans,
                    vars,
                    is_global: true,
                    ..
                } => {
                    let untyped = self.decl_types.get(&s.span).copied();
                    for v in vars {
                        let transform = self.transformation(trans);
                        let (size, type_) = self.check_transformed_sizedtype(
                            ctx.action, &transform, &v.name, ty, untyped,
                        );
                        outs.push((
                            v.name.clone(),
                            loc(s.span),
                            OutVar {
                                unconstrained: transform_sizedtype(&transform, &type_),
                                constrained: type_.clone(),
                                block,
                                trans: transform.clone(),
                            },
                        ));
                        sizes.extend(size);
                        body.extend(self.create_decl_with_assign(
                            &v.name,
                            ctx,
                            type_,
                            v.init.as_ref(),
                            &transform,
                            loc(s.span),
                        ));
                    }
                }
                _ => body.extend(self.stmt(ctx, s)),
            }
        }
        (outs, sizes, body)
    }
}

fn s_of(l: Loc) -> Span {
    l.span().unwrap_or_default()
}

/// Groups consecutive indices: `x[1][2].1[3]` becomes `x[1, 2].1[3]`. The grouped value has
/// the metadata of the outermost indexed value.
fn group_lvalue(carry: Vec<TIndex>, lv: &TLValue) -> TLValue {
    match &lv.kind {
        TLValueKind::Variable { .. } | TLValueKind::Tuple(_) => {
            if carry.is_empty() {
                lv.clone()
            } else {
                TLValue {
                    kind: TLValueKind::Indexed(Box::new(lv.clone()), carry),
                    ..lv.clone()
                }
            }
        }
        TLValueKind::TupleProjection(inner, ix) => {
            let p = TLValue {
                kind: TLValueKind::TupleProjection(Box::new(group_lvalue(vec![], inner)), *ix),
                ..lv.clone()
            };
            if carry.is_empty() {
                p
            } else {
                TLValue {
                    kind: TLValueKind::Indexed(Box::new(p), carry),
                    ..lv.clone()
                }
            }
        }
        TLValueKind::Indexed(inner, idx) => {
            let mut all = idx.clone();
            all.extend(carry);
            TLValue {
                kind: group_lvalue(all, inner).kind,
                ..lv.clone()
            }
        }
    }
}

fn expr_of_lvalue(lv: &TLValue) -> TExpr {
    let kind = match &lv.kind {
        TLValueKind::Variable { name, origin } => TExprKind::Variable {
            name: name.clone(),
            origin: *origin,
            global: false,
        },
        TLValueKind::Indexed(l, idx) => {
            TExprKind::Indexed(Box::new(expr_of_lvalue(l)), idx.clone())
        }
        TLValueKind::TupleProjection(l, i) => {
            TExprKind::TupleProjection(Box::new(expr_of_lvalue(l)), *i)
        }
        TLValueKind::Tuple(ls) => TExprKind::TupleExpr(ls.iter().map(expr_of_lvalue).collect()),
    };
    TExpr {
        kind,
        ty: lv.ty.clone(),
        ad: lv.ad.clone(),
        span: lv.span,
    }
}

/// The unconstrained type of a transformed variable (`transform_sizedtype`).
fn transform_sizedtype(t: &Transformation, st: &SizedType) -> SizedType {
    use Operator::*;
    let minus_one = |d: &Expr| Expr::binop(d.clone(), Minus, Expr::int(1));
    let k_choose_2 = |k: &Expr| {
        Expr::binop(
            Expr::binop(
                k.clone(),
                Times,
                Expr::binop(k.clone(), Minus, Expr::int(1)),
            ),
            Divide,
            Expr::int(2),
        )
    };
    enum Change<'f> {
        Uni(&'f dyn Fn(&Expr) -> Expr),
        Multi(&'f dyn Fn(&Expr, &Expr) -> Expr),
    }
    fn shrink(f: &Change, f2: &dyn Fn(&Expr) -> Expr, st: &SizedType) -> SizedType {
        match st {
            SizedType::Array(t, d) => SizedType::Array(Box::new(shrink(f, f2, t)), d.clone()),
            SizedType::Vector(d) => match f {
                Change::Uni(f) => SizedType::Vector(f(d)),
                Change::Multi(_) => st.clone(),
            },
            SizedType::RowVector(d) => match f {
                Change::Uni(f) => SizedType::RowVector(f(d)),
                Change::Multi(_) => st.clone(),
            },
            SizedType::Matrix(d1, d2) => match f {
                Change::Uni(f) => SizedType::Matrix(f(d1), f2(d2)),
                Change::Multi(f) => SizedType::Vector(f(d1, d2)),
            },
            _ => st.clone(),
        }
    }
    let id = |d: &Expr| d.clone();
    match t {
        Transformation::Tuple(tms) => {
            let (inner, dims) = st.array_dims();
            let dims: Vec<Expr> = dims.into_iter().cloned().collect();
            let SizedType::Tuple(subs) = inner else {
                return st.clone();
            };
            SizedType::build_array(
                dims,
                SizedType::Tuple(
                    subs.iter()
                        .zip(tms)
                        .map(|(s, t)| transform_sizedtype(t, s))
                        .collect(),
                ),
            )
        }
        Transformation::SumToZero => shrink(&Change::Uni(&minus_one), &minus_one, st),
        Transformation::Simplex | Transformation::StochasticColumn => {
            shrink(&Change::Uni(&minus_one), &id, st)
        }
        Transformation::StochasticRow => shrink(&Change::Uni(&id), &minus_one, st),
        Transformation::CholeskyCorr | Transformation::Correlation => {
            let f = |a: &Expr, _: &Expr| k_choose_2(a);
            shrink(&Change::Multi(&f), &id, st)
        }
        Transformation::Covariance => {
            let f = |k: &Expr, _: &Expr| Expr::binop(k.clone(), Plus, k_choose_2(k));
            shrink(&Change::Multi(&f), &id, st)
        }
        Transformation::CholeskyCov => {
            let f = |m: &Expr, n: &Expr| {
                Expr::binop(
                    Expr::binop(k_choose_2(n), Plus, n.clone()),
                    Plus,
                    Expr::binop(Expr::binop(m.clone(), Minus, n.clone()), Times, n.clone()),
                )
            };
            shrink(&Change::Multi(&f), &id, st)
        }
        _ => st.clone(),
    }
}

fn stmt_contains_check(s: &Stmt) -> bool {
    s.contains_fn_kind(&mut |k: &FunKind| {
        matches!(k, FunKind::CompilerInternal(Internal::Check { .. }))
    })
}

fn migrate_checks_to_end_of_block(stmts: Vec<Stmt>) -> Vec<Stmt> {
    let (checks, mut rest): (Vec<Stmt>, Vec<Stmt>) =
        stmts.into_iter().partition(stmt_contains_check);
    rest.extend(checks);
    rest
}

fn collect_decl_types<'a>(stmts: &'a [ast::Stmt], out: &mut HashMap<Span, &'a ast::SizedType>) {
    for s in stmts {
        match &s.kind {
            ast::StmtKind::VarDecl { ty, .. } => {
                out.insert(s.span, ty);
            }
            ast::StmtKind::IfThenElse(_, a, b) => {
                collect_decl_types(std::slice::from_ref(a), out);
                if let Some(b) = b {
                    collect_decl_types(std::slice::from_ref(b), out);
                }
            }
            ast::StmtKind::While(_, b)
            | ast::StmtKind::For { body: b, .. }
            | ast::StmtKind::ForEach { body: b, .. }
            | ast::StmtKind::FunDef { body: b, .. } => {
                collect_decl_types(std::slice::from_ref(b), out)
            }
            ast::StmtKind::Block(ss) | ast::StmtKind::Profile(_, ss) => collect_decl_types(ss, out),
            _ => {}
        }
    }
}

/// Argument types of a function definition, for finding redundant forward declarations.
fn arg_types(s: &TStmt) -> Option<(String, Vec<(ast::AdLevel, UnsizedType)>)> {
    match &s.kind {
        TStmtKind::FunDef { name, args, .. } => Some((
            name.clone(),
            args.iter().map(|a| (a.ad, a.ty.clone())).collect(),
        )),
        _ => None,
    }
}

/// Lowers a typed program. `ast` is the untyped program it was checked from, used to print
/// size expressions as written.
pub fn lower_program(p: &TProgram, untyped: &ast::Program) -> Program {
    let mut decl_types = HashMap::new();
    for (_, b) in untyped.blocks() {
        if let Some(b) = b {
            collect_decl_types(&b.stmts, &mut decl_types);
        }
    }
    // Forward declarations of defined functions are dropped.
    let defined: Vec<(String, Vec<(ast::AdLevel, UnsizedType)>)> = p
        .functions
        .iter()
        .filter(|s| matches!(&s.kind, TStmtKind::FunDef { body: Some(b), .. } if b.kind != TStmtKind::Skip))
        .filter_map(arg_types)
        .collect();
    let functions: Vec<&TStmt> = p
        .functions
        .iter()
        .filter(|s| {
            let is_decl = matches!(&s.kind, TStmtKind::FunDef { body, .. }
                if body.as_ref().is_none_or(|b| b.kind == TStmtKind::Skip));
            !(is_decl && arg_types(s).is_some_and(|a| defined.contains(&a)))
        })
        .collect();
    let ud_dists = functions
        .iter()
        .filter_map(|s| match &s.kind {
            TStmtKind::FunDef { name, args, .. } => {
                args.first().map(|a| (name.clone(), a.ty.clone()))
            }
            _ => None,
        })
        .collect();
    let l = Lower {
        sym: Gensym::default(),
        ud_dists,
        decl_types,
    };
    let data_ctx = DeclCtx {
        action: Action::Ignore,
        ad: Ad2::Data,
    };
    let input_vars = p
        .data
        .iter()
        .flat_map(|s| match &s.kind {
            TStmtKind::VarDecl { ty, vars, .. } => vars
                .iter()
                .map(|v| (v.name.clone(), loc(s.span), l.sized_type(ty)))
                .collect(),
            _ => vec![],
        })
        .collect();
    let check_data = DeclCtx {
        action: Action::Check,
        ..data_ctx
    };
    let datab: Vec<Stmt> = p.data.iter().flat_map(|s| l.stmt(check_data, s)).collect();
    let (_, _, param) = l.block(
        DeclCtx {
            action: Action::Constrain,
            ad: Ad2::Var,
        },
        IoBlock::Parameters,
        &p.parameters,
    );
    let (out_param, paramsizes, param_gq) = l.block(
        DeclCtx {
            action: Action::Constrain,
            ..data_ctx
        },
        IoBlock::Parameters,
        &p.parameters,
    );
    let (_, _, txparam) = l.block(
        DeclCtx {
            action: Action::Check,
            ad: Ad2::Var,
        },
        IoBlock::TransformedParameters,
        &p.transformed_parameters,
    );
    let (out_tparam, tparamsizes, txparam_gq) = l.block(
        check_data,
        IoBlock::TransformedParameters,
        &p.transformed_parameters,
    );
    let (out_gq, gq_sizes, gq_stmts) = l.block(
        check_data,
        IoBlock::GeneratedQuantities,
        &p.generated_quantities,
    );
    let mut output_vars = out_param;
    output_vars.extend(out_tparam);
    output_vars.extend(out_gq);
    let mut prepare_data = datab;
    prepare_data.extend(migrate_checks_to_end_of_block(
        p.transformed_data
            .iter()
            .flat_map(|s| l.stmt(check_data, s))
            .collect(),
    ));
    prepare_data.extend(paramsizes);
    prepare_data.extend(tparamsizes);
    prepare_data.extend(gq_sizes);
    let model_ctx = DeclCtx {
        action: Action::Ignore,
        ad: Ad2::Var,
    };
    let modelb: Vec<Stmt> = p.model.iter().flat_map(|s| l.stmt(model_ctx, s)).collect();
    let mut log_prob = param;
    log_prob.extend(migrate_checks_to_end_of_block(txparam));
    if let Some(hd) = modelb.first() {
        let ml = hd.loc;
        log_prob.push(Stmt::new(StmtP::Block(modelb), ml));
    }
    let (mut decls, mut checks, mut others) = (Vec::new(), Vec::new(), Vec::new());
    for s in txparam_gq {
        if matches!(s.pattern, StmtP::Decl { .. }) {
            decls.push(s);
        } else if stmt_contains_check(&s) {
            checks.push(s);
        } else {
            others.push(s);
        }
    }
    let iexpr = |p: ExprP| Expr::new(p, Meta::empty());
    let fnot = |e: Expr| iexpr(ExprP::FunApp(FunKind::op(Operator::PNot), vec![e]));
    let if_return = |cond: Expr| {
        Stmt::new(
            StmtP::IfElse(
                cond,
                Box::new(Stmt::new(StmtP::Return(None), Loc::EMPTY)),
                None,
            ),
            Loc::EMPTY,
        )
    };
    let var = |n: &str| iexpr(ExprP::Var(n.to_string()));
    let mut gq = param_gq;
    gq.extend(decls);
    gq.push(if_return(fnot(iexpr(ExprP::EOr(
        Box::new(var("emit_transformed_parameters__")),
        Box::new(var("emit_generated_quantities__")),
    )))));
    gq.extend(others);
    gq.extend(checks);
    gq.push(if_return(fnot(var("emit_generated_quantities__"))));
    gq.extend(migrate_checks_to_end_of_block(gq_stmts));
    Program {
        functions: functions.into_iter().filter_map(|s| l.fun_def(s)).collect(),
        input_vars,
        prepare_data,
        log_prob,
        generate_quantities: gq,
        output_vars,
        name: p.name.clone(),
    }
}
