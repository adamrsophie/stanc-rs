//! The passes stanc3 adds at `--Oexperimental`: expression propagation, loop unrolling, loop
//! vectorization, lazy code motion, and the narrowing of autodiff levels.

use std::collections::{BTreeMap, BTreeSet};

use stanc_rs_semantic::types::Ad;

use crate::dataflow::{DualPowerset, EntryExit, Graph, Powerset, mfp};
use crate::inline::subst_args_stmt;
use crate::lower::{Gensym, fill_ad};
use crate::mir::*;
use crate::ocaml_order::{OrdExpr, cmp_expr};
use crate::optimize::{
    call_return_type, cannot_duplicate_expr, expr_any, free_vars_expr, is_autodiffable,
    map_labeled, map_rec, propagation, subst_expr, top_decls, transform_program,
    transform_program_blockwise, try_eval_expr,
};

fn no_loc(pattern: StmtP) -> Stmt {
    Stmt::new(pattern, Loc::EMPTY)
}

fn var(name: &str) -> Expr {
    Expr::new(ExprP::Var(name.to_string()), Meta::empty())
}

/// `expression_propagation`: forward substitution of the expressions assigned to variables.
pub fn expression_propagation(p: &mut Program) {
    let transfer = |s: &Stmt, m: &BTreeMap<String, Expr>| {
        let kill = |m: &BTreeMap<String, Expr>, v: &str| -> BTreeMap<String, Expr> {
            m.iter()
                .filter(|(k, d)| {
                    let mut fv = BTreeSet::new();
                    free_vars_expr(d, &mut fv);
                    k.as_str() != v && !fv.contains(v)
                })
                .map(|(k, d)| (k.clone(), d.clone()))
                .collect()
        };
        match &s.pattern {
            StmtP::Assignment(lv, _, e)
                if lv.idx.is_empty() && matches!(lv.base, LBase::Variable(_)) =>
            {
                let x = lv.variable();
                let mut fv = BTreeSet::new();
                free_vars_expr(e, &mut fv);
                if cannot_duplicate_expr(e) || fv.contains(x) {
                    kill(m, x)
                } else {
                    // stanc3 adds to the map before the kill, so entries that read `x` stay.
                    let mut e = e.clone();
                    subst_expr(m, &mut e);
                    let mut m = m.clone();
                    m.insert(x.to_string(), e);
                    m
                }
            }
            StmtP::Decl { id, .. } => kill(m, id),
            StmtP::Assignment(lv, ..) => kill(m, lv.variable()),
            StmtP::Profile(_, b) | StmtP::Block(b) => {
                let mut ks = Vec::new();
                b.iter().for_each(|c| top_decls(c, &mut ks));
                let mut m = m.clone();
                for k in ks {
                    m = kill(&m, &k);
                }
                m
            }
            _ => m.clone(),
        }
    };
    transform_program(p, &|s| propagation(s, &transfer));
}

/// `contains_top_break_or_continue`.
fn breaks_out(s: &Stmt) -> bool {
    match &s.pattern {
        StmtP::Break | StmtP::Continue => true,
        StmtP::Profile(_, l) | StmtP::Block(l) | StmtP::SList(l) => l.iter().any(breaks_out),
        StmtP::IfElse(_, a, b) => breaks_out(a) || b.as_deref().is_some_and(breaks_out),
        _ => false,
    }
}

/// `top_down_map_rec_stmt_loc`.
fn map_top_down(s: &mut Stmt, f: &impl Fn(StmtP) -> StmtP) {
    let p = std::mem::replace(&mut s.pattern, StmtP::Skip);
    s.pattern = f(p);
    for c in s.children_mut() {
        map_top_down(c, f);
    }
}

/// `static_loop_unrolling`: unrolls loops over at most 33 known iterations.
pub fn static_loop_unrolling(p: &mut Program) {
    transform_program_blockwise(p, &|_, mut s| {
        map_top_down(&mut s, &|p| match p {
            StmtP::For {
                loopvar,
                lower,
                upper,
                body,
            } => {
                let lo = try_eval_expr(&lower, false);
                let up = try_eval_expr(&upper, false);
                match (&lo.pattern, &up.pattern) {
                    (ExprP::Lit(LitType::Int, a), ExprP::Lit(LitType::Int, b))
                        if !breaks_out(&body) =>
                    {
                        let (a, b): (i64, i64) = (a.parse().unwrap(), b.parse().unwrap());
                        if b - a > 32 {
                            return StmtP::For {
                                loopvar,
                                lower,
                                upper,
                                body,
                            };
                        }
                        StmtP::SList(
                            (a..=b)
                                .map(|i| {
                                    let mut st = no_loc(body.pattern.clone());
                                    subst_args_stmt(
                                        std::slice::from_ref(&loopvar),
                                        &[Expr::int(i)],
                                        &mut st,
                                    );
                                    st
                                })
                                .collect(),
                        )
                    }
                    _ => StmtP::For {
                        loopvar,
                        lower,
                        upper,
                        body,
                    },
                }
            }
            p => p,
        });
        s
    });
}

/// `one_step_loop_unrolling`: peels the first iteration off each loop.
pub fn one_step_loop_unrolling(p: &mut Program) {
    transform_program_blockwise(p, &|_, mut s| {
        map_rec(&mut s, &|p| match p {
            StmtP::For {
                loopvar,
                lower,
                upper,
                body,
            } if !breaks_out(&body) => {
                let (lower, decls) = if cannot_duplicate_expr(&lower) {
                    let name = format!("_{loopvar}_lower");
                    let loc = lower.meta.loc;
                    let decl = Stmt::new(
                        StmtP::Decl {
                            ad: stanc_rs_semantic::types::Ad::DataOnly,
                            id: name.clone(),
                            ty: DeclType::Sized(SizedType::Int),
                            init: DeclInit::Assign(lower),
                        },
                        loc,
                    );
                    (var(&name), vec![decl])
                } else {
                    (lower, vec![])
                };
                let mut first = no_loc(body.pattern.clone());
                subst_args_stmt(
                    std::slice::from_ref(&loopvar),
                    std::slice::from_ref(&lower),
                    &mut first,
                );
                let rest = no_loc(StmtP::For {
                    loopvar,
                    upper: upper.clone(),
                    body,
                    lower: Expr {
                        pattern: ExprP::FunApp(
                            FunKind::op(Operator::Plus),
                            vec![lower.clone(), Expr::int(1)],
                        ),
                        meta: lower.meta.clone(),
                    },
                });
                let block = match first.pattern {
                    StmtP::Block(mut l) => {
                        l.push(rest);
                        StmtP::Block(l)
                    }
                    p => StmtP::Block(vec![no_loc(p), rest]),
                };
                let cond = Expr {
                    pattern: ExprP::FunApp(FunKind::op(Operator::Geq), vec![upper, lower.clone()]),
                    meta: lower.meta.clone(),
                };
                let unrolled = StmtP::IfElse(cond, Box::new(no_loc(block)), None);
                if decls.is_empty() {
                    unrolled
                } else {
                    let mut l = decls;
                    l.push(no_loc(unrolled));
                    StmtP::Block(l)
                }
            }
            StmtP::While(e, body) if !breaks_out(&body) => {
                let again = Stmt::new(StmtP::While(e.clone(), body.clone()), body.loc);
                StmtP::IfElse(e, Box::new(no_loc(StmtP::Block(vec![*body, again]))), None)
            }
            p => p,
        });
        s
    });
}

// ----- Lazy code motion -----

type ExprSet = BTreeSet<OrdExpr>;

fn kind_exprs(k: &FunKind) -> Vec<&Expr> {
    match k {
        FunKind::CompilerInternal(f) => f.exprs(),
        _ => vec![],
    }
}

/// `used_subexpressions_expr`: an expression and all of its subexpressions.
fn subexpressions(e: &Expr, out: &mut ExprSet) {
    out.insert(OrdExpr(e.clone()));
    match &e.pattern {
        ExprP::Var(_) | ExprP::Lit(..) => {}
        ExprP::FunApp(k, l) => {
            for x in l.iter().chain(kind_exprs(k)) {
                subexpressions(x, out);
            }
        }
        _ => {
            for x in e.children() {
                subexpressions(x, out);
            }
        }
    }
}

fn lval_exprs<'a>(lv: &'a LValue, out: &mut Vec<&'a Expr>) {
    for i in &lv.idx {
        out.extend(i.exprs());
    }
    if let LBase::TupleProjection(inner, _) = &lv.base {
        lval_exprs(inner, out);
    }
}

/// The expressions a statement evaluates itself, without those of its sub-statements
/// (`top_used_expressions_stmt_help`).
fn top_exprs(s: &Stmt) -> Vec<&Expr> {
    match &s.pattern {
        StmtP::TargetPE(e)
        | StmtP::JacobianPE(e)
        | StmtP::Return(Some(e))
        | StmtP::Decl {
            init: DeclInit::Assign(e),
            ..
        }
        | StmtP::While(e, _)
        | StmtP::IfElse(e, ..) => vec![e],
        StmtP::Assignment(lv, _, e) => {
            let mut v = vec![e];
            lval_exprs(lv, &mut v);
            v
        }
        StmtP::NRFunApp(k, l) => l.iter().chain(kind_exprs(k)).collect(),
        StmtP::For { lower, upper, .. } => vec![lower, upper],
        _ => vec![],
    }
}

/// `used_expressions_stmt_help`, recursing into sub-statements; a loop also uses its variable.
fn used_exprs(s: &Stmt, sub: bool, out: &mut ExprSet) {
    for e in top_exprs(s) {
        if sub {
            subexpressions(e, out);
        } else {
            out.insert(OrdExpr(e.clone()));
        }
    }
    if let StmtP::For { loopvar, .. } = &s.pattern {
        out.insert(OrdExpr(Expr::new(
            ExprP::Var(loopvar.clone()),
            Meta::empty(),
        )));
    }
    for c in s.children() {
        used_exprs(c, sub, out);
    }
}

/// `assigned_or_declared_vars_stmt`.
fn assigned_or_declared(s: &Stmt) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    match &s.pattern {
        StmtP::Assignment(lv, ..) => {
            out.insert(lv.variable().to_string());
        }
        StmtP::Decl { id, .. } => {
            out.insert(id.clone());
        }
        StmtP::TargetPE(_) | StmtP::JacobianPE(_) => {
            out.insert("target".to_string());
        }
        StmtP::NRFunApp(
            FunKind::UserDefined(_, Suffix::Target | Suffix::Jacobian)
            | FunKind::StanLib(_, Suffix::Target | Suffix::Jacobian),
            _,
        ) => {
            out.insert("target".to_string());
        }
        StmtP::For { loopvar, .. } => {
            out.insert(loopvar.clone());
        }
        _ => {}
    }
    out
}

/// `killed_expressions_stmt`: the expressions of `p` that read a variable `s` writes.
fn killed(p: &ExprSet, s: &Stmt) -> ExprSet {
    let written = assigned_or_declared(s);
    p.iter()
        .filter(|e| {
            let mut fv = BTreeSet::new();
            free_vars_expr(&e.0, &mut fv);
            !fv.is_disjoint(&written)
        })
        .cloned()
        .collect()
}

fn set_union(a: &ExprSet, b: &ExprSet) -> ExprSet {
    a.union(b).cloned().collect()
}

fn set_diff(a: &ExprSet, b: &ExprSet) -> ExprSet {
    a.difference(b).cloned().collect()
}

/// `lazy_expressions_mfp`: where each expression is computed at the latest, and which
/// expressions are used later without being computed there.
fn lazy_expressions(
    fwd: &Graph,
    rev: &Graph,
    root: &Stmt,
) -> (Vec<ExprSet>, Vec<EntryExit<ExprSet>>) {
    let mut all = ExprSet::new();
    used_exprs(root, true, &mut all);
    let n = fwd.nodes.len();
    let used: Vec<ExprSet> = (0..n)
        .map(|l| {
            let mut s = ExprSet::new();
            for e in top_exprs(fwd.nodes[l]) {
                subexpressions(e, &mut s);
            }
            s
        })
        .collect();
    let dual = DualPowerset(all);
    let anticipated = mfp(rev, &dual, |l, p: &ExprSet| {
        set_union(&used[l], &set_diff(p, &killed(p, rev.nodes[l])))
    });
    let available = mfp(fwd, &dual, |l, p: &ExprSet| {
        let pg = set_union(p, &anticipated[l].exit);
        set_diff(&pg, &killed(&pg, fwd.nodes[l]))
    });
    let earliest: Vec<ExprSet> = (0..n)
        .map(|l| set_diff(&anticipated[l].exit, &available[l].entry))
        .collect();
    let postponable = mfp(fwd, &dual, |l, p: &ExprSet| {
        set_diff(&set_union(p, &earliest[l]), &used[l])
    });
    let eop = |l: usize| set_union(&earliest[l], &postponable[l].entry);
    let latest: Vec<ExprSet> = (0..n)
        .map(|l| {
            eop(l)
                .into_iter()
                .filter(|e| {
                    used[l].contains(e) || fwd.succs[l].iter().any(|&s| !eop(s).contains(e))
                })
                .collect()
        })
        .collect();
    let used_not_latest = mfp(rev, &Powerset(ExprSet::new()), |l, p: &ExprSet| {
        set_diff(&set_union(p, &used[l]), &latest[l])
    });
    (latest, used_not_latest)
}

/// `expr_depth`.
fn expr_depth(e: &Expr) -> usize {
    match &e.pattern {
        ExprP::Var(_) | ExprP::Lit(..) => 0,
        ExprP::FunApp(k, l) => {
            1 + l
                .iter()
                .chain(kind_exprs(k))
                .map(expr_depth)
                .max()
                .unwrap_or(0)
        }
        _ => 1 + e.children().into_iter().map(expr_depth).max().unwrap_or(0),
    }
}

type ExprMap = BTreeMap<OrdExpr, Expr>;

/// `expr_subst_expr`: replaces the largest subexpressions found in `m`.
fn expr_subst(m: &ExprMap, e: &mut Expr) {
    if let Some(r) = m.get(&OrdExpr(e.clone())) {
        *e = r.clone();
        return;
    }
    for c in e.children_mut() {
        expr_subst(m, c);
    }
}

/// `expr_subst_stmt_base`.
fn expr_subst_stmt_base(m: &ExprMap, s: &mut Stmt) {
    if m.is_empty() {
        return;
    }
    if let StmtP::Assignment(lv, _, e) = &mut s.pattern {
        for i in &mut lv.idx {
            for x in i.exprs_mut() {
                expr_subst(m, x);
            }
        }
        expr_subst(m, e);
    } else {
        for x in s.exprs_mut() {
            expr_subst(m, x);
        }
    }
}

/// `expr_subst_stmt_except_initial_assign`: substitutes everywhere except in the assignment
/// that computes a temporary.
fn subst_except_initial(m: &ExprMap, s: &mut Stmt) {
    for c in s.children_mut() {
        subst_except_initial(m, c);
    }
    let own = match &s.pattern {
        StmtP::Assignment(lv, _, e)
            if lv.idx.is_empty() && matches!(lv.base, LBase::Variable(_)) =>
        {
            Some((lv.variable().to_string(), e))
        }
        StmtP::Decl {
            id,
            init: DeclInit::Assign(e),
            ..
        } => Some((id.clone(), e)),
        _ => None,
    };
    let key = own.and_then(|(x, e)| {
        let k = OrdExpr(e.clone());
        let r = m.get(&k)?;
        let mut v = e.clone();
        v.pattern = ExprP::Var(x);
        (cmp_expr(&v, r) == std::cmp::Ordering::Equal).then_some(k)
    });
    match key {
        Some(k) => {
            let mut m = m.clone();
            m.remove(&k);
            expr_subst_stmt_base(&m, s);
        }
        None => expr_subst_stmt_base(m, s),
    }
}

fn lcm_transform(s: Stmt, sym: &Gensym) -> Stmt {
    let fwd = Graph::build_with(&s, false, false);
    let rev = fwd.reversed();
    let (latest, unl) = lazy_expressions(&fwd, &rev, &s);
    let mut top = ExprSet::new();
    used_exprs(&s, false, &mut top);
    let mut temps: ExprMap = BTreeMap::new();
    fn collect(e: &Expr, temps: &mut ExprMap, sym: &Gensym) {
        match &e.pattern {
            ExprP::Var(_) | ExprP::Lit(..) => {}
            _ if cannot_duplicate_expr(e) => {
                for c in e.children() {
                    collect(c, temps, sym);
                }
            }
            _ => {
                let name = format!("lcm_{}", sym.generate());
                temps.insert(OrdExpr(e.clone()), var(&name));
            }
        }
    }
    for e in &top {
        collect(&e.0, &mut temps, sym);
    }
    // The temporaries keep the type and level of their expression.
    for (k, v) in temps.iter_mut() {
        v.meta = k.0.meta.clone();
    }
    let decls: Vec<Stmt> = temps
        .iter()
        .rev()
        .map(|(k, v)| {
            let ExprP::Var(name) = &v.pattern else {
                unreachable!()
            };
            no_loc(StmtP::Decl {
                ad: k.0.meta.ad.clone(),
                id: name.clone(),
                ty: DeclType::Unsized(k.0.meta.ty.clone()),
                init: DeclInit::Default,
            })
        })
        .collect();
    let mut body = s.clone();
    map_labeled(&mut body, &mut 0, &mut |i, st| {
        let latest_used: ExprSet = latest[i].intersection(&unl[i].entry).cloned().collect();
        let mut to_assign: Vec<&OrdExpr> = latest_used
            .iter()
            .filter(|e| temps.contains_key(e))
            .collect();
        to_assign.sort_by_key(|e| expr_depth(&e.0));
        let assigns: Vec<Stmt> = to_assign
            .iter()
            .map(|e| {
                let ExprP::Var(name) = &temps[*e].pattern else {
                    unreachable!()
                };
                no_loc(StmtP::Assignment(
                    LValue::var(name),
                    e.0.meta.ty.clone(),
                    e.0.clone(),
                ))
            })
            .collect();
        let m: ExprMap = temps
            .iter()
            .filter(|(k, _)| latest_used.contains(*k) || unl[i].exit.contains(*k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        // stanc3 drops the location of a statement that gets assignments before it.
        let mut this = no_loc(std::mem::replace(&mut st.pattern, StmtP::Skip));
        if assigns.is_empty() {
            subst_except_initial(&m, &mut this);
            st.pattern = this.pattern;
        } else {
            let mut l = assigns;
            l.push(this);
            for x in &mut l {
                subst_except_initial(&m, x);
            }
            st.pattern = StmtP::SList(l);
        }
    });
    let mut l = decls;
    l.push(body);
    no_loc(StmtP::SList(l))
}

fn skip() -> Stmt {
    no_loc(StmtP::Skip)
}

/// `lazy_code_motion`: computes each expression once, as late as possible, in a temporary.
pub fn lazy_code_motion(p: &mut Program) {
    lazy_code_motion_with(p, &Gensym::default());
}

/// `lazy_code_motion`, numbering its temporaries after those of earlier passes.
pub(crate) fn lazy_code_motion_with(p: &mut Program, sym: &Gensym) {
    let block_skip = |b: Stmt| no_loc(StmtP::Block(vec![b, skip()]));
    transform_program_blockwise(p, &|_, mut s| {
        map_rec(&mut s, &|p| match p {
            StmtP::IfElse(e, b1, Some(b2)) => StmtP::IfElse(
                e,
                Box::new(block_skip(*b1)),
                Some(Box::new(block_skip(*b2))),
            ),
            StmtP::IfElse(e, b, None) => {
                StmtP::IfElse(e, Box::new(block_skip(*b)), Some(Box::new(skip())))
            }
            StmtP::While(e, b) => StmtP::While(e, Box::new(block_skip(*b))),
            StmtP::For {
                loopvar,
                lower,
                upper,
                body,
            } => StmtP::For {
                loopvar,
                lower,
                upper,
                body: Box::new(block_skip(*body)),
            },
            p => p,
        });
        let mut s = lcm_transform(s, sym);
        let unwrap = |b: &Stmt| -> Option<Stmt> {
            match &b.pattern {
                StmtP::Block(l) if l.len() == 2 && l[1].pattern == StmtP::Skip => {
                    Some(l[0].clone())
                }
                _ => None,
            }
        };
        map_rec(&mut s, &|p| match p {
            StmtP::IfElse(e, b1, Some(b2)) => match (unwrap(&b1), unwrap(&b2)) {
                (Some(x), Some(y)) => StmtP::IfElse(e, Box::new(x), Some(Box::new(y))),
                (Some(x), None) if b2.pattern == StmtP::Skip => StmtP::IfElse(e, Box::new(x), None),
                _ => StmtP::IfElse(e, b1, Some(b2)),
            },
            StmtP::While(e, b) => match unwrap(&b) {
                Some(x) => StmtP::While(e, Box::new(x)),
                None => StmtP::While(e, b),
            },
            StmtP::For {
                loopvar,
                lower,
                upper,
                body,
            } => StmtP::For {
                loopvar,
                lower,
                upper,
                body: Box::new(unwrap(&body).unwrap_or(*body)),
            },
            p => p,
        });
        s
    });
}

// ----- Loop vectorization -----

fn var_names(e: &Expr) -> BTreeSet<String> {
    fn go(e: &Expr, out: &mut BTreeSet<String>) {
        if let ExprP::Var(x) = &e.pattern {
            out.insert(x.clone());
        }
        for c in e.children() {
            go(c, out);
        }
    }
    let mut out = BTreeSet::new();
    go(e, &mut out);
    out
}

fn elementwise_function(name: &str) -> Option<&'static str> {
    match name {
        "Times__" => Some("EltTimes__"),
        "Divide__" => Some("EltDivide__"),
        "Pow__" => Some("EltPow__"),
        _ => None,
    }
}

fn expr_reads_target(e: &Expr) -> bool {
    expr_any(e, &|e| {
        matches!(
            &e.pattern,
            ExprP::FunApp(
                FunKind::UserDefined(_, Suffix::Target) | FunKind::StanLib(_, Suffix::Target),
                _
            )
        )
    })
}

/// What a loop body reads and writes (`conflicts`).
#[derive(Clone)]
struct Conflicts {
    target: bool,
    reads: BTreeSet<String>,
    writes: BTreeSet<String>,
    conflicts: BTreeSet<String>,
    breaks: bool,
}

fn loop_init(upper: &Expr) -> Conflicts {
    Conflicts {
        target: false,
        reads: var_names(upper),
        writes: BTreeSet::new(),
        conflicts: BTreeSet::new(),
        breaks: false,
    }
}

fn merge_info(outer: Conflicts, inner: &Conflicts) -> Conflicts {
    let clash: BTreeSet<String> = outer.writes.intersection(&inner.writes).cloned().collect();
    Conflicts {
        target: outer.target || inner.target,
        reads: outer.reads.union(&inner.reads).cloned().collect(),
        writes: outer.writes.union(&inner.writes).cloned().collect(),
        conflicts: outer.conflicts.union(&clash).cloned().collect(),
        breaks: outer.breaks,
    }
}

/// `vector_state`.
enum Widen<T> {
    /// Only loop-invariant scalars.
    Scalar(T),
    /// Something was vectorized.
    Widened(T),
    Error,
}

struct Vectorizer<'a> {
    loopvar: &'a str,
    lower: &'a Expr,
    upper: &'a Expr,
    dont_read: BTreeSet<String>,
    dont_write: BTreeSet<String>,
    target: bool,
}

impl Vectorizer<'_> {
    fn can_vectorize(&self, e: &Expr) -> bool {
        !cannot_duplicate_expr(e) && var_names(e).is_disjoint(&self.dont_read)
    }

    fn invariant(&self, e: &Expr) -> bool {
        !var_names(e).contains(self.loopvar)
    }

    fn widen(&self, e: &Expr) -> Widen<Expr> {
        use UnsizedType::*;
        match e.meta.ty {
            Int | Real | Complex if self.invariant(e) => Widen::Scalar(e.clone()),
            Int | Real => self.widen_inner(e),
            _ => Widen::Error,
        }
    }

    fn make_call(&self, e: &Expr, name: &str, args: &[Expr]) -> Widen<Expr> {
        use UnsizedType::*;
        match call_return_type(name, args) {
            Some(Some(ty @ (Vector | RowVector))) => self.call(e, name, args, ty),
            Some(Some(ty @ Array(_))) if matches!(&ty, Array(t) if matches!(**t, Int | Real)) => {
                self.call(e, name, args, ty)
            }
            _ => Widen::Error,
        }
    }

    fn call(&self, e: &Expr, name: &str, args: &[Expr], ty: UnsizedType) -> Widen<Expr> {
        let mut meta = e.meta.clone();
        meta.ty = ty;
        Widen::Widened(Expr::new(
            ExprP::FunApp(
                FunKind::StanLib(name.to_string(), Suffix::Plain),
                args.to_vec(),
            ),
            meta,
        ))
    }

    fn widen_inner(&self, e: &Expr) -> Widen<Expr> {
        match &e.pattern {
            ExprP::Indexed(base, idcs) if self.invariant(base) => match self.widen_indices(idcs) {
                None => Widen::Error,
                Some(idcs) => {
                    let mut meta = e.meta.clone();
                    meta.ty = infer_type_of_indexed(&base.meta.ty, &idcs);
                    Widen::Widened(Expr::new(ExprP::Indexed(base.clone(), idcs), meta))
                }
            },
            ExprP::FunApp(FunKind::StanLib(name, Suffix::Plain), args) => {
                match self.widen_all(args) {
                    Widen::Error => Widen::Error,
                    Widen::Scalar(a) | Widen::Widened(a) => {
                        match (self.make_call(e, name, &a), elementwise_function(name)) {
                            (Widen::Error, Some(n)) => self.make_call(e, n, &a),
                            (r, _) => r,
                        }
                    }
                }
            }
            _ => Widen::Error,
        }
    }

    fn widen_all(&self, es: &[Expr]) -> Widen<Vec<Expr>> {
        let mut acc = Widen::Scalar(vec![]);
        for e in es {
            acc = match (acc, self.widen(e)) {
                (Widen::Scalar(mut l), Widen::Scalar(e)) => {
                    l.push(e);
                    Widen::Scalar(l)
                }
                (
                    Widen::Scalar(mut l) | Widen::Widened(mut l),
                    Widen::Scalar(e) | Widen::Widened(e),
                ) => {
                    l.push(e);
                    Widen::Widened(l)
                }
                _ => Widen::Error,
            };
        }
        acc
    }

    /// Widens exactly one index; all others must be invariant scalars.
    fn widen_indices(&self, idxs: &[Index<Expr>]) -> Option<Vec<Index<Expr>>> {
        let mut acc = Widen::Scalar(vec![]);
        for i in idxs {
            acc = match (acc, i) {
                (Widen::Widened(mut l), Index::Single(e)) if self.invariant(e) => {
                    l.push(Index::Single(e.clone()));
                    Widen::Widened(l)
                }
                (Widen::Scalar(mut l), Index::Single(e)) if matches!(&e.pattern, ExprP::Var(v) if v == self.loopvar) =>
                {
                    l.push(Index::Between(self.lower.clone(), self.upper.clone()));
                    Widen::Widened(l)
                }
                (Widen::Scalar(mut l), Index::Single(e)) => match self.widen(e) {
                    Widen::Widened(e) => {
                        l.push(Index::MultiIndex(e));
                        Widen::Widened(l)
                    }
                    Widen::Scalar(e) => {
                        l.push(Index::Single(e));
                        Widen::Scalar(l)
                    }
                    Widen::Error => Widen::Error,
                },
                _ => Widen::Error,
            };
        }
        match acc {
            Widen::Widened(l) => Some(l),
            _ => None,
        }
    }

    /// Moves the vectorizable statements of a loop body into `acc`, returning what is left.
    fn collect(&self, acc: &mut Vec<Stmt>, s: &Stmt) -> Option<Stmt> {
        let wrap = |p: StmtP| Stmt::new(p, s.loc.clone());
        let keep = || Some(s.clone());
        match &s.pattern {
            StmtP::Block(l) => {
                let rest: Vec<Stmt> = l.iter().filter_map(|x| self.collect(acc, x)).collect();
                (!rest.is_empty()).then(|| wrap(StmtP::Block(rest)))
            }
            StmtP::SList(l) => {
                let mut rest: Vec<Stmt> = l.iter().filter_map(|x| self.collect(acc, x)).collect();
                match rest.len() {
                    0 => None,
                    1 => rest.pop(),
                    _ => Some(wrap(StmtP::SList(rest))),
                }
            }
            StmtP::Profile(name, l) => {
                let mut inner = vec![];
                let rest: Vec<Stmt> = l
                    .iter()
                    .filter_map(|x| self.collect(&mut inner, x))
                    .collect();
                if !inner.is_empty() {
                    acc.push(wrap(StmtP::Profile(name.clone(), inner)));
                }
                (!rest.is_empty()).then(|| wrap(StmtP::Profile(name.clone(), rest)))
            }
            StmtP::TargetPE(e) if !self.target && self.can_vectorize(e) => {
                let ExprP::FunApp(
                    FunKind::StanLib(name, suffix @ (Suffix::Lpdf(_) | Suffix::Lpmf(_))),
                    args,
                ) = &e.pattern
                else {
                    return keep();
                };
                let funapp = |args: Vec<Expr>| {
                    let mut f = e.clone();
                    f.pattern = ExprP::FunApp(FunKind::StanLib(name.clone(), *suffix), args);
                    f
                };
                match self.widen_all(args) {
                    Widen::Error => keep(),
                    Widen::Scalar(a) => {
                        let count = Expr::binop(
                            self.upper.clone(),
                            Operator::Minus,
                            Expr::binop(self.lower.clone(), Operator::Minus, Expr::int(1)),
                        );
                        acc.push(wrap(StmtP::TargetPE(Expr::binop(
                            count,
                            Operator::Times,
                            funapp(a),
                        ))));
                        None
                    }
                    Widen::Widened(a) => match call_return_type(name, &a) {
                        Some(Some(UnsizedType::Real)) => {
                            acc.push(wrap(StmtP::TargetPE(funapp(a))));
                            None
                        }
                        _ => keep(),
                    },
                }
            }
            StmtP::Assignment(
                LValue {
                    base: LBase::Variable(v),
                    idx,
                },
                vtype,
                value,
            ) if !self.dont_write.contains(v)
                && self.can_vectorize(value)
                && idx
                    .iter()
                    .all(|i| i.exprs().iter().all(|e| self.can_vectorize(e))) =>
            {
                match (self.widen_indices(idx), self.widen(value)) {
                    (Some(i), Widen::Widened(e))
                        if e.meta.ty == infer_type_of_indexed(vtype, &i) =>
                    {
                        acc.push(wrap(StmtP::Assignment(
                            LValue {
                                base: LBase::Variable(v.clone()),
                                idx: i,
                            },
                            vtype.clone(),
                            e,
                        )));
                        None
                    }
                    _ => keep(),
                }
            }
            _ => keep(),
        }
    }
}

fn vectorized_for(
    loc: Loc,
    info: &Conflicts,
    loopvar: String,
    lower: Expr,
    upper: Expr,
    body: Stmt,
) -> Stmt {
    let make = |body: Stmt| {
        Stmt::new(
            StmtP::For {
                loopvar: loopvar.clone(),
                lower: lower.clone(),
                upper: upper.clone(),
                body: Box::new(body),
            },
            loc.clone(),
        )
    };
    if info.breaks || !info.writes.is_disjoint(&var_names(&upper)) {
        return make(body);
    }
    let v = Vectorizer {
        loopvar: &loopvar,
        lower: &lower,
        upper: &upper,
        dont_read: info.writes.clone(),
        dont_write: info.reads.union(&info.conflicts).cloned().collect(),
        target: info.target,
    };
    let mut acc = vec![];
    let rest = v.collect(&mut acc, &body);
    match rest {
        Some(body) if acc.is_empty() => make(body),
        Some(body) => {
            acc.push(make(body));
            Stmt::new(StmtP::SList(acc), loc.clone())
        }
        None if acc.len() == 1 => acc.pop().unwrap(),
        None => Stmt::new(StmtP::SList(acc), loc.clone()),
    }
}

fn duplicable_bounds(s: &Stmt) -> bool {
    matches!(&s.pattern, StmtP::For { lower, upper, .. }
        if !cannot_duplicate_expr(lower) && !cannot_duplicate_expr(upper))
}

fn vectorize_stmt(s: Stmt) -> Stmt {
    if duplicable_bounds(&s) {
        let StmtP::For {
            loopvar,
            lower,
            upper,
            body,
        } = s.pattern
        else {
            unreachable!()
        };
        let (info, body) = vectorize_inner(loop_init(&upper), *body);
        return vectorized_for(s.loc, &info, loopvar, lower, upper, body);
    }
    let mut s = s;
    for c in s.children_mut() {
        *c = vectorize_stmt(std::mem::replace(c, skip()));
    }
    s
}

/// `collect_reads`: adds the reads of a statement's expressions, then visits its sub-statements.
fn collect_reads(mut c: Conflicts, mut s: Stmt) -> (Conflicts, Stmt) {
    for e in s.exprs() {
        c.target = c.target || expr_reads_target(e);
        c.reads.extend(var_names(e));
    }
    for ch in s.children_mut() {
        let (c2, x) = vectorize_inner(c, std::mem::replace(ch, skip()));
        c = c2;
        *ch = x;
    }
    (c, s)
}

fn vectorize_inner(outer: Conflicts, s: Stmt) -> (Conflicts, Stmt) {
    if duplicable_bounds(&s) {
        let StmtP::For {
            loopvar,
            lower,
            upper,
            body,
        } = s.pattern
        else {
            unreachable!()
        };
        let (info, body) = vectorize_inner(loop_init(&upper), *body);
        let st = vectorized_for(s.loc, &info, loopvar, lower, upper, body);
        return (merge_info(outer, &info), st);
    }
    match &s.pattern {
        StmtP::Assignment(lv, ..) => {
            let lhs = lv.variable().to_string();
            let (mut c, s) = collect_reads(outer, s);
            if c.writes.contains(&lhs) {
                c.conflicts.insert(lhs);
            } else {
                c.writes.insert(lhs);
            }
            (c, s)
        }
        StmtP::Break | StmtP::Continue => {
            let mut c = outer;
            c.breaks = true;
            (c, s)
        }
        StmtP::For { .. } | StmtP::While(..) => {
            let breaks = outer.breaks;
            let (mut c, s) = collect_reads(outer, s);
            c.breaks = breaks;
            (c, s)
        }
        _ => collect_reads(outer, s),
    }
}

/// `vectorize_loops`: turns loops of elementwise statements into vectorized statements.
pub fn vectorize_loops(p: &mut Program) {
    for f in &mut p.functions {
        if let Some(b) = f.body.take() {
            f.body = Some(vectorize_stmt(b));
        }
    }
    for l in [
        &mut p.prepare_data,
        &mut p.log_prob,
        &mut p.reverse_mode_log_prob,
        &mut p.generate_quantities,
        &mut p.transform_inits,
        &mut p.unconstrain_array,
    ] {
        *l = std::mem::take(l).into_iter().map(vectorize_stmt).collect();
    }
}

// ----- Autodiff levels -----

fn max_ad<'a>(l: impl IntoIterator<Item = &'a Expr>) -> Ad {
    l.into_iter()
        .try_fold(Ad::DataOnly, |a, e| a.lub(&e.meta.ad))
        .unwrap_or(Ad::AutoDiffable)
}

/// `update_expr_ad_levels`: recomputes the autodiff level of an expression, given the
/// variables that may carry derivatives.
fn update_expr_ad_levels(vars: &BTreeSet<String>, e: &mut Expr) {
    for c in e.children_mut() {
        update_expr_ad_levels(vars, c);
    }
    e.meta.ad = match &e.pattern {
        ExprP::Var(x) if vars.contains(x) => return,
        ExprP::Var(_) => fill_ad(&Ad::DataOnly, &e.meta.ty),
        ExprP::Lit(..) => Ad::DataOnly,
        ExprP::FunApp(FunKind::CompilerInternal(Internal::MakeTuple), l) => {
            Ad::Tuple(l.iter().map(|x| x.meta.ad.clone()).collect())
        }
        ExprP::FunApp(_, l) => max_ad(l),
        ExprP::TernaryIf(a, b, c) => max_ad([&**a, &**b, &**c]),
        ExprP::EAnd(a, b) | ExprP::EOr(a, b) => max_ad([&**a, &**b]),
        ExprP::Promotion(x, ..) | ExprP::Indexed(x, _) | ExprP::TupleProjection(x, _) => {
            x.meta.ad.clone()
        }
    };
}

/// `optimize_ad_levels`: makes local variables data only when no derivative can reach them.
pub fn optimize_ad_levels(p: &mut Program) {
    let params: BTreeSet<String> = p
        .output_vars
        .iter()
        .filter(|(_, _, v)| v.block == IoBlock::Parameters)
        .map(|(n, ..)| n.clone())
        .collect();
    transform_program_blockwise(p, &|fd, mut s| {
        let mut initial = params.clone();
        if let Some(f) = fd {
            initial.extend(
                f.args
                    .iter()
                    .filter(|(_, _, t)| is_autodiffable(t))
                    .map(|(_, n, _)| n.clone()),
            );
        }
        let fwd = Graph::build(&s, false);
        let rev = fwd.reversed();
        // The exits also lead back to the start, so the analysis runs to a fixed point over
        // repeated executions.
        let mut succs = fwd.succs.clone();
        for &l in &rev.initials {
            succs[l].extend(fwd.initials.iter().copied());
        }
        let circular = Graph {
            nodes: fwd.nodes.clone(),
            children: fwd.children.clone(),
            preds: fwd.preds.clone(),
            succs,
            parents: fwd.parents.clone(),
            initials: fwd.initials.clone(),
            exits: fwd.exits.clone(),
        };
        let vars = mfp(&circular, &Powerset(initial), |l, p: &BTreeSet<String>| {
            let mut out = p.clone();
            let node = circular.nodes[l];
            if let StmtP::Decl {
                id,
                ad: Ad::DataOnly,
                ..
            } = &node.pattern
            {
                out.remove(id);
            }
            if let StmtP::Assignment(lv, _, e) = &node.pattern {
                let mut e = e.clone();
                update_expr_ad_levels(p, &mut e);
                if e.meta.ad.has_autodiff() {
                    out.insert(lv.variable().to_string());
                }
            }
            out
        });
        let sets: Vec<BTreeSet<String>> = vars
            .iter()
            .map(|v| {
                let mut s = v.exit.clone();
                s.extend(v.exit.iter().map(|x| format!("{x}_in__")));
                s
            })
            .collect();
        map_labeled(&mut s, &mut 0, &mut |i, st| {
            let set = &sets[i];
            for e in st.exprs_mut() {
                update_expr_ad_levels(set, e);
            }
            match &mut st.pattern {
                StmtP::Decl { ad, id, ty, .. } => {
                    let level = if set.contains(id.as_str()) {
                        Ad::AutoDiffable
                    } else {
                        Ad::DataOnly
                    };
                    *ad = fill_ad(&level, &ty.to_unsized());
                }
                StmtP::Assignment(lv, ty, e) if !set.contains(lv.variable()) => {
                    if let ExprP::Promotion(_, _, pad) = &mut e.pattern
                        && pad.has_autodiff()
                    {
                        *pad = fill_ad(&Ad::DataOnly, ty);
                    }
                }
                _ => {}
            }
        });
        s
    });
}

/// stanc3's `--Oexperimental` suite: every pass, without preserving numerical stability.
pub fn optimization_suite_experimental(p: &mut Program) {
    use crate::optimize::*;
    // Inlining and lazy code motion share stanc3's global name counter.
    let sym = Gensym::default();
    crate::inline::function_inlining_with(p, &sym);
    constant_propagation(p, false);
    copy_propagation(p);
    constant_propagation(p, false);
    dead_code_elimination(p);
    vectorize_loops(p);
    one_step_loop_unrolling(p);
    expression_propagation(p);
    partial_evaluation(p);
    lazy_code_motion_with(p, &sym);
    copy_propagation(p);
    constant_propagation(p, false);
    static_loop_unrolling(p);
    dead_code_elimination(p);
    list_collapsing(p);
    crate::soa::optimize_soa(p);
    optimize_ad_levels(p);
    allow_uninitialized_decls(p);
    block_fixing(p);
}
