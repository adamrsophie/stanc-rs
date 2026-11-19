//! Function inlining, a port of stanc3 v2.40.0's `function_inlining` (`Optimize.ml`).
//!
//! The generated names come from one counter in the order stanc3 creates them, which follows
//! OCaml's evaluation order: constructor arguments and record fields right to left, list
//! elements left to right.

use std::collections::{BTreeMap, HashMap};

use stanc_rs_semantic::types::Ad;
use stanc_rs_syntax::ast::UnsizedType;

use crate::lower::{Gensym, fill_ad};
use crate::mir::*;
use crate::optimize::subst_stmt_base;

/// The return type, argument names, and body of an inlinable function.
type InlineMap = HashMap<String, (Option<UnsizedType>, Vec<String>, Stmt)>;

type Triple = (Vec<Stmt>, Vec<Stmt>, Expr);

fn no_loc(p: StmtP) -> Stmt {
    Stmt::new(p, Loc::EMPTY)
}

fn int_lit(v: &str) -> Expr {
    Expr::new(ExprP::Lit(LitType::Int, v.to_string()), Meta::empty())
}

/// `slist_concat_no_loc`.
fn slist_concat(mut l: Vec<Stmt>, p: StmtP) -> StmtP {
    if l.is_empty() {
        return p;
    }
    l.push(no_loc(p));
    StmtP::SList(l)
}

/// `Fun_kind.with_unnormalized_suffix`.
fn with_unnormalized_suffix(name: &str) -> Option<String> {
    name.strip_suffix("_lpdf")
        .map(|n| format!("{n}_lupdf"))
        .or_else(|| name.strip_suffix("_lpmf").map(|n| format!("{n}_lupmf")))
}

fn compute_suffix_and_name(propto: bool, suffix: &Suffix, fname: &str) -> (Suffix, String) {
    let unnorm = || with_unnormalized_suffix(fname).unwrap_or_else(|| fname.to_string());
    match suffix {
        Suffix::Lpdf(true) if propto => (Suffix::Lpdf(true), unnorm()),
        Suffix::Lpdf(_) => (Suffix::Lpdf(false), fname.to_string()),
        Suffix::Lpmf(true) if propto => (Suffix::Lpmf(true), unnorm()),
        Suffix::Lpmf(_) => (Suffix::Lpmf(false), fname.to_string()),
        s => (*s, fname.to_string()),
    }
}

/// `map_rec_stmt_loc`: applies `f` bottom up, visiting the children in the order OCaml's
/// derived map does (the else branch of a conditional before the then branch).
fn map_rec(s: &mut Stmt, f: &mut dyn FnMut(StmtP) -> StmtP) {
    match &mut s.pattern {
        StmtP::IfElse(_, t, e) => {
            if let Some(e) = e {
                map_rec(e, f);
            }
            map_rec(t, f);
        }
        StmtP::While(_, b) | StmtP::For { body: b, .. } => map_rec(b, f),
        StmtP::Profile(_, ss) | StmtP::Block(ss) | StmtP::SList(ss) => {
            for c in ss {
                map_rec(c, f);
            }
        }
        _ => {}
    }
    let p = std::mem::replace(&mut s.pattern, StmtP::Skip);
    s.pattern = f(p);
}

/// `count_returns`: the returns below a statement.
fn count_returns(s: &Stmt) -> usize {
    s.children()
        .into_iter()
        .map(|c| {
            if matches!(c.pattern, StmtP::Return(_)) {
                1
            } else {
                count_returns(c)
            }
        })
        .sum()
}

fn rename_lbase(lv: &mut LValue, m: &HashMap<String, String>) {
    match &mut lv.base {
        LBase::Variable(v) => {
            if let Some(n) = m.get(v) {
                *v = n.clone();
            }
        }
        LBase::TupleProjection(inner, _) => rename_lbase(inner, m),
    }
}

fn rename_vars(e: &mut Expr, m: &HashMap<String, String>) {
    if let ExprP::Var(v) = &mut e.pattern
        && let Some(n) = m.get(v)
    {
        *v = n.clone();
        return;
    }
    for c in e.children_mut() {
        rename_vars(c, m);
    }
}

/// `name_subst_stmt`.
fn name_subst_stmt(s: &mut Stmt, m: &HashMap<String, String>) {
    let mut f = |mut p: StmtP| {
        let mut st = no_loc(p);
        if let StmtP::Assignment(lv, _, e) = &mut st.pattern {
            for i in &mut lv.idx {
                for x in i.exprs_mut() {
                    rename_vars(x, m);
                }
            }
            rename_vars(e, m);
        } else {
            for x in st.exprs_mut() {
                rename_vars(x, m);
            }
        }
        p = st.pattern;
        p
    };
    map_rec(s, &mut f);
}

/// `subst_args_stmt`: replaces the parameters by the arguments.
pub(crate) fn subst_args_stmt(args: &[String], es: &[Expr], s: &mut Stmt) {
    let m: BTreeMap<String, Expr> = args.iter().cloned().zip(es.iter().cloned()).collect();
    let mut f = |p: StmtP| {
        let mut st = no_loc(p);
        subst_stmt_base(&m, &mut st);
        st.pattern
    };
    map_rec(s, &mut f);
}

/// `Stmt.Helpers.expr_of_lvalue`.
fn expr_of_lvalue(lv: &LValue, meta: &Meta) -> Expr {
    let base = match &lv.base {
        LBase::Variable(v) => ExprP::Var(v.clone()),
        LBase::TupleProjection(inner, i) => {
            ExprP::TupleProjection(Box::new(expr_of_lvalue(inner, meta)), *i)
        }
    };
    if lv.idx.is_empty() {
        Expr::new(base, meta.clone())
    } else {
        Expr::new(
            ExprP::Indexed(Box::new(Expr::new(base, meta.clone())), lv.idx.clone()),
            meta.clone(),
        )
    }
}

/// `Stmt.Helpers.lvalue_of_expr_opt`.
fn lvalue_of_expr(e: &Expr) -> Option<LValue> {
    let lbase = |e: &Expr| match &e.pattern {
        ExprP::Var(s) => Some(LBase::Variable(s.clone())),
        ExprP::TupleProjection(l, i) => {
            Some(LBase::TupleProjection(Box::new(lvalue_of_expr(l)?), *i))
        }
        _ => None,
    };
    match &e.pattern {
        ExprP::Var(s) => Some(LValue::var(s)),
        ExprP::Indexed(l, i) => Some(LValue {
            base: lbase(l)?,
            idx: i.clone(),
        }),
        ExprP::TupleProjection(l, i) => Some(LValue {
            base: LBase::TupleProjection(Box::new(lvalue_of_expr(l)?), *i),
            idx: vec![],
        }),
        _ => None,
    }
}

/// `Mir_utils.unsafe_unsized_to_sized_type`: a sized type with zero sizes.
pub(crate) fn unsized_to_sized(t: &UnsizedType) -> SizedType {
    use UnsizedType as U;
    let z = || Expr::int(0);
    match t {
        U::Real => SizedType::Real,
        U::Int => SizedType::Int,
        U::Complex => SizedType::Complex,
        U::Array(t) => SizedType::Array(Box::new(unsized_to_sized(t)), z()),
        U::Matrix => SizedType::Matrix(z(), z()),
        U::Vector => SizedType::Vector(z()),
        U::RowVector => SizedType::RowVector(z()),
        U::ComplexMatrix => SizedType::ComplexMatrix(z(), z()),
        U::ComplexVector => SizedType::ComplexVector(z()),
        U::ComplexRowVector => SizedType::ComplexRowVector(z()),
        U::Tuple(ts) => SizedType::Tuple(ts.iter().map(unsized_to_sized).collect()),
        U::Function(f) => unsized_to_sized(f.ret.as_ref().expect("a function returning a value")),
        U::MathLibraryFunction => panic!("the return type of a function is a library function"),
    }
}

/// The declarations and statements an inlined value needs, and the value.
type Inlined<T> = (Vec<Stmt>, Vec<Stmt>, T);

struct Inliner<'a> {
    sym: &'a Gensym,
}

impl Inliner<'_> {
    fn gen_inline_var(&self, name: &str, id: &str) -> String {
        format!("inline_{name}_{id}_{}", self.sym.generate())
    }

    /// `replace_fresh_local_vars`: gives the local variables of an inlined body fresh names.
    fn replace_fresh_local_vars(&self, fname: &str, mut s: Stmt) -> Stmt {
        let mut m: HashMap<String, String> = HashMap::new();
        let mut f = |p: StmtP| match p {
            StmtP::Decl { ad, id, ty, init } => {
                let new = m
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| self.gen_inline_var(fname, &id));
                m.insert(id, new.clone());
                StmtP::Decl {
                    ad,
                    id: new,
                    ty,
                    init,
                }
            }
            StmtP::For {
                loopvar,
                lower,
                upper,
                body,
            } => {
                let new = m
                    .get(&loopvar)
                    .cloned()
                    .unwrap_or_else(|| self.gen_inline_var(fname, &loopvar));
                m.insert(loopvar, new.clone());
                StmtP::For {
                    loopvar: new,
                    lower,
                    upper,
                    body,
                }
            }
            StmtP::Assignment(mut lv, t, e) => {
                rename_lbase(&mut lv, &m);
                StmtP::Assignment(lv, t, e)
            }
            p => p,
        };
        map_rec(&mut s, &mut f);
        name_subst_stmt(&mut s, &m);
        s
    }

    /// `handle_early_returns`: turns the returns of an inlined body into assignments and
    /// breaks out of a one iteration loop.
    fn handle_early_returns(&self, fname: &str, opt_var: Option<&str>, mut s: Stmt) -> StmtP {
        let returned = self.gen_inline_var(fname, "early_ret_check");
        let num_returns = count_returns(&s);
        let mut inner = |p: StmtP| match p {
            StmtP::Return(opt_ret) => match (opt_var, opt_ret) {
                (None, None) if num_returns > 1 => StmtP::Break,
                (None, None) => StmtP::Block(vec![]),
                (Some(name), Some(e)) if num_returns > 1 => StmtP::SList(vec![
                    no_loc(StmtP::Assignment(
                        LValue::var(&returned),
                        UnsizedType::Int,
                        int_lit("1"),
                    )),
                    no_loc(StmtP::Assignment(LValue::var(name), e.meta.ty.clone(), e)),
                    no_loc(StmtP::Break),
                ]),
                (Some(name), Some(e)) => StmtP::Assignment(LValue::var(name), e.meta.ty.clone(), e),
                (Some(_), None) => panic!("a function returning a value has an empty return"),
                (None, Some(_)) => panic!("a void function returns a value"),
            },
            p @ StmtP::For { .. } if num_returns > 1 => StmtP::SList(vec![
                no_loc(p),
                no_loc(StmtP::IfElse(
                    Expr::new(ExprP::Var(returned.clone()), Meta::empty()),
                    Box::new(no_loc(StmtP::Break)),
                    None,
                )),
            ]),
            p => p,
        };
        if num_returns > 1 {
            map_rec(&mut s, &mut inner);
            let iterator = self.gen_inline_var(fname, "iterator");
            StmtP::SList(vec![
                no_loc(StmtP::Decl {
                    ad: Ad::DataOnly,
                    id: returned.clone(),
                    ty: DeclType::Sized(SizedType::Int),
                    init: DeclInit::Default,
                }),
                no_loc(StmtP::Assignment(
                    LValue::var(&returned),
                    UnsizedType::Int,
                    int_lit("0"),
                )),
                no_loc(StmtP::For {
                    loopvar: iterator,
                    lower: int_lit("1"),
                    upper: int_lit("1"),
                    body: Box::new(s),
                }),
            ])
        } else {
            map_rec(&mut s, &mut inner);
            s.pattern
        }
    }

    fn inline_list<T>(
        &self,
        es: Vec<T>,
        f: &mut dyn FnMut(&Self, T) -> Inlined<T>,
    ) -> Inlined<Vec<T>> {
        let dse: Vec<_> = es.into_iter().map(|e| f(self, e)).collect();
        let mut d = Vec::new();
        let mut s = Vec::new();
        let mut out = Vec::new();
        let mut parts = Vec::new();
        for (dl, sl, e) in dse {
            parts.push((dl, sl));
            out.push(e);
        }
        for (dl, _) in parts.iter().rev() {
            d.extend(dl.iter().cloned());
        }
        for (_, sl) in parts.into_iter().rev() {
            s.extend(sl);
        }
        (d, s, out)
    }

    fn expr(&self, propto: bool, adt: &Ad, fim: &InlineMap, e: Expr) -> Triple {
        let Expr { pattern, meta } = e;
        let w = |p: ExprP| Expr::new(p, meta.clone());
        match pattern {
            p @ (ExprP::Var(_) | ExprP::Lit(..)) => (vec![], vec![], w(p)),
            ExprP::Promotion(x, t, ad) => {
                let (d, s, x) = self.expr(propto, adt, fim, *x);
                (d, s, w(ExprP::Promotion(Box::new(x), t, ad)))
            }
            ExprP::FunApp(kind, es) => {
                let (mut d, mut s, es) =
                    self.inline_list(es, &mut |me: &Self, e| me.expr(propto, adt, fim, e));
                match kind {
                    k @ FunKind::CompilerInternal(_) => (d, s, w(ExprP::FunApp(k, es))),
                    FunKind::StanLib(fname, suffix) => {
                        let (suffix, _) = compute_suffix_and_name(propto, &suffix, &fname);
                        (d, s, w(ExprP::FunApp(FunKind::StanLib(fname, suffix), es)))
                    }
                    FunKind::UserDefined(fname, suffix) => {
                        let (suffix, fname2) = compute_suffix_and_name(propto, &suffix, &fname);
                        match fim.get(&fname2) {
                            None => (
                                d,
                                s,
                                w(ExprP::FunApp(FunKind::UserDefined(fname, suffix), es)),
                            ),
                            Some((rt, args, body)) => {
                                let ret_name = self.gen_inline_var(&fname, "return");
                                let ut = rt
                                    .clone()
                                    .expect("an inlined function in an expression returns a value");
                                let st = unsized_to_sized(&ut);
                                let ut = st.to_unsized();
                                let mut b = self.replace_fresh_local_vars(&fname, body.clone());
                                subst_args_stmt(args, &es, &mut b);
                                let handled = self.handle_early_returns(&fname, Some(&ret_name), b);
                                d.push(no_loc(StmtP::Decl {
                                    ad: fill_ad(adt, &ut),
                                    id: ret_name.clone(),
                                    ty: DeclType::Sized(st),
                                    init: DeclInit::Uninit,
                                }));
                                s.push(no_loc(handled));
                                let v = Expr::new(
                                    ExprP::Var(ret_name),
                                    Meta {
                                        ty: ut,
                                        ad: adt.clone(),
                                        loc: Loc::EMPTY,
                                    },
                                );
                                (d, s, v)
                            }
                        }
                    }
                }
            }
            ExprP::TernaryIf(e1, e2, e3) => {
                let (d1, mut s1, e1) = self.expr(propto, adt, fim, *e1);
                let (d2, s2, e2) = self.expr(propto, adt, fim, *e2);
                let (d3, s3, e3) = self.expr(propto, adt, fim, *e3);
                let mut d = d1;
                d.extend(d2);
                d.extend(d3);
                s1.push(no_loc(StmtP::IfElse(
                    e1.clone(),
                    Box::new(no_loc(StmtP::Block(s2))),
                    Some(Box::new(no_loc(StmtP::Block(s3)))),
                )));
                (
                    d,
                    s1,
                    w(ExprP::TernaryIf(Box::new(e1), Box::new(e2), Box::new(e3))),
                )
            }
            ExprP::Indexed(x, idx) => {
                let (dl, sl, x) = self.expr(propto, adt, fim, *x);
                let (mut d, mut s, idx) =
                    self.inline_list(idx, &mut |me: &Self, i| me.index(propto, adt, fim, i));
                d.extend(dl);
                s.extend(sl);
                (d, s, w(ExprP::Indexed(Box::new(x), idx)))
            }
            ExprP::TupleProjection(x, i) => {
                let (d, s, x) = self.expr(propto, adt, fim, *x);
                (d, s, w(ExprP::TupleProjection(Box::new(x), i)))
            }
            ExprP::EAnd(e1, e2) => {
                let (mut d, mut s, e1) = self.expr(propto, adt, fim, *e1);
                let (d2, s2, e2) = self.expr(propto, adt, fim, *e2);
                d.extend(d2);
                s.push(no_loc(StmtP::IfElse(
                    e1.clone(),
                    Box::new(no_loc(StmtP::Block(s2))),
                    None,
                )));
                (d, s, w(ExprP::EAnd(Box::new(e1), Box::new(e2))))
            }
            ExprP::EOr(e1, e2) => {
                let (mut d, mut s, e1) = self.expr(propto, adt, fim, *e1);
                let (d2, s2, e2) = self.expr(propto, adt, fim, *e2);
                d.extend(d2);
                s.push(no_loc(StmtP::IfElse(
                    e1.clone(),
                    Box::new(no_loc(StmtP::Skip)),
                    Some(Box::new(no_loc(StmtP::Block(s2)))),
                )));
                (d, s, w(ExprP::EOr(Box::new(e1), Box::new(e2))))
            }
        }
    }

    fn index(
        &self,
        propto: bool,
        adt: &Ad,
        fim: &InlineMap,
        i: Index<Expr>,
    ) -> (Vec<Stmt>, Vec<Stmt>, Index<Expr>) {
        match i {
            Index::All => (vec![], vec![], Index::All),
            Index::Single(e) => {
                let (d, s, e) = self.expr(propto, adt, fim, e);
                (d, s, Index::Single(e))
            }
            Index::Upfrom(e) => {
                let (d, s, e) = self.expr(propto, adt, fim, e);
                (d, s, Index::Upfrom(e))
            }
            Index::MultiIndex(e) => {
                let (d, s, e) = self.expr(propto, adt, fim, e);
                (d, s, Index::MultiIndex(e))
            }
            Index::Between(a, b) => {
                let (mut d, mut s, a) = self.expr(propto, adt, fim, a);
                let (d2, s2, b) = self.expr(propto, adt, fim, b);
                d.extend(d2);
                s.extend(s2);
                (d, s, Index::Between(a, b))
            }
        }
    }

    fn stmt(&self, propto: bool, adt: &Ad, fim: &InlineMap, st: Stmt) -> Stmt {
        let Stmt { pattern, loc } = st;
        let cat = |a: Vec<Stmt>, b: Vec<Stmt>| {
            let mut a = a;
            a.extend(b);
            a
        };
        let pattern = match pattern {
            StmtP::Assignment(lhs, ut, e2) => {
                let e1 = expr_of_lvalue(&lhs, &e2.meta);
                let (dl1, sl1, e1) = self.expr(propto, adt, fim, e1);
                let (dl2, sl2, e2) = self.expr(propto, adt, fim, e2);
                let lhs = lvalue_of_expr(&e1).expect("an assigned location");
                slist_concat(
                    cat(cat(dl2, dl1), cat(sl2, sl1)),
                    StmtP::Assignment(lhs, ut, e2),
                )
            }
            StmtP::TargetPE(e) => {
                let (d, s, e) = self.expr(propto, adt, fim, e);
                slist_concat(cat(d, s), StmtP::TargetPE(e))
            }
            StmtP::JacobianPE(e) => {
                let (d, s, e) = self.expr(propto, adt, fim, e);
                slist_concat(cat(d, s), StmtP::JacobianPE(e))
            }
            StmtP::NRFunApp(kind, es) => {
                let (d, s, es) =
                    self.inline_list(es, &mut |me: &Self, e| me.expr(propto, adt, fim, e));
                let p = match &kind {
                    FunKind::UserDefined(name, _) => match fim.get(name) {
                        None => StmtP::NRFunApp(kind, es),
                        Some((_, args, b)) => {
                            let b = self.replace_fresh_local_vars(name, b.clone());
                            let b = self.handle_early_returns(name, None, b);
                            let mut b = no_loc(b);
                            subst_args_stmt(args, &es, &mut b);
                            b.pattern
                        }
                    },
                    _ => StmtP::NRFunApp(kind, es),
                };
                slist_concat(cat(d, s), p)
            }
            StmtP::Return(None) => StmtP::Return(None),
            StmtP::Return(Some(e)) => {
                let (d, s, e) = self.expr(propto, adt, fim, e);
                slist_concat(cat(d, s), StmtP::Return(Some(e)))
            }
            StmtP::IfElse(c, t, e) => {
                let (d, s, c) = self.expr(propto, adt, fim, c);
                let e = e.map(|e| Box::new(self.stmt(propto, adt, fim, *e)));
                let t = Box::new(self.stmt(propto, adt, fim, *t));
                slist_concat(cat(d, s), StmtP::IfElse(c, t, e))
            }
            StmtP::While(c, b) => {
                let (d, s, c) = self.expr(propto, adt, fim, c);
                let body = if s.is_empty() {
                    self.stmt(propto, adt, fim, *b)
                } else {
                    let mut l = vec![self.stmt(propto, adt, fim, *b)];
                    l.extend(s.iter().cloned());
                    no_loc(StmtP::Block(l))
                };
                slist_concat(cat(d, s), StmtP::While(c, Box::new(body)))
            }
            StmtP::For {
                loopvar,
                lower,
                upper,
                body,
            } => {
                let (dl, sl, lower) = self.expr(propto, adt, fim, lower);
                let (du, su, upper) = self.expr(propto, adt, fim, upper);
                let body = if su.is_empty() {
                    self.stmt(propto, adt, fim, *body)
                } else {
                    let mut l = vec![self.stmt(propto, adt, fim, *body)];
                    l.extend(su.iter().cloned());
                    no_loc(StmtP::Block(l))
                };
                slist_concat(
                    cat(cat(dl, du), cat(sl, su)),
                    StmtP::For {
                        loopvar,
                        lower,
                        upper,
                        body: Box::new(body),
                    },
                )
            }
            StmtP::Profile(n, l) => StmtP::Profile(n, self.stmts(propto, adt, fim, l)),
            StmtP::Block(l) => StmtP::Block(self.stmts(propto, adt, fim, l)),
            StmtP::SList(l) => StmtP::SList(self.stmts(propto, adt, fim, l)),
            StmtP::Decl {
                ad,
                id,
                ty,
                init: DeclInit::Assign(e),
            } => {
                let (d, s, e) = self.expr(propto, adt, fim, e);
                slist_concat(
                    cat(d, s),
                    StmtP::Decl {
                        ad,
                        id,
                        ty,
                        init: DeclInit::Assign(e),
                    },
                )
            }
            p => p,
        };
        Stmt::new(pattern, loc)
    }

    fn stmts(&self, propto: bool, adt: &Ad, fim: &InlineMap, l: Vec<Stmt>) -> Vec<Stmt> {
        l.into_iter()
            .map(|s| self.stmt(propto, adt, fim, s))
            .collect()
    }

    /// `create_function_inline_map`.
    fn inline_map(&self, adt: &Ad, fs: &[&FunDef]) -> InlineMap {
        let mut accum = InlineMap::new();
        for fd in fs {
            let Some(body) = &fd.body else { continue };
            let args: Vec<String> = fd.args.iter().map(|(_, n, _)| n.clone()).collect();
            let create = |acc: &InlineMap, propto: bool| {
                (
                    fd.rt.clone(),
                    args.clone(),
                    self.stmt(propto, adt, acc, body.clone()),
                )
            };
            match with_unnormalized_suffix(&fd.name) {
                None => {
                    let data = create(&accum, true);
                    accum.entry(fd.name.clone()).or_insert(data);
                }
                Some(unnorm) => {
                    let data = create(&accum, false);
                    let data2 = create(&accum, true);
                    accum.entry(fd.name.clone()).or_insert(data);
                    accum.entry(unnorm).or_insert(data2);
                }
            }
        }
        accum
    }
}

/// `function_inlining`: inlines the calls of user-defined functions that have a single
/// definition into the model's blocks.
pub fn function_inlining(p: &mut Program) {
    function_inlining_with(p, &Gensym::default());
}

/// `function_inlining` with a name counter that later passes continue.
pub(crate) fn function_inlining_with(p: &mut Program, sym: &Gensym) {
    let inl = Inliner { sym };
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for f in &p.functions {
        *counts.entry(&f.name).or_default() += 1;
    }
    let inlinable: Vec<&FunDef> = p
        .functions
        .iter()
        .filter(|f| counts[f.name.as_str()] == 1)
        .collect();
    let data_map = inl.inline_map(&Ad::DataOnly, &inlinable);
    let ad_map = inl.inline_map(&Ad::AutoDiffable, &inlinable);
    let run = |l: Vec<Stmt>, adt: &Ad, m: &InlineMap| inl.stmts(true, adt, m, l);
    // OCaml builds the record's fields right to left.
    p.generate_quantities = run(
        std::mem::take(&mut p.generate_quantities),
        &Ad::DataOnly,
        &data_map,
    );
    p.reverse_mode_log_prob = run(
        std::mem::take(&mut p.reverse_mode_log_prob),
        &Ad::AutoDiffable,
        &ad_map,
    );
    p.log_prob = run(std::mem::take(&mut p.log_prob), &Ad::AutoDiffable, &ad_map);
    p.unconstrain_array = run(
        std::mem::take(&mut p.unconstrain_array),
        &Ad::AutoDiffable,
        &ad_map,
    );
    p.transform_inits = run(
        std::mem::take(&mut p.transform_inits),
        &Ad::AutoDiffable,
        &ad_map,
    );
}
