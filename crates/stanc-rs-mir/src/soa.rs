//! `optimize_soa`: decides which parameters the gradient code may read as struct of arrays
//! (SoA) matrices, a port of stanc3's `Memory_patterns`.
//!
//! A matrix must be an array of structs (AoS) when Stan Math has no SoA overload for a function
//! it is passed to, when single elements are read in a loop, and when it is assigned to or from
//! one that must be AoS. stanc3 computes the last part with a dataflow analysis on a circular
//! flowgraph whose properties only grow, so every reachable statement ends up with the same set;
//! this port computes that set directly as a fixed point.

use std::collections::BTreeSet;

use stanc_rs_semantic::signatures::supports_soa;
use stanc_rs_semantic::types::Ad;

use crate::mir::*;

type Names = BTreeSet<String>;

fn contains_eigen(t: &UnsizedType) -> bool {
    use UnsizedType::*;
    match t {
        Vector | RowVector | Matrix | ComplexVector | ComplexRowVector | ComplexMatrix => true,
        Array(t) => contains_eigen(t),
        Function(f) => f.ret.as_ref().is_some_and(contains_eigen),
        Tuple(ts) => ts.iter().any(contains_eigen),
        Int | Real | Complex | MathLibraryFunction => false,
    }
}

fn is_eigen(t: &UnsizedType) -> bool {
    use UnsizedType::*;
    matches!(
        t,
        Vector | RowVector | Matrix | ComplexVector | ComplexRowVector | ComplexMatrix
    )
}

fn is_complex(t: &UnsizedType) -> bool {
    use UnsizedType::*;
    match t {
        Complex | ComplexVector | ComplexRowVector | ComplexMatrix => true,
        Array(t) => is_complex(t),
        _ => false,
    }
}

fn fun_args(es: &[Expr]) -> Vec<(Ad, UnsizedType)> {
    es.iter()
        .map(|e| (e.meta.ad.clone(), e.meta.ty.clone()))
        .collect()
}

/// `query_var_eigen_names`: the autodiff variables holding matrices that an expression of a
/// matrix type is made of.
fn eigen_names(e: &Expr) -> Names {
    fn go(e: &Expr, out: &mut Names) {
        if !contains_eigen(&e.meta.ty) {
            return;
        }
        match &e.pattern {
            ExprP::Var(s) => {
                if e.meta.ad.has_autodiff() {
                    out.insert(s.clone());
                }
            }
            ExprP::Lit(..) => {}
            ExprP::FunApp(_, es) => es.iter().for_each(|x| go(x, out)),
            ExprP::TernaryIf(_, a, b) | ExprP::EAnd(a, b) | ExprP::EOr(a, b) => {
                go(a, out);
                go(b, out);
            }
            ExprP::Indexed(x, _) | ExprP::Promotion(x, ..) | ExprP::TupleProjection(x, _) => {
                go(x, out)
            }
        }
    }
    let mut out = Names::new();
    go(e, &mut out);
    out
}

fn is_nonzero_subset(set: &Names, subset: &Names) -> bool {
    !set.is_empty() && !subset.is_empty() && subset.is_subset(set)
}

/// `is_uni_eigen_loop_indexing`: reads single elements of a vector or matrix in a loop.
fn single_element_in_loop(in_loop: bool, ut: &UnsizedType, idx: &[Index<Expr>]) -> bool {
    if !in_loop {
        return false;
    }
    let singles = idx.iter().filter(|i| matches!(i, Index::Single(_))).count();
    match ut {
        UnsizedType::Vector | UnsizedType::RowVector if singles > 0 => true,
        UnsizedType::Matrix if singles > 1 => true,
        UnsizedType::Array(t) => !idx.is_empty() && single_element_in_loop(in_loop, t, &idx[1..]),
        UnsizedType::Function(f) => match &f.ret {
            Some(t) => !idx.is_empty() && single_element_in_loop(in_loop, t, &idx[1..]),
            None => false,
        },
        _ => false,
    }
}

fn union(mut a: Names, b: Names) -> Names {
    a.extend(b);
    a
}

/// `query_initial_demotable_expr`.
fn initial_expr(in_loop: bool, acc: &Names, e: &Expr) -> Names {
    let q = |acc: &Names, e: &Expr| initial_expr(in_loop, acc, e);
    match &e.pattern {
        ExprP::FunApp(kind, es) => initial_funs(in_loop, acc, kind, es),
        ExprP::Indexed(x, idx) => {
            let mut index_set = Names::new();
            for i in idx {
                for y in i.exprs() {
                    index_set.extend(q(acc, y));
                }
            }
            let demotes = if single_element_in_loop(in_loop, &x.meta.ty, idx) {
                union(eigen_names(x), index_set)
            } else {
                union(q(acc, x), index_set)
            };
            union(acc.clone(), demotes)
        }
        ExprP::Var(_) | ExprP::Lit(..) => acc.clone(),
        ExprP::Promotion(x, ..) | ExprP::TupleProjection(x, _) => q(acc, x),
        ExprP::TernaryIf(p, t, f) => union(union(q(acc, p), eigen_names(t)), eigen_names(f)),
        ExprP::EAnd(l, r) | ExprP::EOr(l, r) => {
            let full = union(q(acc, l), q(acc, r));
            union(q(&full, l), q(&full, r))
        }
    }
}

/// `query_initial_demotable_funs`.
fn initial_funs(in_loop: bool, acc: &Names, kind: &FunKind, es: &[Expr]) -> Names {
    let top: Names = es.iter().flat_map(eigen_names).collect();
    let demoted = es
        .iter()
        .fold(acc.clone(), |a, e| initial_expr(in_loop, &a, e));
    let all = || union(union(acc.clone(), demoted.clone()), top.clone());
    match kind {
        FunKind::StanLib(name, _) if name == "check_matching_dims" => acc.clone(),
        FunKind::StanLib(name, _) => {
            if supports_soa(name, &fun_args(es)) {
                union(acc.clone(), demoted)
            } else {
                all()
            }
        }
        FunKind::CompilerInternal(
            Internal::MakeArray | Internal::MakeRowVec | Internal::MakeTuple,
        )
        | FunKind::UserDefined(..) => all(),
        FunKind::CompilerInternal(_) => acc.clone(),
    }
}

/// `extract_nonderived_admatrix_types`.
fn nonderived_admatrix_types(e: &Expr) -> Vec<(Ad, UnsizedType)> {
    if !(e.meta.ad.has_autodiff() && contains_eigen(&e.meta.ty)) {
        return vec![(e.meta.ad.clone(), e.meta.ty.clone())];
    }
    match &e.pattern {
        ExprP::FunApp(FunKind::StanLib(name, _), es) => match name.as_str() {
            "check_matching_dims" => vec![],
            "rep_vector" => vec![(Ad::AutoDiffable, UnsizedType::Vector)],
            "rep_row_vector" => vec![(Ad::AutoDiffable, UnsizedType::RowVector)],
            "rep_matrix" if es.len() == 3 && es[0].meta.ty == UnsizedType::Real => {
                vec![(Ad::AutoDiffable, UnsizedType::Matrix)]
            }
            _ => es.iter().flat_map(nonderived_admatrix_types).collect(),
        },
        ExprP::FunApp(FunKind::CompilerInternal(Internal::MakeArray), _) => vec![
            (Ad::AutoDiffable, UnsizedType::Real),
            (
                Ad::DataOnly,
                UnsizedType::Array(Box::new(UnsizedType::Real)),
            ),
        ],
        ExprP::FunApp(FunKind::CompilerInternal(Internal::MakeRowVec), _) => vec![
            (Ad::AutoDiffable, UnsizedType::Real),
            (Ad::DataOnly, UnsizedType::RowVector),
        ],
        ExprP::FunApp(..) => vec![],
        ExprP::Indexed(x, _) | ExprP::Promotion(x, ..) | ExprP::TupleProjection(x, _) => {
            nonderived_admatrix_types(x)
        }
        ExprP::Var(_) | ExprP::Lit(..) => vec![(e.meta.ad.clone(), e.meta.ty.clone())],
        ExprP::TernaryIf(_, a, b) | ExprP::EAnd(a, b) | ExprP::EOr(a, b) => {
            let mut v = nonderived_admatrix_types(a);
            v.extend(nonderived_admatrix_types(b));
            v
        }
    }
}

/// `contains_at_least_one_ad_matrix_or_all_data`.
fn has_ad_matrix_or_all_data(args: &[(Ad, UnsizedType)]) -> bool {
    args.is_empty()
        || args.iter().any(|(a, t)| a.has_autodiff() && is_eigen(t))
        || args.iter().all(|(a, _)| !a.has_autodiff())
}

/// `query_initial_demotable_stmt`.
fn initial_stmt(in_loop: bool, acc: &Names, s: &Stmt) -> Names {
    let q = |acc: &Names, e: &Expr| initial_expr(in_loop, acc, e);
    let all = |l: &[Stmt], in_loop: bool| {
        l.iter()
            .fold(Names::new(), |a, x| union(a, initial_stmt(in_loop, acc, x)))
    };
    match &s.pattern {
        StmtP::Assignment(lv, ut, rhs) => {
            let name = lv.variable().to_string();
            let mut idx_list = acc.clone();
            for i in &lv.idx {
                for e in i.exprs() {
                    idx_list = q(&idx_list, e);
                }
            }
            if single_element_in_loop(in_loop, ut, &lv.idx) {
                idx_list.insert(name.clone());
            }
            let rhs_demotable = q(acc, rhs);
            let mut demotions = union(idx_list, rhs_demotable.clone());
            if matches!(lv.base, LBase::TupleProjection(..)) {
                demotions.extend(eigen_names(rhs));
                demotions.insert(name.clone());
            }
            if contains_eigen(&rhs.meta.ty) {
                let not_promotable = rhs.meta.ad == Ad::AutoDiffable
                    && !has_ad_matrix_or_all_data(&nonderived_admatrix_types(rhs));
                let unsupported = match &rhs.pattern {
                    ExprP::FunApp(FunKind::UserDefined(..), _) => true,
                    ExprP::FunApp(FunKind::StanLib(f, _), es) => !supports_soa(f, &fun_args(es)),
                    _ => false,
                };
                let all_rhs_aos = is_nonzero_subset(&rhs_demotable, &eigen_names(rhs));
                if all_rhs_aos || not_promotable || unsupported {
                    demotions.extend(eigen_names(rhs));
                    demotions.insert(name);
                }
            }
            union(acc.clone(), demotions)
        }
        StmtP::NRFunApp(kind, es) => initial_funs(in_loop, acc, kind, es),
        StmtP::IfElse(p, t, f) => {
            let pa = q(acc, p);
            let mut out = union(acc.clone(), initial_stmt(in_loop, &pa, t));
            if let Some(f) = f {
                out.extend(initial_stmt(in_loop, &pa, f));
            }
            union(out, pa)
        }
        StmtP::Return(e) => e.as_ref().map_or_else(Names::new, |e| q(acc, e)),
        StmtP::SList(l) | StmtP::Block(l) | StmtP::Profile(_, l) => all(l, in_loop),
        StmtP::TargetPE(e) | StmtP::JacobianPE(e) => q(acc, e),
        StmtP::For {
            lower, upper, body, ..
        } => {
            let is_one = |e: &Expr| matches!(&e.pattern, ExprP::Lit(LitType::Int, s) if s == "1");
            if is_one(lower) && is_one(upper) {
                initial_stmt(in_loop, acc, body)
            } else {
                union(
                    union(q(acc, lower), q(acc, upper)),
                    initial_stmt(true, acc, body),
                )
            }
        }
        StmtP::While(p, body) => {
            union(union(acc.clone(), q(acc, p)), initial_stmt(true, acc, body))
        }
        StmtP::Decl {
            id,
            ty: DeclType::Sized(st),
            init,
            ..
        } => {
            let mut out = acc.clone();
            if is_complex(&st.to_unsized()) {
                out.insert(id.clone());
            }
            if let DeclInit::Assign(e) = init {
                out.extend(q(acc, e));
            }
            out
        }
        StmtP::Decl { .. } | StmtP::Skip | StmtP::Break | StmtP::Continue => acc.clone(),
    }
}

/// `query_demotable_stmt`: an assignment involving an AoS matrix makes the others AoS too.
fn demotable(aos: &Names, s: &Stmt) -> Names {
    let (name, rhs) = match &s.pattern {
        StmtP::Assignment(lv, _, rhs) => (lv.variable(), rhs),
        StmtP::Decl {
            id,
            init: DeclInit::Assign(e),
            ..
        } => (id.as_str(), e),
        _ => return Names::new(),
    };
    let names = eigen_names(rhs);
    if aos.contains(name) || is_nonzero_subset(aos, &names) {
        let mut out = names;
        out.insert(name.to_string());
        out
    } else {
        Names::new()
    }
}

fn for_each_stmt(s: &Stmt, f: &mut impl FnMut(&Stmt)) {
    f(s);
    for c in s.children() {
        for_each_stmt(c, f);
    }
}

fn set_read_mem(e: &mut Expr, aos: bool) {
    if let ExprP::FunApp(FunKind::CompilerInternal(Internal::ReadParam { mem_pattern, .. }), _) =
        &mut e.pattern
    {
        *mem_pattern = if aos {
            MemPattern::AoS
        } else {
            MemPattern::SoA
        };
    }
}

/// `SizedType.has_mem_pattern`.
fn has_mem_pattern(t: &SizedType) -> bool {
    match t {
        SizedType::Vector(_) | SizedType::RowVector(_) | SizedType::Matrix(..) => true,
        SizedType::Array(t, _) => has_mem_pattern(t),
        SizedType::Tuple(ts) => ts.iter().any(has_mem_pattern),
        _ => false,
    }
}

fn set_mem(s: &mut Stmt, aos: &Names) {
    if let StmtP::Decl { ty, init, .. } = &mut s.pattern
        && let DeclType::Unsized(ut) = ty
    {
        // Declarations left unsized by earlier passes become uninitialized, and those holding
        // matrices get (zero) sizes to carry their memory pattern.
        *init = DeclInit::Uninit;
        let st = crate::inline::unsized_to_sized(ut);
        if has_mem_pattern(&st) {
            *ty = DeclType::Sized(st);
        }
    }
    match &mut s.pattern {
        StmtP::Decl {
            id,
            init: DeclInit::Assign(e),
            ..
        } => set_read_mem(e, aos.contains(id.as_str())),
        StmtP::Assignment(lv, _, e) => {
            let a = aos.contains(lv.variable());
            set_read_mem(e, a);
        }
        _ => {}
    }
    for c in s.children_mut() {
        set_mem(c, aos);
    }
}

/// `optimize_soa`: marks the parameters read in `reverse_mode_log_prob` as SoA unless one of
/// the rules above makes them AoS.
pub fn optimize_soa(p: &mut Program) {
    let initial = p
        .reverse_mode_log_prob
        .iter()
        .fold(Names::new(), |a, s| initial_stmt(false, &a, s));
    // ponytail: assumes every statement is reachable, as it is in model code; stanc3 leaves
    // unreachable statements (after a `break`) out of the propagation.
    let mut aos = initial;
    loop {
        let mut next = aos.clone();
        for s in &p.reverse_mode_log_prob {
            for_each_stmt(s, &mut |x| next.extend(demotable(&aos, x)));
        }
        if next == aos {
            break;
        }
        aos = next;
    }
    for s in &mut p.reverse_mode_log_prob {
        set_mem(s, &aos);
    }
}
