//! The order stanc3's `[@@deriving compare]` gives MIR expressions: constructors in declaration
//! order, then their arguments left to right, lists lexicographically, and locations ignored.
//! Sets and maps of expressions in stanc3 iterate in this order, which decides where lazy code
//! motion puts its temporaries and how it numbers them.

use std::cmp::Ordering;

use stanc_rs_semantic::types::Ad;

use crate::mir::*;

fn lex<T>(a: &[T], b: &[T], f: impl Fn(&T, &T) -> Ordering) -> Ordering {
    for (x, y) in a.iter().zip(b) {
        let o = f(x, y);
        if o != Ordering::Equal {
            return o;
        }
    }
    a.len().cmp(&b.len())
}

fn ut_rank(t: &UnsizedType) -> u8 {
    use UnsizedType::*;
    match t {
        Int => 0,
        Real => 1,
        Vector => 2,
        Complex => 3,
        RowVector => 4,
        Matrix => 5,
        ComplexVector => 6,
        ComplexRowVector => 7,
        ComplexMatrix => 8,
        Array(_) => 9,
        Tuple(_) => 10,
        Function(_) => 11,
        MathLibraryFunction => 12,
    }
}

/// `UnsizedType.compare`.
pub fn cmp_ut(a: &UnsizedType, b: &UnsizedType) -> Ordering {
    use UnsizedType::*;
    match (a, b) {
        (Array(x), Array(y)) => cmp_ut(x, y),
        (Tuple(x), Tuple(y)) => lex(x, y, cmp_ut),
        (Function(f), Function(g)) => {
            let ad = |a: &stanc_rs_syntax::ast::AdLevel| -> Ad { (*a).into() };
            lex(&f.args, &g.args, |(a1, t1), (a2, t2)| {
                cmp_ad(&ad(a1), &ad(a2)).then_with(|| cmp_ut(t1, t2))
            })
            .then_with(|| match (&f.ret, &g.ret) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (Some(x), Some(y)) => cmp_ut(x, y),
            })
        }
        _ => ut_rank(a).cmp(&ut_rank(b)),
    }
}

/// `UnsizedType.compare_autodifftype`.
pub fn cmp_ad(a: &Ad, b: &Ad) -> Ordering {
    match (a, b) {
        (Ad::Tuple(x), Ad::Tuple(y)) => lex(x, y, cmp_ad),
        _ => {
            let r = |a: &Ad| match a {
                Ad::DataOnly => 0,
                Ad::AutoDiffable => 1,
                Ad::Tuple(_) => 2,
            };
            r(a).cmp(&r(b))
        }
    }
}

fn suffix_rank(s: &Suffix) -> (u8, bool) {
    match s {
        Suffix::Plain => (0, false),
        Suffix::Rng => (1, false),
        Suffix::Lpdf(p) => (2, *p),
        Suffix::Lpmf(p) => (3, *p),
        Suffix::Target => (4, false),
        Suffix::Jacobian => (5, false),
    }
}

fn cmp_trans(a: &Transformation, b: &Transformation) -> Ordering {
    use Transformation::*;
    let rank = |t: &Transformation| match t {
        Identity => 0,
        Lower(_) => 1,
        Upper(_) => 2,
        LowerUpper(..) => 3,
        Offset(_) => 4,
        Multiplier(_) => 5,
        OffsetMultiplier(..) => 6,
        Ordered => 7,
        PositiveOrdered => 8,
        Simplex => 9,
        UnitVector => 10,
        SumToZero => 11,
        CholeskyCorr => 12,
        CholeskyCov => 13,
        Correlation => 14,
        Covariance => 15,
        StochasticRow => 16,
        StochasticColumn => 17,
        Tuple(_) => 18,
    };
    match (a, b) {
        (Lower(x), Lower(y))
        | (Upper(x), Upper(y))
        | (Offset(x), Offset(y))
        | (Multiplier(x), Multiplier(y)) => cmp_expr(x, y),
        (LowerUpper(x1, x2), LowerUpper(y1, y2))
        | (OffsetMultiplier(x1, x2), OffsetMultiplier(y1, y2)) => {
            cmp_expr(x1, y1).then_with(|| cmp_expr(x2, y2))
        }
        (Tuple(x), Tuple(y)) => lex(x, y, cmp_trans),
        _ => rank(a).cmp(&rank(b)),
    }
}

fn internal_rank(i: &Internal) -> u8 {
    use Internal::*;
    match i {
        Length => 0,
        MakeArray => 1,
        MakeTuple => 2,
        MakeRowVec => 3,
        NegInf => 4,
        ReadData => 5,
        ReadDeserializer => 6,
        ReadParam { .. } => 7,
        WriteParam { .. } => 8,
        ValidateSize => 9,
        ValidateSizePositive => 10,
        ValidateSizeUnitVector => 11,
        Check { .. } => 12,
        Print => 13,
        Reject => 14,
        FatalError => 15,
        ResizeToMatch => 16,
        NaN => 17,
        DeepCopy => 18,
    }
}

fn cmp_internal(a: &Internal, b: &Internal) -> Ordering {
    use Internal::*;
    match (a, b) {
        (
            ReadParam {
                constrain: c1,
                dims: d1,
                mem_pattern: m1,
            },
            ReadParam {
                constrain: c2,
                dims: d2,
                mem_pattern: m2,
            },
        ) => cmp_trans(c1, c2)
            .then_with(|| lex(d1, d2, cmp_expr))
            .then_with(|| (*m1 as u8).cmp(&(*m2 as u8))),
        (
            WriteParam {
                unconstrain_opt: u1,
                var: v1,
            },
            WriteParam {
                unconstrain_opt: u2,
                var: v2,
            },
        ) => match (u1, u2) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (Some(x), Some(y)) => cmp_trans(x, y),
        }
        .then_with(|| cmp_expr(v1, v2)),
        (
            Check {
                trans: t1,
                var_name: n1,
                var: v1,
            },
            Check {
                trans: t2,
                var_name: n2,
                var: v2,
            },
        ) => cmp_trans(t1, t2)
            .then_with(|| n1.cmp(n2))
            .then_with(|| cmp_expr(v1, v2)),
        _ => internal_rank(a).cmp(&internal_rank(b)),
    }
}

fn cmp_kind(a: &FunKind, b: &FunKind) -> Ordering {
    match (a, b) {
        (FunKind::StanLib(n1, s1), FunKind::StanLib(n2, s2))
        | (FunKind::UserDefined(n1, s1), FunKind::UserDefined(n2, s2)) => n1
            .cmp(n2)
            .then_with(|| suffix_rank(s1).cmp(&suffix_rank(s2))),
        (FunKind::CompilerInternal(x), FunKind::CompilerInternal(y)) => cmp_internal(x, y),
        _ => {
            let r = |k: &FunKind| match k {
                FunKind::StanLib(..) => 0,
                FunKind::CompilerInternal(_) => 1,
                FunKind::UserDefined(..) => 2,
            };
            r(a).cmp(&r(b))
        }
    }
}

fn cmp_index(a: &Index<Expr>, b: &Index<Expr>) -> Ordering {
    let rank = |i: &Index<Expr>| match i {
        Index::All => 0,
        Index::Single(_) => 1,
        Index::Upfrom(_) => 2,
        Index::Between(..) => 3,
        Index::MultiIndex(_) => 4,
    };
    match (a, b) {
        (Index::Single(x), Index::Single(y))
        | (Index::Upfrom(x), Index::Upfrom(y))
        | (Index::MultiIndex(x), Index::MultiIndex(y)) => cmp_expr(x, y),
        (Index::Between(x1, x2), Index::Between(y1, y2)) => {
            cmp_expr(x1, y1).then_with(|| cmp_expr(x2, y2))
        }
        _ => rank(a).cmp(&rank(b)),
    }
}

fn pattern_rank(p: &ExprP) -> u8 {
    match p {
        ExprP::Var(_) => 0,
        ExprP::Lit(..) => 1,
        ExprP::FunApp(..) => 2,
        ExprP::TernaryIf(..) => 3,
        ExprP::EAnd(..) => 4,
        ExprP::EOr(..) => 5,
        ExprP::Indexed(..) => 6,
        ExprP::Promotion(..) => 7,
        ExprP::TupleProjection(..) => 8,
    }
}

/// `Expr.Typed.compare`.
pub fn cmp_expr(a: &Expr, b: &Expr) -> Ordering {
    use ExprP::*;
    let pat = match (&a.pattern, &b.pattern) {
        (Var(x), Var(y)) => x.cmp(y),
        (Lit(t1, s1), Lit(t2, s2)) => (*t1 as u8).cmp(&(*t2 as u8)).then_with(|| s1.cmp(s2)),
        (FunApp(k1, a1), FunApp(k2, a2)) => cmp_kind(k1, k2).then_with(|| lex(a1, a2, cmp_expr)),
        (TernaryIf(x1, x2, x3), TernaryIf(y1, y2, y3)) => cmp_expr(x1, y1)
            .then_with(|| cmp_expr(x2, y2))
            .then_with(|| cmp_expr(x3, y3)),
        (EAnd(x1, x2), EAnd(y1, y2)) | (EOr(x1, x2), EOr(y1, y2)) => {
            cmp_expr(x1, y1).then_with(|| cmp_expr(x2, y2))
        }
        (Indexed(x, i), Indexed(y, j)) => cmp_expr(x, y).then_with(|| lex(i, j, cmp_index)),
        (Promotion(x, t1, d1), Promotion(y, t2, d2)) => cmp_expr(x, y)
            .then_with(|| cmp_ut(t1, t2))
            .then_with(|| cmp_ad(d1, d2)),
        (TupleProjection(x, i), TupleProjection(y, j)) => cmp_expr(x, y).then_with(|| i.cmp(j)),
        (p, q) => pattern_rank(p).cmp(&pattern_rank(q)),
    };
    pat.then_with(|| cmp_ut(&a.meta.ty, &b.meta.ty))
        .then_with(|| cmp_ad(&a.meta.ad, &b.meta.ad))
}

/// An expression ordered as stanc3 orders it, for sets and maps.
#[derive(Clone, Debug)]
pub struct OrdExpr(pub Expr);

impl PartialEq for OrdExpr {
    fn eq(&self, other: &Self) -> bool {
        cmp_expr(&self.0, &other.0) == Ordering::Equal
    }
}

impl Eq for OrdExpr {}

impl PartialOrd for OrdExpr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdExpr {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_expr(&self.0, &other.0)
    }
}
