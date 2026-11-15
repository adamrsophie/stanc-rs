//! stanc3's canonicalizer (`Canonicalize.ml`), which `--print-canonical` and `--canonicalize`
//! apply before formatting: it removes redundant forward declarations, unneeded parentheses,
//! and adds braces around the bodies of loops and conditionals.

use crate::ast::*;

/// Which rewrites to apply; the field names follow stanc3's `--canonicalize` options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Settings {
    /// `deprecations`: drop forward declarations of functions defined later.
    pub deprecations: bool,
    /// `parentheses`: drop parentheses that do not change the meaning.
    pub parentheses: bool,
    /// `braces`: put the bodies of `if`, `else`, `while`, and `for` in braces.
    pub braces: bool,
    /// `includes`: print included files in place of their `#include`.
    pub inline_includes: bool,
    /// `strip-comments`: drop comments.
    pub strip_comments: bool,
}

impl Settings {
    /// What `--print-canonical` applies.
    pub const LEGACY: Settings = Settings {
        deprecations: true,
        parentheses: true,
        braces: true,
        inline_includes: true,
        strip_comments: false,
    };

    /// Parses a `--canonicalize` value such as `deprecations,braces`.
    pub fn parse(list: &str) -> Result<Settings, String> {
        let mut s = Settings::default();
        for item in list.split(',').filter(|x| !x.is_empty()) {
            match item {
                "deprecations" => s.deprecations = true,
                "parentheses" => s.parentheses = true,
                "braces" => s.braces = true,
                "includes" => s.inline_includes = true,
                "strip-comments" => s.strip_comments = true,
                other => {
                    return Err(format!(
                        "invalid --canonicalize option '{other}'; expected deprecations, \
                         parentheses, braces, includes, or strip-comments"
                    ));
                }
            }
        }
        Ok(s)
    }
}

/// Applies the rewrites of `settings` to a program.
pub fn canonicalize(p: &Program, settings: &Settings) -> Program {
    let mut p = p.clone();
    if settings.deprecations {
        remove_forward_decls(&mut p);
    }
    for b in blocks_mut(&mut p) {
        for s in &mut b.stmts {
            if settings.parentheses {
                parens_stmt(s);
            }
            if settings.braces {
                braces_stmt(s);
            }
        }
    }
    p
}

fn blocks_mut(p: &mut Program) -> impl Iterator<Item = &mut Block> {
    [
        &mut p.functions,
        &mut p.data,
        &mut p.transformed_data,
        &mut p.parameters,
        &mut p.transformed_parameters,
        &mut p.model,
        &mut p.generated_quantities,
    ]
    .into_iter()
    .flatten()
}

fn signature(args: &[FunArg]) -> Vec<(AdLevel, UnsizedType)> {
    args.iter().map(|a| (a.ad, a.ty.clone())).collect()
}

fn remove_forward_decls(p: &mut Program) {
    let Some(b) = &mut p.functions else { return };
    let defined: Vec<(String, Vec<(AdLevel, UnsizedType)>)> = b
        .stmts
        .iter()
        .filter_map(|s| match &s.kind {
            StmtKind::FunDef {
                name, args, body, ..
            } if !matches!(body.kind, StmtKind::Skip) => Some((name.name.clone(), signature(args))),
            _ => None,
        })
        .collect();
    b.stmts.retain(|s| match &s.kind {
        StmtKind::FunDef {
            name, args, body, ..
        } if matches!(body.kind, StmtKind::Skip) => {
            let sig = signature(args);
            !defined.iter().any(|(n, a)| *n == name.name && *a == sig)
        }
        _ => true,
    });
}

fn is_cmp(op: BinOp) -> bool {
    use BinOp::*;
    matches!(op, Equals | NEquals | Less | Leq | Greater | Geq)
}

/// `no_parens`: removes parentheses that only group an operand already grouped by its position.
fn no_parens(e: Expr) -> Expr {
    let span = e.span;
    let kind = match e.kind {
        ExprKind::Paren(inner) => return no_parens(*inner),
        k @ (ExprKind::Variable(_)
        | ExprKind::IntNumeral(_)
        | ExprKind::RealNumeral(_)
        | ExprKind::ImagNumeral(_)
        | ExprKind::GetTarget) => k,
        ExprKind::BinOp(a, op2, b)
            if matches!(&a.kind, ExprKind::BinOp(_, op1, _) if is_cmp(*op1)) && is_cmp(op2) =>
        {
            let a_span = a.span;
            let a = Expr {
                kind: ExprKind::Paren(Box::new(no_parens(*a))),
                span: a_span,
            };
            ExprKind::BinOp(Box::new(a), op2, Box::new(keep_parens(*b)))
        }
        ExprKind::TernaryIf(c, a, b) => ExprKind::TernaryIf(
            Box::new(keep_parens(*c)),
            Box::new(keep_parens(*a)),
            Box::new(keep_parens(*b)),
        ),
        ExprKind::BinOp(a, op, b) => {
            ExprKind::BinOp(Box::new(keep_parens(*a)), op, Box::new(keep_parens(*b)))
        }
        ExprKind::PrefixOp(op, a) => ExprKind::PrefixOp(op, Box::new(keep_parens(*a))),
        ExprKind::Transpose(a) => ExprKind::Transpose(Box::new(keep_parens(*a))),
        ExprKind::Indexed(a, idx) => ExprKind::Indexed(
            Box::new(keep_parens(*a)),
            idx.into_iter()
                .map(|i| match i {
                    Index::Single(e) => Index::Single(no_parens(e)),
                    i => map_index(i, keep_parens),
                })
                .collect(),
        ),
        ExprKind::TupleProjection(a, k) => ExprKind::TupleProjection(Box::new(keep_parens(*a)), k),
        ExprKind::ArrayExpr(xs) => ExprKind::ArrayExpr(xs.into_iter().map(no_parens).collect()),
        ExprKind::RowVectorExpr(xs) => {
            ExprKind::RowVectorExpr(xs.into_iter().map(no_parens).collect())
        }
        ExprKind::TupleExpr(xs) => ExprKind::TupleExpr(xs.into_iter().map(no_parens).collect()),
        ExprKind::FunApp(id, xs) => ExprKind::FunApp(id, xs.into_iter().map(no_parens).collect()),
        ExprKind::CondDistApp(id, xs) => {
            ExprKind::CondDistApp(id, xs.into_iter().map(no_parens).collect())
        }
    };
    Expr { kind, span }
}

/// `keep_parens`: keeps one pair of parentheses around an operator expression.
fn keep_parens(e: Expr) -> Expr {
    match e.kind {
        ExprKind::Paren(inner) if matches!(inner.kind, ExprKind::Paren(_)) => keep_parens(*inner),
        ExprKind::Paren(inner)
            if matches!(
                inner.kind,
                ExprKind::BinOp(..)
                    | ExprKind::PrefixOp(..)
                    | ExprKind::Transpose(_)
                    | ExprKind::TernaryIf(..)
            ) =>
        {
            Expr {
                kind: ExprKind::Paren(Box::new(no_parens(*inner))),
                span: e.span,
            }
        }
        kind => no_parens(Expr { kind, span: e.span }),
    }
}

fn map_index(i: Index, f: fn(Expr) -> Expr) -> Index {
    match i {
        Index::All => Index::All,
        Index::Single(e) => Index::Single(f(e)),
        Index::Upfrom(e) => Index::Upfrom(f(e)),
        Index::Downfrom(e) => Index::Downfrom(f(e)),
        Index::Between(a, b) => Index::Between(f(a), f(b)),
    }
}

fn apply(e: &mut Expr, f: fn(Expr) -> Expr) {
    let taken = std::mem::replace(
        e,
        Expr {
            kind: ExprKind::GetTarget,
            span: e.span,
        },
    );
    *e = f(taken);
}

fn lvalue(l: &mut LValue) {
    match &mut l.kind {
        LValueKind::Variable(_) => {}
        LValueKind::Indexed(a, idx) => {
            lvalue(a);
            for i in idx {
                let taken = std::mem::replace(i, Index::All);
                *i = map_index(taken, no_parens);
            }
        }
        LValueKind::TupleProjection(a, _) => lvalue(a),
        LValueKind::Tuple(ls) => ls.iter_mut().for_each(lvalue),
    }
}

fn sized(t: &mut SizedType) {
    match t {
        SizedType::Int | SizedType::Real | SizedType::Complex => {}
        SizedType::Vector(e)
        | SizedType::RowVector(e)
        | SizedType::ComplexVector(e)
        | SizedType::ComplexRowVector(e) => apply(e, no_parens),
        SizedType::Matrix(a, b) | SizedType::ComplexMatrix(a, b) => {
            apply(a, no_parens);
            apply(b, no_parens);
        }
        SizedType::Array(t, e) => {
            sized(t);
            apply(e, no_parens);
        }
        SizedType::Tuple(ts) => ts.iter_mut().for_each(sized),
    }
}

fn transformation(t: &mut Transformation) {
    match t {
        Transformation::Lower(e)
        | Transformation::Upper(e)
        | Transformation::Offset(e)
        | Transformation::Multiplier(e) => apply(e, keep_parens),
        Transformation::LowerUpper(a, b) | Transformation::OffsetMultiplier(a, b) => {
            apply(a, keep_parens);
            apply(b, keep_parens);
        }
        Transformation::Tuple(ts) => ts.iter_mut().for_each(transformation),
        _ => {}
    }
}

fn parens_stmt(s: &mut Stmt) {
    let ne = |e: &mut Expr| apply(e, no_parens);
    match &mut s.kind {
        StmtKind::VarDecl {
            ty, trans, vars, ..
        } => {
            sized(ty);
            transformation(trans);
            for v in vars {
                if let Some(e) = &mut v.init {
                    ne(e);
                }
            }
        }
        StmtKind::For {
            lower, upper, body, ..
        } => {
            apply(lower, keep_parens);
            apply(upper, keep_parens);
            parens_stmt(body);
        }
        StmtKind::Assign { lhs, rhs, .. } => {
            lvalue(lhs);
            ne(rhs);
        }
        StmtKind::NRFunApp(_, args) => args.iter_mut().for_each(ne),
        StmtKind::TargetPlusAssign(e) | StmtKind::JacobianPlusAssign(e) => ne(e),
        StmtKind::Return(e) => e.iter_mut().for_each(ne),
        StmtKind::Print(ps) | StmtKind::Reject(ps) | StmtKind::FatalError(ps) => {
            for p in ps {
                if let Printable::Expr(e) = p {
                    ne(e);
                }
            }
        }
        StmtKind::Tilde {
            arg,
            args,
            truncation,
            ..
        } => {
            ne(arg);
            args.iter_mut().for_each(ne);
            match truncation {
                Truncation::None => {}
                Truncation::UpFrom(e) | Truncation::DownFrom(e) => ne(e),
                Truncation::Between(a, b) => {
                    ne(a);
                    ne(b);
                }
            }
        }
        StmtKind::IfThenElse(c, a, b) => {
            ne(c);
            parens_stmt(a);
            if let Some(b) = b {
                parens_stmt(b);
            }
        }
        StmtKind::While(c, body) => {
            ne(c);
            parens_stmt(body);
        }
        StmtKind::ForEach { iter, body, .. } => {
            ne(iter);
            parens_stmt(body);
        }
        StmtKind::Profile(_, ss) | StmtKind::Block(ss) => ss.iter_mut().for_each(parens_stmt),
        StmtKind::FunDef { body, .. } => parens_stmt(body),
        StmtKind::Break | StmtKind::Continue | StmtKind::Skip => {}
    }
}

/// Wraps a statement in a block with the statement's location, unless it is one.
fn to_block(s: &mut Stmt) {
    if !matches!(s.kind, StmtKind::Block(_)) {
        let inner = std::mem::replace(
            s,
            Stmt {
                kind: StmtKind::Block(Vec::new()),
                span: s.span,
            },
        );
        s.kind = StmtKind::Block(vec![inner]);
    }
    braces_stmt(s);
}

fn braces_stmt(s: &mut Stmt) {
    match &mut s.kind {
        StmtKind::While(_, body) => to_block(body),
        StmtKind::IfThenElse(_, a, b) => {
            to_block(a);
            if let Some(b) = b {
                // `else if` stays flat, also when written `else { if ... }`.
                let inner = match &mut b.kind {
                    StmtKind::Block(ss)
                        if ss.len() == 1 && matches!(ss[0].kind, StmtKind::IfThenElse(..)) =>
                    {
                        ss.pop()
                    }
                    _ => None,
                };
                if let Some(inner) = inner {
                    **b = inner;
                }
                if matches!(b.kind, StmtKind::IfThenElse(..)) {
                    braces_stmt(b);
                } else {
                    to_block(b);
                }
            }
        }
        StmtKind::For { body, .. } => to_block(body),
        StmtKind::ForEach { body, .. } | StmtKind::FunDef { body, .. } => braces_stmt(body),
        StmtKind::Profile(_, ss) | StmtKind::Block(ss) => ss.iter_mut().for_each(braces_stmt),
        _ => {}
    }
}
