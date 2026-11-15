//! Deprecation warnings, collected after type checking in program order as stanc3's
//! `Deprecation_analysis` does.

use stanc_rs_syntax::ast::{
    Expr, ExprKind, Index, LValue, LValueKind, Printable, Program, SizedType, Stmt, StmtKind,
    Transformation, Truncation,
};
use stanc_rs_syntax::{Diagnostic, DiagnosticKind, Span};

const LKJ_COV: &str = "lkj_cov is deprecated and will be removed in Stan 3.0. Use lkj_corr with an independent lognormal distribution on the scales, see: https://mc-stan.org/docs/reference-manual/deprecations.html#lkj_cov-distribution";

fn deprecated_ode(name: &str) -> Option<&'static str> {
    Some(match name {
        "integrate_ode" | "integrate_ode_rk45" => "ode_rk45",
        "integrate_ode_bdf" => "ode_bdf",
        "integrate_ode_adams" => "ode_adams",
        _ => return None,
    })
}

struct Walk {
    out: Vec<(Span, String)>,
}

impl Walk {
    fn expr(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::FunApp(id, args) | ExprKind::CondDistApp(id, args) => {
                if let Some(new) = deprecated_ode(&id.name) {
                    self.out.push((
                        id.span,
                        format!(
                            "{} is deprecated and will be removed in Stan 3.0. Use {new} instead. The new interface is slightly different, see: https://mc-stan.org/users/documentation/case-studies/convert_odes.html",
                            id.name
                        ),
                    ));
                } else if id.name == "lkj_cov_lpdf" {
                    self.out.push((id.span, LKJ_COV.into()));
                }
                args.iter().for_each(|a| self.expr(a));
            }
            ExprKind::TernaryIf(a, b, c) => {
                self.expr(a);
                self.expr(b);
                self.expr(c);
            }
            ExprKind::BinOp(a, _, b) => {
                self.expr(a);
                self.expr(b);
            }
            ExprKind::PrefixOp(_, a)
            | ExprKind::Transpose(a)
            | ExprKind::Paren(a)
            | ExprKind::TupleProjection(a, _) => self.expr(a),
            ExprKind::Indexed(a, idx) => {
                self.expr(a);
                idx.iter().for_each(|i| self.index(i));
            }
            ExprKind::ArrayExpr(es) | ExprKind::RowVectorExpr(es) | ExprKind::TupleExpr(es) => {
                es.iter().for_each(|a| self.expr(a))
            }
            ExprKind::Variable(_)
            | ExprKind::IntNumeral(_)
            | ExprKind::RealNumeral(_)
            | ExprKind::ImagNumeral(_)
            | ExprKind::GetTarget => {}
        }
    }

    fn index(&mut self, i: &Index) {
        match i {
            Index::All => {}
            Index::Single(e) | Index::Upfrom(e) | Index::Downfrom(e) => self.expr(e),
            Index::Between(a, b) => {
                self.expr(a);
                self.expr(b);
            }
        }
    }

    fn lvalue(&mut self, l: &LValue) {
        match &l.kind {
            LValueKind::Variable(_) => {}
            LValueKind::Indexed(b, idx) => {
                self.lvalue(b);
                idx.iter().for_each(|i| self.index(i));
            }
            LValueKind::TupleProjection(b, _) => self.lvalue(b),
            LValueKind::Tuple(ls) => ls.iter().for_each(|x| self.lvalue(x)),
        }
    }

    fn sized(&mut self, t: &SizedType) {
        match t {
            SizedType::Int | SizedType::Real | SizedType::Complex => {}
            SizedType::Vector(e)
            | SizedType::RowVector(e)
            | SizedType::ComplexVector(e)
            | SizedType::ComplexRowVector(e) => self.expr(e),
            SizedType::Matrix(a, b) | SizedType::ComplexMatrix(a, b) => {
                self.expr(a);
                self.expr(b);
            }
            SizedType::Array(t, e) => {
                self.sized(t);
                self.expr(e);
            }
            SizedType::Tuple(ts) => ts.iter().for_each(|t| self.sized(t)),
        }
    }

    fn transformation(&mut self, t: &Transformation) {
        match t {
            Transformation::Lower(e)
            | Transformation::Upper(e)
            | Transformation::Offset(e)
            | Transformation::Multiplier(e) => self.expr(e),
            Transformation::LowerUpper(a, b) | Transformation::OffsetMultiplier(a, b) => {
                self.expr(a);
                self.expr(b);
            }
            Transformation::Tuple(ts) => ts.iter().for_each(|t| self.transformation(t)),
            _ => {}
        }
    }

    fn printables(&mut self, ps: &[Printable]) {
        for p in ps {
            if let Printable::Expr(e) = p {
                self.expr(e);
            }
        }
    }

    fn stmt(&mut self, s: &Stmt, defined: &[(String, Vec<(String, String)>)]) {
        match &s.kind {
            StmtKind::Assign { lhs, rhs, .. } => {
                self.lvalue(lhs);
                self.expr(rhs);
            }
            StmtKind::NRFunApp(_, args) => args.iter().for_each(|a| self.expr(a)),
            StmtKind::TargetPlusAssign(e) | StmtKind::JacobianPlusAssign(e) => self.expr(e),
            StmtKind::Return(e) => e.iter().for_each(|e| self.expr(e)),
            StmtKind::Print(ps) | StmtKind::Reject(ps) | StmtKind::FatalError(ps) => {
                self.printables(ps)
            }
            StmtKind::Break | StmtKind::Continue | StmtKind::Skip => {}
            StmtKind::Tilde {
                arg,
                dist,
                args,
                truncation,
            } => {
                // stanc3 puts this one at the front of the list.
                if dist.name == "lkj_cov" {
                    self.out.insert(0, (dist.span, LKJ_COV.into()));
                }
                self.expr(arg);
                args.iter().for_each(|a| self.expr(a));
                match truncation {
                    Truncation::None => {}
                    Truncation::UpFrom(e) | Truncation::DownFrom(e) => self.expr(e),
                    Truncation::Between(a, b) => {
                        self.expr(a);
                        self.expr(b);
                    }
                }
            }
            StmtKind::IfThenElse(c, a, b) => {
                self.expr(c);
                self.stmt(a, defined);
                if let Some(b) = b {
                    self.stmt(b, defined);
                }
            }
            StmtKind::While(c, b) => {
                self.expr(c);
                self.stmt(b, defined);
            }
            StmtKind::For {
                lower, upper, body, ..
            } => {
                self.expr(lower);
                self.expr(upper);
                self.stmt(body, defined);
            }
            StmtKind::ForEach { iter, body, .. } => {
                self.expr(iter);
                self.stmt(body, defined);
            }
            StmtKind::Profile(_, ss) | StmtKind::Block(ss) => {
                ss.iter().for_each(|s| self.stmt(s, defined))
            }
            StmtKind::VarDecl {
                ty, trans, vars, ..
            } => {
                self.sized(ty);
                self.transformation(trans);
                for v in vars {
                    if let Some(e) = &v.init {
                        self.expr(e);
                    }
                }
            }
            StmtKind::FunDef {
                name, args, body, ..
            } => {
                if matches!(body.kind, StmtKind::Skip) {
                    let sig = signature(args);
                    if defined.iter().any(|(n, a)| *n == name.name && *a == sig) {
                        self.out.push((
                            name.span,
                            "Functions do not need to be declared before definition; all user defined function names are always in scope regardless of definition order.".into(),
                        ));
                    }
                } else {
                    self.stmt(body, defined);
                }
            }
        }
    }
}

fn signature(args: &[stanc_rs_syntax::ast::FunArg]) -> Vec<(String, String)> {
    args.iter()
        .map(|a| (format!("{:?}", a.ad), format!("{:?}", a.ty)))
        .collect()
}

/// The deprecation warnings of a program that type checks.
pub fn warnings(p: &Program) -> Vec<Diagnostic> {
    let defined: Vec<(String, Vec<(String, String)>)> = p
        .functions
        .iter()
        .flat_map(|b| &b.stmts)
        .filter_map(|s| match &s.kind {
            StmtKind::FunDef {
                name, args, body, ..
            } if !matches!(body.kind, StmtKind::Skip) => Some((name.name.clone(), signature(args))),
            _ => None,
        })
        .collect();
    let mut w = Walk { out: Vec::new() };
    for (_, b) in p.blocks() {
        for s in b.iter().flat_map(|b| &b.stmts) {
            w.stmt(s, &defined);
        }
    }
    w.out
        .into_iter()
        .map(|(span, m)| Diagnostic::new(DiagnosticKind::Warning, span, m))
        .collect()
}
