//! Pretty printer: turns an AST back into Stan source.
//!
//! The output is canonical (two-space indentation, one statement per line) and re-parses to the
//! same AST up to spans. Comments are not preserved.

use crate::ast::*;
use std::fmt::Write as _;

/// Formats a whole program.
pub fn program(p: &Program) -> String {
    let mut out = String::new();
    for (name, block) in p.blocks() {
        if let Some(b) = block {
            let _ = writeln!(out, "{name} {{");
            for s in &b.stmts {
                stmt(&mut out, s, 1);
            }
            let _ = writeln!(out, "}}");
        }
    }
    out
}

/// Formats an expression.
pub fn expr(e: &Expr) -> String {
    let mut s = String::new();
    write_expr(&mut s, e);
    s
}

fn list(out: &mut String, xs: &[Expr]) {
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write_expr(out, x);
    }
}

fn write_expr(out: &mut String, e: &Expr) {
    match &e.kind {
        ExprKind::Variable(id) => out.push_str(&id.name),
        ExprKind::IntNumeral(s) | ExprKind::RealNumeral(s) => out.push_str(s),
        ExprKind::ImagNumeral(s) => {
            out.push_str(s);
            out.push('i');
        }
        ExprKind::FunApp(id, args) => {
            out.push_str(&id.name);
            out.push('(');
            list(out, args);
            out.push(')');
        }
        ExprKind::CondDistApp(id, args) => {
            out.push_str(&id.name);
            out.push('(');
            if let Some((first, rest)) = args.split_first() {
                write_expr(out, first);
                if !rest.is_empty() {
                    out.push_str(" | ");
                    list(out, rest);
                }
            }
            out.push(')');
        }
        ExprKind::GetTarget => out.push_str("target()"),
        // Sub-expressions are parenthesized defensively so the output re-parses identically.
        ExprKind::TernaryIf(c, a, b) => {
            out.push('(');
            write_expr(out, c);
            out.push_str(" ? ");
            write_expr(out, a);
            out.push_str(" : ");
            write_expr(out, b);
            out.push(')');
        }
        ExprKind::BinOp(a, op, b) => {
            out.push('(');
            write_expr(out, a);
            let _ = write!(out, " {} ", op.symbol());
            write_expr(out, b);
            out.push(')');
        }
        ExprKind::PrefixOp(op, a) => {
            out.push('(');
            out.push_str(op.symbol());
            write_expr(out, a);
            out.push(')');
        }
        ExprKind::Transpose(a) => {
            out.push('(');
            write_expr(out, a);
            out.push_str("')");
        }
        ExprKind::Indexed(a, idx) => {
            write_expr(out, a);
            indices(out, idx);
        }
        ExprKind::ArrayExpr(xs) => {
            out.push('{');
            list(out, xs);
            out.push('}');
        }
        ExprKind::RowVectorExpr(xs) => {
            out.push('[');
            list(out, xs);
            out.push(']');
        }
        ExprKind::TupleExpr(xs) => {
            out.push('(');
            list(out, xs);
            out.push(')');
        }
        ExprKind::TupleProjection(a, k) => {
            write_expr(out, a);
            let _ = write!(out, ".{k}");
        }
        ExprKind::Paren(a) => {
            out.push('(');
            write_expr(out, a);
            out.push(')');
        }
    }
}

fn indices(out: &mut String, idx: &[Index]) {
    out.push('[');
    for (i, ix) in idx.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        match ix {
            Index::All => {}
            Index::Single(e) => write_expr(out, e),
            Index::Upfrom(e) => {
                write_expr(out, e);
                out.push(':');
            }
            Index::Downfrom(e) => {
                out.push(':');
                write_expr(out, e);
            }
            Index::Between(a, b) => {
                write_expr(out, a);
                out.push(':');
                write_expr(out, b);
            }
        }
    }
    out.push(']');
}

fn lvalue(out: &mut String, l: &LValue) {
    match &l.kind {
        LValueKind::Variable(id) => out.push_str(&id.name),
        LValueKind::Indexed(a, idx) => {
            lvalue(out, a);
            indices(out, idx);
        }
        LValueKind::TupleProjection(a, k) => {
            lvalue(out, a);
            let _ = write!(out, ".{k}");
        }
        LValueKind::Tuple(xs) => {
            out.push('(');
            for (i, x) in xs.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                lvalue(out, x);
            }
            out.push(')');
        }
    }
}

fn indent(out: &mut String, n: usize) {
    for _ in 0..n {
        out.push_str("  ");
    }
}

fn printables(out: &mut String, ps: &[Printable]) {
    for (i, p) in ps.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        match p {
            Printable::Str(s) => out.push_str(s),
            Printable::Expr(e) => write_expr(out, e),
        }
    }
}

/// Formats an unsized type.
pub fn unsized_type(t: &UnsizedType) -> String {
    t.to_string()
}

fn transformation(out: &mut String, t: &Transformation) {
    let bound = |out: &mut String, items: &[(&str, &Expr)]| {
        out.push('<');
        for (i, (k, e)) in items.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "{k}=");
            write_expr(out, e);
        }
        out.push('>');
    };
    match t {
        Transformation::Lower(e) => bound(out, &[("lower", e)]),
        Transformation::Upper(e) => bound(out, &[("upper", e)]),
        Transformation::LowerUpper(a, b) => bound(out, &[("lower", a), ("upper", b)]),
        Transformation::Offset(e) => bound(out, &[("offset", e)]),
        Transformation::Multiplier(e) => bound(out, &[("multiplier", e)]),
        Transformation::OffsetMultiplier(a, b) => bound(out, &[("offset", a), ("multiplier", b)]),
        _ => {}
    }
}

/// Formats a declaration type with its transformation.
pub fn decl_type(ty: &SizedType, trans: &Transformation) -> String {
    let mut out = String::new();
    write_decl_type(&mut out, ty, trans);
    out
}

fn write_decl_type(out: &mut String, ty: &SizedType, trans: &Transformation) {
    // Arrays: collect the dimensions.
    if let SizedType::Array(..) = ty {
        let mut dims = Vec::new();
        let mut t = ty;
        while let SizedType::Array(inner, d) = t {
            dims.push(d);
            t = inner;
        }
        out.push_str("array[");
        list(out, &dims.into_iter().cloned().collect::<Vec<_>>());
        out.push_str("] ");
        write_decl_type(out, t, trans);
        return;
    }
    let named = |name: &str| -> Option<String> { Some(name.to_string()) };
    let special = match (trans, ty) {
        (Transformation::Ordered, _) => named("ordered"),
        (Transformation::PositiveOrdered, _) => named("positive_ordered"),
        (Transformation::Simplex, _) => named("simplex"),
        (Transformation::UnitVector, _) => named("unit_vector"),
        (Transformation::SumToZero, SizedType::Vector(_)) => named("sum_to_zero_vector"),
        (Transformation::SumToZero, _) => named("sum_to_zero_matrix"),
        (Transformation::CholeskyCorr, _) => named("cholesky_factor_corr"),
        (Transformation::CholeskyCov, _) => named("cholesky_factor_cov"),
        (Transformation::Correlation, _) => named("corr_matrix"),
        (Transformation::Covariance, _) => named("cov_matrix"),
        (Transformation::StochasticColumn, _) => named("column_stochastic_matrix"),
        (Transformation::StochasticRow, _) => named("row_stochastic_matrix"),
        _ => None,
    };
    if let Some(name) = special {
        out.push_str(&name);
        let sizes: Vec<Expr> = match (trans, ty) {
            (
                Transformation::CholeskyCorr
                | Transformation::Correlation
                | Transformation::Covariance,
                SizedType::Matrix(a, _),
            ) => {
                vec![a.clone()]
            }
            _ => ty.size_exprs().into_iter().cloned().collect(),
        };
        out.push('[');
        list(out, &sizes);
        out.push(']');
        return;
    }
    let (name, sizes): (&str, Vec<Expr>) = match ty {
        SizedType::Int => ("int", vec![]),
        SizedType::Real => ("real", vec![]),
        SizedType::Complex => ("complex", vec![]),
        SizedType::Vector(e) => ("vector", vec![e.clone()]),
        SizedType::RowVector(e) => ("row_vector", vec![e.clone()]),
        SizedType::Matrix(a, b) => ("matrix", vec![a.clone(), b.clone()]),
        SizedType::ComplexVector(e) => ("complex_vector", vec![e.clone()]),
        SizedType::ComplexRowVector(e) => ("complex_row_vector", vec![e.clone()]),
        SizedType::ComplexMatrix(a, b) => ("complex_matrix", vec![a.clone(), b.clone()]),
        SizedType::Tuple(ts) => {
            out.push_str("tuple(");
            let empty = Vec::new();
            let trs = if let Transformation::Tuple(trs) = trans {
                trs
            } else {
                &empty
            };
            for (i, t) in ts.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_decl_type(out, t, trs.get(i).unwrap_or(&Transformation::Identity));
            }
            out.push(')');
            return;
        }
        SizedType::Array(..) => unreachable!(),
    };
    out.push_str(name);
    transformation(out, trans);
    if !sizes.is_empty() {
        out.push('[');
        list(out, &sizes);
        out.push(']');
    }
}

fn stmt(out: &mut String, s: &Stmt, depth: usize) {
    indent(out, depth);
    match &s.kind {
        StmtKind::Assign { lhs, op, rhs } => {
            lvalue(out, lhs);
            match op {
                AssignOp::Assign => out.push_str(" = "),
                AssignOp::Compound(b) => {
                    let _ = write!(out, " {}= ", b.symbol());
                }
            }
            write_expr(out, rhs);
            out.push_str(";\n");
        }
        StmtKind::NRFunApp(id, args) => {
            out.push_str(&id.name);
            out.push('(');
            list(out, args);
            out.push_str(");\n");
        }
        StmtKind::TargetPlusAssign(e) => {
            out.push_str("target += ");
            write_expr(out, e);
            out.push_str(";\n");
        }
        StmtKind::JacobianPlusAssign(e) => {
            out.push_str("jacobian += ");
            write_expr(out, e);
            out.push_str(";\n");
        }
        StmtKind::Break => out.push_str("break;\n"),
        StmtKind::Continue => out.push_str("continue;\n"),
        StmtKind::Return(None) => out.push_str("return;\n"),
        StmtKind::Return(Some(e)) => {
            out.push_str("return ");
            write_expr(out, e);
            out.push_str(";\n");
        }
        StmtKind::Print(ps) | StmtKind::Reject(ps) | StmtKind::FatalError(ps) => {
            let kw = match &s.kind {
                StmtKind::Print(_) => "print",
                StmtKind::Reject(_) => "reject",
                _ => "fatal_error",
            };
            let _ = write!(out, "{kw}(");
            printables(out, ps);
            out.push_str(");\n");
        }
        StmtKind::Skip => out.push_str(";\n"),
        StmtKind::Tilde {
            arg,
            dist,
            args,
            truncation,
        } => {
            write_expr(out, arg);
            let _ = write!(out, " ~ {}(", dist.name);
            list(out, args);
            out.push(')');
            match truncation {
                Truncation::None => {}
                Truncation::UpFrom(a) => {
                    out.push_str(" T[");
                    write_expr(out, a);
                    out.push_str(", ]");
                }
                Truncation::DownFrom(b) => {
                    out.push_str(" T[, ");
                    write_expr(out, b);
                    out.push(']');
                }
                Truncation::Between(a, b) => {
                    out.push_str(" T[");
                    write_expr(out, a);
                    out.push_str(", ");
                    write_expr(out, b);
                    out.push(']');
                }
            }
            out.push_str(";\n");
        }
        StmtKind::IfThenElse(c, a, b) => {
            out.push_str("if (");
            write_expr(out, c);
            out.push_str(")\n");
            stmt(out, a, depth + 1);
            if let Some(b) = b {
                indent(out, depth);
                out.push_str("else\n");
                stmt(out, b, depth + 1);
            }
        }
        StmtKind::While(c, body) => {
            out.push_str("while (");
            write_expr(out, c);
            out.push_str(")\n");
            stmt(out, body, depth + 1);
        }
        StmtKind::For {
            var,
            lower,
            upper,
            body,
        } => {
            let _ = write!(out, "for ({} in ", var.name);
            write_expr(out, lower);
            out.push(':');
            write_expr(out, upper);
            out.push_str(")\n");
            stmt(out, body, depth + 1);
        }
        StmtKind::ForEach { var, iter, body } => {
            let _ = write!(out, "for ({} in ", var.name);
            write_expr(out, iter);
            out.push_str(")\n");
            stmt(out, body, depth + 1);
        }
        StmtKind::Profile(name, stmts) => {
            let _ = writeln!(out, "profile({name}) {{");
            for s in stmts {
                stmt(out, s, depth + 1);
            }
            indent(out, depth);
            out.push_str("}\n");
        }
        StmtKind::Block(stmts) => {
            out.push_str("{\n");
            for s in stmts {
                stmt(out, s, depth + 1);
            }
            indent(out, depth);
            out.push_str("}\n");
        }
        StmtKind::VarDecl {
            ty, trans, vars, ..
        } => {
            write_decl_type(out, ty, trans);
            out.push(' ');
            for (i, v) in vars.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&v.ident.name);
                if let Some(e) = &v.init {
                    out.push_str(" = ");
                    write_expr(out, e);
                }
            }
            out.push_str(";\n");
        }
        StmtKind::FunDef {
            ret,
            name,
            args,
            body,
        } => {
            match ret {
                Some(t) => out.push_str(&t.to_string()),
                None => out.push_str("void"),
            }
            let _ = write!(out, " {}(", name.name);
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                if a.ad == AdLevel::DataOnly {
                    out.push_str("data ");
                }
                let _ = write!(out, "{} {}", a.ty, a.ident.name);
            }
            out.push(')');
            if matches!(body.kind, StmtKind::Skip) {
                out.push_str(";\n");
            } else {
                out.push('\n');
                stmt(out, body, depth);
            }
        }
    }
}
