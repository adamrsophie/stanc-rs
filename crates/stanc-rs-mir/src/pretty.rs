//! The MIR printer of stanc3 (`--debug-mir-pretty`), laid out with the port of OCaml's
//! `Format`.

use stanc_rs_semantic::types::Ad;
use stanc_rs_syntax::format::Formatter;

use crate::mir::*;

/// OCaml's `%S`: a quoted string with OCaml escapes.
fn ocaml_quoted(s: &str) -> String {
    format!("\"{}\"", ocaml_escaped(s))
}

/// OCaml's `String.escaped`.
fn ocaml_escaped(s: &str) -> String {
    let mut o = String::new();
    for &b in s.as_bytes() {
        match b {
            b'"' => o.push_str("\\\""),
            b'\\' => o.push_str("\\\\"),
            b'\n' => o.push_str("\\n"),
            b'\t' => o.push_str("\\t"),
            b'\r' => o.push_str("\\r"),
            8 => o.push_str("\\b"),
            b' '..=b'~' => o.push(b as char),
            _ => o.push_str(&format!("\\{b:03}")),
        }
    }
    o
}

/// An S-expression atom as `Sexp.to_string` prints it: escaped and quoted when needed.
fn sexp_atom(s: &str) -> (String, bool) {
    let must = s.is_empty() || {
        let b = s.as_bytes();
        let mut ix = b.len() - 1;
        loop {
            match b[ix] {
                b'"' | b'(' | b')' | b';' | b'\\' => break true,
                b'|' => {
                    if ix > 0 && b[ix - 1] == b'#' {
                        break true;
                    }
                }
                b'#' => {
                    if ix > 0 && b[ix - 1] == b'|' {
                        break true;
                    }
                }
                0..=32 | 127..=255 => break true,
                _ => {}
            }
            if ix == 0 {
                break false;
            }
            ix -= 1;
        }
    };
    if must {
        (ocaml_quoted(s), false)
    } else {
        (s.to_string(), true)
    }
}

/// A minimal S-expression, printed as `Sexp.to_string` (machine format) does.
enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

impl Sexp {
    fn atom(s: &str) -> Sexp {
        Sexp::Atom(s.to_string())
    }

    fn write(&self, out: &mut String, may_need_space: bool) -> bool {
        match self {
            Sexp::Atom(a) => {
                let (s, plain) = sexp_atom(a);
                if may_need_space && plain {
                    out.push(' ');
                }
                out.push_str(&s);
                plain
            }
            Sexp::List(xs) => {
                out.push('(');
                let mut m = false;
                for x in xs {
                    m = x.write(out, m);
                }
                out.push(')');
                false
            }
        }
    }

    fn to_string_mach(&self) -> String {
        let mut s = String::new();
        self.write(&mut s, false);
        s
    }
}

fn sexp_of_trans(t: &Transformation) -> Sexp {
    use Transformation as T;
    let e = |x: &Expr| Sexp::Atom(expr_string(x));
    let one = |n: &str, x: &Expr| Sexp::List(vec![Sexp::atom(n), e(x)]);
    let two = |n: &str, x: &Expr, y: &Expr| Sexp::List(vec![Sexp::atom(n), e(x), e(y)]);
    match t {
        T::Identity => Sexp::atom("Identity"),
        T::Lower(x) => one("Lower", x),
        T::Upper(x) => one("Upper", x),
        T::LowerUpper(x, y) => two("LowerUpper", x, y),
        T::Offset(x) => one("Offset", x),
        T::Multiplier(x) => one("Multiplier", x),
        T::OffsetMultiplier(x, y) => two("OffsetMultiplier", x, y),
        T::Ordered => Sexp::atom("Ordered"),
        T::PositiveOrdered => Sexp::atom("PositiveOrdered"),
        T::Simplex => Sexp::atom("Simplex"),
        T::UnitVector => Sexp::atom("UnitVector"),
        T::SumToZero => Sexp::atom("SumToZero"),
        T::CholeskyCorr => Sexp::atom("CholeskyCorr"),
        T::CholeskyCov => Sexp::atom("CholeskyCov"),
        T::Correlation => Sexp::atom("Correlation"),
        T::Covariance => Sexp::atom("Covariance"),
        T::StochasticRow => Sexp::atom("StochasticRow"),
        T::StochasticColumn => Sexp::atom("StochasticColumn"),
        T::Tuple(ts) => Sexp::List(vec![
            Sexp::atom("TupleTransformation"),
            Sexp::List(ts.iter().map(sexp_of_trans).collect()),
        ]),
    }
}

/// The name of a compiler internal function (`Internal_fun.to_string`).
pub fn internal_name(f: &Internal) -> String {
    let s = match f {
        Internal::Length => Sexp::atom("FnLength"),
        Internal::MakeArray => Sexp::atom("FnMakeArray"),
        Internal::MakeTuple => Sexp::atom("FnMakeTuple"),
        Internal::MakeRowVec => Sexp::atom("FnMakeRowVec"),
        Internal::NegInf => Sexp::atom("FnNegInf"),
        Internal::ReadData => Sexp::atom("FnReadData"),
        Internal::ReadDeserializer => Sexp::atom("FnReadDeserializer"),
        Internal::ReadParam {
            constrain,
            dims,
            mem_pattern,
        } => Sexp::List(vec![
            Sexp::atom("FnReadParam"),
            Sexp::List(vec![Sexp::atom("constrain"), sexp_of_trans(constrain)]),
            Sexp::List(vec![
                Sexp::atom("dims"),
                Sexp::List(dims.iter().map(|d| Sexp::Atom(expr_string(d))).collect()),
            ]),
            Sexp::List(vec![
                Sexp::atom("mem_pattern"),
                Sexp::atom(match mem_pattern {
                    MemPattern::AoS => "AoS",
                    MemPattern::SoA => "SoA",
                }),
            ]),
        ]),
        Internal::WriteParam {
            unconstrain_opt,
            var,
        } => Sexp::List(vec![
            Sexp::atom("FnWriteParam"),
            Sexp::List(vec![
                Sexp::atom("unconstrain_opt"),
                Sexp::List(unconstrain_opt.iter().map(|t| sexp_of_trans(t)).collect()),
            ]),
            Sexp::List(vec![Sexp::atom("var"), Sexp::Atom(expr_string(var))]),
        ]),
        Internal::ValidateSize => Sexp::atom("FnValidateSize"),
        Internal::ValidateSizePositive => Sexp::atom("FnValidateSizePositive"),
        Internal::ValidateSizeUnitVector => Sexp::atom("FnValidateSizeUnitVector"),
        Internal::Check {
            trans,
            var_name,
            var,
        } => Sexp::List(vec![
            Sexp::atom("FnCheck"),
            Sexp::List(vec![Sexp::atom("trans"), sexp_of_trans(trans)]),
            Sexp::List(vec![Sexp::atom("var_name"), Sexp::atom(var_name)]),
            Sexp::List(vec![Sexp::atom("var"), Sexp::Atom(expr_string(var))]),
        ]),
        Internal::Print => Sexp::atom("FnPrint"),
        Internal::Reject => Sexp::atom("FnReject"),
        Internal::FatalError => Sexp::atom("FnFatalError"),
        Internal::ResizeToMatch => Sexp::atom("FnResizeToMatch"),
        Internal::NaN => Sexp::atom("FnNaN"),
        Internal::DeepCopy => Sexp::atom("FnDeepCopy"),
    };
    format!("{}__", s.to_string_mach())
}

/// The printed name of a called function (`Fun_kind.pp`).
pub fn fun_kind_name(k: &FunKind) -> String {
    match k {
        FunKind::StanLib(s, Suffix::Lpdf(true) | Suffix::Lpmf(true))
        | FunKind::UserDefined(s, Suffix::Lpdf(true) | Suffix::Lpmf(true)) => {
            if let Some(p) = s.strip_suffix("_lpdf") {
                format!("{p}_lupdf")
            } else if let Some(p) = s.strip_suffix("_lpmf") {
                format!("{p}_lupmf")
            } else {
                s.clone()
            }
        }
        FunKind::StanLib(s, _) | FunKind::UserDefined(s, _) => s.clone(),
        FunKind::CompilerInternal(f) => internal_name(f),
    }
}

/// `UnsizedType.pp`.
pub fn unsized_type(f: &mut Formatter, t: &UnsizedType) {
    use UnsizedType::*;
    match t {
        Int => f.string("int"),
        Real => f.string("real"),
        Complex => f.string("complex"),
        Vector => f.string("vector"),
        RowVector => f.string("row_vector"),
        Matrix => f.string("matrix"),
        ComplexVector => f.string("complex_vector"),
        ComplexRowVector => f.string("complex_row_vector"),
        ComplexMatrix => f.string("complex_matrix"),
        Array(_) => {
            let (n, inner) = t.array_dims();
            f.string(&format!("array[{}] ", ",".repeat(n - 1)));
            unsized_type(f, inner);
        }
        Tuple(ts) => {
            f.string("tuple(");
            f.pf("@[");
            for (i, t) in ts.iter().enumerate() {
                if i > 0 {
                    f.pf(",@ ");
                }
                unsized_type(f, t);
            }
            if ts.len() == 1 {
                f.string(",");
            }
            f.pf("@]");
            f.string(")");
        }
        Function(ft) => {
            f.pf("@[<h>(");
            for (i, (ad, t)) in ft.args.iter().enumerate() {
                if i > 0 {
                    f.pf(",@ ");
                }
                if *ad == stanc_rs_syntax::ast::AdLevel::DataOnly
                    && !matches!(t, Function(_) | MathLibraryFunction)
                {
                    f.string("data ");
                }
                unsized_type(f, t);
            }
            f.string(") => ");
            match &ft.ret {
                Some(t) => unsized_type(f, t),
                None => f.string("void"),
            }
            f.pf("@]");
        }
        MathLibraryFunction => f.string("<Stan Math function>"),
    }
}

/// `UnsizedType.pp_tuple_autodifftype`.
fn tuple_ad(f: &mut Formatter, ad: &Ad) {
    match ad {
        Ad::DataOnly => f.string("data"),
        Ad::AutoDiffable => f.string("var"),
        Ad::Tuple(ads) => {
            f.string("tuple_ad");
            f.pf("@[(");
            for (i, a) in ads.iter().enumerate() {
                if i > 0 {
                    f.pf(",@ ");
                }
                tuple_ad(f, a);
            }
            f.pf(")@]");
        }
    }
}

/// `UnsizedType.pp_autodifftype`.
fn autodifftype(f: &mut Formatter, ad: &Ad) {
    match ad {
        Ad::DataOnly => f.string("data "),
        Ad::AutoDiffable => {}
        t => tuple_ad(f, t),
    }
}

fn comma_list<T>(f: &mut Formatter, xs: &[T], mut pp: impl FnMut(&mut Formatter, &T)) {
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            f.pf(",@ ");
        }
        pp(f, x);
    }
}

fn index(f: &mut Formatter, i: &Index<Expr>) {
    match i {
        Index::All => f.string(":"),
        Index::Single(e) | Index::MultiIndex(e) => expr(f, e),
        Index::Upfrom(e) => {
            expr(f, e);
            f.string(":");
        }
        Index::Between(a, b) => {
            expr(f, a);
            f.string(":");
            expr(f, b);
        }
    }
}

fn indices(f: &mut Formatter, idx: &[Index<Expr>]) {
    f.pf("@[");
    if !idx.is_empty() {
        f.pf("@[<1>[");
        comma_list(f, idx, index);
        f.pf("]@]");
    }
    f.pf("@]");
}

/// `Expr.pp`.
pub fn expr(f: &mut Formatter, e: &Expr) {
    match &e.pattern {
        ExprP::Var(v) => f.string(v),
        ExprP::Lit(LitType::Str, s) => f.string(&ocaml_quoted(s)),
        ExprP::Lit(_, s) => f.string(s),
        ExprP::FunApp(FunKind::StanLib(name, Suffix::Plain), args)
            if args.len() == 2 && Operator::of_name(name).is_some() =>
        {
            let op = Operator::of_name(name).expect("operator");
            f.string("(");
            expr(f, &args[0]);
            f.string(" ");
            f.string(op.symbol());
            f.string(" ");
            expr(f, &args[1]);
            f.string(")");
        }
        ExprP::FunApp(k, args) => {
            f.string(&fun_kind_name(k));
            f.pf("(@[<hov>");
            comma_list(f, args, expr);
            f.pf("@])");
        }
        ExprP::TernaryIf(c, t, e) => {
            f.pf("(@[");
            expr(f, c);
            f.pf("@ ?@ ");
            expr(f, t);
            f.pf("@ :@ ");
            expr(f, e);
            f.pf("@])");
        }
        ExprP::Indexed(e, idx) => {
            f.pf("@[");
            expr(f, e);
            indices(f, idx);
            f.pf("@]");
        }
        ExprP::TupleProjection(e, i) => {
            f.pf("@[");
            expr(f, e);
            f.string(&format!(".{i}"));
            f.pf("@]");
        }
        ExprP::EAnd(a, b) => {
            expr(f, a);
            f.string(" && ");
            expr(f, b);
        }
        ExprP::EOr(a, b) => {
            expr(f, a);
            f.string(" || ");
            expr(f, b);
        }
        ExprP::Promotion(e, ut, ad) => {
            f.pf("promote(@[<hov>");
            expr(f, e);
            f.pf(",@ ");
            unsized_type(f, ut);
            f.pf(",@ ");
            tuple_ad(f, ad);
            f.pf("@])");
        }
    }
}

/// An expression as `Fmt.str "%a" Expr.pp` prints it.
pub fn expr_string(e: &Expr) -> String {
    let mut f = Formatter::new();
    expr(&mut f, e);
    f.finish()
}

/// `SizedType.pp`.
pub fn sized_type(f: &mut Formatter, st: &SizedType) {
    let br = |f: &mut Formatter, e: &Expr| {
        f.pf("@[<1>[");
        expr(f, e);
        f.pf("]@]");
    };
    let br2 = |f: &mut Formatter, a: &Expr, b: &Expr| {
        f.pf("@[<1>[");
        expr(f, a);
        f.pf(",@ ");
        expr(f, b);
        f.pf("]@]");
    };
    match st {
        SizedType::Int => f.string("int"),
        SizedType::Real => f.string("real"),
        SizedType::Complex => f.string("complex"),
        SizedType::Vector(e) => {
            f.string("vector");
            br(f, e);
        }
        SizedType::RowVector(e) => {
            f.string("row_vector");
            br(f, e);
        }
        SizedType::ComplexVector(e) => {
            f.string("complex_vector");
            br(f, e);
        }
        SizedType::ComplexRowVector(e) => {
            f.string("complex_row_vector");
            br(f, e);
        }
        SizedType::Matrix(a, b) => {
            f.string("matrix");
            br2(f, a, b);
        }
        SizedType::ComplexMatrix(a, b) => {
            f.string("complex_matrix");
            br2(f, a, b);
        }
        SizedType::Array(t, e) => {
            f.string("array");
            f.pf("@[<1>[");
            sized_type(f, t);
            f.pf(",@ ");
            expr(f, e);
            f.pf("]@]");
        }
        SizedType::Tuple(ts) => {
            f.pf("tuple(@[");
            comma_list(f, ts, sized_type);
            f.pf("@])");
        }
    }
}

fn decl_type(f: &mut Formatter, t: &DeclType) {
    match t {
        DeclType::Sized(st) => sized_type(f, st),
        DeclType::Unsized(ut) => unsized_type(f, ut),
    }
}

fn lvalue(f: &mut Formatter, lv: &LValue) {
    match &lv.base {
        LBase::Variable(v) => f.string(v),
        LBase::TupleProjection(inner, i) => {
            lvalue(f, inner);
            f.string(&format!(".{i}"));
        }
    }
    indices(f, &lv.idx);
}

fn stmt_list(f: &mut Formatter, ss: &[Stmt]) {
    for (i, s) in ss.iter().enumerate() {
        if i > 0 {
            f.pf("@,");
        }
        stmt(f, s);
    }
}

/// `Stmt.pp`.
pub fn stmt(f: &mut Formatter, s: &Stmt) {
    match &s.pattern {
        StmtP::Assignment(lv, _, rhs) => {
            f.pf("@[<hov>");
            lvalue(f, lv);
            f.pf(" =@[<h>@ ");
            expr(f, rhs);
            f.pf("@];@]");
        }
        StmtP::TargetPE(e) => {
            f.pf("@[<h>target +=@ ");
            expr(f, e);
            f.pf(";@]");
        }
        StmtP::JacobianPE(e) => {
            f.pf("@[<h>jacobian +=@ ");
            expr(f, e);
            f.pf(";@]");
        }
        StmtP::NRFunApp(k, args) => {
            f.pf("@[");
            f.string(&fun_kind_name(k));
            f.pf("@[<1>(");
            comma_list(f, args, expr);
            f.pf(")@]");
            f.pf(";@]");
        }
        StmtP::Break => f.string("break;"),
        StmtP::Continue => f.string("continue;"),
        StmtP::Skip => f.string(";"),
        StmtP::Return(Some(e)) => {
            f.string("return ");
            expr(f, e);
            f.string(";");
        }
        StmtP::Return(None) => f.string("return;"),
        StmtP::IfElse(c, t, e) => {
            f.string("if(");
            expr(f, c);
            f.string(") ");
            stmt(f, t);
            if let Some(e) = e {
                f.string(" else ");
                stmt(f, e);
            }
        }
        StmtP::While(c, b) => {
            f.string("while(");
            expr(f, c);
            f.string(") ");
            stmt(f, b);
        }
        StmtP::For {
            loopvar,
            lower,
            upper,
            body,
        } => {
            f.string(&format!("for({loopvar} in "));
            expr(f, lower);
            f.string(":");
            expr(f, upper);
            f.string(") ");
            stmt(f, body);
        }
        StmtP::Profile(name, ss) => {
            f.string(&format!("profile({name}){{"));
            f.pf("@;<1 2>@[<v>");
            stmt_list(f, ss);
            f.pf("@]@;}");
        }
        StmtP::Block(ss) => {
            f.pf("{@;<1 2>@[<v>");
            stmt_list(f, ss);
            f.pf("@]@;}");
        }
        StmtP::SList(ss) => {
            f.pf("@[<v>");
            stmt_list(f, ss);
            f.pf("@]");
        }
        StmtP::Decl { ad, id, ty, init } => {
            f.pf("@[<hov 2>");
            autodifftype(f, ad);
            decl_type(f, ty);
            f.pf("@ ");
            f.string(id);
            if let DeclInit::Assign(e) = init {
                f.string(" = ");
                expr(f, e);
            }
            f.pf(";@]");
        }
    }
}

fn block<T>(f: &mut Formatter, label: &str, xs: &[T], mut pp: impl FnMut(&mut Formatter, &T)) {
    if xs.is_empty() {
        return;
    }
    f.pf("@[<v2>");
    f.string(&format!("{label} {{"));
    f.pf("@ ");
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            f.pf("@,");
        }
        pp(f, x);
    }
    f.pf("@]@ }@\n");
}

fn fun_def(f: &mut Formatter, d: &FunDef) {
    let args = |f: &mut Formatter| {
        f.pf("@[<1>(");
        comma_list(f, &d.args, |f, (ad, name, ut)| {
            autodifftype(f, ad);
            unsized_type(f, ut);
            f.string(&format!(" {name}"));
        });
        f.pf(")@]");
    };
    let rt = |f: &mut Formatter| match &d.rt {
        Some(t) => unsized_type(f, t),
        None => f.string("void"),
    };
    match &d.body {
        None => {
            f.pf("@[<v2>extern ");
            rt(f);
            f.string(&format!(" {}", d.name));
            args(f);
            f.pf(";@]");
        }
        Some(b) => {
            f.pf("@[<v2>");
            rt(f);
            f.string(&format!(" {}", d.name));
            args(f);
            f.pf(" {@ ");
            stmt(f, b);
            f.pf("@]@ }");
        }
    }
}

/// `Program.pp`: the program as `--debug-mir-pretty` prints it.
pub fn program(p: &Program) -> String {
    // stanc3 opens its vertical box on the standard formatter rather than on the one it prints
    // to, so the blocks sit in the top-level box.
    let mut f = Formatter::new();
    block(&mut f, "functions", &p.functions, fun_def);
    f.pf("@,");
    block(&mut f, "input_vars", &p.input_vars, |f, (name, _, st)| {
        f.pf("@[<h>");
        sized_type(f, st);
        f.string(&format!(" {name};"));
        f.pf("@]");
    });
    f.pf("@,");
    block(&mut f, "prepare_data", &p.prepare_data, stmt);
    f.pf("@,");
    block(&mut f, "log_prob", &p.log_prob, stmt);
    f.pf("@,");
    block(&mut f, "rev_log_prob", &p.reverse_mode_log_prob, stmt);
    f.pf("@,");
    block(&mut f, "generate_quantities", &p.generate_quantities, stmt);
    f.pf("@,");
    block(&mut f, "transform_inits", &p.transform_inits, stmt);
    f.pf("@,");
    block(&mut f, "output_vars", &p.output_vars, |f, (name, _, ov)| {
        f.pf("@[<hov 2>");
        f.string(match ov.block {
            IoBlock::Parameters => "parameters",
            IoBlock::TransformedParameters => "transformed_parameters",
            IoBlock::GeneratedQuantities => "generated_quantities",
        });
        f.string(" ");
        sized_type(f, &ov.constrained);
        f.string(&format!(" {name};"));
        f.pf("@ //");
        sized_type(f, &ov.unconstrained);
        f.pf("@]");
    });
    f.finish()
}
