//! Model information as JSON, following stanc3's `--info`.

use std::collections::BTreeSet;

use stanc_rs_semantic::typed::{
    FunKind, TExpr, TExprKind, TIndex, TLValue, TLValueKind, TPrintable, TProgram, TSizedType,
    TStmt, TStmtKind, TTransformation, TTruncation,
};
use stanc_rs_syntax::ast::UnsizedType as U;

/// The JSON document: declared variables of the data, parameters, transformed parameters, and
/// generated quantities blocks; the library functions and distributions the program calls; and
/// the included files.
pub fn info_json(p: &TProgram, included_files: &[String]) -> String {
    let mut calls = Calls::default();
    for s in [
        &p.functions,
        &p.data,
        &p.transformed_data,
        &p.parameters,
        &p.transformed_parameters,
        &p.model,
        &p.generated_quantities,
    ]
    .into_iter()
    .flatten()
    {
        calls.stmt(s);
    }
    let list = |xs: &mut dyn Iterator<Item = &String>| {
        let v: Vec<String> = xs.map(|x| string(x)).collect();
        format!("[{}]", v.join(", "))
    };
    let blocks = [
        ("inputs", &p.data),
        ("parameters", &p.parameters),
        ("transformed parameters", &p.transformed_parameters),
        ("generated quantities", &p.generated_quantities),
    ];
    let mut out = String::from("{\n");
    for (name, block) in blocks {
        let vars: Vec<String> = block
            .iter()
            .filter_map(|s| match &s.kind {
                TStmtKind::VarDecl { ty, vars, .. } => Some((ty, vars)),
                _ => None,
            })
            .flat_map(|(ty, vars)| {
                let t = type_json(&ty.unsized_type());
                vars.iter()
                    .map(move |v| format!("{}: {t}", string(&v.name)))
            })
            .collect();
        out += &format!("  {}: {{{}}},\n", string(name), vars.join(", "));
    }
    out += &format!("  \"functions\": {},\n", list(&mut calls.functions.iter()));
    out += &format!(
        "  \"distributions\": {},\n",
        list(&mut calls.distributions.iter())
    );
    out += &format!(
        "  \"included_files\": {}\n}}\n",
        list(&mut included_files.iter())
    );
    out
}

/// `{"type": ..., "dimensions": n}`, with vectors and matrices counted as dimensions of their
/// scalars and tuples listing their elements.
fn type_json(t: &U) -> String {
    let mut dims = 0;
    let mut t = t;
    while let U::Array(inner) = t {
        dims += 1;
        t = inner;
    }
    let (ty, extra) = match t {
        U::Int => ("\"int\"".to_string(), 0),
        U::Real => ("\"real\"".to_string(), 0),
        U::Complex => ("\"complex\"".to_string(), 0),
        U::Vector | U::RowVector => ("\"real\"".to_string(), 1),
        U::ComplexVector | U::ComplexRowVector => ("\"complex\"".to_string(), 1),
        U::Matrix => ("\"real\"".to_string(), 2),
        U::ComplexMatrix => ("\"complex\"".to_string(), 2),
        U::Tuple(ts) => {
            let v: Vec<String> = ts.iter().map(type_json).collect();
            (format!("[{}]", v.join(", ")), 0)
        }
        // Declarations never have these types.
        other => (string(&format!("{other:?}")), 0),
    };
    format!("{{\"type\": {ty}, \"dimensions\": {}}}", dims + extra)
}

/// A JSON string literal.
fn string(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o += "\\\"",
            '\\' => o += "\\\\",
            '\n' => o += "\\n",
            '\r' => o += "\\r",
            '\t' => o += "\\t",
            c if (c as u32) < 0x20 => o += &format!("\\u{:04x}", c as u32),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Library calls, split as stanc3 does: applications written with `|` and the distributions of
/// `~` statements (as `_lupdf` or `_lupmf`) are distributions; everything else is a function.
#[derive(Default)]
struct Calls {
    functions: BTreeSet<String>,
    distributions: BTreeSet<String>,
}

impl Calls {
    fn stmt(&mut self, s: &TStmt) {
        match &s.kind {
            TStmtKind::Assign { lhs, rhs, .. } => {
                self.lvalue(lhs);
                self.expr(rhs);
            }
            TStmtKind::NRFunApp { kind, name, args } => {
                if *kind == FunKind::StanLib {
                    self.functions.insert(name.clone());
                }
                args.iter().for_each(|a| self.expr(a));
            }
            TStmtKind::TargetPlusAssign(e) | TStmtKind::JacobianPlusAssign(e) => self.expr(e),
            TStmtKind::Tilde {
                arg,
                dist,
                dist_kind,
                args,
                truncation,
            } => {
                if *dist_kind == FunKind::StanLib {
                    let d = if let Some(b) = dist.strip_suffix("_lpdf") {
                        format!("{b}_lupdf")
                    } else if let Some(b) = dist.strip_suffix("_lpmf") {
                        format!("{b}_lupmf")
                    } else {
                        dist.clone()
                    };
                    self.distributions.insert(d);
                }
                self.expr(arg);
                args.iter().for_each(|a| self.expr(a));
                match truncation {
                    TTruncation::None => {}
                    TTruncation::UpFrom(e) | TTruncation::DownFrom(e) => self.expr(e),
                    TTruncation::Between(a, b) => {
                        self.expr(a);
                        self.expr(b);
                    }
                }
            }
            TStmtKind::Return(Some(e)) => self.expr(e),
            TStmtKind::Print(ps) | TStmtKind::Reject(ps) | TStmtKind::FatalError(ps) => {
                let f = match &s.kind {
                    TStmtKind::Print(_) => "print",
                    TStmtKind::Reject(_) => "reject",
                    _ => "fatal_error",
                };
                self.functions.insert(f.to_string());
                for p in ps {
                    if let TPrintable::Expr(e) = p {
                        self.expr(e);
                    }
                }
            }
            TStmtKind::IfThenElse(c, a, b) => {
                self.expr(c);
                self.stmt(a);
                if let Some(b) = b {
                    self.stmt(b);
                }
            }
            TStmtKind::While(c, b) => {
                self.expr(c);
                self.stmt(b);
            }
            TStmtKind::For {
                lower, upper, body, ..
            } => {
                self.expr(lower);
                self.expr(upper);
                self.stmt(body);
            }
            TStmtKind::ForEach { iter, body, .. } => {
                self.expr(iter);
                self.stmt(body);
            }
            TStmtKind::Profile(_, b) | TStmtKind::Block(b) => b.iter().for_each(|s| self.stmt(s)),
            TStmtKind::VarDecl {
                ty, trans, vars, ..
            } => {
                self.sized(ty);
                self.transformation(trans);
                vars.iter()
                    .filter_map(|v| v.init.as_ref())
                    .for_each(|e| self.expr(e));
            }
            TStmtKind::FunDef { body: Some(b), .. } => self.stmt(b),
            _ => {}
        }
    }

    fn sized(&mut self, t: &TSizedType) {
        match t {
            TSizedType::Vector(e)
            | TSizedType::RowVector(e)
            | TSizedType::ComplexVector(e)
            | TSizedType::ComplexRowVector(e) => self.expr(e),
            TSizedType::Matrix(a, b) | TSizedType::ComplexMatrix(a, b) => {
                self.expr(a);
                self.expr(b);
            }
            TSizedType::Array(t, e) => {
                self.sized(t);
                self.expr(e);
            }
            TSizedType::Tuple(ts) => ts.iter().for_each(|t| self.sized(t)),
            TSizedType::Int | TSizedType::Real | TSizedType::Complex => {}
        }
    }

    fn transformation(&mut self, t: &TTransformation) {
        match t {
            TTransformation::Lower(e)
            | TTransformation::Upper(e)
            | TTransformation::Offset(e)
            | TTransformation::Multiplier(e) => self.expr(e),
            TTransformation::LowerUpper(a, b) | TTransformation::OffsetMultiplier(a, b) => {
                self.expr(a);
                self.expr(b);
            }
            TTransformation::Tuple(ts) => ts.iter().for_each(|t| self.transformation(t)),
            _ => {}
        }
    }

    fn lvalue(&mut self, l: &TLValue) {
        match &l.kind {
            TLValueKind::Variable { .. } => {}
            TLValueKind::Indexed(i, idx) => {
                self.lvalue(i);
                idx.iter().for_each(|i| self.index(i));
            }
            TLValueKind::TupleProjection(i, _) => self.lvalue(i),
            TLValueKind::Tuple(ls) => ls.iter().for_each(|x| self.lvalue(x)),
        }
    }

    fn index(&mut self, i: &TIndex) {
        match i {
            TIndex::All => {}
            TIndex::Single(e) | TIndex::Upfrom(e) | TIndex::Downfrom(e) => self.expr(e),
            TIndex::Between(a, b) => {
                self.expr(a);
                self.expr(b);
            }
        }
    }

    fn expr(&mut self, e: &TExpr) {
        match &e.kind {
            TExprKind::FunApp {
                kind,
                name,
                args,
                cond,
            } => {
                if *kind == FunKind::StanLib {
                    let set = if *cond {
                        &mut self.distributions
                    } else {
                        &mut self.functions
                    };
                    set.insert(name.clone());
                }
                args.iter().for_each(|a| self.expr(a));
            }
            TExprKind::ArrayExpr(a) | TExprKind::RowVectorExpr(a) | TExprKind::TupleExpr(a) => {
                a.iter().for_each(|x| self.expr(x))
            }
            TExprKind::TernaryIf(a, b, c) => {
                self.expr(a);
                self.expr(b);
                self.expr(c);
            }
            TExprKind::BinOp(a, _, b) => {
                self.expr(a);
                self.expr(b);
            }
            TExprKind::PrefixOp(_, a)
            | TExprKind::Transpose(a)
            | TExprKind::TupleProjection(a, _)
            | TExprKind::Promotion(a, _) => self.expr(a),
            TExprKind::Indexed(a, idx) => {
                self.expr(a);
                idx.iter().for_each(|i| self.index(i));
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn variables_and_calls() {
        let src = "functions { real f(real x) { return x; } }
data { int<lower=0> N; vector[N] y; array[2] complex z; }
parameters { real mu; real<lower=0> sigma; matrix[2, 3] B; }
model { y ~ normal(mu, sigma); target += normal_lpdf(mu | 0, f(1)); }
generated quantities { tuple(real, array[2] int) t = (1.0, {2, 3}); print(sqrt(2)); }";
        let c = crate::check_source("m.stan", src, &crate::Options::default()).unwrap();
        let expected = r#"{
  "inputs": {"N": {"type": "int", "dimensions": 0}, "y": {"type": "real", "dimensions": 1}, "z": {"type": "complex", "dimensions": 1}},
  "parameters": {"mu": {"type": "real", "dimensions": 0}, "sigma": {"type": "real", "dimensions": 0}, "B": {"type": "real", "dimensions": 2}},
  "transformed parameters": {},
  "generated quantities": {"t": {"type": [{"type": "real", "dimensions": 0}, {"type": "int", "dimensions": 1}], "dimensions": 0}},
  "functions": ["print", "sqrt"],
  "distributions": ["normal_lpdf", "normal_lupdf"],
  "included_files": []
}
"#;
        assert_eq!(c.info(), expected);
    }
}
