//! Rust code generation for typed Stan programs.
//!
//! [`generate`] turns a checked program into one self-contained Rust module that depends on
//! `stanc-rs-runtime` (and through it on stanmath-rs). The module defines `pub struct Model`, a
//! constructor `Model::new(&Data, seed)`, and an implementation of
//! `stanc_rs_runtime::GeneratedModel`, whose log density is generic in the scalar type so that the
//! same code evaluates values and reverse-mode gradients.
//!
//! Conventions of the generated code:
//!
//! * Stan `int` is `i64`; data reals are `f64`; values that may depend on parameters have the
//!   scalar type `T` (`f64` outside the log density). Vectors and row vectors are `Vec`, matrices
//!   are stanmath-rs `Matrix` (column-major), arrays are nested `Vec`, tuples are Rust tuples.
//! * Every fallible operation returns `rt::Result` and is propagated with `?`; nothing panics on
//!   bad input. Indexing goes through `rt::index`, which checks Stan's one-based bounds.
//! * Internal names end in `__`, which Stan identifiers cannot.

#![allow(missing_docs)]

pub mod expr;
pub mod library;
pub mod program;
pub mod stmt;

use std::collections::{HashMap, HashSet};

use stanc_rs_semantic::typed::{TFunArg, TProgram, TSizedType};
use stanc_rs_semantic::types::Ad;
use stanc_rs_syntax::ast::UnsizedType as U;
use stanc_rs_syntax::{Diagnostic, DiagnosticKind, Span};

pub(crate) type R<T> = Result<T, Diagnostic>;

pub(crate) fn unsupported(span: Span, msg: impl Into<String>) -> Diagnostic {
    Diagnostic::new(DiagnosticKind::Unsupported, span, msg.into())
}

const RUST_KEYWORDS: &[&str] = &[
    "as", "async", "await", "box", "break", "const", "continue", "dyn", "else", "enum", "extern",
    "false", "fn", "for", "gen", "if", "impl", "in", "let", "loop", "macro", "match", "mod",
    "move", "mut", "override", "priv", "pub", "ref", "return", "static", "struct", "trait", "true",
    "try", "type", "typeof", "unsafe", "unsized", "use", "virtual", "where", "while", "yield",
    "abstract", "become", "do", "final",
];

/// The Rust identifier for a Stan identifier.
pub fn mangle(name: &str) -> String {
    match name {
        "self" | "Self" | "super" | "crate" | "T" | "Ok" | "Err" | "Some" | "None" | "sm"
        | "rt" | "_" => {
            format!("{name}__")
        }
        n if RUST_KEYWORDS.contains(&n) => format!("r#{n}"),
        n => n.to_string(),
    }
}

/// A user-defined function as generated.
#[derive(Clone, Debug)]
pub struct UdfDef {
    pub rust_name: String,
    pub args: Vec<TFunArg>,
    pub ret: Option<U>,
}

/// Code generation state.
pub struct Gen {
    out: String,
    pub(crate) indent: usize,
    /// Names accessed as fields of `self`.
    pub(crate) fields: HashSet<String>,
    pub(crate) tmp: usize,
    pub(crate) udfs: HashMap<String, Vec<UdfDef>>,
    /// Return type of the function being generated.
    pub(crate) ret: Option<U>,
}

impl Gen {
    fn new() -> Gen {
        Gen {
            out: String::new(),
            indent: 0,
            fields: HashSet::new(),
            tmp: 0,
            udfs: HashMap::new(),
            ret: None,
        }
    }

    /// Appends one line at the current indentation.
    pub fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    pub(crate) fn internal(&self, msg: &str) -> Diagnostic {
        Diagnostic::new(
            DiagnosticKind::Internal,
            Span::default(),
            format!("code generation: {msg}"),
        )
    }

    /// `usize` from a Stan size expression.
    pub(crate) fn size(&mut self, e: &stanc_rs_semantic::typed::TExpr) -> R<String> {
        Ok(format!("rt::stan::usize_of({})?", self.expr(e)?))
    }

    /// The value a declared variable starts with: NaN for reals, the smallest integer for ints.
    pub(crate) fn default_value(&mut self, ty: &TSizedType, ad: &Ad) -> R<String> {
        let nan = if ad.has_autodiff() {
            "T::from_f64(f64::NAN)"
        } else {
            "f64::NAN"
        };
        let cnan = format!("sm::Complex::new({nan}, {nan})");
        Ok(match ty {
            TSizedType::Int => "i64::MIN".into(),
            TSizedType::Real => nan.into(),
            TSizedType::Complex => cnan,
            TSizedType::Vector(n) | TSizedType::RowVector(n) => {
                format!("vec![{nan}; {}]", self.size(n)?)
            }
            TSizedType::ComplexVector(n) | TSizedType::ComplexRowVector(n) => {
                format!("vec![{cnan}; {}]", self.size(n)?)
            }
            TSizedType::Matrix(r, c) => format!(
                "rt::stan::mat_filled({}, {}, {nan})",
                self.size(r)?,
                self.size(c)?
            ),
            TSizedType::ComplexMatrix(r, c) => {
                format!(
                    "rt::stan::mat_filled({}, {}, {cnan})",
                    self.size(r)?,
                    self.size(c)?
                )
            }
            TSizedType::Array(t, n) => {
                let n = self.size(n)?;
                let inner = self.default_value(t, ad)?;
                format!("vec![{inner}; {n}]")
            }
            TSizedType::Tuple(ts) => {
                let parts = ts
                    .iter()
                    .enumerate()
                    .map(|(i, t)| self.default_value(t, &expr::slot_ad(ad, i)))
                    .collect::<R<Vec<_>>>()?;
                format!("({},)", parts.join(", "))
            }
        })
    }
}

/// Generates the Rust module for a checked program.
pub fn generate(p: &TProgram) -> Result<String, Diagnostic> {
    let mut g = Gen::new();
    g.program(p)?;
    Ok(g.out)
}
