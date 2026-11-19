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
pub mod laplace;
pub mod library;
pub mod program;
pub mod solvers;
pub mod stmt;

use std::collections::{HashMap, HashSet};

use stanc_rs_semantic::typed::{TFunArg, TProgram, TSizedType};
use stanc_rs_semantic::types::Ad;
use stanc_rs_syntax::ast::UnsizedType as U;
use stanc_rs_syntax::{Diagnostic, DiagnosticKind, SourceMap, Span};

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
    /// Set while lowering an operator written as a function call, so that the operator's own
    /// fallback to a library call does not loop.
    pub(crate) in_operator: Option<Span>,
    /// The Stan variable being assigned, for size error messages.
    pub(crate) assign_name: String,
    /// The program's sources, to print statement locations.
    sources: SourceMap,
    /// The locations of the statements, as the generated `LOCATIONS__` table lists them.
    locs: Vec<String>,
    loc_ids: HashMap<String, usize>,
    /// Set inside a function body that records its current statement in `loc__`.
    pub(crate) track_loc: bool,
}

impl Gen {
    fn new(sources: &SourceMap) -> Gen {
        Gen {
            sources: sources.clone(),
            locs: Vec::new(),
            loc_ids: HashMap::new(),
            track_loc: false,
            out: String::new(),
            indent: 0,
            fields: HashSet::new(),
            tmp: 0,
            udfs: HashMap::new(),
            in_operator: None,
            ret: None,
            assign_name: String::new(),
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

/// Generates the Rust module for a checked program. `sources` resolves the spans of the
/// program's statements, which the generated code attaches to runtime errors.
pub fn generate(p: &TProgram, sources: &SourceMap) -> Result<String, Diagnostic> {
    let mut g = Gen::new(sources);
    g.program(p)?;
    g.line("");
    g.line("/// The source locations of the statements, for runtime errors.");
    g.line("const LOCATIONS__: &[&str] = &[");
    for l in std::mem::take(&mut g.locs) {
        g.line(&format!("    {l:?},"));
    }
    g.line("];");
    Ok(g.out)
}

impl Gen {
    /// Records that the statement at `span` starts, when inside a tracked body.
    pub(crate) fn mark_statement(&mut self, span: Span) {
        if !self.track_loc || self.sources.files.get(span.file as usize).is_none() {
            return;
        }
        let loc = self.sources.pp_span(span);
        let next = self.locs.len();
        let k = *self.loc_ids.entry(loc.clone()).or_insert(next);
        if k == next {
            self.locs.push(loc);
        }
        self.line(&format!("loc__.set({k});"));
    }

    /// Opens a function body whose errors get the location of the statement that was running:
    /// the body runs in an immediately called closure, and statements record themselves in
    /// `loc__`. Returns the previous tracking state for [`Gen::end_located`].
    pub(crate) fn begin_located(&mut self, ret: &str) -> bool {
        self.line("let loc__ = std::cell::Cell::new(usize::MAX);");
        self.line(&format!("let r__ = (|| -> rt::Result<{ret}> {{"));
        self.indent += 1;
        std::mem::replace(&mut self.track_loc, true)
    }

    /// Closes a body opened by [`Gen::begin_located`]; the result is the function's value.
    pub(crate) fn end_located(&mut self, prev: bool) {
        self.indent -= 1;
        self.line("})();");
        self.line("r__.map_err(|e| rt::locate(e, loc__.get(), LOCATIONS__))");
        self.track_loc = prev;
    }
}
