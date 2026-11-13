//! Lexer, parser, abstract syntax tree, and pretty printer of the Stan language for stanc-rs.
//!
//! stanc-rs is an independent, unofficial implementation. The grammar follows stanc3 v2.40.0.
//!
//! ```
//! use stanc_rs_syntax::{parse_source, NoIncludes};
//!
//! let (program, _sources) =
//!     parse_source("m.stan", "parameters { real mu; } model { mu ~ normal(0, 1); }", &NoIncludes).unwrap();
//! assert!(program.model.is_some());
//! ```

pub mod ast;
pub mod lexer;
pub mod parser;
pub mod pretty;
pub mod source;

pub use lexer::{IncludeResolver, NoIncludes};
pub use source::{Diagnostic, DiagnosticKind, SourceFile, SourceMap, Span};

/// Lexes and parses a program. On error, the source map is returned with the diagnostic so that
/// it can be rendered.
pub fn parse_source(
    name: &str,
    text: &str,
    resolver: &dyn IncludeResolver,
) -> Result<(ast::Program, SourceMap), (Diagnostic, SourceMap)> {
    parse_with(name, text, resolver, false)
}

/// Like [`parse_source`]; with `functions_only`, parses a `.stanfunctions` file.
pub fn parse_with(
    name: &str,
    text: &str,
    resolver: &dyn IncludeResolver,
    functions_only: bool,
) -> Result<(ast::Program, SourceMap), (Diagnostic, SourceMap)> {
    let mut sources = SourceMap::default();
    let file = sources.add(SourceFile::new(name, text));
    let (tokens, lex_error) = lexer::tokenize_partial(&mut sources, file, resolver);
    let r = if functions_only { parser::parse_functions_only(&tokens) } else { parser::parse_program(&tokens) };
    let eof = tokens.last().map(|t| t.span);
    match (r, lex_error) {
        (Ok(p), None) => Ok((p, sources)),
        (Err(d), None) => Err((d, sources)),
        // A syntax error strictly before the lexical error wins, as in stanc3's lazy lexer.
        (Err(d), Some(_)) if d.span != eof => Err((d, sources)),
        (_, Some(l)) => Err((l, sources)),
    }
}
