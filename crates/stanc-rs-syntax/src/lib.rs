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
pub mod autoformat;
pub mod canonicalize;
pub mod format;
pub mod lexer;
mod lr;
pub mod parser;
pub mod pretty;
pub mod source;

pub use lexer::{IncludeResolver, NoIncludes};
pub use source::{Comment, CommentKind, Diagnostic, DiagnosticKind, SourceFile, SourceMap, Span};

/// The errors stanc3 raises in the actions of its grammar, before its parser would fail.
const STANC3_ACTION_ERRORS: &[&str] = &[
    "Ill-formed identifier.",
    "Ill-formed index.",
    "Ill-formed assignment.",
    "Ill-formed declaration. \";\" expected",
];

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
    let r = if functions_only {
        parser::parse_functions_only(&tokens)
    } else {
        parser::parse_program(&tokens)
    };
    let eof = tokens.last().map(|t| t.span);
    // Word syntax errors as stanc3 does, from the state of its parser at the same token.
    let r = r.map_err(|mut d| {
        if d.kind == DiagnosticKind::Syntax
            && let Some((i, msg)) = lr::syntax_error_message(&tokens, functions_only)
        {
            // Where stanc3's parser stops later, its location wins; at the same token, only
            // errors from the grammar's actions keep their own wording.
            if tokens.get(i).map(|t| t.span) == d.span {
                d.message = msg;
            } else if d.span.is_some_and(|s| tokens[i].span.start > s.start)
                && !STANC3_ACTION_ERRORS
                    .iter()
                    .any(|p| d.message.starts_with(p))
            {
                d.message = msg;
                d.span = Some(tokens[i].span);
            }
        }
        d
    });
    match (r, lex_error) {
        (Ok(p), None) => Ok((p, sources)),
        (Err(d), None) => Err((d, sources)),
        // A syntax error strictly before the lexical error wins, as in stanc3's lazy lexer.
        (Err(d), Some(_)) if d.span != eof => Err((d, sources)),
        (_, Some(l)) => Err((l, sources)),
    }
}
