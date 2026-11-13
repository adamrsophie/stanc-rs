//! Tokenizer for Stan source, following the lexer of stanc3 v2.40.0.
//!
//! The lexer is handwritten. Stan's lexical grammar is small, and a handwritten scanner gives
//! exact control over the few context-sensitive spots: the two-word block keywords
//! (`transformed data`), `.1` as either a real literal or a tuple projection, and `#include`.

use crate::source::{Diagnostic, DiagnosticKind, SourceFile, SourceMap, Span};

/// Token kinds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tok {
    // Block keywords.
    Functions,
    Data,
    TransformedData,
    Parameters,
    TransformedParameters,
    Model,
    GeneratedQuantities,
    // Punctuation.
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBrack,
    RBrack,
    LAngle,
    RAngle,
    Comma,
    Semi,
    Bar,
    // Control flow.
    Return,
    If,
    Else,
    While,
    Profile,
    For,
    In,
    Break,
    Continue,
    // Types.
    Void,
    Int,
    Real,
    Complex,
    Vector,
    RowVector,
    ComplexVector,
    ComplexRowVector,
    Tuple,
    Array,
    Matrix,
    ComplexMatrix,
    Ordered,
    PositiveOrdered,
    Simplex,
    UnitVector,
    SumToZeroVector,
    SumToZeroMatrix,
    CholeskyFactorCorr,
    CholeskyFactorCov,
    CorrMatrix,
    CovMatrix,
    ColumnStochasticMatrix,
    RowStochasticMatrix,
    // Transformation keywords.
    Lower,
    Upper,
    Offset,
    Multiplier,
    Jacobian,
    // Operators.
    QMark,
    Colon,
    Bang,
    Minus,
    Plus,
    Hat,
    EltPow,
    Transpose,
    Times,
    Divide,
    Modulo,
    IDivide,
    LDivide,
    EltTimes,
    EltDivide,
    Or,
    And,
    Equals,
    NEquals,
    Leq,
    Geq,
    Tilde,
    // Assignment.
    Assign,
    PlusAssign,
    MinusAssign,
    TimesAssign,
    DivideAssign,
    EltTimesAssign,
    EltDivideAssign,
    // Effects.
    Print,
    Reject,
    FatalError,
    /// `T`, the truncation marker; also usable as an identifier.
    Truncate,
    Target,
    // Literals and names.
    IntLit(String),
    RealLit(String),
    /// `.5`: a real literal or a tuple projection, depending on context.
    DotNumeral(String),
    /// Imaginary literal without its trailing `i`.
    ImagLit(String),
    /// String literal including its quotes.
    StringLit(String),
    Ident(String),
    Eof,
}

impl Tok {
    /// How the token is written in source, for error messages.
    pub fn describe(&self) -> String {
        match self {
            Tok::IntLit(s) | Tok::RealLit(s) | Tok::DotNumeral(s) | Tok::StringLit(s) => s.clone(),
            Tok::ImagLit(s) => format!("{s}i"),
            Tok::Ident(s) => s.clone(),
            Tok::Eof => "end of file".into(),
            other => keyword_text(other).unwrap_or("?").to_string(),
        }
    }

    /// True for keywords that name a type (used for better error messages).
    pub fn is_type_keyword(&self) -> bool {
        matches!(
            self,
            Tok::Int
                | Tok::Real
                | Tok::Complex
                | Tok::Vector
                | Tok::RowVector
                | Tok::ComplexVector
                | Tok::ComplexRowVector
                | Tok::Tuple
                | Tok::Array
                | Tok::Matrix
                | Tok::ComplexMatrix
                | Tok::Ordered
                | Tok::PositiveOrdered
                | Tok::Simplex
                | Tok::UnitVector
                | Tok::SumToZeroVector
                | Tok::SumToZeroMatrix
                | Tok::CholeskyFactorCorr
                | Tok::CholeskyFactorCov
                | Tok::CorrMatrix
                | Tok::CovMatrix
                | Tok::ColumnStochasticMatrix
                | Tok::RowStochasticMatrix
        )
    }

    /// True for reserved words (keywords that cannot be identifiers).
    pub fn reserved_name(&self) -> Option<&'static str> {
        match self {
            Tok::Truncate | Tok::Ident(_) => None,
            Tok::TransformedData | Tok::TransformedParameters | Tok::GeneratedQuantities => None,
            other => keyword_text(other)
                .filter(|s| s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')),
        }
    }
}

const KEYWORDS: &[(&str, Tok)] = &[
    ("functions", Tok::Functions),
    ("data", Tok::Data),
    ("parameters", Tok::Parameters),
    ("model", Tok::Model),
    ("return", Tok::Return),
    ("if", Tok::If),
    ("else", Tok::Else),
    ("while", Tok::While),
    ("profile", Tok::Profile),
    ("for", Tok::For),
    ("in", Tok::In),
    ("break", Tok::Break),
    ("continue", Tok::Continue),
    ("void", Tok::Void),
    ("int", Tok::Int),
    ("real", Tok::Real),
    ("complex", Tok::Complex),
    ("vector", Tok::Vector),
    ("row_vector", Tok::RowVector),
    ("complex_vector", Tok::ComplexVector),
    ("complex_row_vector", Tok::ComplexRowVector),
    ("tuple", Tok::Tuple),
    ("array", Tok::Array),
    ("matrix", Tok::Matrix),
    ("complex_matrix", Tok::ComplexMatrix),
    ("ordered", Tok::Ordered),
    ("positive_ordered", Tok::PositiveOrdered),
    ("simplex", Tok::Simplex),
    ("unit_vector", Tok::UnitVector),
    ("sum_to_zero_vector", Tok::SumToZeroVector),
    ("sum_to_zero_matrix", Tok::SumToZeroMatrix),
    ("cholesky_factor_corr", Tok::CholeskyFactorCorr),
    ("cholesky_factor_cov", Tok::CholeskyFactorCov),
    ("corr_matrix", Tok::CorrMatrix),
    ("cov_matrix", Tok::CovMatrix),
    ("column_stochastic_matrix", Tok::ColumnStochasticMatrix),
    ("row_stochastic_matrix", Tok::RowStochasticMatrix),
    ("lower", Tok::Lower),
    ("upper", Tok::Upper),
    ("offset", Tok::Offset),
    ("multiplier", Tok::Multiplier),
    ("jacobian", Tok::Jacobian),
    ("print", Tok::Print),
    ("reject", Tok::Reject),
    ("fatal_error", Tok::FatalError),
    ("T", Tok::Truncate),
    ("target", Tok::Target),
];

const SYMBOLS: &[(&str, Tok)] = &[
    // Longest first.
    ("%/%", Tok::IDivide),
    (".*=", Tok::EltTimesAssign),
    ("./=", Tok::EltDivideAssign),
    (".*", Tok::EltTimes),
    (".^", Tok::EltPow),
    ("./", Tok::EltDivide),
    ("||", Tok::Or),
    ("&&", Tok::And),
    ("==", Tok::Equals),
    ("!=", Tok::NEquals),
    ("<=", Tok::Leq),
    (">=", Tok::Geq),
    ("+=", Tok::PlusAssign),
    ("-=", Tok::MinusAssign),
    ("*=", Tok::TimesAssign),
    ("/=", Tok::DivideAssign),
    ("{", Tok::LBrace),
    ("}", Tok::RBrace),
    ("(", Tok::LParen),
    (")", Tok::RParen),
    ("[", Tok::LBrack),
    ("]", Tok::RBrack),
    ("<", Tok::LAngle),
    (">", Tok::RAngle),
    (",", Tok::Comma),
    (";", Tok::Semi),
    ("|", Tok::Bar),
    ("?", Tok::QMark),
    (":", Tok::Colon),
    ("!", Tok::Bang),
    ("-", Tok::Minus),
    ("+", Tok::Plus),
    ("^", Tok::Hat),
    ("'", Tok::Transpose),
    ("*", Tok::Times),
    ("/", Tok::Divide),
    ("%", Tok::Modulo),
    ("\\", Tok::LDivide),
    ("~", Tok::Tilde),
    ("=", Tok::Assign),
];

fn keyword_text(t: &Tok) -> Option<&'static str> {
    match t {
        Tok::TransformedData => return Some("transformed data"),
        Tok::TransformedParameters => return Some("transformed parameters"),
        Tok::GeneratedQuantities => return Some("generated quantities"),
        _ => {}
    }
    KEYWORDS
        .iter()
        .chain(SYMBOLS)
        .find(|(_, k)| k == t)
        .map(|(s, _)| *s)
}

/// A token with its location.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    /// Kind and payload.
    pub tok: Tok,
    /// Location.
    pub span: Span,
}

/// Finds the files named by `#include` directives.
pub trait IncludeResolver {
    /// Returns the display name and contents of the file named `name`, or `None` if not found.
    fn resolve(&self, name: &str) -> Option<(String, String)>;
}

/// A resolver that finds nothing.
pub struct NoIncludes;

impl IncludeResolver for NoIncludes {
    fn resolve(&self, _name: &str) -> Option<(String, String)> {
        None
    }
}

/// Maximum nesting depth of `#include`.
const MAX_INCLUDE_DEPTH: usize = 64;

/// Tokenizes file `file` of `sources`, expanding includes. The result ends with [`Tok::Eof`].
pub fn tokenize(
    sources: &mut SourceMap,
    file: u32,
    resolver: &dyn IncludeResolver,
) -> Result<Vec<Token>, Diagnostic> {
    match tokenize_partial(sources, file, resolver) {
        (toks, None) => Ok(toks),
        (_, Some(e)) => Err(e),
    }
}

/// Like [`tokenize`], but on error also returns the tokens before the error, terminated by an
/// [`Tok::Eof`] placed at the error. stanc3 lexes lazily, so a syntax error before a lexical error
/// is reported first; the parser driver uses this to do the same.
pub fn tokenize_partial(
    sources: &mut SourceMap,
    file: u32,
    resolver: &dyn IncludeResolver,
) -> (Vec<Token>, Option<Diagnostic>) {
    let mut out = Vec::new();
    let mut stack = vec![sources.files[file as usize].name.clone()];
    match lex_file(sources, file, resolver, &mut stack, &mut out) {
        Ok(()) => {
            let end = sources.files[file as usize].text.len();
            out.push(Token {
                tok: Tok::Eof,
                span: Span::new(file, end, end),
            });
            (out, None)
        }
        Err(e) => {
            let at = e.span.unwrap_or_default();
            out.push(Token {
                tok: Tok::Eof,
                span: Span {
                    end: at.start,
                    ..at
                },
            });
            (out, Some(e))
        }
    }
}

fn lex_file(
    sources: &mut SourceMap,
    file: u32,
    resolver: &dyn IncludeResolver,
    stack: &mut Vec<String>,
    out: &mut Vec<Token>,
) -> Result<(), Diagnostic> {
    let text = sources.files[file as usize].text.clone();
    let b = text.as_bytes();
    let mut i = 0;
    let err =
        |kind, s: usize, e: usize, msg: String| Diagnostic::new(kind, Span::new(file, s, e), msg);
    while i < b.len() {
        let c = b[i];
        // Whitespace.
        if matches!(c, b' ' | b'\t' | b'\x0c' | b'\r' | b'\n') {
            i += 1;
            continue;
        }
        // Comments.
        if text[i..].starts_with("/*") {
            match text[i + 2..].find("*/") {
                Some(k) => i = i + 2 + k + 2,
                None => {
                    // Reported at the end of the file, where stanc3 reports it.
                    return Err(err(
                        DiagnosticKind::Lexical,
                        b.len(),
                        b.len(),
                        "Comment opened by \"/*\" is never closed by \"*/\".".into(),
                    ));
                }
            }
            continue;
        }
        if text[i..].starts_with("//") {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Include directive.
        if text[i..].starts_with("#include") {
            let start = i;
            let mut j = i + "#include".len();
            let ws_start = j;
            while j < b.len() && matches!(b[j], b' ' | b'\t' | b'\x0c' | b'\r' | b'\n') {
                j += 1;
            }
            if j == ws_start {
                return Err(err(
                    DiagnosticKind::Lexical,
                    start,
                    j,
                    "Invalid character found: '#'.".into(),
                ));
            }
            let (name, end) = if j < b.len() && (b[j] == b'"' || b[j] == b'<') {
                let close = if b[j] == b'"' { b'"' } else { b'>' };
                let mut k = j + 1;
                while k < b.len() && b[k] != close && b[k] != b'\n' && b[k] != b'\r' {
                    k += 1;
                }
                if k >= b.len() || b[k] != close {
                    return Err(err(
                        DiagnosticKind::Lexical,
                        start,
                        k,
                        "Unterminated #include file name.".into(),
                    ));
                }
                (text[j + 1..k].to_string(), k + 1)
            } else {
                let mut k = j;
                while k < b.len() && !matches!(b[k], b' ' | b'\t' | b'\x0c' | b'\r' | b'\n') {
                    k += 1;
                }
                (text[j..k].to_string(), k)
            };
            if stack.contains(&name) {
                return Err(err(
                    DiagnosticKind::Include,
                    start,
                    end,
                    format!("File {name} recursively included itself."),
                ));
            }
            if stack.len() > MAX_INCLUDE_DEPTH {
                return Err(err(
                    DiagnosticKind::Include,
                    start,
                    end,
                    "Includes are nested too deeply.".into(),
                ));
            }
            let Some((display, contents)) = resolver.resolve(&name) else {
                return Err(err(
                    DiagnosticKind::Include,
                    start,
                    end,
                    format!("Could not find include file '{name}' in specified include paths."),
                ));
            };
            let id = sources.add(SourceFile::new(display, contents));
            stack.push(name);
            lex_file(sources, id, resolver, stack, out)?;
            stack.pop();
            i = end;
            continue;
        }
        let start = i;
        // Identifiers and keywords.
        if c.is_ascii_alphabetic() {
            let mut j = i + 1;
            while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                j += 1;
            }
            let word = &text[i..j];
            // Two-word block names: "transformed" spaces "data"/"parameters", "generated" spaces "quantities".
            let two_word = |first: &str, second: &str| -> Option<usize> {
                if word != first {
                    return None;
                }
                let mut k = j;
                while k < b.len() && matches!(b[k], b' ' | b'\t' | b'\x0c') {
                    k += 1;
                }
                if k == j || !text[k..].starts_with(second) {
                    return None;
                }
                let e = k + second.len();
                if e < b.len() && (b[e].is_ascii_alphanumeric() || b[e] == b'_') {
                    return None;
                }
                Some(e)
            };
            let block = [
                ("transformed", "data", Tok::TransformedData),
                ("transformed", "parameters", Tok::TransformedParameters),
                ("generated", "quantities", Tok::GeneratedQuantities),
            ]
            .into_iter()
            .find_map(|(a, s, t)| two_word(a, s).map(|e| (e, t)));
            if let Some((e, t)) = block {
                out.push(Token {
                    tok: t,
                    span: Span::new(file, i, e),
                });
                i = e;
                continue;
            }
            let tok = KEYWORDS
                .iter()
                .find(|(k, _)| *k == word)
                .map(|(_, t)| t.clone())
                .unwrap_or_else(|| Tok::Ident(word.to_string()));
            out.push(Token {
                tok,
                span: Span::new(file, i, j),
            });
            i = j;
            continue;
        }
        // Numbers.
        if c.is_ascii_digit() || (c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            let (tok, j) =
                lex_number(&text, i).map_err(|(e, msg)| err(DiagnosticKind::Lexical, i, e, msg))?;
            out.push(Token {
                tok,
                span: Span::new(file, i, j),
            });
            i = j;
            continue;
        }
        // Strings.
        if c == b'"' {
            let mut j = i + 1;
            while j < b.len() && b[j] != b'"' && b[j] != b'\n' && b[j] != b'\r' {
                j += 1;
            }
            if j >= b.len() || b[j] != b'"' {
                return Err(err(
                    DiagnosticKind::Lexical,
                    i,
                    j,
                    "Unterminated string literal.".into(),
                ));
            }
            out.push(Token {
                tok: Tok::StringLit(text[i..=j].to_string()),
                span: Span::new(file, i, j + 1),
            });
            i = j + 1;
            continue;
        }
        // Symbols.
        if let Some((s, t)) = SYMBOLS.iter().find(|(s, _)| text[i..].starts_with(s)) {
            out.push(Token {
                tok: t.clone(),
                span: Span::new(file, i, i + s.len()),
            });
            i += s.len();
            continue;
        }
        let ch = text[i..].chars().next().unwrap_or('?');
        return Err(err(
            DiagnosticKind::Lexical,
            start,
            start + ch.len_utf8(),
            format!("Invalid character found: '{ch}'."),
        ));
    }
    Ok(())
}

fn digits(b: &[u8], mut j: usize) -> Option<usize> {
    // integer_constant = [0-9]+ ('_' [0-9]+)*
    if j >= b.len() || !b[j].is_ascii_digit() {
        return None;
    }
    while j < b.len() && b[j].is_ascii_digit() {
        j += 1;
    }
    while j + 1 < b.len() && b[j] == b'_' && b[j + 1].is_ascii_digit() {
        j += 1;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
    }
    Some(j)
}

fn exponent(b: &[u8], j: usize) -> Option<usize> {
    if j < b.len() && (b[j] == b'e' || b[j] == b'E') {
        let mut k = j + 1;
        if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
            k += 1;
        }
        return digits(b, k);
    }
    None
}

/// Longest match among the numeric token rules of stanc3.
fn lex_number(text: &str, i: usize) -> Result<(Tok, usize), (usize, String)> {
    let b = text.as_bytes();
    let (mut tok_kind, mut end);
    if b[i] == b'.' {
        // real_constant2 = '.' int exp | real_constant_dot = '.' int
        let j = digits(b, i + 1).expect("checked by caller");
        match exponent(b, j) {
            Some(e) => {
                tok_kind = 1;
                end = e;
            }
            None => {
                tok_kind = 2;
                end = j;
            }
        }
    } else {
        let j = digits(b, i).expect("checked by caller");
        tok_kind = 0;
        end = j;
        if j < b.len() && b[j] == b'.' {
            // real_constant1 = int '.' int? exp?
            let mut k = j + 1;
            if let Some(d) = digits(b, k) {
                k = d;
            }
            if let Some(e) = exponent(b, k) {
                k = e;
            }
            tok_kind = 1;
            end = k;
        } else if let Some(e) = exponent(b, j) {
            tok_kind = 1;
            end = e;
        }
    }
    // Imaginary suffix.
    if end < b.len() && b[end] == b'i' {
        let lit = text[i..end].to_string();
        return Ok((Tok::ImagLit(lit), end + 1));
    }
    // A number immediately followed by an identifier character or another digit separator is
    // lexed by stanc3 as a number then an identifier, which the parser rejects. We do the same.
    let lit = text[i..end].to_string();
    let tok = match tok_kind {
        0 => Tok::IntLit(lit),
        1 => Tok::RealLit(lit),
        _ => Tok::DotNumeral(lit),
    };
    Ok((tok, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(src: &str) -> Vec<Tok> {
        let mut sm = SourceMap::default();
        let f = sm.add(SourceFile::new("t.stan", src));
        tokenize(&mut sm, f, &NoIncludes)
            .unwrap()
            .into_iter()
            .map(|t| t.tok)
            .collect()
    }

    fn lex_err(src: &str) -> Diagnostic {
        let mut sm = SourceMap::default();
        let f = sm.add(SourceFile::new("t.stan", src));
        tokenize(&mut sm, f, &NoIncludes).unwrap_err()
    }

    #[test]
    fn block_keywords() {
        assert_eq!(
            toks("transformed  data transformed\ndata generated quantities transformedx"),
            vec![
                Tok::TransformedData,
                Tok::Ident("transformed".into()),
                Tok::Data,
                Tok::GeneratedQuantities,
                Tok::Ident("transformedx".into()),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn numbers() {
        assert_eq!(
            toks("1 1_000 1.5 1. .5 .5e3 1e-3 2i 2.5i x.1"),
            vec![
                Tok::IntLit("1".into()),
                Tok::IntLit("1_000".into()),
                Tok::RealLit("1.5".into()),
                Tok::RealLit("1.".into()),
                Tok::DotNumeral(".5".into()),
                Tok::RealLit(".5e3".into()),
                Tok::RealLit("1e-3".into()),
                Tok::ImagLit("2".into()),
                Tok::ImagLit("2.5".into()),
                Tok::Ident("x".into()),
                Tok::DotNumeral(".1".into()),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn operators_longest_match() {
        assert_eq!(
            toks("a.*=b %/% c ./ d .^ e <= f"),
            vec![
                Tok::Ident("a".into()),
                Tok::EltTimesAssign,
                Tok::Ident("b".into()),
                Tok::IDivide,
                Tok::Ident("c".into()),
                Tok::EltDivide,
                Tok::Ident("d".into()),
                Tok::EltPow,
                Tok::Ident("e".into()),
                Tok::Leq,
                Tok::Ident("f".into()),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn comments_and_locations() {
        let mut sm = SourceMap::default();
        let f = sm.add(SourceFile::new("t.stan", "/* a\n b */ // c\n  mu"));
        let t = tokenize(&mut sm, f, &NoIncludes).unwrap();
        assert_eq!(t[0].tok, Tok::Ident("mu".into()));
        assert_eq!(sm.location(t[0].span), "t.stan:3:3");
    }

    #[test]
    fn errors() {
        assert!(lex_err("/* never closed").message.contains("never closed"));
        assert!(lex_err("x @ y").message.contains("'@'"));
        assert!(lex_err("x = \"abc").message.contains("Unterminated"));
        assert!(lex_err("é").message.contains("Invalid character"));
        assert!(
            lex_err("# comment").message.contains("'#'")
                || lex_err("# comment").kind == DiagnosticKind::Lexical
        );
    }

    struct Map(Vec<(&'static str, &'static str)>);
    impl IncludeResolver for Map {
        fn resolve(&self, name: &str) -> Option<(String, String)> {
            self.0
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(n, t)| (n.to_string(), t.to_string()))
        }
    }

    #[test]
    fn includes() {
        let mut sm = SourceMap::default();
        let f = sm.add(SourceFile::new("m.stan", "a\n#include \"b.stan\"\nc"));
        let r = Map(vec![("b.stan", "b")]);
        let t = tokenize(&mut sm, f, &r).unwrap();
        let names: Vec<_> = t.iter().map(|t| t.tok.describe()).collect();
        assert_eq!(names, ["a", "b", "c", "end of file"]);
        assert_eq!(sm.location(t[1].span), "b.stan:1:1");

        let mut sm = SourceMap::default();
        let f = sm.add(SourceFile::new("m.stan", "#include <loop.stan>"));
        let r = Map(vec![("loop.stan", "#include loop.stan")]);
        assert!(
            tokenize(&mut sm, f, &r)
                .unwrap_err()
                .message
                .contains("recursively")
        );
    }
}
