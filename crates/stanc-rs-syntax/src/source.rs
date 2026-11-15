//! Source files, spans, and diagnostics.

use std::fmt::Write as _;

/// A byte range in one source file of a [`SourceMap`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Span {
    /// Index of the file in the source map.
    pub file: u32,
    /// Byte offset of the first character.
    pub start: u32,
    /// Byte offset one past the last character.
    pub end: u32,
}

impl Span {
    /// Creates a span.
    pub fn new(file: u32, start: usize, end: usize) -> Self {
        Span {
            file,
            start: start as u32,
            end: end as u32,
        }
    }

    /// The smallest span covering `self` and `other` (same file assumed; otherwise `self`).
    pub fn to(self, other: Span) -> Span {
        if self.file != other.file {
            return self;
        }
        Span {
            file: self.file,
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// One source file.
#[derive(Clone, Debug)]
pub struct SourceFile {
    /// Name used in diagnostics.
    pub name: String,
    /// Full text.
    pub text: String,
    /// Where this file was included from: the `#include` directive in the including file.
    pub included_from: Option<Span>,
    line_starts: Vec<usize>,
}

impl SourceFile {
    /// Creates a source file.
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        let mut line_starts = vec![0];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i + 1);
            }
        }
        SourceFile {
            name: name.into(),
            text,
            included_from: None,
            line_starts,
        }
    }

    /// One-based line and column (in characters) of a byte offset.
    pub fn line_col(&self, offset: usize) -> (usize, usize) {
        let offset = offset.min(self.text.len());
        let line = match self.line_starts.binary_search(&offset) {
            Ok(l) => l,
            Err(l) => l - 1,
        };
        let start = self.line_starts[line];
        let col = self.text[start..offset].chars().count();
        (line + 1, col + 1)
    }

    /// One-based line and zero-based byte column of a byte offset, as stanc3 reports them.
    pub fn line_byte_col(&self, offset: usize) -> (usize, usize) {
        let offset = offset.min(self.text.len());
        let line = match self.line_starts.binary_search(&offset) {
            Ok(l) => l,
            Err(l) => l - 1,
        };
        (line + 1, offset - self.line_starts[line])
    }

    /// Number of lines, not counting an empty line after a final terminator.
    pub fn line_count(&self) -> usize {
        if self.text.is_empty() {
            0
        } else if self.text.ends_with('\n') {
            self.line_starts.len() - 1
        } else {
            self.line_starts.len()
        }
    }

    /// Text of a one-based line without its terminator.
    pub fn line_text(&self, line: usize) -> &str {
        let start = self.line_starts[line - 1];
        let end = self
            .line_starts
            .get(line)
            .copied()
            .unwrap_or(self.text.len());
        self.text[start..end].trim_end_matches(['\n', '\r'])
    }
}

/// What a [`Comment`] records.
#[derive(Clone, Debug, PartialEq)]
pub enum CommentKind {
    /// `// text`, without the slashes and the line terminator.
    Line(String),
    /// `/* lines */`, the text between the delimiters split at line terminators.
    Block(Vec<String>),
    /// An `#include` directive and the file name it names.
    Include(String),
    /// The end of a comma, an operator, `|`, `?`, `<`, `>`, or `else`: tokens whose location
    /// the syntax tree does not keep, which the formatter uses to place comments.
    Separator,
}

/// A comment or another piece of the source the formatter needs, in source order. As in
/// stanc3, a comment's span starts after its opening delimiter; a line comment's span ends at
/// its line terminator and a block comment's after the closing delimiter.
#[derive(Clone, Debug, PartialEq)]
pub struct Comment {
    /// What it is.
    pub kind: CommentKind,
    /// Where it is.
    pub span: Span,
}

/// All files of a compilation: the main file and its includes.
#[derive(Clone, Debug, Default)]
pub struct SourceMap {
    /// Files, indexed by [`Span::file`].
    pub files: Vec<SourceFile>,
    /// Comments, includes, and separators of all files, in the order the lexer met them.
    pub comments: Vec<Comment>,
}

impl SourceMap {
    /// Adds a file and returns its index.
    pub fn add(&mut self, file: SourceFile) -> u32 {
        self.files.push(file);
        (self.files.len() - 1) as u32
    }

    /// `name:line:col` of the start of a span.
    pub fn location(&self, span: Span) -> String {
        match self.files.get(span.file as usize) {
            Some(f) => {
                let (l, c) = f.line_col(span.start as usize);
                format!("{}:{}:{}", f.name, l, c)
            }
            None => "<unknown>".into(),
        }
    }

    /// One-based line and column of the start of a span.
    pub fn line_col(&self, span: Span) -> Option<(usize, usize)> {
        self.files
            .get(span.file as usize)
            .map(|f| f.line_col(span.start as usize))
    }
}

/// What kind of problem a diagnostic reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticKind {
    /// Invalid characters, unterminated comments, malformed numbers.
    Lexical,
    /// Input that does not match the grammar.
    Syntax,
    /// Type errors, scope errors, and other violations of the language rules.
    Semantic,
    /// Valid Stan that this compiler does not support yet.
    Unsupported,
    /// A bug in the compiler.
    Internal,
    /// A warning; compilation continues.
    Warning,
    /// An include file could not be read.
    Include,
}

impl DiagnosticKind {
    fn label(self) -> &'static str {
        match self {
            DiagnosticKind::Lexical | DiagnosticKind::Syntax | DiagnosticKind::Include => {
                "Syntax error"
            }
            DiagnosticKind::Semantic => "Semantic error",
            DiagnosticKind::Unsupported => "Unsupported feature",
            DiagnosticKind::Internal => "Internal compiler error",
            DiagnosticKind::Warning => "Warning",
        }
    }

    /// The kind of syntax error stanc3 names after the location.
    fn syntax_kind(self) -> Option<&'static str> {
        match self {
            DiagnosticKind::Lexical => Some("lexing error"),
            DiagnosticKind::Syntax => Some("parsing error"),
            DiagnosticKind::Include => Some("include error"),
            _ => None,
        }
    }
}

impl SourceMap {
    /// A location as stanc3 prints it: `'file', line L, column C`, followed by the chain of
    /// files it was included from.
    fn pp_location(&self, file: u32, offset: usize, out: &mut String) {
        let f = &self.files[file as usize];
        let (l, c) = f.line_byte_col(offset);
        let _ = write!(out, "'{}', line {l}, column {c}", f.name);
        if let Some(inc) = f.included_from {
            out.push_str(", included from\n");
            self.pp_location(inc.file, inc.start as usize, out);
        }
    }

    /// A span as stanc3 prints it: the start, then ` to` the end unless the two are equal or
    /// the start is in an included file.
    pub fn pp_span(&self, span: Span) -> String {
        let mut out = String::new();
        self.pp_location(span.file, span.start as usize, &mut out);
        let f = &self.files[span.file as usize];
        let (l0, c0) = f.line_byte_col(span.start as usize);
        let (l1, c1) = f.line_byte_col(span.end as usize);
        if f.included_from.is_none() && (l0, c0) != (l1, c1) {
            if l0 == l1 {
                let _ = write!(out, " to column {c1}");
            } else {
                let _ = write!(out, " to line {l1}, column {c1}");
            }
        }
        out
    }

    /// The source excerpt stanc3 prints under a location: two lines either side and a caret.
    pub fn pp_context(&self, file: u32, offset: usize) -> String {
        let f = &self.files[file as usize];
        let (line, col) = f.line_byte_col(offset);
        let n = f.line_count();
        let at_eof = line == n + 1;
        let get = |i: usize| -> Option<&str> {
            if at_eof && i == line {
                Some("")
            } else if i == 0 || i > n {
                None
            } else {
                Some(f.line_text(i))
            }
        };
        let bars = "   -------------------------------------------------\n";
        let mut out = String::from(bars);
        let row = |i: usize, out: &mut String| {
            if let Some(t) = get(i) {
                let _ = writeln!(out, "{i:>6}:  {t}");
            }
        };
        for i in [line.wrapping_sub(2), line.wrapping_sub(1), line] {
            row(i, &mut out);
        }
        let here = get(line).unwrap_or("");
        let pad: String = here
            .bytes()
            .take(col.min(here.len()))
            .map(|b| if b == b'\t' { '\t' } else { ' ' })
            .collect();
        let _ = writeln!(
            out,
            "         {pad}^{}",
            if at_eof {
                " (error at end of file)"
            } else {
                ""
            }
        );
        for i in [line + 1, line + 2] {
            row(i, &mut out);
        }
        out.push_str(bars);
        out
    }
}

/// An error or warning with an optional source location.
#[derive(Clone, Debug, PartialEq)]
pub struct Diagnostic {
    /// Category.
    pub kind: DiagnosticKind,
    /// Main message.
    pub message: String,
    /// Location, when the problem has one.
    pub span: Option<Span>,
    /// Extra explanations or suggestions.
    pub notes: Vec<String>,
}

impl Diagnostic {
    /// Creates a diagnostic with a location.
    pub fn new(kind: DiagnosticKind, span: Span, message: impl Into<String>) -> Self {
        Diagnostic {
            kind,
            message: message.into(),
            span: Some(span),
            notes: Vec::new(),
        }
    }

    /// Creates a diagnostic without a location.
    pub fn global(kind: DiagnosticKind, message: impl Into<String>) -> Self {
        Diagnostic {
            kind,
            message: message.into(),
            span: None,
            notes: Vec::new(),
        }
    }

    /// Adds a note.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// True for everything except warnings.
    pub fn is_error(&self) -> bool {
        self.kind != DiagnosticKind::Warning
    }

    /// Renders the diagnostic as stanc3 does: a header with the location, a source excerpt
    /// with a caret, and the message.
    pub fn render(&self, sources: &SourceMap) -> String {
        let mut out = String::new();
        let span = self
            .span
            .filter(|s| (s.file as usize) < sources.files.len());
        if self.kind == DiagnosticKind::Warning {
            // stanc3: `@[<v4>Warning in @[<location>@]:@ @[<text>@]@]`.
            let mut f = crate::format::Formatter::new();
            f.open(crate::format::BoxKind::V, 4);
            match span {
                Some(s) => {
                    f.string("Warning in ");
                    f.open(crate::format::BoxKind::B, 0);
                    for (i, part) in sources.pp_span(s).split('\n').enumerate() {
                        if i > 0 {
                            f.force_newline();
                        }
                        f.string(part);
                    }
                    f.close();
                    f.string(":");
                }
                None => f.string("Warning:"),
            }
            f.space();
            f.open(crate::format::BoxKind::B, 0);
            f.text(&self.message);
            f.close();
            f.close();
            out.push_str(&f.finish());
            out.push('\n');
            return out;
        }
        match span {
            Some(s) => {
                let _ = write!(out, "{} in {}", self.kind.label(), sources.pp_span(s));
                if let Some(k) = self.kind.syntax_kind() {
                    let _ = write!(out, ", {k}");
                }
                out.push_str(":\n");
                out.push_str(&sources.pp_context(s.file, s.start as usize));
                out.push('\n');
            }
            None => {
                let _ = writeln!(out, "{}:", self.kind.label());
            }
        }
        let _ = writeln!(out, "{}", self.message);
        for n in &self.notes {
            let _ = writeln!(out, "{n}");
        }
        out
    }
}

/// A JSON string literal.
fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl DiagnosticKind {
    fn json_name(self) -> &'static str {
        match self {
            DiagnosticKind::Lexical => "lexing",
            DiagnosticKind::Syntax => "parsing",
            DiagnosticKind::Include => "include",
            DiagnosticKind::Semantic => "semantic",
            DiagnosticKind::Unsupported => "unsupported",
            DiagnosticKind::Internal => "internal",
            DiagnosticKind::Warning => "warning",
        }
    }
}

impl Diagnostic {
    /// The diagnostic as a JSON object: `severity` (`error` or `warning`), `kind`, the
    /// `message` without location, `file`, `begin` and `end` (one-based lines, zero-based byte
    /// columns, as in the rendered header; absent without a location), `included_from` (the
    /// chain of `#include` sites, innermost first), and the `rendered` text.
    pub fn to_json(&self, sources: &SourceMap) -> String {
        let mut out = String::from("{");
        let _ = write!(
            out,
            "\"severity\": {}, \"kind\": {}, \"message\": {}",
            json_string(if self.is_error() { "error" } else { "warning" }),
            json_string(self.kind.json_name()),
            json_string(&self.message)
        );
        let span = self
            .span
            .filter(|s| (s.file as usize) < sources.files.len());
        if let Some(s) = span {
            let f = &sources.files[s.file as usize];
            let (l0, c0) = f.line_byte_col(s.start as usize);
            let (l1, c1) = f.line_byte_col(s.end as usize);
            let _ = write!(
                out,
                ", \"file\": {}, \"begin\": {{\"line\": {l0}, \"column\": {c0}}}, \"end\": {{\"line\": {l1}, \"column\": {c1}}}",
                json_string(&f.name)
            );
            let mut chain = Vec::new();
            let mut inc = f.included_from;
            while let Some(i) = inc {
                let g = &sources.files[i.file as usize];
                let (l, c) = g.line_byte_col(i.start as usize);
                chain.push(format!(
                    "{{\"file\": {}, \"line\": {l}, \"column\": {c}}}",
                    json_string(&g.name)
                ));
                inc = g.included_from;
            }
            let _ = write!(out, ", \"included_from\": [{}]", chain.join(", "));
        }
        let _ = write!(
            out,
            ", \"rendered\": {}}}",
            json_string(&self.render(sources))
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_and_column() {
        let f = SourceFile::new("m.stan", "ab\ncd\n\nefg");
        assert_eq!(f.line_col(0), (1, 1));
        assert_eq!(f.line_col(3), (2, 1));
        assert_eq!(f.line_col(4), (2, 2));
        assert_eq!(f.line_col(7), (4, 1));
        assert_eq!(f.line_text(4), "efg");
    }
}
