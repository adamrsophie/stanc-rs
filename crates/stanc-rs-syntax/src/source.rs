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

/// All files of a compilation: the main file and its includes.
#[derive(Clone, Debug, Default)]
pub struct SourceMap {
    /// Files, indexed by [`Span::file`].
    pub files: Vec<SourceFile>,
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
            DiagnosticKind::Lexical => "Syntax error (lexical)",
            DiagnosticKind::Syntax => "Syntax error",
            DiagnosticKind::Semantic => "Semantic error",
            DiagnosticKind::Unsupported => "Unsupported feature",
            DiagnosticKind::Internal => "Internal compiler error",
            DiagnosticKind::Warning => "Warning",
            DiagnosticKind::Include => "Include error",
        }
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

    /// Renders the diagnostic with a source excerpt and a caret line.
    pub fn render(&self, sources: &SourceMap) -> String {
        let mut out = String::new();
        let span = self
            .span
            .and_then(|s| sources.files.get(s.file as usize).map(|f| (s, f)));
        match span {
            Some((s, f)) => {
                let (line, col) = f.line_col(s.start as usize);
                // Columns are zero-based in messages, as in stanc3.
                let _ = writeln!(
                    out,
                    "{} in '{}', line {}, column {}:",
                    self.kind.label(),
                    f.name,
                    line,
                    col - 1
                );
                let first = line.saturating_sub(2).max(1);
                let _ = writeln!(out, "   -------------------------------------------------");
                for l in first..=line {
                    let _ = writeln!(out, "{:>5}:  {}", l, f.line_text(l));
                }
                let (end_line, end_col) = f.line_col((s.end as usize).max(s.start as usize));
                let width = if end_line == line {
                    end_col.saturating_sub(col).max(1)
                } else {
                    1
                };
                let _ = writeln!(out, "         {}{}", " ".repeat(col - 1), "^".repeat(width));
                let _ = writeln!(out, "   -------------------------------------------------");
                let _ = writeln!(out);
            }
            None => {
                let _ = writeln!(out, "{}:", self.kind.label());
            }
        }
        let _ = writeln!(out, "{}", self.message);
        for n in &self.notes {
            let _ = writeln!(out, "  note: {n}");
        }
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
