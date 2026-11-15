//! A port of OCaml's `Format` pretty printing engine, which stanc3 uses for its formatter and
//! its messages. Boxes, break hints, the margin, and the maximum indentation behave as in OCaml,
//! so output laid out through this module breaks lines where stanc3's does.

use std::collections::VecDeque;

/// Kinds of pretty printing boxes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxKind {
    /// `@[<h>`: never breaks.
    H,
    /// `@[<v>`: every break hint breaks.
    V,
    /// `@[<hv>`: all break hints break if the box does not fit on the line.
    HV,
    /// `@[<hov>`: fills lines, breaking only where needed.
    HOV,
    /// `@[` or `@[<b>`: like `HOV`, but also breaks where that reduces the indentation.
    B,
    /// A box that fits on the line (internal).
    Fits,
}

#[derive(Clone, Debug)]
enum Tok {
    Text(String),
    Break {
        fits: (String, isize, String),
        breaks: (String, isize, String),
    },
    Begin(isize, BoxKind),
    End,
    Newline,
}

#[derive(Clone, Debug)]
struct Elem {
    size: isize,
    tok: Tok,
    length: isize,
}

const INFINITY: isize = 1_000_000_010;

/// A pretty printer writing to a string.
pub struct Formatter {
    margin: isize,
    min_space_left: isize,
    max_indent: isize,
    space_left: isize,
    current_indent: isize,
    is_new_line: bool,
    left_total: isize,
    right_total: isize,
    curr_depth: usize,
    elems: Vec<Elem>,
    queue: VecDeque<usize>,
    scan_stack: Vec<(isize, usize)>,
    format_stack: Vec<(BoxKind, isize)>,
    out: String,
}

impl Default for Formatter {
    fn default() -> Self {
        Self::new()
    }
}

impl Formatter {
    /// A formatter with OCaml's defaults: margin 78, maximum indentation 68.
    pub fn new() -> Self {
        let mut f = Formatter {
            margin: 78,
            min_space_left: 10,
            max_indent: 68,
            space_left: 78,
            current_indent: 0,
            is_new_line: true,
            left_total: 1,
            right_total: 1,
            curr_depth: 0,
            elems: Vec::new(),
            queue: VecDeque::new(),
            scan_stack: Vec::new(),
            format_stack: Vec::new(),
            out: String::new(),
        };
        f.rinit();
        f
    }

    /// The right margin.
    pub fn margin(&self) -> isize {
        self.margin
    }

    /// `Format.pp_set_margin`.
    pub fn set_margin(&mut self, n: isize) {
        if n >= 1 {
            let n = n.min(INFINITY - 1);
            self.margin = n;
            let new_max_indent = if self.max_indent <= self.margin {
                self.max_indent
            } else {
                (self.margin - self.min_space_left)
                    .max(self.margin / 2)
                    .max(1)
            };
            if new_max_indent > 1 {
                self.set_min_space_left(self.margin - new_max_indent);
            }
        }
    }

    fn set_min_space_left(&mut self, n: isize) {
        if n >= 1 {
            let n = n.min(INFINITY - 1);
            self.min_space_left = n;
            self.max_indent = self.margin - self.min_space_left;
            self.rinit();
        }
    }

    fn rinit(&mut self) {
        self.left_total = 1;
        self.right_total = 1;
        self.queue.clear();
        self.elems.clear();
        self.init_scan_stack();
        self.format_stack.clear();
        self.current_indent = 0;
        self.curr_depth = 0;
        self.space_left = self.margin;
        self.open(BoxKind::HOV, 0);
    }

    fn init_scan_stack(&mut self) {
        self.scan_stack.clear();
        let i = self.alloc(Elem {
            size: -1,
            tok: Tok::Text(String::new()),
            length: 0,
        });
        self.scan_stack.push((-1, i));
    }

    fn alloc(&mut self, e: Elem) -> usize {
        self.elems.push(e);
        self.elems.len() - 1
    }

    fn output(&mut self, s: &str) {
        self.out.push_str(s);
    }

    fn blanks(&mut self, n: isize) {
        for _ in 0..n.max(0) {
            self.out.push(' ');
        }
    }

    fn break_new_line(&mut self, before: &str, offset: isize, after: &str, width: isize) {
        self.output(before);
        self.out.push('\n');
        self.is_new_line = true;
        let indent = self.margin - width + offset;
        let real = self.max_indent.min(indent);
        self.current_indent = real;
        self.space_left = self.margin - self.current_indent;
        self.blanks(self.current_indent);
        self.output(after);
    }

    fn break_same_line(&mut self, before: &str, width: isize, after: &str) {
        self.output(before);
        self.space_left -= width;
        self.blanks(width);
        self.output(after);
    }

    fn force_break_line(&mut self) {
        match self.format_stack.last().copied() {
            None => self.out.push('\n'),
            Some((kind, width)) => {
                if width > self.space_left && !matches!(kind, BoxKind::Fits | BoxKind::H) {
                    self.break_new_line("", 0, "", width);
                }
            }
        }
    }

    fn format_token(&mut self, size: isize, tok: Tok) {
        match tok {
            Tok::Text(s) => {
                self.space_left -= size;
                self.output(&s);
                self.is_new_line = false;
            }
            Tok::Begin(off, kind) => {
                let insertion_point = self.margin - self.space_left;
                if insertion_point > self.max_indent {
                    self.force_break_line();
                }
                let width = self.space_left - off;
                let kind = match kind {
                    BoxKind::V => BoxKind::V,
                    k => {
                        if size > self.space_left {
                            k
                        } else {
                            BoxKind::Fits
                        }
                    }
                };
                self.format_stack.push((kind, width));
            }
            Tok::End => {
                self.format_stack.pop();
            }
            Tok::Newline => match self.format_stack.last().copied() {
                None => self.out.push('\n'),
                Some((_, width)) => self.break_new_line("", 0, "", width),
            },
            Tok::Break { fits, breaks } => {
                let (before, off, after) = breaks;
                if let Some((kind, width)) = self.format_stack.last().copied() {
                    let new_line = match kind {
                        BoxKind::HOV => size + before.len() as isize > self.space_left,
                        BoxKind::B => {
                            if self.is_new_line {
                                false
                            } else if size + before.len() as isize > self.space_left {
                                true
                            } else {
                                self.current_indent > self.margin - width + off
                            }
                        }
                        BoxKind::HV | BoxKind::V => true,
                        BoxKind::Fits | BoxKind::H => false,
                    };
                    if new_line {
                        self.break_new_line(&before, off, &after, width);
                    } else {
                        self.break_same_line(&fits.0, fits.1, &fits.2);
                    }
                }
            }
        }
    }

    fn advance_left(&mut self) {
        while let Some(&i) = self.queue.front() {
            let (size, length) = (self.elems[i].size, self.elems[i].length);
            let pending = self.right_total - self.left_total;
            if size >= 0 || pending >= self.space_left {
                self.queue.pop_front();
                let size = if size >= 0 { size } else { INFINITY };
                let tok = std::mem::replace(&mut self.elems[i].tok, Tok::End);
                self.format_token(size, tok);
                self.left_total += length;
            } else {
                break;
            }
        }
    }

    fn enqueue(&mut self, e: Elem) -> usize {
        self.right_total += e.length;
        let i = self.alloc(e);
        self.queue.push_back(i);
        i
    }

    fn enqueue_advance(&mut self, e: Elem) {
        self.enqueue(e);
        self.advance_left();
    }

    fn set_size(&mut self, is_break: bool) {
        let Some(&(left_total, i)) = self.scan_stack.last() else {
            return;
        };
        if left_total < self.left_total {
            self.init_scan_stack();
            return;
        }
        let size = self.elems[i].size;
        let matches = match self.elems[i].tok {
            Tok::Break { .. } => is_break,
            Tok::Begin(..) => !is_break,
            _ => return,
        };
        if matches {
            self.elems[i].size = self.right_total + size;
            self.scan_stack.pop();
        }
    }

    fn scan_push(&mut self, is_break: bool, e: Elem) {
        let i = self.enqueue(e);
        if is_break {
            self.set_size(true);
        }
        self.scan_stack.push((self.right_total, i));
    }

    /// Opens a box with the given indentation.
    pub fn open(&mut self, kind: BoxKind, indent: isize) {
        self.curr_depth += 1;
        let size = -self.right_total;
        self.scan_push(
            false,
            Elem {
                size,
                tok: Tok::Begin(indent, kind),
                length: 0,
            },
        );
    }

    /// Closes the innermost box.
    pub fn close(&mut self) {
        if self.curr_depth > 1 {
            self.enqueue(Elem {
                size: 0,
                tok: Tok::End,
                length: 0,
            });
            self.set_size(true);
            self.set_size(false);
            self.curr_depth -= 1;
        }
    }

    /// Prints a string, which must not contain a newline to be laid out correctly.
    pub fn string(&mut self, s: &str) {
        let n = s.len() as isize;
        self.enqueue_advance(Elem {
            size: n,
            tok: Tok::Text(s.to_string()),
            length: n,
        });
    }

    /// A break hint: `nspaces` blanks if the line is not broken, otherwise a new line indented
    /// by `offset` relative to the box.
    pub fn brk(&mut self, nspaces: isize, offset: isize) {
        let size = -self.right_total;
        self.scan_push(
            true,
            Elem {
                size,
                tok: Tok::Break {
                    fits: (String::new(), nspaces, String::new()),
                    breaks: (String::new(), offset, String::new()),
                },
                length: nspaces,
            },
        );
    }

    /// `@ `.
    pub fn space(&mut self) {
        self.brk(1, 0);
    }

    /// `@,`.
    pub fn cut(&mut self) {
        self.brk(0, 0);
    }

    /// `@\n`.
    pub fn force_newline(&mut self) {
        self.enqueue_advance(Elem {
            size: 0,
            tok: Tok::Newline,
            length: 0,
        });
    }

    /// Flushes all pending material, closing open boxes; `@.` with a final newline, `@?`
    /// without.
    pub fn flush(&mut self, newline: bool) {
        while self.curr_depth > 1 {
            self.close();
        }
        self.right_total = INFINITY;
        self.advance_left();
        if newline {
            self.out.push('\n');
        }
        self.rinit();
    }

    /// Interprets OCaml format directives in a literal: `@[`, `@[<kind n>`, `@]`, `@ `, `@,`,
    /// `@;`, `@;<n m>`, `@\n`, `@.`, `@?`, and `@@`. Other text is printed as is, each run between
    /// directives as one string.
    pub fn pf(&mut self, fmt: &str) {
        let b = fmt.as_bytes();
        let mut i = 0;
        let mut lit = String::new();
        while i < b.len() {
            if b[i] != b'@' || i + 1 >= b.len() {
                let c = fmt[i..].chars().next().unwrap_or(' ');
                lit.push(c);
                i += c.len_utf8();
                continue;
            }
            let d = b[i + 1];
            if d == b'@' {
                lit.push('@');
                i += 2;
                continue;
            }
            self.lit(&mut lit);
            i += 2;
            match d {
                b'[' => {
                    let (mut kind, mut indent) = (BoxKind::B, 0);
                    if b.get(i) == Some(&b'<') {
                        let end = fmt[i..].find('>').map(|k| i + k).unwrap_or(b.len());
                        let spec = &fmt[i + 1..end];
                        // `<v2>` is `<v 2>`.
                        let split = spec
                            .find(|c: char| !c.is_ascii_alphabetic())
                            .filter(|&k| k > 0 && !spec[k..].starts_with(' '));
                        let spaced;
                        let spec = match split {
                            Some(k) => {
                                spaced = format!("{} {}", &spec[..k], &spec[k..]);
                                spaced.as_str()
                            }
                            None => spec,
                        };
                        let mut parts = spec.split_whitespace();
                        let first = parts.next().unwrap_or("");
                        let num = |s: Option<&str>| s.and_then(|s| s.parse().ok()).unwrap_or(0);
                        match first {
                            "h" => kind = BoxKind::H,
                            "v" => kind = BoxKind::V,
                            "hv" => kind = BoxKind::HV,
                            "hov" => kind = BoxKind::HOV,
                            "b" => kind = BoxKind::B,
                            n => {
                                indent = n.parse().unwrap_or(0);
                                parts = "".split_whitespace();
                            }
                        }
                        if first.parse::<isize>().is_err() {
                            indent = num(parts.next());
                        }
                        i = end + 1;
                    }
                    self.open(kind, indent);
                }
                b']' => self.close(),
                b' ' => self.space(),
                b',' => self.cut(),
                b';' => {
                    if b.get(i) == Some(&b'<') {
                        let end = fmt[i..].find('>').map(|k| i + k).unwrap_or(b.len());
                        let mut parts = fmt[i + 1..end].split_whitespace();
                        let n = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
                        let m = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                        self.brk(n, m);
                        i = end + 1;
                    } else {
                        self.brk(1, 0);
                    }
                }
                b'\n' => self.force_newline(),
                b'.' => self.flush(true),
                b'?' => self.flush(false),
                _ => {
                    lit.push('@');
                    lit.push(d as char);
                }
            }
        }
        self.lit(&mut lit);
    }

    fn lit(&mut self, lit: &mut String) {
        if !lit.is_empty() {
            self.string(&std::mem::take(lit));
        }
    }

    /// `Fmt.text`: spaces are break hints and newlines are forced.
    pub fn text(&mut self, s: &str) {
        for (k, line) in s.split('\n').enumerate() {
            if k > 0 {
                self.force_newline();
            }
            for (j, word) in line.split(' ').enumerate() {
                if j > 0 {
                    self.space();
                }
                self.string(word);
            }
        }
    }

    /// Flushes and returns the output.
    pub fn finish(mut self) -> String {
        self.flush(false);
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(f: impl FnOnce(&mut Formatter)) -> String {
        let mut p = Formatter::new();
        f(&mut p);
        p.finish()
    }

    #[test]
    fn hov_fills_and_hv_breaks_all() {
        let words = "aaaa bbbb cccc dddd";
        let s = render(|p| {
            p.set_margin(12);
            p.pf("@[<hov 2>");
            p.text(words);
            p.pf("@]");
        });
        assert_eq!(s, "aaaa bbbb\n  cccc dddd");
        let s = render(|p| {
            p.set_margin(12);
            p.pf("@[<hv 2>");
            p.text(words);
            p.pf("@]");
        });
        assert_eq!(s, "aaaa\n  bbbb\n  cccc\n  dddd");
        let s = render(|p| {
            p.pf("@[<hv 2>");
            p.text(words);
            p.pf("@]");
        });
        assert_eq!(s, words);
    }

    #[test]
    fn vertical_boxes_indent() {
        let s = render(|p| p.pf("@[<v 0>a {@,@[<v 2>  b;@,c;@]@,}@]"));
        assert_eq!(s, "a {\n  b;\n  c;\n}");
    }
}
