//! stanc3's formatter (`stanc --auto-format`): a port of `Pretty_printing.ml` and
//! `Pretty_print_prog.ml` on the [`Formatter`](crate::format::Formatter) engine. Comments are
//! placed by comparing their locations with those of the syntax tree, as stanc3 does, so the
//! output keeps them where stanc3 keeps them.

use crate::ast::*;
use crate::format::{BoxKind, Formatter};
use crate::source::{CommentKind, SourceMap, Span};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::rc::Rc;

/// Formatter settings.
#[derive(Clone, Debug)]
pub struct Options {
    /// The right margin, stanc's `--max-line-length`.
    pub line_length: isize,
    /// Print the contents of included files instead of the `#include` directives.
    pub inline_includes: bool,
    /// Drop comments.
    pub strip_comments: bool,
    /// The program is a `.stanfunctions` file: print the functions without a block.
    pub bare_functions: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            line_length: 78,
            inline_includes: false,
            strip_comments: false,
            bare_functions: false,
        }
    }
}

/// A position as stanc3 records it: line (from one), column (in bytes, from zero), file, and
/// the position of the `#include` the file came from.
#[derive(Clone, Debug)]
struct Loc {
    line: i64,
    col: i64,
    file: i64,
    included_from: Option<Rc<Loc>>,
}

impl Loc {
    const EMPTY: Loc = Loc {
        line: -1,
        col: -1,
        file: -1,
        included_from: None,
    };

    fn chain(&self) -> Vec<&Loc> {
        let mut v = vec![self];
        let mut cur = self;
        while let Some(p) = &cur.included_from {
            v.push(p);
            cur = p;
        }
        v.reverse();
        v
    }

    /// `Location.compare`: outermost file first, then line and column.
    fn cmp(&self, other: &Loc) -> Ordering {
        let (a, b) = (self.chain(), other.chain());
        for (x, y) in a.iter().zip(&b) {
            match x.line.cmp(&y.line).then(x.col.cmp(&y.col)) {
                Ordering::Equal => {}
                o => return o,
            }
        }
        a.len().cmp(&b.len())
    }

    fn initial_file_loc(&self) -> &Loc {
        let mut cur = self;
        while let Some(p) = &cur.included_from {
            cur = p;
        }
        cur
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Style {
    Line,
    Block,
    Include,
}

#[derive(Clone, Debug)]
struct Cm {
    style: Style,
    lines: Vec<String>,
    begin: Loc,
    end: Loc,
}

#[derive(Clone, Debug)]
enum Item {
    Comment(Cm),
    Separator(Loc),
}

/// Which comments `pp_comments_spacing` takes.
#[derive(Clone, Copy)]
enum Take {
    All,
    UntilSeparator,
}

/// What `pp_comments_spacing` prints before comments, or instead of them.
#[derive(Clone, Copy, PartialEq)]
enum Fill {
    Nothing,
    Space,
    Blank,
}

struct Printer<'a> {
    f: Formatter,
    sources: &'a SourceMap,
    includes: Vec<Option<Rc<Loc>>>,
    comments: VecDeque<Item>,
    skipped: Vec<Cm>,
    print_included: bool,
    error: Option<String>,
}

/// Formats a program as `stanc --auto-format` does. Fails, as stanc3 does, on an `#include`
/// inside a construct the formatter cannot keep it in.
pub fn format_program(p: &Program, sources: &SourceMap, opts: &Options) -> Result<String, String> {
    let mut pr = Printer::new(sources, opts);
    pr.f.open(BoxKind::V, 0);
    if opts.bare_functions {
        if let Some(b) = &p.functions {
            pr.f.open(BoxKind::H, 0);
            pr.f.open(BoxKind::B, 0);
            let end = pr.end(b.span);
            pr.statements(&b.stmts, &end);
            pr.f.close();
            pr.f.close();
        }
    } else {
        for (name, block) in p.blocks() {
            let Some(b) = block else { continue };
            let (begin, end) = (pr.begin(b.span), pr.end(b.span));
            if pr.should_skip(&begin) && pr.should_skip(&end) {
                continue;
            }
            let cs = pr.get_comments(&begin);
            pr.spacing(true, None, Some(&begin), cs);
            pr.f.string(&format!("{name} {{"));
            pr.f.cut();
            pr.indented_box(0, |pr| pr.statements(&b.stmts, &end));
            pr.f.cut();
            pr.f.string("}");
            pr.f.cut();
        }
        let rest = pr.remaining_comments();
        pr.spacing(true, None, None, rest);
    }
    if let Some(e) = pr.error {
        return Err(e);
    }
    Ok(pr.f.finish())
}

static NO_SOURCES: SourceMap = SourceMap {
    files: Vec::new(),
    comments: Vec::new(),
};

/// Lays out a message as stanc3's type checker does with `@[<v>...@]`: literal text, and
/// expressions in indented boxes printed by stanc3's printer, without comments.
pub struct Message(Printer<'static>);

impl Default for Message {
    fn default() -> Self {
        Self::new()
    }
}

impl Message {
    /// An empty message.
    pub fn new() -> Self {
        let mut p = Printer::new(&NO_SOURCES, &Options::default());
        p.f.open(BoxKind::V, 0);
        Message(p)
    }

    /// Literal text; its spaces are not break hints.
    pub fn text(&mut self, s: &str) -> &mut Self {
        self.0.f.string(s);
        self
    }

    /// A break hint of the vertical box: a new line.
    pub fn newline(&mut self) -> &mut Self {
        self.0.f.space();
        self
    }

    /// An expression.
    pub fn expr(&mut self, e: &Expr) -> &mut Self {
        self.0.expr(e);
        self
    }

    /// stanc3's `pp_indented_box`: `@,    @[<hov 2>...@]@,`.
    pub fn indented(&mut self, body: impl FnOnce(&mut Message)) -> &mut Self {
        self.0.f.cut();
        self.0.f.string("    ");
        self.0.f.open(BoxKind::HOV, 2);
        body(self);
        self.0.f.close();
        self.0.f.cut();
        self
    }

    /// The laid out message.
    pub fn finish(self) -> String {
        self.0.f.finish()
    }
}

impl<'a> Printer<'a> {
    fn new(sources: &'a SourceMap, opts: &Options) -> Self {
        let mut f = Formatter::new();
        f.set_margin(opts.line_length);
        let mut includes: Vec<Option<Rc<Loc>>> = Vec::with_capacity(sources.files.len());
        for file in &sources.files {
            let inc = file.included_from.map(|s| {
                let (line, col) = sources.files[s.file as usize].line_byte_col(s.start as usize);
                Rc::new(Loc {
                    line: line as i64,
                    col: col as i64,
                    file: s.file as i64,
                    included_from: includes[s.file as usize].clone(),
                })
            });
            includes.push(inc);
        }
        let mut pr = Printer {
            f,
            sources,
            includes,
            comments: VecDeque::new(),
            skipped: Vec::new(),
            print_included: opts.inline_includes,
            error: None,
        };
        for c in &sources.comments {
            let begin = pr.begin(c.span);
            let item = match &c.kind {
                CommentKind::Separator => Item::Separator(begin),
                k => {
                    let (style, lines) = match k {
                        CommentKind::Line(s) => (Style::Line, vec![s.clone()]),
                        CommentKind::Block(l) => (Style::Block, l.clone()),
                        CommentKind::Include(s) => (Style::Include, vec![s.clone()]),
                        CommentKind::Separator => unreachable!(),
                    };
                    // stanc3 records a line comment's end before counting its line
                    // terminator, so the end stays on the comment's line.
                    let end = if style == Style::Line {
                        Loc {
                            col: begin.col + (c.span.end - c.span.start) as i64,
                            ..begin.clone()
                        }
                    } else {
                        pr.end(c.span)
                    };
                    Item::Comment(Cm {
                        style,
                        lines,
                        begin,
                        end,
                    })
                }
            };
            let from_include = match &item {
                Item::Comment(c) => c.begin.included_from.is_some(),
                Item::Separator(l) => l.included_from.is_some(),
            };
            let keep = match &item {
                Item::Comment(c) if c.style == Style::Include => {
                    !opts.inline_includes && !from_include
                }
                Item::Comment(_) if opts.strip_comments => false,
                _ => opts.inline_includes || !from_include,
            };
            if keep {
                pr.comments.push_back(item);
            }
        }
        pr
    }

    fn loc(&self, file: u32, offset: usize) -> Loc {
        let Some(f) = self.sources.files.get(file as usize) else {
            // Messages print expressions without their sources; locations do not matter there.
            return Loc::EMPTY;
        };
        let (line, col) = f.line_byte_col(offset);
        Loc {
            line: line as i64,
            col: col as i64,
            file: file as i64,
            included_from: self.includes[file as usize].clone(),
        }
    }

    fn begin(&self, s: Span) -> Loc {
        self.loc(s.file, s.start as usize)
    }

    fn end(&self, s: Span) -> Loc {
        self.loc(s.file, s.end as usize)
    }

    // Comment queue.

    fn take(&mut self, end: &Loc, mode: Take) -> Vec<Cm> {
        let mut out = Vec::new();
        while let Some(item) = self.comments.front() {
            match item {
                Item::Comment(c) if c.begin.cmp(end) == Ordering::Less => {
                    if let Some(Item::Comment(c)) = self.comments.pop_front() {
                        out.push(c);
                    }
                }
                Item::Separator(l)
                    if matches!(mode, Take::All) && l.cmp(end) != Ordering::Greater =>
                {
                    self.comments.pop_front();
                }
                _ => break,
            }
        }
        out
    }

    fn get_comments(&mut self, end: &Loc) -> Vec<Cm> {
        self.take(end, Take::All)
    }

    fn skip_comments(&mut self, loc: &Loc) {
        for mut c in self.get_comments(loc) {
            if c.style == Style::Include {
                self.error.get_or_insert_with(|| {
                    format!("Unable to format #include of {}!", c.lines.join(""))
                });
                continue;
            }
            if let Some(first) = c.lines.first_mut() {
                *first = format!(" ^^^:{first}");
                self.skipped.push(c);
            }
        }
    }

    fn remaining_comments(&mut self) -> Vec<Cm> {
        let mut out = std::mem::take(&mut self.skipped);
        for item in std::mem::take(&mut self.comments) {
            if let Item::Comment(c) = item {
                out.push(c);
            }
        }
        out
    }

    fn should_skip(&self, loc: &Loc) -> bool {
        !self.print_included && loc.included_from.is_some()
    }

    // Spacing and comments.

    fn space(&mut self, newline: bool, prev: &Loc, begin: &Loc) {
        let prev = if self.should_skip(prev) {
            prev.initial_file_loc()
        } else {
            prev
        };
        if self.should_skip(begin) {
        } else if prev.file != begin.file || prev.line + 1 < begin.line {
            self.f.cut();
            self.f.cut();
        } else if newline || prev.line < begin.line {
            self.f.cut();
        } else {
            self.f.string(" ");
        }
    }

    fn comment(&mut self, c: &Cm) {
        match c.style {
            Style::Block => {
                self.f.string("/*");
                self.f.open(BoxKind::V, -2);
                let lines = trim_tail(c.begin.col, &c.lines);
                for (i, l) in lines.iter().enumerate() {
                    if i > 0 {
                        self.f.cut();
                    }
                    self.f.string(l);
                }
                self.f.close();
                self.f.string("*/");
            }
            Style::Line => self.f.string(&format!("//{}", c.lines[0])),
            Style::Include => {
                self.f.open(BoxKind::B, 0);
                self.f.string(&format!("#include {}", c.lines[0]));
                self.f.close();
            }
        }
    }

    fn recurse(&mut self, mut prev: Loc, ls: &[Cm]) -> Loc {
        for c in ls {
            self.space(false, &prev, &c.begin);
            self.comment(c);
            prev = c.end.clone();
        }
        prev
    }

    /// `pp_spacing`.
    fn spacing(&mut self, newline: bool, prev: Option<&Loc>, next: Option<&Loc>, ls: Vec<Cm>) {
        let newline = newline || matches!(ls.last(), Some(c) if c.style != Style::Block);
        let skipped = std::mem::take(&mut self.skipped);
        match ls.split_first() {
            Some((hd, tl)) => {
                if let Some(p) = prev {
                    self.space(false, p, &hd.begin);
                }
                self.comment(hd);
                self.recurse(Loc::EMPTY, &skipped);
                let last = self.recurse(hd.end.clone(), tl);
                if let Some(n) = next {
                    self.space(newline, &last, n);
                }
            }
            None => {
                self.recurse(Loc::EMPTY, &skipped);
                if let (Some(p), Some(n)) = (prev, next) {
                    self.space(newline, p, n);
                }
            }
        }
    }

    /// `pp_comments_spacing`.
    fn comments_spacing(&mut self, take: Take, before: Fill, no_comment: Fill, loc: &Loc) {
        let cs = self.take(loc, take);
        if cs.is_empty() {
            self.fill(no_comment);
            return;
        }
        self.fill(before);
        let mut was_block = true;
        for c in &cs {
            self.comment(c);
            was_block = c.style == Style::Block;
            if !was_block {
                self.f.force_newline();
            }
        }
        if was_block && before == Fill::Nothing {
            self.f.space();
        }
    }

    fn fill(&mut self, f: Fill) {
        match f {
            Fill::Nothing => {}
            Fill::Space => self.f.space(),
            Fill::Blank => self.f.string(" "),
        }
    }

    fn indented_box(&mut self, offset: isize, body: impl FnOnce(&mut Self)) {
        self.f.open(BoxKind::H, 0);
        self.f.string("  ");
        self.f.open(BoxKind::B, offset);
        body(self);
        self.f.close();
        self.f.close();
    }

    // Expressions.

    fn start_funapp(&mut self, id: &Ident) {
        let long = id.name.len() > 16;
        if long {
            self.f.open(BoxKind::B, 2);
        }
        self.f.string(&id.name);
        if !long {
            self.f.open(BoxKind::B, 1);
        }
    }

    /// `pp_list_of`: items separated by commas, with the comments between them.
    fn list_of<T>(
        &mut self,
        items: &[T],
        span_of: fn(&T) -> Span,
        pp: fn(&mut Self, &T),
        span: Span,
    ) {
        let (begin, end) = (self.begin(span), self.end(span));
        self.skip_comments(&begin);
        if let Some((first, rest)) = items.split_first() {
            let l = self.begin(span_of(first));
            self.comments_spacing(Take::All, Fill::Nothing, Fill::Nothing, &l);
            let mut cur = first;
            for next in rest {
                let next_loc = self.begin(span_of(next));
                pp(self, cur);
                self.comments_spacing(Take::UntilSeparator, Fill::Space, Fill::Nothing, &next_loc);
                self.f.string(",");
                self.comments_spacing(Take::All, Fill::Blank, Fill::Space, &next_loc);
                cur = next;
            }
            pp(self, cur);
        }
        self.comments_spacing(Take::All, Fill::Blank, Fill::Nothing, &end);
    }

    fn exprs(&mut self, es: &[Expr], span: Span) {
        self.list_of(es, |e| e.span, |p, e| p.expr(e), span);
    }

    fn index(&mut self, i: &Index) {
        match i {
            Index::All => self.f.string(" : "),
            Index::Single(e) => self.expr(e),
            Index::Upfrom(e) => {
                self.expr(e);
                self.f.string(" : ");
            }
            Index::Downfrom(e) => {
                self.f.string(" : ");
                self.expr(e);
            }
            Index::Between(a, b) => {
                self.expr(a);
                self.f.string(" : ");
                self.expr(b);
            }
        }
    }

    fn indices(&mut self, idx: &[Index]) {
        self.f.string("[");
        for (k, i) in idx.iter().enumerate() {
            if k > 0 {
                self.f.string(", ");
            }
            self.index(i);
        }
        self.f.string("]");
    }

    fn expr(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::TernaryIf(c, a, b) => {
                let (then_loc, else_loc) = (self.begin(a.span), self.begin(b.span));
                self.f.open(BoxKind::B, 0);
                self.expr(c);
                self.f.space();
                self.comments_spacing(
                    Take::UntilSeparator,
                    Fill::Nothing,
                    Fill::Nothing,
                    &then_loc,
                );
                self.f.string("? ");
                self.comments_spacing(Take::All, Fill::Nothing, Fill::Nothing, &then_loc);
                self.expr(a);
                self.f.space();
                self.comments_spacing(
                    Take::UntilSeparator,
                    Fill::Nothing,
                    Fill::Nothing,
                    &else_loc,
                );
                self.f.string(": ");
                self.comments_spacing(Take::All, Fill::Nothing, Fill::Nothing, &else_loc);
                self.expr(b);
                self.f.close();
            }
            ExprKind::BinOp(a, op, b) => {
                let next = self.begin(b.span);
                self.f.open(BoxKind::B, 0);
                self.expr(a);
                self.f.space();
                self.comments_spacing(Take::UntilSeparator, Fill::Nothing, Fill::Nothing, &next);
                self.f.string(op.symbol());
                self.f.string(" ");
                self.comments_spacing(Take::All, Fill::Nothing, Fill::Nothing, &next);
                self.expr(b);
                self.f.close();
            }
            ExprKind::PrefixOp(op, a) => {
                let l = self.begin(a.span);
                self.comments_spacing(Take::All, Fill::Nothing, Fill::Nothing, &l);
                self.f.string(op.symbol());
                self.expr(a);
            }
            ExprKind::Transpose(a) => {
                self.expr(a);
                self.f.string("'");
            }
            ExprKind::Variable(id) => self.f.string(&id.name),
            ExprKind::IntNumeral(s) | ExprKind::RealNumeral(s) => self.f.string(s),
            ExprKind::ImagNumeral(s) => self.f.string(&format!("{s}i")),
            ExprKind::FunApp(id, args) => {
                self.start_funapp(id);
                self.f.string("(");
                self.f.cut();
                self.exprs(args, e.span);
                self.f.string(")");
                self.f.close();
            }
            ExprKind::CondDistApp(id, args) => {
                self.start_funapp(id);
                self.f.string("(");
                self.f.cut();
                if let [only] = &args[..] {
                    self.expr(only);
                    let end = self.end(e.span);
                    self.comments_spacing(Take::All, Fill::Space, Fill::Nothing, &end);
                } else if let Some((first, rest)) = args.split_first() {
                    let b = self.begin(rest[0].span);
                    self.expr(first);
                    self.comments_spacing(Take::UntilSeparator, Fill::Space, Fill::Nothing, &b);
                    self.f.string(" |");
                    self.f.space();
                    self.comments_spacing(Take::All, Fill::Nothing, Fill::Nothing, &b);
                    self.exprs(rest, e.span);
                }
                self.f.string(")");
                self.f.close();
            }
            ExprKind::GetTarget => self.f.string("target()"),
            ExprKind::ArrayExpr(xs) => {
                self.f.string("{");
                self.f.open(BoxKind::B, 0);
                self.exprs(xs, e.span);
                self.f.string("}");
                self.f.close();
            }
            ExprKind::RowVectorExpr(xs) => {
                self.f.string("[");
                self.f.open(BoxKind::B, 0);
                self.exprs(xs, e.span);
                self.f.string("]");
                self.f.close();
            }
            ExprKind::Paren(a) => {
                self.f.string("(");
                self.expr(a);
                self.f.string(")");
            }
            ExprKind::Indexed(a, idx) => {
                self.expr(a);
                self.indices(idx);
            }
            ExprKind::TupleProjection(a, k) => {
                self.expr(a);
                self.f.string(&format!(".{k}"));
            }
            ExprKind::TupleExpr(xs) => {
                self.f.string("(");
                self.f.open(BoxKind::HV, 0);
                self.exprs(xs, e.span);
                if xs.len() == 1 {
                    self.f.string(",");
                }
                self.f.close();
                self.f.string(")");
            }
        }
    }

    fn lvalue(&mut self, l: &LValue) {
        match &l.kind {
            LValueKind::Variable(id) => self.f.string(&id.name),
            LValueKind::Indexed(a, idx) => {
                self.lvalue(a);
                self.indices(idx);
            }
            LValueKind::TupleProjection(a, k) => {
                self.lvalue(a);
                self.f.string(&format!(".{k}"));
            }
            LValueKind::Tuple(ls) => {
                self.f.string("(");
                self.f.open(BoxKind::B, 0);
                self.list_of(ls, |l| l.span, |p, l| p.lvalue(l), l.span);
                self.f.close();
                self.f.string(")");
            }
        }
    }

    // Types.

    fn unsized_ty(&mut self, t: &UnsizedType) {
        match t {
            UnsizedType::Array(_) => {
                let (n, inner) = t.array_dims();
                self.f.string(&format!("array[{}] ", ",".repeat(n - 1)));
                self.unsized_ty(inner);
            }
            UnsizedType::Tuple(ts) => {
                self.f.string("tuple(");
                self.f.open(BoxKind::B, 0);
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 {
                        self.f.string(",");
                        self.f.space();
                    }
                    self.unsized_ty(t);
                }
                if ts.len() == 1 {
                    self.f.string(",");
                }
                self.f.close();
                self.f.string(")");
            }
            t => self.f.string(&t.to_string()),
        }
    }

    fn sized(&mut self, t: &SizedType) {
        let brackets = |p: &mut Self, es: &[&Expr]| {
            p.f.open(BoxKind::B, 1);
            p.f.string("[");
            for (i, e) in es.iter().enumerate() {
                if i > 0 {
                    p.f.string(",");
                    p.f.space();
                }
                p.expr(e);
            }
            p.f.string("]");
            p.f.close();
        };
        match t {
            SizedType::Int => self.f.string("int"),
            SizedType::Real => self.f.string("real"),
            SizedType::Complex => self.f.string("complex"),
            SizedType::Vector(e) => {
                self.f.string("vector");
                brackets(self, &[e]);
            }
            SizedType::RowVector(e) => {
                self.f.string("row_vector");
                brackets(self, &[e]);
            }
            SizedType::ComplexVector(e) => {
                self.f.string("complex_vector");
                brackets(self, &[e]);
            }
            SizedType::ComplexRowVector(e) => {
                self.f.string("complex_row_vector");
                brackets(self, &[e]);
            }
            SizedType::Matrix(a, b) => {
                self.f.string("matrix");
                brackets(self, &[a, b]);
            }
            SizedType::ComplexMatrix(a, b) => {
                self.f.string("complex_matrix");
                brackets(self, &[a, b]);
            }
            SizedType::Array(inner, e) => {
                self.f.string("array");
                self.f.open(BoxKind::B, 1);
                self.f.string("[");
                self.sized(inner);
                self.f.string(",");
                self.f.space();
                self.expr(e);
                self.f.string("]");
                self.f.close();
            }
            SizedType::Tuple(ts) => {
                self.f.string("tuple(");
                self.f.open(BoxKind::B, 0);
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 {
                        self.f.string(",");
                        self.f.space();
                    }
                    self.sized(t);
                }
                self.f.close();
                self.f.string(")");
            }
        }
    }

    fn bracketed(&mut self, t: &Transformation) {
        let (a, b): ((&str, &Expr), Option<(&str, &Expr)>) = match t {
            Transformation::Lower(e) => (("lower", e), None),
            Transformation::Upper(e) => (("upper", e), None),
            Transformation::LowerUpper(x, y) => (("lower", x), Some(("upper", y))),
            Transformation::Offset(e) => (("offset", e), None),
            Transformation::Multiplier(e) => (("multiplier", e), None),
            Transformation::OffsetMultiplier(x, y) => (("offset", x), Some(("multiplier", y))),
            _ => return,
        };
        self.f.string("<");
        self.f.open(BoxKind::B, 0);
        self.f.string(&format!("{}=", a.0));
        self.expr(a.1);
        if let Some((n, e)) = b {
            self.f.string(",");
            self.f.space();
            self.f.string(&format!("{n}="));
            self.expr(e);
        }
        self.f.close();
        self.f.string(">");
    }

    fn sizes(&mut self, t: &SizedType, cov: bool) {
        let es: Vec<&Expr> = match t {
            SizedType::Vector(e)
            | SizedType::RowVector(e)
            | SizedType::ComplexVector(e)
            | SizedType::ComplexRowVector(e)
                if !cov =>
            {
                vec![e]
            }
            SizedType::Matrix(a, b) if cov && a == b => vec![a],
            SizedType::Matrix(a, b) => vec![a, b],
            SizedType::ComplexMatrix(a, b) if !cov => vec![a, b],
            _ => return,
        };
        self.f.string("[");
        for (i, e) in es.iter().enumerate() {
            if i > 0 {
                self.f.string(", ");
            }
            self.expr(e);
        }
        self.f.string("]");
    }

    fn transformed_type(&mut self, st: &SizedType, trans: &Transformation) {
        if let SizedType::Array(..) = st {
            let mut dims = Vec::new();
            let mut ty = st;
            while let SizedType::Array(inner, e) = ty {
                dims.push(e.clone());
                ty = inner;
            }
            let span = dims[0].span.to(dims[dims.len() - 1].span);
            self.f.string("array[");
            self.f.open(BoxKind::B, 0);
            self.exprs(&dims, span);
            self.f.close();
            self.f.string("]");
            self.f.space();
            self.possibly_transformed(ty, trans);
        } else {
            self.possibly_transformed(st, trans);
        }
    }

    fn possibly_transformed(&mut self, st: &SizedType, trans: &Transformation) {
        use Transformation as T;
        let name = |s: &str, p: &mut Self, cov: bool| {
            p.f.string(s);
            p.sizes(st, cov);
        };
        match trans {
            T::Identity => self.sized(st),
            T::Lower(_)
            | T::Upper(_)
            | T::LowerUpper(..)
            | T::Offset(_)
            | T::Multiplier(_)
            | T::OffsetMultiplier(..) => {
                self.unsized_ty(&st.unsized_type());
                self.bracketed(trans);
                self.sizes(st, false);
            }
            T::Ordered => name("ordered", self, false),
            T::PositiveOrdered => name("positive_ordered", self, false),
            T::Simplex => name("simplex", self, false),
            T::UnitVector => name("unit_vector", self, false),
            T::SumToZero => {
                let s = if matches!(st, SizedType::Matrix(..)) {
                    "sum_to_zero_matrix"
                } else {
                    "sum_to_zero_vector"
                };
                name(s, self, false)
            }
            T::CholeskyCorr => name("cholesky_factor_corr", self, true),
            T::CholeskyCov => name("cholesky_factor_cov", self, true),
            T::Correlation => name("corr_matrix", self, true),
            T::Covariance => name("cov_matrix", self, true),
            T::StochasticColumn => name("column_stochastic_matrix", self, false),
            T::StochasticRow => name("row_stochastic_matrix", self, false),
            T::Tuple(ts) => {
                let SizedType::Tuple(types) = st else {
                    return self.sized(st);
                };
                self.f.string("tuple(");
                self.f.open(BoxKind::B, 0);
                for (i, (t, tr)) in types.iter().zip(ts).enumerate() {
                    if i > 0 {
                        self.f.string(",");
                        self.f.space();
                    }
                    self.transformed_type(t, tr);
                }
                if ts.len() == 1 {
                    self.f.string(",");
                }
                self.f.close();
                self.f.string(")");
            }
        }
    }

    // Statements.

    fn first_loc(&self, s: &Stmt) -> Loc {
        let span = match &s.kind {
            StmtKind::NRFunApp(id, _)
            | StmtKind::For { var: id, .. }
            | StmtKind::ForEach { var: id, .. }
            | StmtKind::FunDef { name: id, .. } => id.span,
            StmtKind::TargetPlusAssign(e)
            | StmtKind::JacobianPlusAssign(e)
            | StmtKind::Return(Some(e))
            | StmtKind::IfThenElse(e, _, _)
            | StmtKind::While(e, _) => e.span,
            StmtKind::VarDecl {
                ty, trans, vars, ..
            } => {
                let dim = match ty {
                    SizedType::Vector(e)
                    | SizedType::RowVector(e)
                    | SizedType::Matrix(e, _)
                    | SizedType::ComplexVector(e)
                    | SizedType::ComplexRowVector(e)
                    | SizedType::ComplexMatrix(e, _)
                    | SizedType::Array(_, e) => Some(e.span),
                    _ => None,
                };
                let tf = match trans {
                    Transformation::Lower(e)
                    | Transformation::Upper(e)
                    | Transformation::LowerUpper(e, _)
                    | Transformation::Offset(e)
                    | Transformation::Multiplier(e)
                    | Transformation::OffsetMultiplier(e, _) => Some(e.span),
                    _ => None,
                };
                dim.or(tf).unwrap_or(vars[0].ident.span)
            }
            _ => s.span,
        };
        self.begin(span)
    }

    fn statements(&mut self, ss: &[Stmt], block_end: &Loc) {
        self.f.open(BoxKind::V, 0);
        let mut prev: Option<Loc> = None;
        for s in ss {
            if let Some(p) = &prev {
                self.skip_comments(p);
            }
            let begin = self.first_loc(s);
            let cs = self.get_comments(&begin);
            self.spacing(true, prev.as_ref(), Some(&begin), cs);
            if !self.should_skip(&begin) {
                self.stmt(s);
            }
            prev = Some(self.end(s.span));
        }
        if let Some(p) = &prev {
            self.skip_comments(p);
        }
        let cs = self.get_comments(block_end);
        self.spacing(true, prev.as_ref(), None, cs);
        self.f.close();
    }

    fn indent_unless_block(&mut self, s: &Stmt, loc: &Loc) {
        if let StmtKind::Block(_) = s.kind {
            return self.stmt(s);
        }
        let begin = self.begin(s.span);
        let cs = self.get_comments(&begin);
        self.spacing(true, Some(loc), Some(&begin), cs);
        self.indented_box(0, |p| p.stmt(s));
    }

    fn if_chain(&mut self, s: &Stmt, loc: &Loc) {
        match &s.kind {
            StmtKind::IfThenElse(c, then, els) => {
                self.f.string("if (");
                self.expr(c);
                self.f.string(") ");
                let ce = self.end(c.span);
                self.indent_unless_block(then, &ce);
                if let Some(els) = els {
                    let newline = !matches!(then.kind, StmtKind::Block(_));
                    let mut loc = self.end(then.span);
                    let eb = self.begin(els.span);
                    let cs = self.take(&eb, Take::UntilSeparator);
                    self.spacing(newline, Some(&loc), Some(&loc), cs);
                    self.f.string("else ");
                    self.comments_spacing(Take::All, Fill::Nothing, Fill::Nothing, &eb);
                    loc.line += 1;
                    self.if_chain(els, &loc);
                }
            }
            _ => self.indent_unless_block(s, loc),
        }
    }

    fn printables(&mut self, ps: &[Printable]) {
        self.f.open(BoxKind::HOV, 0);
        for (i, p) in ps.iter().enumerate() {
            if i > 0 {
                self.f.string(",");
                self.f.space();
            }
            match p {
                Printable::Str(s) => self.f.string(s),
                Printable::Expr(e) => self.expr(e),
            }
        }
        self.f.close();
    }

    fn block_body(&mut self, ss: &[Stmt], span: Span) {
        self.f.cut();
        let end = self.end(span);
        self.indented_box(0, |p| p.statements(ss, &end));
        self.f.cut();
        self.f.string("}");
    }

    fn stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Assign { lhs, op, rhs } => {
                self.f.open(BoxKind::H, 0);
                self.lvalue(lhs);
                self.f.string(" ");
                match op {
                    AssignOp::Assign => self.f.string("="),
                    AssignOp::Compound(op) => self.f.string(&format!("{}=", op.symbol())),
                }
                self.f.string(" ");
                self.expr(rhs);
                self.f.string(";");
                self.f.close();
            }
            StmtKind::NRFunApp(id, args) => {
                self.start_funapp(id);
                self.f.string("(");
                self.f.cut();
                self.exprs(args, s.span);
                self.f.string(");");
                self.f.close();
            }
            StmtKind::TargetPlusAssign(e) => {
                self.f.string("target += ");
                self.expr(e);
                self.f.string(";");
            }
            StmtKind::JacobianPlusAssign(e) => {
                self.f.string("jacobian += ");
                self.expr(e);
                self.f.string(";");
            }
            StmtKind::Tilde {
                arg,
                dist,
                args,
                truncation,
            } => {
                self.expr(arg);
                self.f.string(" ~ ");
                self.start_funapp(dist);
                self.f.string("(");
                self.f.cut();
                self.exprs(args, s.span);
                self.f.string(")");
                self.f.close();
                match truncation {
                    Truncation::None => {}
                    Truncation::UpFrom(e) => {
                        self.f.string(" T[");
                        self.expr(e);
                        self.f.string(", ]");
                    }
                    Truncation::DownFrom(e) => {
                        self.f.string(" T[ , ");
                        self.expr(e);
                        self.f.string("]");
                    }
                    Truncation::Between(a, b) => {
                        self.f.string(" T[");
                        self.expr(a);
                        self.f.string(", ");
                        self.expr(b);
                        self.f.string("]");
                    }
                }
                self.f.string(";");
            }
            StmtKind::Break => self.f.string("break;"),
            StmtKind::Continue => self.f.string("continue;"),
            StmtKind::Return(Some(e)) => {
                self.f.string("return ");
                self.f.open(BoxKind::H, 0);
                self.expr(e);
                self.f.close();
                self.f.string(";");
            }
            StmtKind::Return(None) => self.f.string("return;"),
            StmtKind::Print(ps) | StmtKind::Reject(ps) | StmtKind::FatalError(ps) => {
                let name = match &s.kind {
                    StmtKind::Print(_) => "print",
                    StmtKind::Reject(_) => "reject",
                    _ => "fatal_error",
                };
                self.f.string(&format!("{name}("));
                self.printables(ps);
                self.f.string(");");
            }
            StmtKind::Skip => self.f.string(";"),
            StmtKind::IfThenElse(..) => {
                self.f.open(BoxKind::V, 0);
                let b = self.begin(s.span);
                self.if_chain(s, &b);
                self.f.close();
            }
            StmtKind::While(c, body) => {
                self.f.string("while (");
                self.expr(c);
                self.f.string(") ");
                self.stmt(body);
            }
            StmtKind::For {
                var,
                lower,
                upper,
                body,
            } => {
                self.f.open(BoxKind::V, 0);
                self.f.string(&format!("for ({} in ", var.name));
                self.expr(lower);
                self.f.string(" : ");
                self.expr(upper);
                self.f.string(") ");
                let ue = self.end(upper.span);
                self.indent_unless_block(body, &ue);
                self.f.close();
            }
            StmtKind::ForEach { var, iter, body } => {
                self.f.string(&format!("for ({} in ", var.name));
                self.expr(iter);
                self.f.string(") ");
                let ie = self.end(iter.span);
                self.indent_unless_block(body, &ie);
            }
            StmtKind::Block(ss) => {
                self.f.string("{");
                self.block_body(ss, s.span);
            }
            StmtKind::Profile(name, ss) => {
                self.f.string(&format!("profile({name}) {{"));
                self.block_body(ss, s.span);
            }
            StmtKind::VarDecl {
                ty, trans, vars, ..
            } => {
                self.f.open(BoxKind::H, 0);
                self.transformed_type(ty, trans);
                self.f.string(" ");
                self.list_of(
                    vars,
                    |v| v.ident.span,
                    |p, v| {
                        p.f.string(&v.ident.name);
                        if let Some(e) = &v.init {
                            p.f.string(" = ");
                            p.expr(e);
                        }
                    },
                    s.span,
                );
                self.f.string(";");
                self.f.close();
            }
            StmtKind::FunDef {
                ret,
                name,
                args,
                body,
            } => self.fundef(s, ret.as_ref(), name, args, body),
        }
    }

    fn fundef(
        &mut self,
        s: &Stmt,
        ret: Option<&UnsizedType>,
        name: &Ident,
        args: &[FunArg],
        body: &Stmt,
    ) {
        let margin = self.f.margin();
        let long = name.name.len() as isize > margin / 3;
        let many = args.len() as isize > margin / 10;
        let args_span = Span::new(s.span.file, s.span.start as usize, body.span.start as usize);
        let pp_args = |p: &mut Self| {
            p.list_of(
                args,
                |a| a.ident.span,
                |p, a| {
                    if a.ad == AdLevel::DataOnly {
                        p.f.string("data ");
                    }
                    p.unsized_ty(&a.ty);
                    p.f.string(" ");
                    p.f.string(&a.ident.name);
                },
                args_span,
            )
        };
        self.f.open(BoxKind::B, 0);
        match ret {
            Some(t) => self.unsized_ty(t),
            None => self.f.string("void"),
        }
        self.f.space();
        self.f.close();
        if long {
            self.f.open(BoxKind::B, 0);
            self.f.open(BoxKind::HV, 2);
            self.f.string(&name.name);
            self.f.string("(");
            self.f.cut();
            pp_args(self);
            self.f.close();
            self.f.cut();
            self.f.string(")");
            self.f.close();
        } else {
            self.f.string(&name.name);
            self.f.string("(");
            self.f.open(if many { BoxKind::HV } else { BoxKind::B }, 0);
            pp_args(self);
            self.f.string(")");
            self.f.close();
        }
        if let StmtKind::Skip = body.kind {
            self.f.string(";");
        } else {
            self.f.string(" ");
            self.stmt(body);
        }
    }
}

/// `pp_comment`'s `trim_tail`: the lines after the first lose the indentation they share, but
/// no more than the comment's own column. stanc3 measures each line by the index of its last
/// non-blank character, and this keeps that rule.
fn trim_tail(col: i64, lines: &[String]) -> Vec<String> {
    let Some((hd, tl)) = lines.split_first() else {
        return vec![];
    };
    let mut padding = (col - 2).max(0) as usize;
    for l in tl {
        if let Some(i) = l.bytes().rposition(|c| c != b' ') {
            padding = padding.min(i);
        }
    }
    let mut out = vec![hd.clone()];
    for l in tl {
        let mut k = padding.min(l.len());
        while !l.is_char_boundary(k) {
            k += 1;
        }
        out.push(l[k..].to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NoIncludes, parse_source};

    fn fmt(src: &str, line_length: isize) -> String {
        let (p, sources) = parse_source("t.stan", src, &NoIncludes).unwrap();
        let opts = Options {
            line_length,
            ..Options::default()
        };
        format_program(&p, &sources, &opts).unwrap()
    }

    // Expected outputs are those of stanc 2.40.0 `--auto-format`, without its final newline.
    #[test]
    fn keeps_comments_where_stanc3_does() {
        let src = "model {\n  real y;\n\n  real z; // hi\n  /* a */ real w;   /* b\n  */\n  \
                   y = 1 + // c\n  2;\n}\n";
        let want = "model {\n  real y;\n  \n  real z; // hi\n  /* a */\n  real w; /* b\n          \
                    */\n  y = 1 + // c\n      2;\n}\n";
        assert_eq!(fmt(src, 78), want);
    }

    #[test]
    fn breaks_long_lines_at_the_margin() {
        let src = "functions { real foo_bar2(real x, real y) { return 1.0; } }";
        let want = "functions {\n  real \n  foo_bar2(\n    real x,\n    real y\n  ) {\n    \
                    return \n    1.0;\n  }\n}\n";
        assert_eq!(fmt(src, 12), want);
    }
}
