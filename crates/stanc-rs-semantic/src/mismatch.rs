//! Explanations of signature mismatches, laid out as stanc3's `SignatureMismatch` prints them.

use std::cmp::Ordering;

use stanc_rs_syntax::ast::{AdLevel, FnSuffix, FunctionType, UnsizedType};
use stanc_rs_syntax::format::{BoxKind, Formatter};

use crate::types::{Details, Mismatch};

/// A candidate signature: return type and arguments.
pub type Sig = (Option<UnsizedType>, Vec<(AdLevel, UnsizedType)>);

/// Most signatures stanc3 lists before "(Additional signatures omitted)".
pub const MAX_ERRORS: usize = 5;

fn tag(t: &UnsizedType) -> u8 {
    use UnsizedType::*;
    match t {
        Int => 0,
        Real => 1,
        Vector => 2,
        Complex => 3,
        RowVector => 4,
        Matrix => 5,
        ComplexVector => 6,
        ComplexRowVector => 7,
        ComplexMatrix => 8,
        Array(_) => 9,
        Tuple(_) => 10,
        Function(_) => 11,
        MathLibraryFunction => 12,
    }
}

fn suffix_tag(s: FnSuffix) -> u8 {
    match s {
        FnSuffix::Plain => 0,
        FnSuffix::Rng => 1,
        FnSuffix::Lpdf => 2,
        FnSuffix::Lpmf => 3,
        FnSuffix::Lp => 4,
        FnSuffix::Jacobian => 5,
    }
}

/// stanc3's derived `UnsizedType.compare` (constructors in declaration order).
pub fn compare_type(a: &UnsizedType, b: &UnsizedType) -> Ordering {
    use UnsizedType::*;
    match (a, b) {
        (Array(x), Array(y)) => compare_type(x, y),
        (Tuple(xs), Tuple(ys)) => compare_list(xs, ys, compare_type),
        (Function(f), Function(g)) => compare_list(&f.args, &g.args, |x, y| {
            x.0.cmp(&y.0).then_with(|| compare_type(&x.1, &y.1))
        })
        .then_with(|| compare_return(&f.ret, &g.ret))
        .then_with(|| suffix_tag(f.suffix).cmp(&suffix_tag(g.suffix))),
        _ => tag(a).cmp(&tag(b)),
    }
}

fn compare_list<T>(xs: &[T], ys: &[T], f: impl Fn(&T, &T) -> Ordering) -> Ordering {
    for (x, y) in xs.iter().zip(ys) {
        let c = f(x, y);
        if c != Ordering::Equal {
            return c;
        }
    }
    xs.len().cmp(&ys.len())
}

/// stanc3's `compare_returntype`: `void` first.
pub fn compare_return(a: &Option<UnsizedType>, b: &Option<UnsizedType>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(x), Some(y)) => compare_type(x, y),
    }
}

fn count_dims(t: &UnsizedType) -> usize {
    use UnsizedType::*;
    match t {
        Array(t) => 1 + count_dims(t),
        Matrix | ComplexMatrix => 2,
        Vector | RowVector | ComplexVector | ComplexRowVector => 1,
        _ => 0,
    }
}

fn internal_scalar(t: &UnsizedType) -> &UnsizedType {
    use UnsizedType::*;
    match t {
        Vector | Matrix | RowVector | Real => &Real,
        Complex | ComplexVector | ComplexMatrix | ComplexRowVector => &Complex,
        Array(t) => internal_scalar(t),
        t => t,
    }
}

fn compare_types(x: &UnsizedType, t1: &UnsizedType, t2: &UnsizedType) -> Ordering {
    let d = count_dims(x);
    match (d == count_dims(t1), d == count_dims(t2)) {
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        _ => {}
    }
    let s = internal_scalar(x);
    match (s == internal_scalar(t1), s == internal_scalar(t2)) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        _ => compare_type(t1, t2),
    }
}

/// stanc3's `compare_errors`: more informative errors first.
pub fn compare_errors(e1: &Mismatch, e2: &Mismatch) -> Ordering {
    use Mismatch::*;
    match (e1, e2) {
        (ArgCount(..), ArgCount(..)) => Ordering::Equal,
        (Arg(..), ArgCount(..)) => Ordering::Less,
        (ArgCount(..), Arg(..)) => Ordering::Greater,
        (Arg(x, _), Arg(y, _)) if x != y => y.cmp(x),
        (Arg(_, a), Arg(_, b)) => match (&**a, &**b) {
            (DataOnly, DataOnly) => Ordering::Equal,
            (DataOnly, Type(..)) => Ordering::Less,
            (Type(..), DataOnly) => Ordering::Greater,
            (Type(t1, x1, None), Type(t2, x2, None)) => {
                compare_type(x1, x2).then_with(|| compare_types(x1, t1, t2))
            }
            (Type(_, _, Some(_)), Type(_, _, None)) => Ordering::Greater,
            (Type(_, _, None), Type(_, _, Some(_))) => Ordering::Less,
            (Type(_, _, Some(a)), Type(_, _, Some(b))) => match (&**a, &**b) {
                (Details::Suffix(..), Details::Suffix(..))
                | (Details::ReturnType(..), Details::ReturnType(..)) => Ordering::Equal,
                (Details::Input(a), Details::Input(b)) => compare_errors(a, b),
                (Details::Suffix(..), _) | (_, Details::Input(..)) => Ordering::Less,
                (Details::Input(..), _) | (_, Details::Suffix(..)) => Ordering::Greater,
            },
            _ => Ordering::Equal,
        },
        _ => Ordering::Equal,
    }
}

/// Opaque names (`F1`, `F2`, ...) of the function types printed so far.
#[derive(Default)]
struct Names(Vec<(UnsizedType, String)>);

impl Names {
    fn get(&mut self, t: &UnsizedType) -> String {
        if let Some((_, n)) = self.0.iter().find(|(u, _)| u == t) {
            return n.clone();
        }
        let n = format!("F{}", self.0.len() + 1);
        self.0.push((t.clone(), n.clone()));
        n
    }
}

struct P {
    f: Formatter,
    names: Names,
}

/// `UnsizedType.pp`.
fn pp_plain(f: &mut Formatter, t: &UnsizedType) {
    match t {
        UnsizedType::Array(_) => {
            let (n, inner) = t.array_dims();
            f.string(&format!("array[{}] ", ",".repeat(n - 1)));
            pp_plain(f, inner);
        }
        UnsizedType::Tuple(ts) => {
            f.string("tuple(");
            f.open(BoxKind::B, 0);
            comma_list(f, ts, pp_plain);
            if ts.len() == 1 {
                f.string(",");
            }
            f.close();
            f.string(")");
        }
        UnsizedType::Function(ft) => {
            f.open(BoxKind::H, 0);
            f.string("(");
            comma_list(f, &ft.args, |f, (ad, t)| {
                if *ad == AdLevel::DataOnly && !matches!(t, UnsizedType::Function(_)) {
                    f.string("data ");
                }
                pp_plain(f, t);
            });
            f.string(") => ");
            match &ft.ret {
                Some(t) => pp_plain(f, t),
                None => f.string("void"),
            }
            f.close();
        }
        UnsizedType::MathLibraryFunction => f.string("<Stan Math function>"),
        t => f.string(&t.to_string()),
    }
}

fn comma_list<T>(f: &mut Formatter, xs: &[T], mut pp: impl FnMut(&mut Formatter, &T)) {
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            f.string(",");
            f.space();
        }
        pp(f, x);
    }
}

impl P {
    /// `pp_unsized_type`: function types by name.
    fn ty(&mut self, t: &UnsizedType) {
        match t {
            UnsizedType::Array(_) => {
                let (n, inner) = t.array_dims();
                self.f.string(&format!("array[{}] ", ",".repeat(n - 1)));
                self.ty(inner);
            }
            UnsizedType::Function(_) => {
                let n = self.names.get(t);
                self.f.string(&format!("<{n}>"));
            }
            t => pp_plain(&mut self.f, t),
        }
    }

    fn args(&mut self, args: &[(AdLevel, UnsizedType)]) {
        for (i, (ad, t)) in args.iter().enumerate() {
            if i > 0 {
                self.f.string(",");
                self.f.space();
            }
            if *ad == AdLevel::DataOnly {
                self.f.string("data ");
            }
            self.ty(t);
        }
    }

    fn fundef(&mut self, ft: &FunctionType) {
        self.f.open(BoxKind::HOV, 0);
        self.f.string("(");
        self.f.open(BoxKind::HOV, 0);
        self.args(&ft.args);
        self.f.close();
        self.f.string(") => ");
        match &ft.ret {
            Some(t) => self.ty(t),
            None => self.f.string("void"),
        }
        self.f.close();
    }

    fn fundef_ty(&mut self, t: &UnsizedType) {
        match t {
            UnsizedType::Function(ft) => self.fundef(ft),
            t => self.ty(t),
        }
    }

    /// `pp_with_where`: `body`, then the definitions of the function names it introduced.
    fn with_where(&mut self, body: impl FnOnce(&mut P)) {
        let old = self.names.0.len();
        self.f.open(BoxKind::V, 0);
        body(self);
        let (old, new) = self.new_names(old);
        if !new.is_empty() {
            self.f.cut();
            self.f.string("where ");
            self.f.open(BoxKind::V, 0);
            self.where_defs(old, new);
            self.f.close();
        }
        self.f.close();
    }

    fn new_names(&self, old: usize) -> (usize, Vec<(UnsizedType, String)>) {
        let mut new = self.names.0[old..].to_vec();
        new.sort_by(|a, b| a.1.cmp(&b.1));
        (self.names.0.len(), new)
    }

    fn where_defs(&mut self, old: usize, new: Vec<(UnsizedType, String)>) {
        for (i, (t, n)) in new.iter().enumerate() {
            if i > 0 {
                self.f.cut();
            }
            self.f.string(n);
            self.f.string(" = ");
            self.f.open(BoxKind::HOV, 0);
            self.fundef_ty(t);
            self.f.close();
        }
        let (old, new) = self.new_names(old);
        if !new.is_empty() {
            self.f.cut();
            self.where_defs(old, new);
        }
    }

    fn data_only_msg(&mut self) {
        self.f.string("(");
        self.f.text(
            "Local variables are assumed to depend on parameters; same goes for function inputs unless they are marked with the keyword",
        );
        self.f.space();
        self.f.string("\"data\"");
        self.f.string(".)");
    }

    fn explain(&mut self, e: &Mismatch) {
        let Mismatch::Arg(n, inner) = e else {
            if let Mismatch::ArgCount(x, y) = e {
                self.f.string(&format!(
                    "Expected {x} {} but found {y} {}.",
                    arguments(*x),
                    arguments(*y)
                ));
            }
            return;
        };
        let idx = index_str(*n);
        match &**inner {
            Mismatch::DataOnly => {
                self.f
                    .pf(&format!("@[<hov>The@ {idx}@ argument must be data-only.@ "));
                self.data_only_msg();
                self.f.close();
            }
            Mismatch::Type(x, y, None) => {
                self.f.pf(&format!("@[<hv>The {idx} argument must be@, "));
                self.ty(x);
                self.f.pf("@ but got@, ");
                self.ty(y);
                self.f.close();
            }
            Mismatch::Type(_, _, Some(d)) if matches!(**d, Details::Suffix(..)) => {
                let Details::Suffix(x, y) = **d else {
                    unreachable!()
                };
                self.f.pf(&format!(
                    "@[<v>The {idx} argument must be {} but got {}. These function types are not compatible.@]",
                    suffix_str(x),
                    suffix_str(y)
                ));
            }
            Mismatch::Type(x, y, Some(d)) => {
                self.f.pf(&format!("@[<v>The {idx} argument must be@, "));
                self.fundef_ty(x);
                self.f.pf("@ but got@, ");
                self.fundef_ty(y);
                self.f.space();
                match &**d {
                    Details::Input(err) => {
                        self.f
                            .pf("@[<v 2>These are not compatible because:@ @[<hov>");
                        self.explain_rec(err);
                        self.f.pf("@]@]");
                    }
                    _ => self.f.string("The return types are not compatible."),
                }
                self.f.close();
            }
            _ => {}
        }
    }

    fn explain_rec(&mut self, e: &Mismatch) {
        let Mismatch::Arg(n, inner) = e else {
            if let Mismatch::ArgCount(x, y) = e {
                self.f.string(&format!(
                    "One takes {x} {} but the other takes {y} {}.",
                    arguments(*x),
                    arguments(*y)
                ));
            }
            return;
        };
        let idx = index_str(*n);
        match &**inner {
            Mismatch::DataOnly => {
                self.f.pf(&format!("@[<hov>The@ {idx}@ argument"));
                self.f.text(" has an incompatible data-qualifier.");
                self.f.close();
            }
            Mismatch::Type(x, y, None) => {
                self.f.pf(&format!(
                    "@[<hv>The types for the {idx} argument are incompatible: one is@, "
                ));
                self.ty(x);
                self.f.pf("@ but the other is@, ");
                self.ty(y);
                self.f.close();
            }
            Mismatch::Type(_, _, Some(d)) if matches!(**d, Details::Suffix(..)) => {
                let Details::Suffix(x, y) = **d else {
                    unreachable!()
                };
                self.f.pf(&format!(
                    "@[<v>The {idx} argument is {} but the other is {}. These function types are not compatible.@]",
                    suffix_str(x),
                    suffix_str(y)
                ));
            }
            Mismatch::Type(x, y, Some(d)) => {
                match &**d {
                    Details::Input(_) => self.f.pf(&format!(
                        "@[<v>The types for the {idx} argument are incompatible: one is@, "
                    )),
                    _ => self.f.pf(&format!("@[<v>The {idx} argument must be@, ")),
                }
                self.fundef_ty(x);
                match &**d {
                    Details::Input(_) => self.f.pf("@ but the other is@, "),
                    _ => self.f.pf("@ but got@, "),
                }
                self.fundef_ty(y);
                self.f.space();
                match &**d {
                    Details::Input(err) => {
                        self.f.pf("@[<v>These are not compatible because:@ @[<hov>");
                        self.explain_rec(err);
                        self.f.pf("@]@]");
                    }
                    _ => self.f.string("The return types are different."),
                }
                self.f.close();
            }
            _ => {}
        }
    }
}

fn arguments(n: usize) -> &'static str {
    if n == 1 { "argument" } else { "arguments" }
}

fn index_str(n: usize) -> String {
    match n {
        1 => "first".into(),
        2 => "second".into(),
        3 => "third".into(),
        4 => "fourth".into(),
        n => {
            let s = match (n % 10, n % 100) {
                (1, x) if x != 11 => "st",
                (2, x) if x != 12 => "nd",
                (3, x) if x != 13 => "rd",
                _ => "th",
            };
            format!("{n}{s}")
        }
    }
}

fn suffix_str(s: FnSuffix) -> &'static str {
    match s {
        FnSuffix::Plain => "a pure function",
        FnSuffix::Rng => "an rng function",
        FnSuffix::Lpdf => "a probability density function",
        FnSuffix::Lpmf => "a probability mass function",
        FnSuffix::Lp => "an _lp function",
        FnSuffix::Jacobian => "a _jacobian function",
    }
}

/// stanc3's `pp_signature_mismatch`: the supplied argument types, then each candidate signature
/// with why it does not match.
pub fn signature_mismatch(
    name: &str,
    arg_tys: &[UnsizedType],
    sigs: &[(Sig, Mismatch)],
    omitted: bool,
) -> String {
    let mut p = P {
        f: Formatter::new(),
        names: Names::default(),
    };
    p.f.open(BoxKind::V, 0);
    p.f.pf(&format!(
        "Ill-typed arguments supplied to function @[<1>\"{name}\"@]:"
    ));
    p.f.space();
    p.with_where(|p| {
        p.f.string("(");
        p.f.open(BoxKind::HOV, 0);
        for (i, t) in arg_tys.iter().enumerate() {
            if i > 0 {
                p.f.string(",");
                p.f.space();
            }
            p.ty(t);
        }
        p.f.close();
        p.f.string(")");
    });
    p.f.space();
    p.f.string("Available signatures:");
    p.f.space();
    for (i, ((ret, args), e)) in sigs.iter().enumerate() {
        if i > 0 {
            p.f.cut();
        }
        let ft = FunctionType {
            ret: ret.clone(),
            args: args.clone(),
            suffix: FnSuffix::Plain,
        };
        p.with_where(|p| p.fundef(&ft));
        p.f.space();
        p.f.open(BoxKind::HOV, 2);
        p.f.string("  ");
        p.explain(e);
        p.f.close();
    }
    if omitted {
        p.f.cut();
        p.f.string("(Additional signatures omitted)");
    }
    p.f.close();
    p.f.finish()
}

/// Orders signature errors as stanc3 does (candidates sorted by return type, then errors by
/// [`compare_errors`]), keeping at most [`MAX_ERRORS`]. Returns whether some were omitted.
pub fn select(mut errs: Vec<(Sig, Mismatch)>) -> (Vec<(Sig, Mismatch)>, bool) {
    errs.sort_by(|a, b| compare_return(&a.0.0, &b.0.0));
    errs.sort_by(|a, b| compare_errors(&a.1, &b.1));
    let omitted = errs.len() > MAX_ERRORS;
    errs.truncate(MAX_ERRORS);
    (errs, omitted)
}

/// stanc3's `pp_mismatch_details` for an argument list mismatch, with no skipped arguments.
pub fn input_mismatch(e: &Mismatch) -> String {
    let mut p = P {
        f: Formatter::new(),
        names: Names::default(),
    };
    match e {
        Mismatch::ArgCount(x, y) => p.f.pf(&format!(
            "@[<hov>Expected {x} {}@ but got {y} {}.@]",
            arguments(*x),
            arguments(*y)
        )),
        Mismatch::Arg(n, inner) => {
            let idx = index_str(*n);
            match &**inner {
                Mismatch::DataOnly => {
                    p.f.pf(&format!("@[<hov>The@ {idx} argument is marked data-only. "));
                    p.data_only_msg();
                    p.f.close();
                }
                Mismatch::Type(x, y, _) => p.with_where(|p| {
                    p.f.pf(&format!("@[<hov>The {idx} argument must be@ "));
                    p.ty(x);
                    p.f.pf(" but got@ ");
                    p.ty(y);
                    p.f.pf(".@]");
                }),
                _ => {}
            }
        }
        _ => {}
    }
    p.f.finish()
}
