//! The backend preparation of the MIR, a port of stanc3 v2.40.0's `Transform_Mir` (without
//! the OpenCL rewrites): C++ keyword mangling, reading data and parameters, writing
//! parameters, and the `transform_inits` and `unconstrain_array` blocks.

use std::collections::{HashMap, HashSet};

use stanc_rs_semantic::types::Ad;
use stanc_rs_syntax::ast::UnsizedType;

use crate::lower::{Gensym, add_int_index, add_tuple_index, fill_ad};
use crate::mir::*;

const KWRDS_PREFIX: &str = "_stan_";
const POS: &str = "pos__";

/// C++ keywords and names that are not Stan keywords (`Mangle.cpp_kwrds`).
const CPP_KWRDS: &[&str] = &[
    "alignas",
    "alignof",
    "and",
    "and_eq",
    "asm",
    "bitand",
    "bitor",
    "bool",
    "case",
    "catch",
    "char",
    "char16_t",
    "char32_t",
    "class",
    "compl",
    "const",
    "constexpr",
    "const_cast",
    "decltype",
    "default",
    "delete",
    "do",
    "double",
    "dynamic_cast",
    "enum",
    "explicit",
    "float",
    "friend",
    "goto",
    "inline",
    "long",
    "mutable",
    "namespace",
    "new",
    "noexcept",
    "not",
    "not_eq",
    "nullptr",
    "operator",
    "or",
    "or_eq",
    "private",
    "protected",
    "public",
    "register",
    "reinterpret_cast",
    "short",
    "signed",
    "sizeof",
    "static_assert",
    "static_cast",
    "switch",
    "template",
    "this",
    "thread_local",
    "throw",
    "try",
    "typeid",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "volatile",
    "wchar_t",
    "xor",
    "xor_eq",
    "fvar",
    "STAN_MAJOR",
    "STAN_MINOR",
    "STAN_PATCH",
    "STAN_MATH_MAJOR",
    "STAN_MATH_MINOR",
    "STAN_MATH_PATCH",
    "BSD",
    "BSD4_2",
    "BSD4_3",
    "BSD4_4",
    "EMSCRIPTEN",
    "hpux",
    "sun",
    "linux",
    "VMS",
    "i386",
    "mips",
    "WIN32",
    "clamp",
    "max",
    "min",
    "minmax",
];

/// `Mangle.add_prefix_to_kwrds`.
pub fn add_prefix_to_kwrds(s: &str) -> String {
    if CPP_KWRDS.contains(&s) {
        format!("{KWRDS_PREFIX}{s}")
    } else {
        s.to_string()
    }
}

/// `Mangle.remove_prefix`.
pub fn remove_prefix(s: &str) -> &str {
    s.strip_prefix(KWRDS_PREFIX).unwrap_or(s)
}

fn rename_expr(e: &mut Expr) {
    for c in e.children_mut() {
        rename_expr(c);
    }
    match &mut e.pattern {
        ExprP::FunApp(FunKind::UserDefined(f, _), _) | ExprP::Var(f) => {
            *f = add_prefix_to_kwrds(f);
        }
        _ => {}
    }
}

fn rename_lbase(lv: &mut LValue) {
    match &mut lv.base {
        LBase::Variable(v) => *v = add_prefix_to_kwrds(v),
        LBase::TupleProjection(inner, _) => rename_lbase(inner),
    }
}

fn rename_stmt(s: &mut Stmt) {
    for e in s.exprs_mut() {
        rename_expr(e);
    }
    match &mut s.pattern {
        StmtP::Decl { id, .. } => *id = add_prefix_to_kwrds(id),
        StmtP::NRFunApp(FunKind::UserDefined(f, _), _) => *f = add_prefix_to_kwrds(f),
        StmtP::Assignment(lv, ..) => rename_lbase(lv),
        StmtP::For { loopvar, .. } => *loopvar = add_prefix_to_kwrds(loopvar),
        _ => {}
    }
    for c in s.children_mut() {
        rename_stmt(c);
    }
}

fn sized_exprs_mut(st: &mut SizedType, f: &mut impl FnMut(&mut Expr)) {
    for e in st.exprs_mut() {
        f(e);
    }
}

/// Functions returning Eigen block expressions (`eigen_block_expr_fns`).
const EIGEN_BLOCK_EXPR_FNS: &[&str] = &[
    "head", "tail", "segment", "col", "row", "block", "sub_row", "sub_col",
];

fn is_eigen_type(t: &UnsizedType) -> bool {
    use UnsizedType::*;
    matches!(
        t,
        Vector | RowVector | Matrix | ComplexRowVector | ComplexVector | ComplexMatrix
    )
}

/// `break_eigen_cycles`: wraps Eigen arguments of recursive calls in `eval()` so that C++
/// template expansion terminates.
fn break_eigen_cycles(functions: Vec<FunDef>) -> Vec<FunDef> {
    let mut callgraph: HashMap<String, HashSet<String>> = HashMap::new();
    let mut out = Vec::new();
    for mut fd in functions {
        let fun_args: HashSet<String> = fd
            .args
            .iter()
            .filter(|(_, _, t)| is_eigen_type(t))
            .map(|(_, n, _)| n.clone())
            .collect();
        if fun_args.is_empty() {
            out.push(fd);
            continue;
        }
        let mut calls: HashSet<String> = HashSet::new();
        let cx = EigenCx {
            fun_args: &fun_args,
            fdname: &fd.name,
            callgraph: &callgraph,
        };
        if let Some(body) = &mut fd.body {
            cx.stmt(body, &mut calls);
        }
        if !calls.is_empty() {
            for x in callgraph.values_mut() {
                if x.contains(&fd.name) {
                    x.extend(calls.iter().cloned());
                }
            }
            callgraph.entry(fd.name.clone()).or_default().extend(calls);
        }
        out.push(fd);
    }
    out
}

struct EigenCx<'a> {
    fun_args: &'a HashSet<String>,
    fdname: &'a str,
    callgraph: &'a HashMap<String, HashSet<String>>,
}

impl EigenCx<'_> {
    fn is_potentially_recursive(&self, e: &Expr) -> bool {
        match &e.pattern {
            ExprP::Var(n) => self.fun_args.contains(n),
            ExprP::Indexed(e, _) => self.is_potentially_recursive(e),
            ExprP::FunApp(FunKind::StanLib(f, _), args) if !args.is_empty() => {
                EIGEN_BLOCK_EXPR_FNS.contains(&f.as_str())
                    && self.is_potentially_recursive(&args[0])
            }
            _ => false,
        }
    }

    fn map_args(&self, name: &str, args: &mut [Expr], calls: &mut HashSet<String>) {
        for a in args.iter_mut() {
            self.expr(a, calls);
        }
        if !args.iter().any(|a| self.is_potentially_recursive(a)) {
            return;
        }
        let eval = if name == self.fdname {
            true
        } else {
            match self.callgraph.get(name) {
                Some(nested) if nested.contains(self.fdname) => true,
                Some(nested) => {
                    calls.insert(name.to_string());
                    calls.extend(nested.iter().cloned());
                    false
                }
                None => {
                    calls.insert(name.to_string());
                    false
                }
            }
        };
        if eval {
            for a in args.iter_mut() {
                if self.is_potentially_recursive(a) {
                    let meta = a.meta.clone();
                    let inner = std::mem::replace(a, Expr::int(0));
                    *a = Expr::new(
                        ExprP::FunApp(
                            FunKind::StanLib("eval".to_string(), Suffix::Plain),
                            vec![inner],
                        ),
                        meta,
                    );
                }
            }
        }
    }

    fn expr(&self, e: &mut Expr, calls: &mut HashSet<String>) {
        match &mut e.pattern {
            ExprP::FunApp(FunKind::UserDefined(name, _), args) => {
                let name = name.clone();
                self.map_args(&name, args, calls);
            }
            ExprP::FunApp(FunKind::StanLib(..), args)
                if matches!(
                    args.first(),
                    Some(Expr { pattern: ExprP::Var(_), meta }) if matches!(meta.ty, UnsizedType::Function(_))
                ) =>
            {
                let ExprP::Var(name) = &args[0].pattern else {
                    unreachable!()
                };
                let name = name.clone();
                self.map_args(&name, &mut args[1..], calls);
            }
            _ => {
                for c in e.children_mut() {
                    self.expr(c, calls);
                }
            }
        }
    }

    fn stmt(&self, s: &mut Stmt, calls: &mut HashSet<String>) {
        if let StmtP::NRFunApp(FunKind::UserDefined(name, _), args) = &mut s.pattern {
            let name = name.clone();
            self.map_args(&name, args, calls);
            return;
        }
        for e in s.exprs_mut() {
            self.expr(e, calls);
        }
        for c in s.children_mut() {
            self.stmt(c, calls);
        }
    }
}

// Sized type helpers.

fn base_type(st: &SizedType) -> UnsizedType {
    match st {
        SizedType::Array(t, _) => base_type(t),
        SizedType::Vector(_) | SizedType::RowVector(_) | SizedType::Matrix(..) => UnsizedType::Real,
        SizedType::ComplexVector(_)
        | SizedType::ComplexRowVector(_)
        | SizedType::ComplexMatrix(..) => UnsizedType::Complex,
        x => x.to_unsized(),
    }
}

fn contains_tuple(st: &SizedType) -> bool {
    match st {
        SizedType::Tuple(_) => true,
        SizedType::Array(t, _) => contains_tuple(t),
        _ => false,
    }
}

fn ut_contains_tuple(t: &UnsizedType) -> bool {
    match t {
        UnsizedType::Tuple(_) => true,
        UnsizedType::Array(t) => ut_contains_tuple(t),
        _ => false,
    }
}

fn is_recursive_container(st: &SizedType) -> bool {
    match st {
        SizedType::Array(t, _) => !matches!(**t, SizedType::Int | SizedType::Real),
        SizedType::Tuple(_) => true,
        _ => false,
    }
}

/// `SizedType.get_dims`.
fn get_dims(st: &SizedType) -> Vec<Expr> {
    use SizedType::*;
    match st {
        Tuple(_) | Int | Real | Complex => vec![],
        Matrix(a, b) | ComplexMatrix(a, b) => vec![a.clone(), b.clone()],
        RowVector(d) | Vector(d) | ComplexRowVector(d) | ComplexVector(d) => vec![d.clone()],
        Array(t, d) => {
            let mut v = vec![d.clone()];
            v.extend(get_dims(t));
            v
        }
    }
}

/// `SizedType.get_scalar_and_dims`.
fn get_scalar_and_dims(st: &SizedType) -> (SizedType, Vec<Expr>) {
    use SizedType::*;
    match st {
        Int | Real | Complex | Tuple(_) => (st.clone(), vec![]),
        Vector(d) | RowVector(d) | ComplexVector(d) | ComplexRowVector(d) => {
            (st.clone(), vec![d.clone()])
        }
        Matrix(a, b) | ComplexMatrix(a, b) => (st.clone(), vec![a.clone(), b.clone()]),
        Array(t, d) => {
            let (s, mut dims) = get_scalar_and_dims(t);
            dims.insert(0, d.clone());
            (s, dims)
        }
    }
}

fn get_array_dims(st: &SizedType) -> (SizedType, Vec<Expr>) {
    let (t, dims) = st.array_dims();
    (t.clone(), dims.into_iter().cloned().collect())
}

/// `SizedType.internal_scalar`.
fn internal_scalar(st: &SizedType) -> SizedType {
    use SizedType::*;
    match st {
        Int | Real | Complex | Tuple(_) => st.clone(),
        Vector(_) | RowVector(_) | Matrix(..) => Real,
        ComplexVector(_) | ComplexRowVector(_) | ComplexMatrix(..) => Complex,
        Array(t, _) => internal_scalar(t),
    }
}

fn ut_internal_scalar(t: &UnsizedType) -> UnsizedType {
    use UnsizedType::*;
    match t {
        Vector | Matrix | RowVector | Real => Real,
        Int => Int,
        Complex | ComplexVector | ComplexMatrix | ComplexRowVector => Complex,
        Array(t) => ut_internal_scalar(t),
        t => t.clone(),
    }
}

fn binop_list(es: Vec<Expr>, op: Operator, default: Expr) -> Expr {
    let mut it = es.into_iter();
    match it.next() {
        None => default,
        Some(h) => it.fold(h, |acc, e| Expr::binop(acc, op, e)),
    }
}

/// `SizedType.io_size`.
fn io_size(st: &SizedType) -> Expr {
    use SizedType::*;
    let two = || Expr::int(2);
    match st {
        Int | Real => Expr::int(1),
        Tuple(ts) => binop_list(
            ts.iter().map(io_size).collect(),
            Operator::Plus,
            Expr::int(0),
        ),
        Complex => two(),
        Vector(d) | RowVector(d) => d.clone(),
        Matrix(a, b) => Expr::binop(a.clone(), Operator::Times, b.clone()),
        ComplexVector(d) | ComplexRowVector(d) => Expr::binop(d.clone(), Operator::Times, two()),
        ComplexMatrix(a, b) => Expr::binop(
            a.clone(),
            Operator::Times,
            Expr::binop(b.clone(), Operator::Times, two()),
        ),
        Array(t, d) if matches!(**t, Real | Int) => d.clone(),
        Array(t, d) => Expr::binop(d.clone(), Operator::Times, io_size(t)),
    }
}

/// `SizedType.flatten_tuple_io`.
fn flatten_tuple_io(st: &SizedType) -> Vec<SizedType> {
    match st {
        SizedType::Tuple(ts) => ts.iter().flat_map(flatten_tuple_io).collect(),
        SizedType::Array(..) if contains_tuple(st) => {
            let (tupl, dims) = get_array_dims(st);
            flatten_tuple_io(&tupl)
                .into_iter()
                .map(|t| SizedType::build_array(dims.clone(), t))
                .collect()
        }
        _ => vec![st.clone()],
    }
}

/// `UnsizedType.enumerate_tuple_names_io`.
fn enumerate_tuple_names_io(name: &str, ut: &UnsizedType) -> Vec<String> {
    match ut {
        UnsizedType::Tuple(ts) => ts
            .iter()
            .enumerate()
            .flat_map(|(i, t)| enumerate_tuple_names_io(&format!("{name}.{}", i + 1), t))
            .collect(),
        UnsizedType::Array(_) if ut_contains_tuple(ut) => {
            let mut t = ut;
            while let UnsizedType::Array(inner) = t {
                t = inner;
            }
            enumerate_tuple_names_io(name, t)
        }
        _ => vec![name.to_string()],
    }
}

fn wind_array_type(t: UnsizedType, n: usize) -> UnsizedType {
    (0..n).fold(t, |t, _| UnsizedType::Array(Box::new(t)))
}

fn meta_from_sizedtype(st: &SizedType) -> Meta {
    let ty = st.to_unsized();
    Meta {
        ad: fill_ad(&Ad::DataOnly, &ty),
        ty,
        ..Meta::empty()
    }
}

fn munge_tuple_name(name: &str) -> String {
    name.replace('.', "_dot_")
}

fn make_tuple_temp(name: &str) -> String {
    format!("{}_temp__", munge_tuple_name(name))
}

fn var(name: &str) -> Expr {
    Expr::new(ExprP::Var(name.to_string()), Meta::empty())
}

fn typed_var(name: &str, ty: UnsizedType, ad: Ad, loc: Loc) -> Expr {
    Expr::new(ExprP::Var(name.to_string()), Meta { ty, loc, ad })
}

fn s(pattern: StmtP, loc: Loc) -> Stmt {
    Stmt::new(pattern, loc)
}

fn decl(ad: Ad, id: &str, ty: DeclType, init: DeclInit, loc: Loc) -> Stmt {
    s(
        StmtP::Decl {
            ad,
            id: id.to_string(),
            ty,
            init,
        },
        loc,
    )
}

fn with_idx(lv: &LValue, extra: Vec<Index<Expr>>) -> LValue {
    let mut lv = lv.clone();
    lv.idx.extend(extra);
    lv
}

fn tuple_proj(lv: &LValue, i: usize) -> LValue {
    LValue {
        base: LBase::TupleProjection(Box::new(lv.clone()), i),
        idx: vec![],
    }
}

fn singles_rev(loopvars: &[Expr]) -> Vec<Index<Expr>> {
    loopvars
        .iter()
        .rev()
        .map(|e| Index::Single(e.clone()))
        .collect()
}

fn tuple_expr(es: Vec<Expr>) -> Expr {
    let ty = UnsizedType::Tuple(es.iter().map(|e| e.meta.ty.clone()).collect());
    let ad = Ad::Tuple(es.iter().map(|e| e.meta.ad.clone()).collect());
    Expr::internal(
        Internal::MakeTuple,
        es,
        Meta {
            ty,
            ad,
            ..Meta::empty()
        },
    )
}

fn write_param(unconstrain_opt: Option<Transformation>, var: Expr) -> Stmt {
    s(
        StmtP::NRFunApp(
            FunKind::CompilerInternal(Internal::WriteParam {
                unconstrain_opt: unconstrain_opt.map(Box::new),
                var: Box::new(var),
            }),
            vec![],
        ),
        Loc::EMPTY,
    )
}

type Body<'a> = &'a dyn Fn(&T, Vec<Expr>) -> Stmt;
type ScalarBody<'a> = &'a dyn Fn(&T, &SizedType, Expr) -> Stmt;

#[derive(Default)]
struct T {
    sym: Gensym,
}

impl T {
    /// `Stmt.Helpers.mk_for`.
    fn mk_for(&self, upper: Expr, body: &dyn Fn(&T, Expr) -> Stmt, l: Loc) -> Stmt {
        let (loopvar, old) = self.sym.enter();
        let lv = Expr::new(
            ExprP::Var(loopvar.clone()),
            Meta {
                ty: UnsizedType::Int,
                loc: l,
                ad: Ad::DataOnly,
            },
        );
        let inner = body(self, lv);
        let body = s(StmtP::Block(vec![inner]), l);
        self.sym.reset(old);
        s(
            StmtP::For {
                loopvar,
                lower: Expr::int(1),
                upper,
                body: Box::new(body),
            },
            l,
        )
    }

    /// `Stmt.Helpers.mk_nested_for`.
    fn mk_nested_for(&self, uppers: &[Expr], body: Body, l: Loc) -> Stmt {
        match uppers {
            [] => body(self, vec![]),
            [upper, rest @ ..] => self.mk_for(
                upper.clone(),
                &|me: &T, lv: Expr| {
                    me.mk_nested_for(
                        rest,
                        &|me: &T, mut lvs: Vec<Expr>| {
                            lvs.insert(0, lv.clone());
                            body(me, lvs)
                        },
                        Loc::EMPTY,
                    )
                },
                l,
            ),
        }
    }

    fn mk_for_iteratee(
        &self,
        upper: Expr,
        body: &dyn Fn(&T, Expr) -> Stmt,
        it: Expr,
        l: Loc,
    ) -> Stmt {
        self.mk_for(
            upper,
            &|me: &T, lv: Expr| body(me, add_int_index(it.clone(), Index::Single(lv))),
            l,
        )
    }

    /// `Stmt.Helpers.for_scalar`.
    fn for_scalar(&self, st: &SizedType, body: ScalarBody, v: Expr, l: Loc) -> Stmt {
        use SizedType::*;
        match st {
            Int | Real | Complex | Tuple(_) => body(self, st, v),
            Vector(d) | RowVector(d) | ComplexVector(d) | ComplexRowVector(d) => {
                self.mk_for_iteratee(d.clone(), &|me: &T, e: Expr| body(me, st, e), v, l)
            }
            Matrix(d1, d2) => {
                let row = RowVector(d2.clone());
                self.mk_for_iteratee(
                    d1.clone(),
                    &|me: &T, e: Expr| me.for_scalar(&row, body, e, l),
                    v,
                    l,
                )
            }
            ComplexMatrix(d1, d2) => {
                let row = ComplexRowVector(d2.clone());
                self.mk_for_iteratee(
                    d1.clone(),
                    &|me: &T, e: Expr| me.for_scalar(&row, body, e, l),
                    v,
                    l,
                )
            }
            Array(t, d) => self.mk_for_iteratee(
                d.clone(),
                &|me: &T, e: Expr| me.for_scalar(t, body, e, l),
                v,
                l,
            ),
        }
    }

    /// `Stmt.Helpers.for_scalar_inv`: like `for_scalar`, with the array dimensions iterated
    /// in the inverted order.
    fn for_scalar_inv(&self, st: &SizedType, body: ScalarBody, v: Expr, l: Loc) -> Stmt {
        let meta = v.meta.clone();
        let v = Expr::new(ExprP::Indexed(Box::new(v), vec![]), meta);
        let invert = |e: Expr| match e.pattern {
            ExprP::Indexed(obj, idx) if idx.is_empty() => *obj,
            ExprP::Indexed(obj, mut idx) => {
                idx.reverse();
                Expr::new(ExprP::Indexed(obj, idx), e.meta)
            }
            _ => e,
        };
        self.for_scalar_inv_go(
            st,
            &|me: &T, st: &SizedType, v: Expr| body(me, st, invert(v)),
            v,
            l,
        )
    }

    fn for_scalar_inv_go(&self, st: &SizedType, body: ScalarBody, v: Expr, l: Loc) -> Stmt {
        use SizedType::*;
        match st {
            Array(t, d) => self.for_scalar_inv_go(
                t,
                &|me: &T, _: &SizedType, v: Expr| {
                    me.mk_for_iteratee(d.clone(), &|me: &T, e: Expr| body(me, st, e), v, l)
                },
                v,
                l,
            ),
            Matrix(d1, d2) => self.for_scalar_inv_go(
                &RowVector(d2.clone()),
                &|me: &T, _: &SizedType, v: Expr| {
                    me.mk_for_iteratee(d1.clone(), &|me: &T, e: Expr| body(me, st, e), v, l)
                },
                v,
                l,
            ),
            ComplexMatrix(d1, d2) => self.for_scalar_inv_go(
                &ComplexRowVector(d2.clone()),
                &|me: &T, _: &SizedType, v: Expr| {
                    me.mk_for_iteratee(d1.clone(), &|me: &T, e: Expr| body(me, st, e), v, l)
                },
                v,
                l,
            ),
            _ => self.for_scalar(st, body, v, l),
        }
    }

    fn read_data(decl_id: &str, decl_var: &Expr, flat_type: &UnsizedType) -> Expr {
        let lit = Expr::new(
            ExprP::Lit(LitType::Str, remove_prefix(decl_id).to_string()),
            decl_var.meta.clone(),
        );
        Expr::internal(
            Internal::ReadData,
            vec![lit],
            Meta {
                ty: flat_type.clone(),
                ..decl_var.meta.clone()
            },
        )
    }

    /// The loop copying a flattened array into `lval`, element by element.
    #[allow(clippy::too_many_arguments)]
    fn flat_copy(
        &self,
        st: &SizedType,
        lval: &LValue,
        decl_var: Expr,
        flat_var: &Expr,
        scalar: &UnsizedType,
        assign_loc: Loc,
        pos_loc: Loc,
        loop_loc: Loc,
    ) -> Stmt {
        let ut = st.to_unsized();
        let body = |_: &T, _: &SizedType, v: Expr| {
            let read = Expr::new(
                ExprP::Indexed(Box::new(flat_var.clone()), vec![Index::Single(var(POS))]),
                Meta {
                    ty: scalar.clone(),
                    ..flat_var.meta.clone()
                },
            );
            let assign = s(
                StmtP::Assignment(with_idx(lval, v.collect_indices()), ut.clone(), read),
                assign_loc,
            );
            let incr = s(
                StmtP::Assignment(
                    LValue::var(POS),
                    UnsizedType::Int,
                    Expr::binop(var(POS), Operator::Plus, Expr::int(1)),
                ),
                pos_loc,
            );
            s(StmtP::SList(vec![assign, incr]), pos_loc)
        };
        self.for_scalar_inv(st, &body, decl_var, loop_loc)
    }

    /// `var_context_read_inside_tuple`.
    fn read_inside_tuple(
        &self,
        enclosing: &str,
        origin_type: &UnsizedType,
        lval: &LValue,
        st: &SizedType,
    ) -> Vec<Stmt> {
        let sm = Loc::EMPTY;
        let ut = st.to_unsized();
        let scalar = base_type(st);
        let flat_type = UnsizedType::Array(Box::new(scalar.clone()));
        let decl_id = lval.name();
        let decl_var = typed_var(&decl_id, ut.clone(), Ad::DataOnly, sm);
        let flat_name = |id: &str| format!("{}_flat__", munge_tuple_name(id));
        let munged = munge_tuple_name(enclosing);
        let (enc_flat, enc_pos) = (format!("{munged}_flat__"), format!("{munged}_flat__pos__"));
        let origin_name = Expr::new(
            ExprP::Var(enc_flat),
            Meta {
                ty: origin_type.clone(),
                ad: fill_ad(&Ad::DataOnly, origin_type),
                ..Meta::empty()
            },
        );
        let type_size = Expr::binop(var(&enc_pos), Operator::Plus, io_size(st));
        let end_position = Expr::binop(type_size.clone(), Operator::Minus, Expr::int(1));
        let origin = match ut {
            UnsizedType::Int | UnsizedType::Real | UnsizedType::Complex => {
                add_int_index(origin_name, Index::Single(var(&enc_pos)))
            }
            _ => add_int_index(origin_name, Index::Between(var(&enc_pos), end_position)),
        };
        let incr_tuple_pos = s(
            StmtP::Assignment(LValue::var(&enc_pos), UnsizedType::Int, type_size),
            sm,
        );
        match st {
            SizedType::Int | SizedType::Real | SizedType::Complex => vec![
                s(StmtP::Assignment(lval.clone(), ut, origin), sm),
                incr_tuple_pos,
            ],
            SizedType::Array(t, _) if matches!(**t, SizedType::Int | SizedType::Real) => vec![
                s(StmtP::Assignment(lval.clone(), flat_type, origin), sm),
                incr_tuple_pos,
            ],
            SizedType::Tuple(subtypes) => subtypes
                .iter()
                .enumerate()
                .flat_map(|(i, x)| {
                    let name = format!("{enclosing}.{}", i + 1);
                    self.read_inside_tuple(&name, origin_type, &tuple_proj(lval, i + 1), x)
                })
                .collect(),
            SizedType::Array(..) if contains_tuple(st) => {
                let (tupl, dims) = get_array_dims(st);
                let SizedType::Tuple(types) = &tupl else {
                    unreachable!()
                };
                let names: Vec<String> = (1..=types.len())
                    .map(|i| format!("{enclosing}.{i}"))
                    .collect();
                let mut stmts: Vec<Stmt> = names
                    .iter()
                    .zip(types)
                    .map(|(n, t)| {
                        decl(
                            fill_ad(&Ad::DataOnly, &t.to_unsized()),
                            &make_tuple_temp(n),
                            DeclType::Sized(t.clone()),
                            DeclInit::Default,
                            sm,
                        )
                    })
                    .collect();
                let mut rdims = dims.clone();
                rdims.reverse();
                let ndims = dims.len();
                stmts.push(self.mk_nested_for(
                    &rdims,
                    &|me: &T, loopvars: Vec<Expr>| {
                        let mut body: Vec<Stmt> = names
                            .iter()
                            .zip(types)
                            .flat_map(|(n, t)| {
                                me.read_inside_tuple(
                                    n,
                                    &wind_array_type(t.to_unsized(), ndims),
                                    &LValue::var(&make_tuple_temp(n)),
                                    t,
                                )
                            })
                            .collect();
                        let elems = names
                            .iter()
                            .zip(types)
                            .map(|(n, t)| {
                                Expr::new(ExprP::Var(make_tuple_temp(n)), meta_from_sizedtype(t))
                            })
                            .collect();
                        body.push(s(
                            StmtP::Assignment(
                                with_idx(lval, singles_rev(&loopvars)),
                                ut.clone(),
                                tuple_expr(elems),
                            ),
                            sm,
                        ));
                        s(StmtP::SList(body), sm)
                    },
                    sm,
                ));
                vec![s(StmtP::Block(stmts), sm)]
            }
            _ => {
                let id_flat = flat_name(&decl_id);
                let d = decl(
                    Ad::AutoDiffable,
                    &id_flat,
                    DeclType::Unsized(flat_type.clone()),
                    DeclInit::Default,
                    sm,
                );
                let assign = s(
                    StmtP::Assignment(LValue::var(&id_flat), flat_type.clone(), origin),
                    sm,
                );
                let flat_var = typed_var(&id_flat, flat_type, Ad::DataOnly, sm);
                let pos_reset = s(
                    StmtP::Assignment(LValue::var(POS), UnsizedType::Int, Expr::int(1)),
                    sm,
                );
                let lp = self.flat_copy(st, lval, decl_var, &flat_var, &scalar, sm, sm, sm);
                vec![s(
                    StmtP::Block(vec![d, assign, pos_reset, lp, incr_tuple_pos]),
                    sm,
                )]
            }
        }
    }

    /// `var_context_read_internal`: reads a variable from the data or init context.
    fn var_context_read(&self, lval: &LValue, sm: Loc, st: &SizedType) -> Vec<Stmt> {
        let ut = st.to_unsized();
        let scalar = base_type(st);
        let flat_type = UnsizedType::Array(Box::new(scalar.clone()));
        let decl_id = lval.name();
        let decl_var = typed_var(&decl_id, ut.clone(), Ad::DataOnly, sm);
        let flat_name = |id: &str| format!("{}_flat__", munge_tuple_name(id));
        match st {
            SizedType::Int | SizedType::Real | SizedType::Complex => {
                let e = Expr::new(
                    ExprP::Indexed(
                        Box::new(Self::read_data(&decl_id, &decl_var, &flat_type)),
                        vec![Index::Single(Expr::int(1))],
                    ),
                    Meta {
                        ty: ut.clone(),
                        ..decl_var.meta.clone()
                    },
                );
                vec![s(StmtP::Assignment(lval.clone(), ut, e), sm)]
            }
            SizedType::Array(t, _) if matches!(**t, SizedType::Int | SizedType::Real) => {
                vec![s(
                    StmtP::Assignment(
                        lval.clone(),
                        flat_type.clone(),
                        Self::read_data(&decl_id, &decl_var, &flat_type),
                    ),
                    sm,
                )]
            }
            SizedType::Tuple(subtypes) => subtypes
                .iter()
                .enumerate()
                .flat_map(|(i, x)| {
                    let l = if i == 0 { sm } else { Loc::EMPTY };
                    self.var_context_read(&tuple_proj(lval, i + 1), l, x)
                })
                .collect(),
            SizedType::Array(..) if contains_tuple(st) => {
                let (tupl, dims) = get_array_dims(st);
                let io_names = enumerate_tuple_names_io(&decl_id, &tupl.to_unsized());
                let flat_types = flatten_tuple_io(&tupl);
                let mut stmts = Vec::new();
                for (io_name, fst) in io_names.iter().zip(&flat_types) {
                    let vname = flat_name(io_name);
                    let typ = fst.to_unsized();
                    let array_type = UnsizedType::Array(Box::new(ut_internal_scalar(&typ)));
                    stmts.push(decl(
                        Ad::AutoDiffable,
                        &vname,
                        DeclType::Unsized(array_type.clone()),
                        DeclInit::Default,
                        Loc::EMPTY,
                    ));
                    stmts.push(s(
                        StmtP::Assignment(
                            LValue::var(&vname),
                            typ,
                            Self::read_data(io_name, &decl_var, &array_type),
                        ),
                        sm,
                    ));
                    let pos = format!("{vname}pos__");
                    stmts.push(decl(
                        Ad::DataOnly,
                        &pos,
                        DeclType::Unsized(UnsizedType::Int),
                        DeclInit::Default,
                        Loc::EMPTY,
                    ));
                    stmts.push(s(
                        StmtP::Assignment(LValue::var(&pos), UnsizedType::Int, Expr::int(1)),
                        Loc::EMPTY,
                    ));
                }
                let SizedType::Tuple(types) = &tupl else {
                    unreachable!()
                };
                let names: Vec<String> = (1..=types.len())
                    .map(|i| format!("{decl_id}.{i}"))
                    .collect();
                for (n, t) in names.iter().zip(types) {
                    stmts.push(decl(
                        fill_ad(&Ad::DataOnly, &t.to_unsized()),
                        &make_tuple_temp(n),
                        DeclType::Sized(t.clone()),
                        DeclInit::Default,
                        Loc::EMPTY,
                    ));
                }
                let mut rdims = dims.clone();
                rdims.reverse();
                let ndims = dims.len();
                stmts.push(self.mk_nested_for(
                    &rdims,
                    &|me: &T, loopvars: Vec<Expr>| {
                        let mut body: Vec<Stmt> = names
                            .iter()
                            .zip(types)
                            .flat_map(|(n, t)| {
                                me.read_inside_tuple(
                                    n,
                                    &wind_array_type(t.to_unsized(), ndims),
                                    &LValue::var(&make_tuple_temp(n)),
                                    t,
                                )
                            })
                            .collect();
                        let elems = names
                            .iter()
                            .zip(types)
                            .map(|(n, t)| {
                                Expr::new(ExprP::Var(make_tuple_temp(n)), meta_from_sizedtype(t))
                            })
                            .collect();
                        body.push(s(
                            StmtP::Assignment(
                                with_idx(lval, singles_rev(&loopvars)),
                                ut.clone(),
                                tuple_expr(elems),
                            ),
                            Loc::EMPTY,
                        ));
                        s(StmtP::SList(body), Loc::EMPTY)
                    },
                    Loc::EMPTY,
                ));
                vec![s(StmtP::Block(stmts), sm)]
            }
            _ => {
                let id_flat = flat_name(&decl_id);
                let d = decl(
                    Ad::AutoDiffable,
                    &id_flat,
                    DeclType::Unsized(flat_type.clone()),
                    DeclInit::Uninit,
                    sm,
                );
                let assign = s(
                    StmtP::Assignment(
                        LValue::var(&id_flat),
                        flat_type.clone(),
                        Self::read_data(&decl_id, &decl_var, &flat_type),
                    ),
                    sm,
                );
                let flat_var = typed_var(&id_flat, flat_type, Ad::DataOnly, sm);
                let pos_reset = s(
                    StmtP::Assignment(LValue::var(POS), UnsizedType::Int, Expr::int(1)),
                    Loc::EMPTY,
                );
                let lp = self.flat_copy(
                    st,
                    lval,
                    decl_var,
                    &flat_var,
                    &scalar,
                    Loc::EMPTY,
                    Loc::EMPTY,
                    Loc::EMPTY,
                );
                vec![s(StmtP::Block(vec![d, assign, pos_reset, lp]), sm)]
            }
        }
    }

    /// `param_deserializer_read`.
    fn param_read(&self, lval: &LValue, sm: Loc, ov: &OutVar) -> (Vec<Stmt>, Option<Expr>) {
        if ov.block != IoBlock::Parameters {
            return (vec![], None);
        }
        self.param_read_stmt(lval, sm, &ov.constrained, &ov.trans)
    }

    fn param_read_stmt(
        &self,
        lval: &LValue,
        sm: Loc,
        cst: &SizedType,
        trans: &Transformation,
    ) -> (Vec<Stmt>, Option<Expr>) {
        match cst {
            SizedType::Array(..) if contains_tuple(cst) => {
                let (tupl, mut dims) = get_array_dims(cst);
                dims.reverse();
                let st = self.mk_nested_for(
                    &dims,
                    &|me: &T, loopvars: Vec<Expr>| {
                        let lv = with_idx(lval, singles_rev(&loopvars));
                        s(
                            StmtP::SList(me.param_read_stmt(&lv, sm, &tupl, trans).0),
                            sm,
                        )
                    },
                    sm,
                );
                (vec![st], None)
            }
            SizedType::Tuple(sts) => {
                let trans: Vec<Transformation> = match trans {
                    Transformation::Identity => vec![Transformation::Identity; sts.len()],
                    Transformation::Tuple(ts) => ts.clone(),
                    _ => panic!("a tuple must have a tuple transformation"),
                };
                let stmts = sts
                    .iter()
                    .zip(&trans)
                    .enumerate()
                    .flat_map(|(i, (st, t))| {
                        self.param_read_stmt(&tuple_proj(lval, i + 1), sm, st, t).0
                    })
                    .collect();
                (stmts, None)
            }
            _ => {
                let ut = cst.to_unsized();
                let meta = Meta {
                    ad: fill_ad(&Ad::AutoDiffable, &ut),
                    ty: ut.clone(),
                    loc: sm,
                };
                let read = Expr::internal(
                    Internal::ReadParam {
                        constrain: Box::new(trans.clone()),
                        dims: read_constrain_dims(trans, cst),
                        mem_pattern: MemPattern::AoS,
                    },
                    vec![],
                    meta,
                );
                let top = matches!(lval.base, LBase::Variable(_)) && lval.idx.is_empty();
                (
                    vec![s(StmtP::Assignment(lval.clone(), ut, read.clone()), sm)],
                    top.then_some(read),
                )
            }
        }
    }

    /// `param_serializer_write`.
    fn param_write(&self, decl_id: &str, ov: &OutVar, unconstrain: bool) -> Vec<Stmt> {
        let v = typed_var(
            decl_id,
            ov.constrained.to_unsized(),
            Ad::DataOnly,
            Loc::EMPTY,
        );
        self.param_write_go(v, &ov.constrained, &ov.trans, unconstrain)
    }

    fn param_write_go(
        &self,
        v: Expr,
        st: &SizedType,
        trans: &Transformation,
        unconstrain: bool,
    ) -> Vec<Stmt> {
        match (unconstrain, st, trans) {
            (true, SizedType::Tuple(subtypes), Transformation::Tuple(ts)) => subtypes
                .iter()
                .zip(ts)
                .enumerate()
                .flat_map(|(i, (st, t))| {
                    self.param_write_go(add_tuple_index(v.clone(), i + 1), st, t, unconstrain)
                })
                .collect(),
            (true, SizedType::Array(..), Transformation::Tuple(_)) => {
                let (tupl, mut dims) = get_array_dims(st);
                dims.reverse();
                vec![self.mk_nested_for(
                    &dims,
                    &|me: &T, loopvars: Vec<Expr>| {
                        let e = singles_rev(&loopvars)
                            .into_iter()
                            .fold(v.clone(), add_int_index);
                        s(
                            StmtP::SList(me.param_write_go(e, &tupl, trans, unconstrain)),
                            Loc::EMPTY,
                        )
                    },
                    Loc::EMPTY,
                )]
            }
            (true, ..) => vec![write_param(Some(trans.clone()), v)],
            (false, ..) => vec![write_param(None, v)],
        }
    }

    /// `param_unconstrained_serializer_write`.
    fn param_unconstrained_write(&self, decl_id: &str, sm: Loc, ov: &OutVar) -> Vec<Stmt> {
        let v = typed_var(
            decl_id,
            ov.constrained.to_unsized(),
            Ad::DataOnly,
            Loc::EMPTY,
        );
        self.unconstrained_write_go(v, &ov.constrained, sm)
    }

    fn unconstrained_write_go(&self, v: Expr, st: &SizedType, sm: Loc) -> Vec<Stmt> {
        match st {
            SizedType::Tuple(subtypes) => subtypes
                .iter()
                .enumerate()
                .flat_map(|(i, x)| {
                    self.unconstrained_write_go(add_tuple_index(v.clone(), i + 1), x, sm)
                })
                .collect(),
            _ if is_recursive_container(st) => {
                let (inner, mut dims) = get_scalar_and_dims(st);
                dims.reverse();
                vec![self.mk_nested_for(
                    &dims,
                    &|me: &T, loopvars: Vec<Expr>| {
                        let e = singles_rev(&loopvars)
                            .into_iter()
                            .fold(v.clone(), add_int_index);
                        s(
                            StmtP::SList(me.unconstrained_write_go(e, &inner, sm)),
                            Loc::EMPTY,
                        )
                    },
                    sm,
                )]
            }
            _ => vec![write_param(None, v)],
        }
    }

    /// `var_context_unconstrain_transform`.
    fn var_context_unconstrain(&self, decl_id: &str, sm: Loc, ov: &OutVar) -> Vec<Stmt> {
        let st = &ov.constrained;
        let mut v = vec![decl(
            fill_ad(&Ad::AutoDiffable, &st.to_unsized()),
            decl_id,
            DeclType::Sized(st.clone()),
            DeclInit::Default,
            sm,
        )];
        v.extend(self.var_context_read(&LValue::var(decl_id), sm, st));
        v.extend(self.param_write(decl_id, ov, true));
        v
    }

    /// `array_unconstrain_transform`.
    fn array_unconstrain(&self, decl_id: &str, sm: Loc, ov: &OutVar) -> Vec<Stmt> {
        let st = &ov.constrained;
        let mut v = vec![decl(
            fill_ad(&Ad::AutoDiffable, &st.to_unsized()),
            decl_id,
            DeclType::Sized(st.clone()),
            DeclInit::Default,
            sm,
        )];
        v.extend(self.array_read(&LValue::var(decl_id), st, sm));
        v.extend(self.param_write(decl_id, ov, true));
        v
    }

    fn array_read(&self, lval: &LValue, st: &SizedType, sm: Loc) -> Vec<Stmt> {
        match st {
            SizedType::Tuple(subtypes) => subtypes
                .iter()
                .enumerate()
                .flat_map(|(i, x)| self.array_read(&tuple_proj(lval, i + 1), x, sm))
                .collect(),
            _ if contains_tuple(st) => {
                let (tupl, mut dims) = get_scalar_and_dims(st);
                dims.reverse();
                vec![self.mk_nested_for(
                    &dims,
                    &|me: &T, loopvars: Vec<Expr>| {
                        let lv = with_idx(lval, singles_rev(&loopvars));
                        s(StmtP::SList(me.array_read(&lv, &tupl, sm)), Loc::EMPTY)
                    },
                    sm,
                )]
            }
            _ if is_recursive_container(st) => {
                let (inner, mut dims) = get_scalar_and_dims(st);
                dims.reverse();
                vec![self.mk_nested_for(
                    &dims,
                    &|_: &T, loopvars: Vec<Expr>| {
                        s(
                            StmtP::Assignment(
                                with_idx(lval, singles_rev(&loopvars)),
                                inner.to_unsized(),
                                plain_deserializer_read(sm, &internal_scalar(&inner)),
                            ),
                            sm,
                        )
                    },
                    sm,
                )]
            }
            _ => vec![s(
                StmtP::Assignment(
                    lval.clone(),
                    st.to_unsized(),
                    plain_deserializer_read(sm, st),
                ),
                sm,
            )],
        }
    }
}

/// `read_constrain_dims`: the sizes the constraining functions take.
fn read_constrain_dims(trans: &Transformation, st: &SizedType) -> Vec<Expr> {
    fn constrain_get_dims(st: &SizedType) -> Vec<Expr> {
        use SizedType::*;
        match st {
            Int | Real | Complex | Tuple(_) => vec![],
            Array(t, d) => {
                let mut v = vec![d.clone()];
                v.extend(constrain_get_dims(t));
                v
            }
            Vector(d) | RowVector(d) | ComplexVector(d) | ComplexRowVector(d) => vec![d.clone()],
            Matrix(_, d2) | ComplexMatrix(_, d2) => vec![d2.clone()],
        }
    }
    match trans {
        Transformation::CholeskyCorr | Transformation::Correlation | Transformation::Covariance => {
            constrain_get_dims(st)
        }
        _ => get_dims(st),
    }
}

fn plain_deserializer_read(loc: Loc, st: &SizedType) -> Expr {
    Expr::internal(
        Internal::ReadDeserializer,
        get_dims(st),
        Meta {
            ty: st.to_unsized(),
            loc,
            ad: Ad::AutoDiffable,
        },
    )
}

/// `escape_name`: the model name as a C++ identifier.
fn escape_name(name: &str) -> String {
    let no_keywords = add_prefix_to_kwrds(name);
    let only_ascii: String = no_keywords
        .bytes()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == b'_' {
                (c as char).to_string()
            } else if c == b'-' || c == b'.' {
                "_".to_string()
            } else {
                format!("x{c}")
            }
        })
        .collect();
    match only_ascii.bytes().next() {
        None => "_model".to_string(),
        Some(c) if !(c.is_ascii_alphabetic() || c == b'_') => format!("_{only_ascii}"),
        _ => only_ascii,
    }
}

/// `ensure_body_in_block`: wraps the bodies of conditionals and loops in blocks.
fn ensure_body_in_block(st: &mut Stmt) {
    for c in st.children_mut() {
        ensure_body_in_block(c);
    }
    let in_block = |b: &mut Stmt| {
        let p = std::mem::replace(&mut b.pattern, StmtP::Skip);
        b.pattern = match p {
            StmtP::Block(l) | StmtP::SList(l) => StmtP::Block(l),
            p => StmtP::Block(vec![Stmt::new(p, b.loc)]),
        };
    };
    match &mut st.pattern {
        StmtP::IfElse(_, t, e) => {
            in_block(t);
            if let Some(e) = e {
                in_block(e);
            }
        }
        StmtP::While(_, b) | StmtP::For { body: b, .. } => in_block(b),
        _ => {}
    }
}

/// `flatten_slists_list`.
fn flatten_slists_list(ls: Vec<Stmt>) -> Vec<Stmt> {
    fn flatten_stmt(mut st: Stmt) -> Stmt {
        match st.pattern {
            StmtP::Block(ls) => {
                let mut out = Vec::new();
                for x in ls {
                    let x = flatten_stmt(x);
                    match x.pattern {
                        StmtP::SList(inner) => out.extend(inner),
                        _ => out.push(x),
                    }
                }
                st.pattern = StmtP::Block(out);
                st
            }
            _ => {
                for c in st.children_mut() {
                    let x = std::mem::replace(c, Stmt::new(StmtP::Skip, Loc::EMPTY));
                    *c = flatten_stmt(x);
                }
                st
            }
        }
    }
    fn top(ls: Vec<Stmt>) -> Vec<Stmt> {
        ls.into_iter()
            .flat_map(|st| match st.pattern {
                StmtP::SList(inner) => top(inner),
                _ => vec![st],
            })
            .collect()
    }
    top(ls).into_iter().map(flatten_stmt).collect()
}

/// `add_reads`: inserts the reads after the top level declarations of the variables.
fn add_reads<V>(
    stmts: Vec<Stmt>,
    vars: &[(String, Loc, V)],
    mut mkread: impl FnMut(&LValue, Loc, &V) -> (Vec<Stmt>, Option<Expr>),
) -> Vec<Stmt> {
    let mut out = Vec::new();
    for mut st in stmts {
        let found = match &st.pattern {
            StmtP::Decl { id, .. } => vars.iter().find(|(n, ..)| n == id),
            _ => None,
        };
        match found {
            None => out.push(st),
            Some((id, l, v)) => {
                let (reader, op) = mkread(&LValue::var(id), *l, v);
                match op {
                    Some(e) => {
                        if let StmtP::Decl { init, .. } = &mut st.pattern {
                            *init = DeclInit::Assign(e);
                        }
                        out.push(st);
                    }
                    None => {
                        out.push(st);
                        out.extend(reader);
                    }
                }
            }
        }
    }
    out
}

fn insert_before(mut l: Vec<Stmt>, f: impl Fn(&Stmt) -> bool, to_insert: Vec<Stmt>) -> Vec<Stmt> {
    let at = l.iter().position(f).unwrap_or(l.len());
    l.splice(at..at, to_insert);
    l
}

fn contains_var(e: &Expr, name: &str) -> bool {
    e.any(&mut |e: &Expr| matches!(&e.pattern, ExprP::Var(v) if v == name))
}

/// `Transform_Mir.trans_prog` without OpenCL.
pub fn trans_prog(mut p: Program) -> Program {
    let t = T::default();
    // Name mangling of C++ keywords.
    for (n, _, st) in &mut p.input_vars {
        *n = add_prefix_to_kwrds(n);
        sized_exprs_mut(st, &mut rename_expr);
    }
    for (n, _, ov) in &mut p.output_vars {
        *n = add_prefix_to_kwrds(n);
        sized_exprs_mut(&mut ov.constrained, &mut rename_expr);
        sized_exprs_mut(&mut ov.unconstrained, &mut rename_expr);
        for e in ov.trans.exprs_mut() {
            rename_expr(e);
        }
    }
    for f in &mut p.functions {
        f.name = add_prefix_to_kwrds(&f.name);
        for (_, n, _) in &mut f.args {
            *n = add_prefix_to_kwrds(n);
        }
        if let Some(b) = &mut f.body {
            rename_stmt(b);
        }
    }
    for b in [
        &mut p.prepare_data,
        &mut p.log_prob,
        &mut p.generate_quantities,
    ] {
        for st in b.iter_mut() {
            rename_stmt(st);
        }
    }
    p.functions = break_eigen_cycles(std::mem::take(&mut p.functions));

    let maybe_add_pos = |stmts: Vec<Stmt>| -> Vec<Stmt> {
        let needs = stmts.iter().any(|st| match &st.pattern {
            StmtP::Decl { ty, .. } => {
                use UnsizedType::*;
                match ty.to_unsized() {
                    Int | Real | Complex | Function(_) | MathLibraryFunction => false,
                    Array(t) => !matches!(*t, Real | Int),
                    _ => true,
                }
            }
            _ => false,
        });
        if !needs {
            return stmts;
        }
        let mut v = vec![
            decl(
                Ad::DataOnly,
                POS,
                DeclType::Sized(SizedType::Int),
                DeclInit::Default,
                Loc::EMPTY,
            ),
            s(
                StmtP::Assignment(LValue::var(POS), UnsizedType::Int, Expr::int(1)),
                Loc::EMPTY,
            ),
        ];
        v.extend(stmts);
        v
    };

    let (mut param_writes, mut tparam_writes, mut gq_writes) = (vec![], vec![], vec![]);
    for (id, l, ov) in &p.output_vars {
        let w = s(StmtP::SList(t.param_unconstrained_write(id, *l, ov)), *l);
        match ov.block {
            IoBlock::Parameters => param_writes.push(w),
            IoBlock::TransformedParameters => tparam_writes.push(w),
            IoBlock::GeneratedQuantities => gq_writes.push(w),
        }
    }
    let tparam_start = |st: &Stmt| matches!(&st.pattern, StmtP::IfElse(c, ..) if contains_var(c, "emit_transformed_parameters__"));
    let gq_start = |st: &Stmt| match &st.pattern {
        StmtP::IfElse(c, ..) => matches!(
            &c.pattern,
            ExprP::FunApp(_, args) if matches!(args.as_slice(), [Expr { pattern: ExprP::Var(v), .. }] if v == "emit_generated_quantities__")
        ),
        _ => false,
    };
    let tparam_writes_cond = if tparam_writes.is_empty() {
        vec![]
    } else {
        vec![s(
            StmtP::IfElse(
                var("emit_transformed_parameters__"),
                Box::new(s(StmtP::SList(tparam_writes), Loc::EMPTY)),
                None,
            ),
            Loc::EMPTY,
        )]
    };
    let read = |lv: &LValue, l: Loc, ov: &OutVar| t.param_read(lv, l, ov);
    let gq = add_reads(
        std::mem::take(&mut p.generate_quantities),
        &p.output_vars,
        read,
    );
    let gq = insert_before(gq, tparam_start, param_writes);
    let mut gq = insert_before(gq, gq_start, tparam_writes_cond);
    gq.extend(gq_writes);
    let log_prob = add_reads(std::mem::take(&mut p.log_prob), &p.output_vars, read);
    let params: Vec<&(String, Loc, OutVar)> = p
        .output_vars
        .iter()
        .filter(|(_, _, ov)| ov.block == IoBlock::Parameters)
        .collect();
    let prepare_data = add_reads(
        std::mem::take(&mut p.prepare_data),
        &p.input_vars,
        |lv, l, st| (t.var_context_read(lv, l, st), None),
    );
    p.prepare_data = maybe_add_pos(prepare_data);
    p.transform_inits = maybe_add_pos(
        params
            .iter()
            .flat_map(|(id, l, ov)| t.var_context_unconstrain(id, *l, ov))
            .collect(),
    );
    p.unconstrain_array = params
        .iter()
        .flat_map(|(id, l, ov)| t.array_unconstrain(id, *l, ov))
        .collect();
    p.reverse_mode_log_prob = log_prob.clone();
    p.log_prob = log_prob;
    p.generate_quantities = gq;
    p.name = escape_name(&p.name);

    for f in &mut p.functions {
        if let Some(b) = &mut f.body {
            ensure_body_in_block(b);
        }
    }
    for b in [
        &mut p.prepare_data,
        &mut p.log_prob,
        &mut p.reverse_mode_log_prob,
        &mut p.generate_quantities,
        &mut p.transform_inits,
        &mut p.unconstrain_array,
    ] {
        for st in b.iter_mut() {
            ensure_body_in_block(st);
        }
        *b = flatten_slists_list(std::mem::take(b));
    }
    p
}
