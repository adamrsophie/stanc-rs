//! The middle intermediate representation (MIR), a port of stanc3 v2.40.0's `Middle` modules.
//!
//! Operators are function applications of Stan Math library functions named after the operator
//! (`Plus__`, `PMinus__`, ...), compiler generated code calls compiler internal functions, and
//! declarations are separate from their initialization.

use stanc_rs_semantic::types::Ad;
use stanc_rs_syntax::Span;
pub use stanc_rs_syntax::ast::UnsizedType;

/// A position: a file of the source map and a byte offset in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pos {
    /// Index of the file in the source map; `u32::MAX` for no position.
    pub file: u32,
    /// Byte offset.
    pub off: u32,
}

impl Pos {
    /// The position of compiler generated code.
    pub const EMPTY: Pos = Pos {
        file: u32::MAX,
        off: u32::MAX,
    };
}

/// A source range, which unlike [`Span`] may begin and end in different files (stanc3's
/// `Location_span.t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Loc {
    /// Where the range begins.
    pub begin: Pos,
    /// Where it ends.
    pub end: Pos,
}

impl Loc {
    /// The location of compiler generated code (`Location_span.empty`).
    pub const EMPTY: Loc = Loc {
        begin: Pos::EMPTY,
        end: Pos::EMPTY,
    };

    /// The range of a span.
    pub fn of_span(s: Span) -> Loc {
        Loc {
            begin: Pos {
                file: s.file,
                off: s.start,
            },
            end: Pos {
                file: s.file,
                off: s.end,
            },
        }
    }

    /// `Location_span.merge`: from the beginning of `self` to the end of `other`.
    pub fn merge(self, other: Loc) -> Loc {
        Loc {
            begin: self.begin,
            end: other.end,
        }
    }

    /// True for compiler generated code.
    pub fn is_empty(&self) -> bool {
        *self == Loc::EMPTY
    }

    /// The span, when the range lies in one file.
    pub fn span(&self) -> Option<Span> {
        (self.begin.file == self.end.file && self.begin.file != u32::MAX).then_some(Span {
            file: self.begin.file,
            start: self.begin.off,
            end: self.end.off,
        })
    }
}

/// Function suffixes; the flag of the density kinds is set for the unnormalized (`_lupdf`)
/// forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Suffix {
    Plain,
    Rng,
    Lpdf(bool),
    Lpmf(bool),
    Target,
    Jacobian,
}

impl Suffix {
    /// `Fun_kind.suffix_from_name`.
    pub fn from_name(name: &str) -> Suffix {
        if name.ends_with("_rng") {
            Suffix::Rng
        } else if name.ends_with("_lp") {
            Suffix::Target
        } else if name.ends_with("_jacobian") {
            Suffix::Jacobian
        } else if name.ends_with("_lupdf") {
            Suffix::Lpdf(true)
        } else if name.ends_with("_lupmf") {
            Suffix::Lpmf(true)
        } else if name.ends_with("_lpdf") {
            Suffix::Lpdf(false)
        } else if name.ends_with("_lpmf") {
            Suffix::Lpmf(false)
        } else {
            Suffix::Plain
        }
    }

    /// The suffix with the normalization flag cleared (`Fun_kind.without_propto`).
    pub fn without_propto(self) -> Suffix {
        match self {
            Suffix::Lpdf(_) => Suffix::Lpdf(false),
            Suffix::Lpmf(_) => Suffix::Lpmf(false),
            s => s,
        }
    }
}

/// Stan's operators, which the MIR calls as functions named `<Operator>__`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operator {
    Plus,
    PPlus,
    Minus,
    PMinus,
    Times,
    Divide,
    IntDivide,
    Modulo,
    LDivide,
    EltTimes,
    EltDivide,
    Pow,
    EltPow,
    Or,
    And,
    Equals,
    NEquals,
    Less,
    Leq,
    Greater,
    Geq,
    PNot,
    Transpose,
}

impl Operator {
    const ALL: [Operator; 23] = [
        Operator::Plus,
        Operator::PPlus,
        Operator::Minus,
        Operator::PMinus,
        Operator::Times,
        Operator::Divide,
        Operator::IntDivide,
        Operator::Modulo,
        Operator::LDivide,
        Operator::EltTimes,
        Operator::EltDivide,
        Operator::Pow,
        Operator::EltPow,
        Operator::Or,
        Operator::And,
        Operator::Equals,
        Operator::NEquals,
        Operator::Less,
        Operator::Leq,
        Operator::Greater,
        Operator::Geq,
        Operator::PNot,
        Operator::Transpose,
    ];

    /// The name of the function implementing the operator, e.g. `Plus__`.
    pub fn name(self) -> &'static str {
        use Operator::*;
        match self {
            Plus => "Plus__",
            PPlus => "PPlus__",
            Minus => "Minus__",
            PMinus => "PMinus__",
            Times => "Times__",
            Divide => "Divide__",
            IntDivide => "IntDivide__",
            Modulo => "Modulo__",
            LDivide => "LDivide__",
            EltTimes => "EltTimes__",
            EltDivide => "EltDivide__",
            Pow => "Pow__",
            EltPow => "EltPow__",
            Or => "Or__",
            And => "And__",
            Equals => "Equals__",
            NEquals => "NEquals__",
            Less => "Less__",
            Leq => "Leq__",
            Greater => "Greater__",
            Geq => "Geq__",
            PNot => "PNot__",
            Transpose => "Transpose__",
        }
    }

    /// The operator a function name stands for (`Operator.of_string_opt`).
    pub fn of_name(name: &str) -> Option<Operator> {
        Operator::ALL.into_iter().find(|o| o.name() == name)
    }

    /// Source spelling.
    pub fn symbol(self) -> &'static str {
        use Operator::*;
        match self {
            Plus | PPlus => "+",
            Minus | PMinus => "-",
            Times => "*",
            Divide => "/",
            IntDivide => "%/%",
            Modulo => "%",
            LDivide => "\\",
            EltTimes => ".*",
            EltDivide => "./",
            Pow => "^",
            EltPow => ".^",
            Or => "||",
            And => "&&",
            Equals => "==",
            NEquals => "!=",
            Less => "<",
            Leq => "<=",
            Greater => ">",
            Geq => ">=",
            PNot => "!",
            Transpose => "'",
        }
    }

    /// True for the comparison operators.
    pub fn is_cmp(self) -> bool {
        use Operator::*;
        matches!(self, Equals | NEquals | Less | Leq | Greater | Geq)
    }
}

/// Indexing.
#[derive(Clone, Debug, PartialEq)]
pub enum Index<E> {
    All,
    Single(E),
    Upfrom(E),
    Between(E, E),
    MultiIndex(E),
}

impl<E> Index<E> {
    /// The index expressions.
    pub fn exprs(&self) -> Vec<&E> {
        match self {
            Index::All => vec![],
            Index::Single(e) | Index::Upfrom(e) | Index::MultiIndex(e) => vec![e],
            Index::Between(a, b) => vec![a, b],
        }
    }

    /// The index expressions, mutably.
    pub fn exprs_mut(&mut self) -> Vec<&mut E> {
        match self {
            Index::All => vec![],
            Index::Single(e) | Index::Upfrom(e) | Index::MultiIndex(e) => vec![e],
            Index::Between(a, b) => vec![a, b],
        }
    }
}

/// Constraint transformations.
#[derive(Clone, Debug, PartialEq)]
pub enum Transformation {
    Identity,
    Lower(Expr),
    Upper(Expr),
    LowerUpper(Expr, Expr),
    Offset(Expr),
    Multiplier(Expr),
    OffsetMultiplier(Expr, Expr),
    Ordered,
    PositiveOrdered,
    Simplex,
    UnitVector,
    SumToZero,
    CholeskyCorr,
    CholeskyCov,
    Correlation,
    Covariance,
    StochasticRow,
    StochasticColumn,
    Tuple(Vec<Transformation>),
}

impl Transformation {
    /// Whether values must be checked against the transformation (`Transformation.has_check`).
    pub fn has_check(&self) -> bool {
        match self {
            Transformation::Identity
            | Transformation::Offset(_)
            | Transformation::Multiplier(_)
            | Transformation::OffsetMultiplier(..) => false,
            Transformation::Tuple(ts) => ts.iter().any(Transformation::has_check),
            _ => true,
        }
    }

    /// The bound and scale expressions.
    pub fn exprs(&self) -> Vec<&Expr> {
        use Transformation::*;
        match self {
            Lower(a) | Upper(a) | Offset(a) | Multiplier(a) => vec![a],
            LowerUpper(a, b) | OffsetMultiplier(a, b) => vec![a, b],
            Tuple(ts) => ts.iter().flat_map(|t| t.exprs()).collect(),
            _ => vec![],
        }
    }

    /// The bound and scale expressions, mutably.
    pub fn exprs_mut(&mut self) -> Vec<&mut Expr> {
        use Transformation::*;
        match self {
            Lower(a) | Upper(a) | Offset(a) | Multiplier(a) => vec![a],
            LowerUpper(a, b) | OffsetMultiplier(a, b) => vec![a, b],
            Tuple(ts) => ts.iter_mut().flat_map(|t| t.exprs_mut()).collect(),
            _ => vec![],
        }
    }
}

/// Functions the compiler defines.
#[derive(Clone, Debug, PartialEq)]
pub enum Internal {
    Length,
    MakeArray,
    MakeTuple,
    MakeRowVec,
    NegInf,
    ReadData,
    ValidateSize,
    ValidateSizePositive,
    ValidateSizeUnitVector,
    Check {
        trans: Box<Transformation>,
        var_name: String,
        var: Box<Expr>,
    },
    Print,
    Reject,
    FatalError,
    ResizeToMatch,
    NaN,
    DeepCopy,
}

impl Internal {
    /// Whether a call changes state, so that calling it twice differs from calling it once.
    pub fn can_side_effect(&self) -> bool {
        matches!(
            self,
            Internal::ReadData
                | Internal::ValidateSize
                | Internal::ValidateSizePositive
                | Internal::ValidateSizeUnitVector
        )
    }
}

/// Which function is called.
#[derive(Clone, Debug, PartialEq)]
pub enum FunKind {
    /// A Stan Math library function (or operator); the name is the normalized one
    /// (`normal_lpdf` for `normal_lupdf`).
    StanLib(String, Suffix),
    CompilerInternal(Internal),
    UserDefined(String, Suffix),
}

impl FunKind {
    /// The function name, for library and user-defined functions.
    pub fn name(&self) -> Option<&str> {
        match self {
            FunKind::StanLib(n, _) | FunKind::UserDefined(n, _) => Some(n),
            FunKind::CompilerInternal(_) => None,
        }
    }

    /// A library operator.
    pub fn op(op: Operator) -> FunKind {
        FunKind::StanLib(op.name().to_string(), Suffix::Plain)
    }
}

/// Kinds of literals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LitType {
    Int,
    Real,
    Imaginary,
    Str,
}

/// Type, autodiff level, and location of an expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Meta {
    pub ty: UnsizedType,
    pub loc: Loc,
    pub ad: Ad,
}

impl Meta {
    /// `Expr.Typed.Meta.empty`: an integer data value of compiler generated code.
    pub fn empty() -> Meta {
        Meta {
            ty: UnsizedType::Int,
            loc: Loc::EMPTY,
            ad: Ad::DataOnly,
        }
    }
}

/// An expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub pattern: ExprP,
    pub meta: Meta,
}

/// Expression forms.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprP {
    Var(String),
    Lit(LitType, String),
    FunApp(FunKind, Vec<Expr>),
    TernaryIf(Box<Expr>, Box<Expr>, Box<Expr>),
    EAnd(Box<Expr>, Box<Expr>),
    EOr(Box<Expr>, Box<Expr>),
    Indexed(Box<Expr>, Vec<Index<Expr>>),
    Promotion(Box<Expr>, UnsizedType, Ad),
    TupleProjection(Box<Expr>, usize),
}

impl Expr {
    /// An expression with empty metadata.
    pub fn new(pattern: ExprP, meta: Meta) -> Expr {
        Expr { pattern, meta }
    }

    /// `Expr.Helpers.int`.
    pub fn int(i: i64) -> Expr {
        Expr::new(ExprP::Lit(LitType::Int, i.to_string()), Meta::empty())
    }

    /// `Expr.Helpers.str`.
    pub fn str(s: &str) -> Expr {
        Expr::new(ExprP::Lit(LitType::Str, s.to_string()), Meta::empty())
    }

    /// `Expr.Helpers.binop`, with empty metadata.
    pub fn binop(a: Expr, op: Operator, b: Expr) -> Expr {
        Expr::new(ExprP::FunApp(FunKind::op(op), vec![a, b]), Meta::empty())
    }

    /// `Expr.Helpers.unary_op`, with empty metadata.
    pub fn unary_op(op: Operator, a: Expr) -> Expr {
        Expr::new(ExprP::FunApp(FunKind::op(op), vec![a]), Meta::empty())
    }

    /// A call of a compiler internal function.
    pub fn internal(f: Internal, args: Vec<Expr>, meta: Meta) -> Expr {
        Expr::new(ExprP::FunApp(FunKind::CompilerInternal(f), args), meta)
    }

    /// The sub-expressions, in the order `Pattern.fold` visits them.
    pub fn children(&self) -> Vec<&Expr> {
        match &self.pattern {
            ExprP::Var(_) | ExprP::Lit(..) => vec![],
            ExprP::FunApp(k, args) => {
                let mut v: Vec<&Expr> = Vec::new();
                if let FunKind::CompilerInternal(Internal::Check { trans, var, .. }) = k {
                    v.extend(trans.exprs());
                    v.push(var);
                }
                v.extend(args.iter());
                v
            }
            ExprP::TernaryIf(a, b, c) => vec![a, b, c],
            ExprP::EAnd(a, b) | ExprP::EOr(a, b) => vec![a, b],
            ExprP::Indexed(e, idx) => {
                let mut v = vec![&**e];
                for i in idx {
                    v.extend(i.exprs());
                }
                v
            }
            ExprP::Promotion(e, ..) | ExprP::TupleProjection(e, _) => vec![e],
        }
    }

    /// The sub-expressions, mutably.
    pub fn children_mut(&mut self) -> Vec<&mut Expr> {
        match &mut self.pattern {
            ExprP::Var(_) | ExprP::Lit(..) => vec![],
            ExprP::FunApp(k, args) => {
                let mut v: Vec<&mut Expr> = Vec::new();
                if let FunKind::CompilerInternal(Internal::Check { trans, var, .. }) = k {
                    v.extend(trans.exprs_mut());
                    v.push(var);
                }
                v.extend(args.iter_mut());
                v
            }
            ExprP::TernaryIf(a, b, c) => vec![a, b, c],
            ExprP::EAnd(a, b) | ExprP::EOr(a, b) => vec![a, b],
            ExprP::Indexed(e, idx) => {
                let mut v = vec![&mut **e];
                for i in idx {
                    v.extend(i.exprs_mut());
                }
                v
            }
            ExprP::Promotion(e, ..) | ExprP::TupleProjection(e, _) => vec![e],
        }
    }

    /// True if `f` holds for this expression or a sub-expression.
    pub fn any(&self, f: &mut impl FnMut(&Expr) -> bool) -> bool {
        f(self) || self.children().into_iter().any(|c| c.any(f))
    }

    /// The indices of nested `Indexed` patterns (`Expr.Helpers.collect_indices`).
    pub fn collect_indices(&self) -> Vec<Index<Expr>> {
        match &self.pattern {
            ExprP::Indexed(obj, idx) => {
                let mut v = obj.collect_indices();
                v.extend(idx.iter().cloned());
                v
            }
            _ => vec![],
        }
    }
}

/// The type of an indexed value (`Expr.Helpers.infer_type_of_indexed`).
pub fn infer_type_of_indexed(ut: &UnsizedType, indices: &[Index<Expr>]) -> UnsizedType {
    use UnsizedType::*;
    let single = |i: &Index<Expr>| matches!(i, Index::Single(_));
    match indices {
        [] => ut.clone(),
        [i] if !single(i) => ut.clone(),
        [a, b] if matches!(ut, Matrix | ComplexMatrix) && !single(a) && !single(b) => ut.clone(),
        [a, b] if single(a) && !single(b) && matches!(ut, Matrix) => RowVector,
        [a, b] if single(a) && !single(b) && matches!(ut, ComplexMatrix) => ComplexRowVector,
        [a, b] if !single(a) && single(b) && matches!(ut, Matrix) => Vector,
        [a, b] if !single(a) && single(b) && matches!(ut, ComplexMatrix) => ComplexVector,
        [_] if matches!(ut, Matrix) => RowVector,
        [_] if matches!(ut, ComplexMatrix) => ComplexRowVector,
        [i, rest @ ..] if matches!(ut, Array(_)) => {
            let Array(t) = ut else { unreachable!() };
            if single(i) {
                infer_type_of_indexed(t, rest)
            } else {
                Array(Box::new(infer_type_of_indexed(t, rest)))
            }
        }
        [_, _] if matches!(ut, Matrix) => Real,
        [_] if matches!(ut, Vector | RowVector) => Real,
        [_, _] if matches!(ut, ComplexMatrix) => Complex,
        [_] if matches!(ut, ComplexVector | ComplexRowVector) => Complex,
        _ => ut.clone(),
    }
}

/// The base of an assigned location.
#[derive(Clone, Debug, PartialEq)]
pub enum LBase {
    Variable(String),
    TupleProjection(Box<LValue>, usize),
}

/// An assigned location: a base with indices.
#[derive(Clone, Debug, PartialEq)]
pub struct LValue {
    pub base: LBase,
    pub idx: Vec<Index<Expr>>,
}

impl LValue {
    /// `Stmt.Helpers.lvariable`.
    pub fn var(name: &str) -> LValue {
        LValue {
            base: LBase::Variable(name.to_string()),
            idx: vec![],
        }
    }

    /// The assigned variable (`Stmt.Helpers.lhs_variable`).
    pub fn variable(&self) -> &str {
        match &self.base {
            LBase::Variable(v) => v,
            LBase::TupleProjection(lv, _) => lv.variable(),
        }
    }

    /// The name with tuple projections (`Stmt.Helpers.get_lhs_name`).
    pub fn name(&self) -> String {
        match &self.base {
            LBase::Variable(v) => v.clone(),
            LBase::TupleProjection(lv, i) => format!("{}.{i}", lv.name()),
        }
    }

    /// The index expressions of all levels.
    pub fn exprs(&self) -> Vec<&Expr> {
        let mut v = match &self.base {
            LBase::Variable(_) => vec![],
            LBase::TupleProjection(lv, _) => lv.exprs(),
        };
        for i in &self.idx {
            v.extend(i.exprs());
        }
        v
    }

    /// The index expressions of all levels, mutably.
    pub fn exprs_mut(&mut self) -> Vec<&mut Expr> {
        let mut v = match &mut self.base {
            LBase::Variable(_) => vec![],
            LBase::TupleProjection(lv, _) => lv.exprs_mut(),
        };
        for i in &mut self.idx {
            v.extend(i.exprs_mut());
        }
        v
    }
}

/// Sized types.
#[derive(Clone, Debug, PartialEq)]
pub enum SizedType {
    Int,
    Real,
    Complex,
    Vector(Expr),
    RowVector(Expr),
    Matrix(Expr, Expr),
    ComplexVector(Expr),
    ComplexRowVector(Expr),
    ComplexMatrix(Expr, Expr),
    Array(Box<SizedType>, Expr),
    Tuple(Vec<SizedType>),
}

impl SizedType {
    /// The unsized type.
    pub fn to_unsized(&self) -> UnsizedType {
        match self {
            SizedType::Int => UnsizedType::Int,
            SizedType::Real => UnsizedType::Real,
            SizedType::Complex => UnsizedType::Complex,
            SizedType::Vector(_) => UnsizedType::Vector,
            SizedType::RowVector(_) => UnsizedType::RowVector,
            SizedType::Matrix(..) => UnsizedType::Matrix,
            SizedType::ComplexVector(_) => UnsizedType::ComplexVector,
            SizedType::ComplexRowVector(_) => UnsizedType::ComplexRowVector,
            SizedType::ComplexMatrix(..) => UnsizedType::ComplexMatrix,
            SizedType::Array(t, _) => UnsizedType::Array(Box::new(t.to_unsized())),
            SizedType::Tuple(ts) => UnsizedType::Tuple(ts.iter().map(|t| t.to_unsized()).collect()),
        }
    }

    /// The size expressions.
    pub fn exprs(&self) -> Vec<&Expr> {
        use SizedType::*;
        match self {
            Int | Real | Complex => vec![],
            Vector(e) | RowVector(e) | ComplexVector(e) | ComplexRowVector(e) => vec![e],
            Matrix(a, b) | ComplexMatrix(a, b) => vec![a, b],
            Array(t, e) => {
                let mut v = t.exprs();
                v.push(e);
                v
            }
            Tuple(ts) => ts.iter().flat_map(|t| t.exprs()).collect(),
        }
    }

    /// The size expressions, mutably.
    pub fn exprs_mut(&mut self) -> Vec<&mut Expr> {
        use SizedType::*;
        match self {
            Int | Real | Complex => vec![],
            Vector(e) | RowVector(e) | ComplexVector(e) | ComplexRowVector(e) => vec![e],
            Matrix(a, b) | ComplexMatrix(a, b) => vec![a, b],
            Array(t, e) => {
                let mut v = t.exprs_mut();
                v.push(e);
                v
            }
            Tuple(ts) => ts.iter_mut().flat_map(|t| t.exprs_mut()).collect(),
        }
    }

    /// The element type below the array dimensions, and the dimensions, outermost first.
    pub fn array_dims(&self) -> (&SizedType, Vec<&Expr>) {
        match self {
            SizedType::Array(t, d) => {
                let (inner, mut dims) = t.array_dims();
                dims.insert(0, d);
                (inner, dims)
            }
            t => (t, vec![]),
        }
    }

    /// Wraps `self` in arrays of the dimensions, outermost first (`SizedType.build_sarray`).
    pub fn build_array(dims: Vec<Expr>, st: SizedType) -> SizedType {
        dims.into_iter()
            .rev()
            .fold(st, |t, d| SizedType::Array(Box::new(t), d))
    }
}

/// The type of a declaration.
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum DeclType {
    Sized(SizedType),
    Unsized(UnsizedType),
}

impl DeclType {
    /// The unsized type.
    pub fn to_unsized(&self) -> UnsizedType {
        match self {
            DeclType::Sized(s) => s.to_unsized(),
            DeclType::Unsized(u) => u.clone(),
        }
    }
}

/// How a declaration is initialized.
#[derive(Clone, Debug, PartialEq)]
pub enum DeclInit {
    Uninit,
    Default,
    Assign(Expr),
}

/// A statement.
#[derive(Clone, Debug, PartialEq)]
pub struct Stmt {
    pub pattern: StmtP,
    pub loc: Loc,
}

/// Statement forms.
#[derive(Clone, Debug, PartialEq)]
pub enum StmtP {
    /// The assigned location, the type of its unindexed base, and the value.
    Assignment(LValue, UnsizedType, Expr),
    TargetPE(Expr),
    JacobianPE(Expr),
    NRFunApp(FunKind, Vec<Expr>),
    Break,
    Continue,
    Return(Option<Expr>),
    Skip,
    IfElse(Expr, Box<Stmt>, Option<Box<Stmt>>),
    While(Expr, Box<Stmt>),
    For {
        loopvar: String,
        lower: Expr,
        upper: Expr,
        body: Box<Stmt>,
    },
    Profile(String, Vec<Stmt>),
    Block(Vec<Stmt>),
    SList(Vec<Stmt>),
    Decl {
        ad: Ad,
        id: String,
        ty: DeclType,
        init: DeclInit,
    },
}

impl Stmt {
    /// A statement.
    pub fn new(pattern: StmtP, loc: Loc) -> Stmt {
        Stmt { pattern, loc }
    }

    /// The expressions directly in this statement, in the order `Pattern.fold` visits them.
    pub fn exprs(&self) -> Vec<&Expr> {
        match &self.pattern {
            StmtP::Assignment(lv, _, e) => {
                let mut v = lv.exprs();
                v.push(e);
                v
            }
            StmtP::TargetPE(e) | StmtP::JacobianPE(e) => vec![e],
            StmtP::NRFunApp(k, args) => {
                let mut v: Vec<&Expr> = Vec::new();
                if let FunKind::CompilerInternal(Internal::Check { trans, var, .. }) = k {
                    v.extend(trans.exprs());
                    v.push(var);
                }
                v.extend(args.iter());
                v
            }
            StmtP::Return(Some(e)) => vec![e],
            StmtP::IfElse(c, ..) | StmtP::While(c, _) => vec![c],
            StmtP::For { lower, upper, .. } => vec![lower, upper],
            StmtP::Decl { ty, init, .. } => {
                let mut v = match ty {
                    DeclType::Sized(st) => st.exprs(),
                    DeclType::Unsized(_) => vec![],
                };
                if let DeclInit::Assign(e) = init {
                    v.push(e);
                }
                v
            }
            _ => vec![],
        }
    }

    /// The expressions directly in this statement, mutably.
    pub fn exprs_mut(&mut self) -> Vec<&mut Expr> {
        match &mut self.pattern {
            StmtP::Assignment(lv, _, e) => {
                let mut v = lv.exprs_mut();
                v.push(e);
                v
            }
            StmtP::TargetPE(e) | StmtP::JacobianPE(e) => vec![e],
            StmtP::NRFunApp(k, args) => {
                let mut v: Vec<&mut Expr> = Vec::new();
                if let FunKind::CompilerInternal(Internal::Check { trans, var, .. }) = k {
                    v.extend(trans.exprs_mut());
                    v.push(var);
                }
                v.extend(args.iter_mut());
                v
            }
            StmtP::Return(Some(e)) => vec![e],
            StmtP::IfElse(c, ..) | StmtP::While(c, _) => vec![c],
            StmtP::For { lower, upper, .. } => vec![lower, upper],
            StmtP::Decl { ty, init, .. } => {
                let mut v = match ty {
                    DeclType::Sized(st) => st.exprs_mut(),
                    DeclType::Unsized(_) => vec![],
                };
                if let DeclInit::Assign(e) = init {
                    v.push(e);
                }
                v
            }
            _ => vec![],
        }
    }

    /// The statements directly inside this one.
    pub fn children(&self) -> Vec<&Stmt> {
        match &self.pattern {
            StmtP::IfElse(_, t, e) => {
                let mut v = vec![&**t];
                if let Some(e) = e {
                    v.push(e);
                }
                v
            }
            StmtP::While(_, b) | StmtP::For { body: b, .. } => vec![b],
            StmtP::Profile(_, ss) | StmtP::Block(ss) | StmtP::SList(ss) => ss.iter().collect(),
            _ => vec![],
        }
    }

    /// The statements directly inside this one, mutably.
    pub fn children_mut(&mut self) -> Vec<&mut Stmt> {
        match &mut self.pattern {
            StmtP::IfElse(_, t, e) => {
                let mut v = vec![&mut **t];
                if let Some(e) = e {
                    v.push(e);
                }
                v
            }
            StmtP::While(_, b) | StmtP::For { body: b, .. } => vec![b],
            StmtP::Profile(_, ss) | StmtP::Block(ss) | StmtP::SList(ss) => ss.iter_mut().collect(),
            _ => vec![],
        }
    }

    /// True if `f` holds for this statement or one inside it.
    pub fn any_stmt(&self, f: &mut impl FnMut(&Stmt) -> bool) -> bool {
        f(self) || self.children().into_iter().any(|c| c.any_stmt(f))
    }

    /// True if the statement or one inside it calls a function of a kind for which `f` holds
    /// (`Stmt.Helpers.contains_fn_kind`).
    pub fn contains_fn_kind(&self, f: &mut impl FnMut(&FunKind) -> bool) -> bool {
        self.any_stmt(&mut |s: &Stmt| {
            if let StmtP::NRFunApp(k, _) = &s.pattern
                && f(k)
            {
                return true;
            }
            s.exprs()
                .into_iter()
                .any(|e| e.any(&mut |e: &Expr| matches!(&e.pattern, ExprP::FunApp(k, _) if f(k))))
        })
    }
}

/// A user-defined function.
#[derive(Clone, Debug, PartialEq)]
pub struct FunDef {
    /// Return type; `None` for `void`.
    pub rt: Option<UnsizedType>,
    pub name: String,
    pub suffix: Suffix,
    pub args: Vec<(Ad, String, UnsizedType)>,
    /// `None` for a declaration without definition.
    pub body: Option<Stmt>,
    pub loc: Loc,
}

/// The block an output variable comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoBlock {
    Parameters,
    TransformedParameters,
    GeneratedQuantities,
}

/// An output variable.
#[derive(Clone, Debug, PartialEq)]
pub struct OutVar {
    pub unconstrained: SizedType,
    pub constrained: SizedType,
    pub block: IoBlock,
    pub trans: Transformation,
}

/// Output variables with their declaration locations.
pub type OutputVars = Vec<(String, Loc, OutVar)>;

/// A program.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Program {
    pub functions: Vec<FunDef>,
    pub input_vars: Vec<(String, Loc, SizedType)>,
    pub prepare_data: Vec<Stmt>,
    pub log_prob: Vec<Stmt>,
    pub generate_quantities: Vec<Stmt>,
    pub output_vars: OutputVars,
    pub name: String,
}
