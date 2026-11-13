//! Untyped abstract syntax tree of a Stan program.
//!
//! The structure follows the stanc3 v2.40.0 `Ast` module: every expression and statement carries
//! its source span, and the parser performs no numerical transformation.

use crate::source::Span;

/// An identifier with its location.
#[derive(Clone, Debug, PartialEq)]
pub struct Ident {
    /// The name.
    pub name: String,
    /// Location.
    pub span: Span,
}

/// Binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Plus,
    Minus,
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
}

impl BinOp {
    /// Source spelling.
    pub fn symbol(self) -> &'static str {
        use BinOp::*;
        match self {
            Plus => "+",
            Minus => "-",
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
        }
    }

    /// Name of the Stan Math function implementing the operator (as in the signature dump).
    pub fn function_name(self) -> &'static str {
        use BinOp::*;
        match self {
            Plus => "add",
            Minus => "subtract",
            Times => "multiply",
            Divide => "divide",
            IntDivide => "divide",
            Modulo => "modulus",
            LDivide => "mdivide_left",
            EltTimes => "elt_multiply",
            EltDivide => "elt_divide",
            Pow => "pow",
            EltPow => "pow",
            Or => "logical_or",
            And => "logical_and",
            Equals => "logical_eq",
            NEquals => "logical_neq",
            Less => "logical_lt",
            Leq => "logical_lte",
            Greater => "logical_gt",
            Geq => "logical_gte",
        }
    }
}

/// Prefix operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrefixOp {
    /// `!`
    Not,
    /// `-`
    Minus,
    /// `+`
    Plus,
}

impl PrefixOp {
    /// Source spelling.
    pub fn symbol(self) -> &'static str {
        match self {
            PrefixOp::Not => "!",
            PrefixOp::Minus => "-",
            PrefixOp::Plus => "+",
        }
    }

    /// Name of the Stan Math function implementing the operator.
    pub fn function_name(self) -> &'static str {
        match self {
            PrefixOp::Not => "logical_negation",
            PrefixOp::Minus => "minus",
            PrefixOp::Plus => "plus",
        }
    }
}

/// One index inside `[...]`.
#[derive(Clone, Debug, PartialEq)]
pub enum Index {
    /// `x[]` or `x[:]`.
    All,
    /// `x[i]` (also multi-indexing when `i` is an integer array).
    Single(Expr),
    /// `x[i:]`.
    Upfrom(Expr),
    /// `x[:j]`.
    Downfrom(Expr),
    /// `x[i:j]`.
    Between(Expr, Expr),
}

/// An expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    /// What the expression is.
    pub kind: ExprKind,
    /// Location.
    pub span: Span,
}

/// Expression forms.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    /// Variable reference.
    Variable(Ident),
    /// Integer literal as written (may contain `_`).
    IntNumeral(String),
    /// Real literal as written.
    RealNumeral(String),
    /// Imaginary literal, without the `i`.
    ImagNumeral(String),
    /// `f(args)`.
    FunApp(Ident, Vec<Expr>),
    /// `f(y | args)`, or `f_lpdf(y)` with a conditioning suffix and one argument.
    CondDistApp(Ident, Vec<Expr>),
    /// `target()`.
    GetTarget,
    /// `c ? a : b`.
    TernaryIf(Box<Expr>, Box<Expr>, Box<Expr>),
    /// `a op b`.
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    /// `op a`.
    PrefixOp(PrefixOp, Box<Expr>),
    /// `a'`.
    Transpose(Box<Expr>),
    /// `a[indices]`.
    Indexed(Box<Expr>, Vec<Index>),
    /// `{a, b}`.
    ArrayExpr(Vec<Expr>),
    /// `[a, b]`.
    RowVectorExpr(Vec<Expr>),
    /// `(a, b)`.
    TupleExpr(Vec<Expr>),
    /// `a.1` (one-based slot).
    TupleProjection(Box<Expr>, usize),
    /// `(a)`.
    Paren(Box<Expr>),
}

/// Assignment operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignOp {
    /// `=`
    Assign,
    /// `op=`, with the corresponding binary operator.
    Compound(BinOp),
}

/// The left side of an assignment.
#[derive(Clone, Debug, PartialEq)]
pub struct LValue {
    /// Form.
    pub kind: LValueKind,
    /// Location.
    pub span: Span,
}

/// Left side forms.
#[derive(Clone, Debug, PartialEq)]
pub enum LValueKind {
    /// `x`.
    Variable(Ident),
    /// `x[...]`.
    Indexed(Box<LValue>, Vec<Index>),
    /// `x.1`.
    TupleProjection(Box<LValue>, usize),
    /// `(a, b)`: tuple unpacking.
    Tuple(Vec<LValue>),
}

impl LValue {
    /// The variable being assigned (the leftmost identifier), if there is exactly one.
    pub fn base(&self) -> Option<&Ident> {
        match &self.kind {
            LValueKind::Variable(id) => Some(id),
            LValueKind::Indexed(l, _) | LValueKind::TupleProjection(l, _) => l.base(),
            LValueKind::Tuple(_) => None,
        }
    }
}

/// Truncation of a sampling statement.
#[derive(Clone, Debug, PartialEq)]
pub enum Truncation {
    /// None.
    None,
    /// `T[lo, ]`.
    UpFrom(Expr),
    /// `T[, hi]`.
    DownFrom(Expr),
    /// `T[lo, hi]`.
    Between(Expr, Expr),
}

/// An item of `print`, `reject`, or `fatal_error`.
#[derive(Clone, Debug, PartialEq)]
pub enum Printable {
    /// String literal contents including quotes.
    Str(String),
    /// An expression.
    Expr(Expr),
}

/// Unsized types, used for function arguments and return types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnsizedType {
    Int,
    Real,
    Complex,
    Vector,
    RowVector,
    Matrix,
    ComplexVector,
    ComplexRowVector,
    ComplexMatrix,
    Array(Box<UnsizedType>),
    Tuple(Vec<UnsizedType>),
    /// A function value (higher-order arguments of built-ins).
    Function(Box<FunctionType>),
}

/// The type of a function value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionType {
    /// Return type; `None` for `void`.
    pub ret: Option<UnsizedType>,
    /// Argument autodiff levels and types.
    pub args: Vec<(AdLevel, UnsizedType)>,
}

/// Whether a value may carry derivatives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AdLevel {
    /// Data only.
    DataOnly,
    /// May depend on parameters.
    AutoDiffable,
}

impl UnsizedType {
    /// Wraps `self` in `n` array levels.
    pub fn array_of(self, n: usize) -> UnsizedType {
        (0..n).fold(self, |t, _| UnsizedType::Array(Box::new(t)))
    }

    /// Number of array dimensions and the innermost element type.
    pub fn array_dims(&self) -> (usize, &UnsizedType) {
        match self {
            UnsizedType::Array(t) => {
                let (n, inner) = t.array_dims();
                (n + 1, inner)
            }
            t => (0, t),
        }
    }
}

impl std::fmt::Display for UnsizedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use UnsizedType::*;
        match self {
            Int => write!(f, "int"),
            Real => write!(f, "real"),
            Complex => write!(f, "complex"),
            Vector => write!(f, "vector"),
            RowVector => write!(f, "row_vector"),
            Matrix => write!(f, "matrix"),
            ComplexVector => write!(f, "complex_vector"),
            ComplexRowVector => write!(f, "complex_row_vector"),
            ComplexMatrix => write!(f, "complex_matrix"),
            Array(_) => {
                let (n, inner) = self.array_dims();
                write!(f, "array[{}] {}", ",".repeat(n - 1), inner)
            }
            Tuple(ts) => {
                write!(f, "tuple(")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{t}")?;
                }
                write!(f, ")")
            }
            Function(ft) => {
                write!(f, "(")?;
                for (i, (ad, t)) in ft.args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    if *ad == AdLevel::DataOnly {
                        write!(f, "data ")?;
                    }
                    write!(f, "{t}")?;
                }
                write!(f, ") => ")?;
                match &ft.ret {
                    Some(t) => write!(f, "{t}"),
                    None => write!(f, "void"),
                }
            }
        }
    }
}

/// Sized types of declarations; sizes are expressions.
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
    /// Element type and size.
    Array(Box<SizedType>, Expr),
    Tuple(Vec<SizedType>),
}

impl SizedType {
    /// The corresponding unsized type.
    pub fn unsized_type(&self) -> UnsizedType {
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
            SizedType::Array(t, _) => UnsizedType::Array(Box::new(t.unsized_type())),
            SizedType::Tuple(ts) => UnsizedType::Tuple(ts.iter().map(|t| t.unsized_type()).collect()),
        }
    }

    /// Size expressions, outermost first.
    pub fn size_exprs(&self) -> Vec<&Expr> {
        match self {
            SizedType::Int | SizedType::Real | SizedType::Complex | SizedType::Tuple(_) => vec![],
            SizedType::Vector(e) | SizedType::RowVector(e) | SizedType::ComplexVector(e) | SizedType::ComplexRowVector(e) => {
                vec![e]
            }
            SizedType::Matrix(a, b) | SizedType::ComplexMatrix(a, b) => vec![a, b],
            SizedType::Array(t, e) => {
                let mut v = vec![e];
                v.extend(t.size_exprs());
                v
            }
        }
    }
}

/// Constraint transformations of top-level declarations.
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
    StochasticColumn,
    StochasticRow,
    /// One transformation per tuple slot.
    Tuple(Vec<Transformation>),
}

/// A declared variable and its optional initial value.
#[derive(Clone, Debug, PartialEq)]
pub struct DeclVar {
    /// Name.
    pub ident: Ident,
    /// Initializer.
    pub init: Option<Expr>,
}

/// A function argument.
#[derive(Clone, Debug, PartialEq)]
pub struct FunArg {
    /// `data` qualifier present.
    pub ad: AdLevel,
    /// Type.
    pub ty: UnsizedType,
    /// Name.
    pub ident: Ident,
}

/// A statement.
#[derive(Clone, Debug, PartialEq)]
pub struct Stmt {
    /// Form.
    pub kind: StmtKind,
    /// Location.
    pub span: Span,
}

/// Statement forms.
#[derive(Clone, Debug, PartialEq)]
pub enum StmtKind {
    Assign { lhs: LValue, op: AssignOp, rhs: Expr },
    /// Function call used as a statement.
    NRFunApp(Ident, Vec<Expr>),
    TargetPlusAssign(Expr),
    JacobianPlusAssign(Expr),
    Break,
    Continue,
    Return(Option<Expr>),
    Print(Vec<Printable>),
    Reject(Vec<Printable>),
    FatalError(Vec<Printable>),
    /// `;`
    Skip,
    Tilde { arg: Expr, dist: Ident, args: Vec<Expr>, truncation: Truncation },
    IfThenElse(Expr, Box<Stmt>, Option<Box<Stmt>>),
    While(Expr, Box<Stmt>),
    For { var: Ident, lower: Expr, upper: Expr, body: Box<Stmt> },
    ForEach { var: Ident, iter: Expr, body: Box<Stmt> },
    Profile(String, Vec<Stmt>),
    Block(Vec<Stmt>),
    VarDecl { ty: SizedType, trans: Transformation, vars: Vec<DeclVar>, is_global: bool },
    FunDef { ret: Option<UnsizedType>, name: Ident, args: Vec<FunArg>, body: Box<Stmt> },
}

/// A program block.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    /// Statements.
    pub stmts: Vec<Stmt>,
    /// Location of the whole block.
    pub span: Span,
}

/// A whole Stan program.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Program {
    pub functions: Option<Block>,
    pub data: Option<Block>,
    pub transformed_data: Option<Block>,
    pub parameters: Option<Block>,
    pub transformed_parameters: Option<Block>,
    pub model: Option<Block>,
    pub generated_quantities: Option<Block>,
}

impl Program {
    /// Blocks in program order with their names.
    pub fn blocks(&self) -> [(&'static str, Option<&Block>); 7] {
        [
            ("functions", self.functions.as_ref()),
            ("data", self.data.as_ref()),
            ("transformed data", self.transformed_data.as_ref()),
            ("parameters", self.parameters.as_ref()),
            ("transformed parameters", self.transformed_parameters.as_ref()),
            ("model", self.model.as_ref()),
            ("generated quantities", self.generated_quantities.as_ref()),
        ]
    }
}

/// Distribution suffixes that make a one-argument call a conditional application.
pub const CONDITIONING_SUFFIXES: &[&str] = &["_lpdf", "_lupdf", "_lpmf", "_lupmf", "_cdf", "_lcdf", "_lccdf"];
