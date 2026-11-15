//! Typed abstract syntax tree: the output of the type checker.
//!
//! Every expression carries its unsized type and autodiff level. Implicit conversions are explicit
//! [`TExprKind::Promotion`] nodes, so later stages never infer conversions themselves.

use stanc_rs_syntax::Span;
use stanc_rs_syntax::ast::{AdLevel, AssignOp, BinOp, Ident, PrefixOp, UnsizedType};

use crate::types::{Ad, Promotion};

/// Where a variable was declared.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Origin {
    MathLibrary,
    Functions,
    Data,
    TData,
    Param,
    TParam,
    Model,
    GQuant,
}

impl Origin {
    /// Block name as written in Stan.
    pub fn block_name(self) -> &'static str {
        match self {
            Origin::MathLibrary => "Stan Math Library",
            Origin::Functions => "functions block",
            Origin::Data => "data block",
            Origin::TData => "transformed data block",
            Origin::Param => "parameters block",
            Origin::TParam => "transformed parameters block",
            Origin::Model => "model block",
            Origin::GQuant => "generated quantities block",
        }
    }
}

/// Which function an application calls.
#[derive(Clone, Debug, PartialEq)]
pub enum FunKind {
    /// A Stan Math library function (including operators' named forms).
    StanLib,
    /// A user-defined function.
    UserDefined,
}

/// A typed expression.
#[derive(Clone, Debug, PartialEq)]
pub struct TExpr {
    /// Form.
    pub kind: TExprKind,
    /// Type.
    pub ty: UnsizedType,
    /// Autodiff level.
    pub ad: Ad,
    /// Location.
    pub span: Span,
}

/// Typed index.
#[derive(Clone, Debug, PartialEq)]
pub enum TIndex {
    All,
    Single(TExpr),
    Upfrom(TExpr),
    Downfrom(TExpr),
    Between(TExpr, TExpr),
}

impl TIndex {
    /// True unless the index is a single integer.
    pub fn is_multi(&self) -> bool {
        !matches!(self, TIndex::Single(e) if e.ty == UnsizedType::Int)
    }
}

/// Typed expression forms.
#[derive(Clone, Debug, PartialEq)]
pub enum TExprKind {
    /// A variable, with the block it comes from (`Functions` for function arguments and locals).
    Variable {
        name: String,
        origin: Origin,
        global: bool,
    },
    IntLit(i64),
    /// Real literal as written.
    RealLit(String),
    /// Imaginary literal as written, without the `i`.
    ImagLit(String),
    /// Function application. `name` is as written (may end in `_lupdf`).
    FunApp {
        kind: FunKind,
        name: String,
        args: Vec<TExpr>,
        cond: bool,
    },
    GetTarget,
    TernaryIf(Box<TExpr>, Box<TExpr>, Box<TExpr>),
    BinOp(Box<TExpr>, BinOp, Box<TExpr>),
    PrefixOp(PrefixOp, Box<TExpr>),
    Transpose(Box<TExpr>),
    Indexed(Box<TExpr>, Vec<TIndex>),
    ArrayExpr(Vec<TExpr>),
    RowVectorExpr(Vec<TExpr>),
    TupleExpr(Vec<TExpr>),
    /// One-based slot.
    TupleProjection(Box<TExpr>, usize),
    /// Implicit conversion of the inner expression to `ty`/`ad` of this node.
    Promotion(Box<TExpr>, Promotion),
}

/// Typed assignable value.
#[derive(Clone, Debug, PartialEq)]
pub struct TLValue {
    /// Form.
    pub kind: TLValueKind,
    /// Type of the assigned location.
    pub ty: UnsizedType,
    /// Autodiff level of the assigned location.
    pub ad: Ad,
    /// Location.
    pub span: Span,
}

/// Typed lvalue forms.
#[derive(Clone, Debug, PartialEq)]
pub enum TLValueKind {
    Variable { name: String, origin: Origin },
    Indexed(Box<TLValue>, Vec<TIndex>),
    TupleProjection(Box<TLValue>, usize),
    Tuple(Vec<TLValue>),
}

/// Typed printable.
#[derive(Clone, Debug, PartialEq)]
pub enum TPrintable {
    Str(String),
    Expr(TExpr),
}

/// Typed truncation bounds.
#[derive(Clone, Debug, PartialEq)]
pub enum TTruncation {
    None,
    UpFrom(TExpr),
    DownFrom(TExpr),
    Between(TExpr, TExpr),
}

/// A sized type with typed size expressions.
#[derive(Clone, Debug, PartialEq)]
pub enum TSizedType {
    Int,
    Real,
    Complex,
    Vector(TExpr),
    RowVector(TExpr),
    Matrix(TExpr, TExpr),
    ComplexVector(TExpr),
    ComplexRowVector(TExpr),
    ComplexMatrix(TExpr, TExpr),
    Array(Box<TSizedType>, TExpr),
    Tuple(Vec<TSizedType>),
}

impl TSizedType {
    /// The corresponding unsized type.
    pub fn unsized_type(&self) -> UnsizedType {
        match self {
            TSizedType::Int => UnsizedType::Int,
            TSizedType::Real => UnsizedType::Real,
            TSizedType::Complex => UnsizedType::Complex,
            TSizedType::Vector(_) => UnsizedType::Vector,
            TSizedType::RowVector(_) => UnsizedType::RowVector,
            TSizedType::Matrix(..) => UnsizedType::Matrix,
            TSizedType::ComplexVector(_) => UnsizedType::ComplexVector,
            TSizedType::ComplexRowVector(_) => UnsizedType::ComplexRowVector,
            TSizedType::ComplexMatrix(..) => UnsizedType::ComplexMatrix,
            TSizedType::Array(t, _) => UnsizedType::Array(Box::new(t.unsized_type())),
            TSizedType::Tuple(ts) => {
                UnsizedType::Tuple(ts.iter().map(|t| t.unsized_type()).collect())
            }
        }
    }

    /// Array sizes, outermost first, and the element type.
    pub fn array_parts(&self) -> (Vec<&TExpr>, &TSizedType) {
        match self {
            TSizedType::Array(t, e) => {
                let (mut dims, inner) = t.array_parts();
                dims.insert(0, e);
                (dims, inner)
            }
            t => (vec![], t),
        }
    }
}

/// Typed constraint transformation.
#[derive(Clone, Debug, PartialEq)]
pub enum TTransformation {
    Identity,
    Lower(TExpr),
    Upper(TExpr),
    LowerUpper(TExpr, TExpr),
    Offset(TExpr),
    Multiplier(TExpr),
    OffsetMultiplier(TExpr, TExpr),
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
    Tuple(Vec<TTransformation>),
}

/// A typed variable declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct TDeclVar {
    pub name: String,
    pub span: Span,
    pub init: Option<TExpr>,
}

/// A typed function argument.
#[derive(Clone, Debug, PartialEq)]
pub struct TFunArg {
    pub ad: AdLevel,
    pub ty: UnsizedType,
    pub name: String,
}

/// A typed statement.
#[derive(Clone, Debug, PartialEq)]
pub struct TStmt {
    pub kind: TStmtKind,
    pub span: Span,
}

/// Typed statement forms.
#[derive(Clone, Debug, PartialEq)]
pub enum TStmtKind {
    Assign {
        lhs: TLValue,
        op: AssignOp,
        rhs: TExpr,
    },
    NRFunApp {
        kind: FunKind,
        name: String,
        args: Vec<TExpr>,
    },
    TargetPlusAssign(TExpr),
    JacobianPlusAssign(TExpr),
    /// `arg ~ dist(args)`; `dist_name` is the resolved density function (e.g. `normal_lpdf`).
    Tilde {
        arg: TExpr,
        dist: String,
        dist_kind: FunKind,
        args: Vec<TExpr>,
        truncation: TTruncation,
    },
    Break,
    Continue,
    Return(Option<TExpr>),
    Print(Vec<TPrintable>),
    Reject(Vec<TPrintable>),
    FatalError(Vec<TPrintable>),
    Skip,
    IfThenElse(TExpr, Box<TStmt>, Option<Box<TStmt>>),
    While(TExpr, Box<TStmt>),
    For {
        var: String,
        lower: TExpr,
        upper: TExpr,
        body: Box<TStmt>,
    },
    ForEach {
        var: String,
        iter: TExpr,
        elem_ty: UnsizedType,
        body: Box<TStmt>,
    },
    Profile(String, Vec<TStmt>),
    Block(Vec<TStmt>),
    VarDecl {
        ty: TSizedType,
        trans: TTransformation,
        vars: Vec<TDeclVar>,
        is_global: bool,
        ad: Ad,
    },
    FunDef {
        ret: Option<UnsizedType>,
        name: String,
        args: Vec<TFunArg>,
        body: Option<Box<TStmt>>,
    },
}

/// A typed program.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TProgram {
    /// Model name (used in diagnostics and generated code).
    pub name: String,
    pub functions: Vec<TStmt>,
    pub data: Vec<TStmt>,
    pub transformed_data: Vec<TStmt>,
    pub parameters: Vec<TStmt>,
    pub transformed_parameters: Vec<TStmt>,
    pub model: Vec<TStmt>,
    pub generated_quantities: Vec<TStmt>,
}

/// Re-export for convenience.
pub type Identifier = Ident;
