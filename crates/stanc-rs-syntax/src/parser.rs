//! Recursive descent parser for the Stan grammar of stanc3 v2.40.0 (`parser.mly`).
//!
//! Operator precedence, lowest first:
//!
//! | Level | Operators | Associativity |
//! | --- | --- | --- |
//! | ternary | `? :` | right |
//! | 1 | `\|\|` | left |
//! | 2 | `&&` | left |
//! | 3 | `==` `!=` | left |
//! | 4 | `<` `<=` `>` `>=` | left |
//! | 5 | `+` `-` | left |
//! | 6 | `*` `/` `%` `.*` `./` | left |
//! | 7 | `%/%` `\` | left |
//! | prefix | `!` `-` `+` | |
//! | power | `^` `.^` | right |
//! | postfix | `'` | |
//!
//! Constraint expressions (inside `<lower=..., upper=...>`) use only the arithmetic levels, so
//! that `>` closes the constraint.

use crate::ast::*;
use crate::lexer::{Tok, Token};
use crate::source::{Diagnostic, DiagnosticKind, Span};

type PResult<T> = Result<T, Diagnostic>;

/// Parses a full program from tokens produced by [`crate::lexer::tokenize`].
pub fn parse_program(tokens: &[Token]) -> PResult<Program> {
    let mut p = Parser {
        toks: tokens,
        pos: 0,
    };
    p.program()
}

/// Parses a `.stanfunctions` file: function definitions only.
pub fn parse_functions_only(tokens: &[Token]) -> PResult<Program> {
    let mut p = Parser {
        toks: tokens,
        pos: 0,
    };
    let start = p.span();
    let mut stmts = Vec::new();
    while p.peek() != &Tok::Eof {
        stmts.push(p.function_def()?);
    }
    let span = start.to(p.prev_span());
    Ok(Program {
        functions: Some(Block { stmts, span }),
        ..Program::default()
    })
}

#[derive(Clone, Copy, PartialEq)]
enum DeclCtx {
    /// Data and parameters blocks: constraints allowed, no initializer.
    TopNoAssign,
    /// Transformed data, transformed parameters, generated quantities: constraints and initializers.
    Top,
    /// Model block, nested blocks, and function bodies: no constraints.
    Local,
}

struct Parser<'a> {
    toks: &'a [Token],
    pos: usize,
}

const BINARY_LEVELS: &[&[(Tok, BinOp)]] = &[
    &[(Tok::Or, BinOp::Or)],
    &[(Tok::And, BinOp::And)],
    &[(Tok::Equals, BinOp::Equals), (Tok::NEquals, BinOp::NEquals)],
    &[
        (Tok::LAngle, BinOp::Less),
        (Tok::Leq, BinOp::Leq),
        (Tok::RAngle, BinOp::Greater),
        (Tok::Geq, BinOp::Geq),
    ],
    &[(Tok::Plus, BinOp::Plus), (Tok::Minus, BinOp::Minus)],
    &[
        (Tok::Times, BinOp::Times),
        (Tok::Divide, BinOp::Divide),
        (Tok::Modulo, BinOp::Modulo),
        (Tok::EltTimes, BinOp::EltTimes),
        (Tok::EltDivide, BinOp::EltDivide),
    ],
    &[
        (Tok::IDivide, BinOp::IntDivide),
        (Tok::LDivide, BinOp::LDivide),
    ],
];

/// First arithmetic level (`+ -`), the lowest level allowed in constraint expressions.
const ARITH_LEVEL: usize = 4;

impl<'a> Parser<'a> {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn peek_at(&self, k: usize) -> &Tok {
        let i = (self.pos + k).min(self.toks.len() - 1);
        &self.toks[i].tok
    }

    fn span(&self) -> Span {
        self.toks[self.pos].span
    }

    fn prev_span(&self) -> Span {
        self.toks[self.pos.saturating_sub(1)].span
    }

    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == t {
            self.bump();
            true
        } else {
            false
        }
    }

    fn error<T>(&self, msg: impl Into<String>) -> PResult<T> {
        Err(Diagnostic::new(DiagnosticKind::Syntax, self.span(), msg))
    }

    fn unexpected<T>(&self, expected: &str) -> PResult<T> {
        self.error(format!(
            "Ill-formed phrase. Found \"{}\". Expected {}.",
            self.peek().describe(),
            expected
        ))
    }

    fn expect(&mut self, t: Tok, what: &str) -> PResult<Span> {
        if self.peek() == &t {
            Ok(self.bump().span)
        } else {
            self.unexpected(what)
        }
    }

    // ----- Program and blocks -----

    fn program(&mut self) -> PResult<Program> {
        let mut prog = Program::default();
        let order = [
            Tok::Functions,
            Tok::Data,
            Tok::TransformedData,
            Tok::Parameters,
            Tok::TransformedParameters,
            Tok::Model,
            Tok::GeneratedQuantities,
        ];
        let mut next = 0;
        while self.peek() != &Tok::Eof {
            let Some(k) = order.iter().position(|t| t == self.peek()) else {
                return self.unexpected(
                    "\"functions {\", \"data {\", \"transformed data {\", \"parameters {\", \"transformed parameters {\", \"model {\" or \"generated quantities {\"",
                );
            };
            if k < next {
                return self.error(format!(
                    "Ill-formed program. Found block \"{}\" out of order or repeated; blocks must appear in the order functions, data, transformed data, parameters, transformed parameters, model, generated quantities.",
                    self.peek().describe()
                ));
            }
            next = k + 1;
            let start = self.bump().span;
            self.expect(Tok::LBrace, "\"{\" to open the block")?;
            let mut stmts = Vec::new();
            while self.peek() != &Tok::RBrace {
                if self.peek() == &Tok::Eof {
                    return self.unexpected("\"}\" to close the block");
                }
                let s = match k {
                    0 => self.function_def()?,
                    1 | 3 => self.top_no_assign()?,
                    5 => self.vardecl_or_statement(DeclCtx::Local)?,
                    _ => self.vardecl_or_statement(DeclCtx::Top)?,
                };
                stmts.push(s);
            }
            let end = self.bump().span;
            let block = Some(Block {
                stmts,
                span: start.to(end),
            });
            match k {
                0 => prog.functions = block,
                1 => prog.data = block,
                2 => prog.transformed_data = block,
                3 => prog.parameters = block,
                4 => prog.transformed_parameters = block,
                5 => prog.model = block,
                _ => prog.generated_quantities = block,
            }
        }
        Ok(prog)
    }

    fn top_no_assign(&mut self) -> PResult<Stmt> {
        if self.peek() == &Tok::Semi {
            let span = self.bump().span;
            return Ok(Stmt {
                kind: StmtKind::Skip,
                span,
            });
        }
        if !self.peek().is_type_keyword() {
            return self
                .unexpected("a variable declaration (statements are not allowed in this block)");
        }
        self.var_decl(DeclCtx::TopNoAssign)
    }

    // ----- Functions -----

    fn function_def(&mut self) -> PResult<Stmt> {
        let start = self.span();
        let ret = if self.eat(&Tok::Void) {
            None
        } else if self.peek().is_type_keyword() {
            Some(self.unsized_type()?)
        } else {
            return self.unexpected("a function return type (\"void\" or a type)");
        };
        let name = self.decl_identifier()?;
        self.expect(Tok::LParen, "\"(\" to begin the argument list")?;
        let mut args = Vec::new();
        if self.peek() != &Tok::RParen {
            loop {
                let ad = if self.eat(&Tok::Data) {
                    AdLevel::DataOnly
                } else {
                    AdLevel::AutoDiffable
                };
                if !self.peek().is_type_keyword() {
                    return self.unexpected("an argument type");
                }
                let ty = self.unsized_type()?;
                let ident = self.decl_identifier()?;
                args.push(FunArg { ad, ty, ident });
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        self.expect(Tok::RParen, "\",\" or \")\" in the argument list")?;
        let body = if self.peek() == &Tok::Semi {
            let span = self.bump().span;
            Stmt {
                kind: StmtKind::Skip,
                span,
            }
        } else if self.peek() == &Tok::LBrace {
            self.statement(DeclCtx::Local)?
        } else {
            return self
                .unexpected("\"{\" to begin the function body or \";\" for a forward declaration");
        };
        let span = start.to(body.span);
        Ok(Stmt {
            kind: StmtKind::FunDef {
                ret,
                name,
                args,
                body: Box::new(body),
            },
            span,
        })
    }

    fn basic_unsized(&mut self) -> PResult<UnsizedType> {
        let t = match self.peek() {
            Tok::Int => UnsizedType::Int,
            Tok::Real => UnsizedType::Real,
            Tok::Complex => UnsizedType::Complex,
            Tok::Vector => UnsizedType::Vector,
            Tok::RowVector => UnsizedType::RowVector,
            Tok::Matrix => UnsizedType::Matrix,
            Tok::ComplexVector => UnsizedType::ComplexVector,
            Tok::ComplexRowVector => UnsizedType::ComplexRowVector,
            Tok::ComplexMatrix => UnsizedType::ComplexMatrix,
            t if t.is_type_keyword() => {
                return self.error(format!(
                    "Ill-formed type. Constrained type \"{}\" cannot be used for function arguments or return types.",
                    t.describe()
                ));
            }
            _ => return self.unexpected("a type"),
        };
        self.bump();
        if self.peek() == &Tok::LBrack {
            return self.error(
                "Ill-formed type. Sizes are not allowed in function argument or return types; arrays use \"array[] type\".",
            );
        }
        Ok(t)
    }

    fn unsized_type(&mut self) -> PResult<UnsizedType> {
        match self.peek() {
            Tok::Array => {
                self.bump();
                self.expect(Tok::LBrack, "\"[\" after \"array\"")?;
                let mut n = 1;
                while self.eat(&Tok::Comma) {
                    n += 1;
                }
                self.expect(Tok::RBrack, "\",\" or \"]\" in unsized array dimensions")?;
                let inner = if self.peek() == &Tok::Tuple {
                    self.unsized_tuple()?
                } else {
                    self.basic_unsized()?
                };
                Ok(inner.array_of(n))
            }
            Tok::Tuple => self.unsized_tuple(),
            _ => self.basic_unsized(),
        }
    }

    fn unsized_tuple(&mut self) -> PResult<UnsizedType> {
        self.bump();
        self.expect(Tok::LParen, "\"(\" after \"tuple\"")?;
        let mut ts = vec![self.unsized_type()?];
        self.expect(Tok::Comma, "\",\": tuples need at least two elements")?;
        if self.peek() != &Tok::RParen {
            loop {
                ts.push(self.unsized_type()?);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        self.expect(Tok::RParen, "\",\" or \")\" in the tuple type")?;
        Ok(UnsizedType::Tuple(ts))
    }

    // ----- Identifiers -----

    fn identifier(&mut self) -> PResult<Ident> {
        match self.peek().clone() {
            Tok::Ident(name) => {
                let span = self.bump().span;
                Ok(Ident { name, span })
            }
            Tok::Truncate => {
                let span = self.bump().span;
                Ok(Ident {
                    name: "T".into(),
                    span,
                })
            }
            _ => self.unexpected("an identifier"),
        }
    }

    fn decl_identifier(&mut self) -> PResult<Ident> {
        if let Some(name) = self.peek().reserved_name() {
            return self.error(format!(
                "Ill-formed identifier. Expected a new identifier but found reserved keyword \"{name}\"."
            ));
        }
        self.identifier()
    }

    // ----- Declarations -----

    fn var_decl(&mut self, ctx: DeclCtx) -> PResult<Stmt> {
        let start = self.span();
        if self.peek() == &Tok::Data {
            return self.error("Ill-formed declaration. The \"data\" qualifier is only allowed on function arguments.");
        }
        let (ty, trans) = self.higher_type(ctx)?;
        let mut vars = Vec::new();
        loop {
            if !vars.is_empty() && self.peek().is_type_keyword() {
                return self.error(format!(
                    "Ill-formed identifier. Found a type (\"{}\") where an identifier was expected. All variables declared in a comma-separated list must be of the same type.",
                    self.peek().describe()
                ));
            }
            let ident = self.decl_identifier()?;
            if self.peek() == &Tok::LBrack {
                return self.error(
                    "Ill-formed declaration. \";\" expected after variable declaration. It looks like you are trying to use the old array syntax. Please use the new syntax: array[size] type name;",
                );
            }
            let init = if self.peek() == &Tok::Assign {
                if ctx == DeclCtx::TopNoAssign {
                    return self.error("Ill-formed declaration. Variables in this block cannot be assigned a value.");
                }
                self.bump();
                Some(self.expr()?)
            } else {
                None
            };
            vars.push(DeclVar { ident, init });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        let end = self.expect(Tok::Semi, "\";\" or \",\" after the declaration")?;
        let is_global = ctx != DeclCtx::Local;
        Ok(Stmt {
            kind: StmtKind::VarDecl {
                ty,
                trans,
                vars,
                is_global,
            },
            span: start.to(end),
        })
    }

    fn higher_type(&mut self, ctx: DeclCtx) -> PResult<(SizedType, Transformation)> {
        match self.peek() {
            Tok::Array => {
                self.bump();
                self.expect(Tok::LBrack, "\"[\" after \"array\"")?;
                let mut dims = vec![self.expr()?];
                while self.eat(&Tok::Comma) {
                    dims.push(self.expr()?);
                }
                self.expect(Tok::RBrack, "\",\" or \"]\" in array dimensions")?;
                if self.peek() == &Tok::Array {
                    return self.error(
                        "Ill-formed type. Arrays cannot directly contain arrays; use array[N, M].",
                    );
                }
                let (inner, trans) = if self.peek() == &Tok::Tuple {
                    self.tuple_type(ctx)?
                } else {
                    self.base_type(ctx)?
                };
                let ty = dims
                    .into_iter()
                    .rev()
                    .fold(inner, |t, d| SizedType::Array(Box::new(t), d));
                Ok((ty, trans))
            }
            Tok::Tuple => self.tuple_type(ctx),
            _ => self.base_type(ctx),
        }
    }

    fn tuple_type(&mut self, ctx: DeclCtx) -> PResult<(SizedType, Transformation)> {
        self.bump();
        self.expect(Tok::LParen, "\"(\" after \"tuple\"")?;
        let mut ts = vec![self.higher_type(ctx)?];
        self.expect(Tok::Comma, "\",\": tuples need at least two elements")?;
        if self.peek() != &Tok::RParen {
            loop {
                ts.push(self.higher_type(ctx)?);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        self.expect(Tok::RParen, "\",\" or \")\" in the tuple type")?;
        let (types, trans): (Vec<_>, Vec<_>) = ts.into_iter().unzip();
        Ok((SizedType::Tuple(types), Transformation::Tuple(trans)))
    }

    fn bracket1(&mut self) -> PResult<Expr> {
        self.expect(Tok::LBrack, "\"[\" and a size")?;
        let e = self.expr()?;
        self.expect(Tok::RBrack, "\"]\"")?;
        Ok(e)
    }

    fn bracket2(&mut self) -> PResult<(Expr, Expr)> {
        self.expect(Tok::LBrack, "\"[\" and sizes")?;
        let a = self.expr()?;
        self.expect(Tok::Comma, "\",\" and a second size")?;
        let b = self.expr()?;
        self.expect(Tok::RBrack, "\"]\"")?;
        Ok((a, b))
    }

    fn base_type(&mut self, ctx: DeclCtx) -> PResult<(SizedType, Transformation)> {
        let local = ctx == DeclCtx::Local;
        let tok = self.peek().clone();
        let constrained = matches!(
            tok,
            Tok::Ordered
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
        );
        if local && constrained {
            return self.error(format!(
                "Ill-formed type. Constrained type \"{}\" is only allowed for top-level block variables, not local variables.",
                tok.describe()
            ));
        }
        if !tok.is_type_keyword() || tok == Tok::Array || tok == Tok::Tuple {
            return self.unexpected("a type");
        }
        self.bump();
        let trans_for = |p: &mut Self, allow_offset: bool| -> PResult<Transformation> {
            if p.peek() != &Tok::LAngle {
                return Ok(Transformation::Identity);
            }
            if local {
                return p.error("Ill-formed type. Constraints are only allowed on top-level block variables, not local variables.");
            }
            p.constraint(allow_offset)
        };
        Ok(match tok {
            Tok::Int => (SizedType::Int, trans_for(self, false)?),
            Tok::Real => (SizedType::Real, trans_for(self, true)?),
            Tok::Complex => (SizedType::Complex, trans_for(self, true)?),
            Tok::Vector => {
                let t = trans_for(self, true)?;
                (SizedType::Vector(self.bracket1()?), t)
            }
            Tok::RowVector => {
                let t = trans_for(self, true)?;
                (SizedType::RowVector(self.bracket1()?), t)
            }
            Tok::Matrix => {
                let t = trans_for(self, true)?;
                let (a, b) = self.bracket2()?;
                (SizedType::Matrix(a, b), t)
            }
            Tok::ComplexVector => {
                let t = trans_for(self, true)?;
                (SizedType::ComplexVector(self.bracket1()?), t)
            }
            Tok::ComplexRowVector => {
                let t = trans_for(self, true)?;
                (SizedType::ComplexRowVector(self.bracket1()?), t)
            }
            Tok::ComplexMatrix => {
                let t = trans_for(self, true)?;
                let (a, b) = self.bracket2()?;
                (SizedType::ComplexMatrix(a, b), t)
            }
            Tok::Ordered => (SizedType::Vector(self.bracket1()?), Transformation::Ordered),
            Tok::PositiveOrdered => (
                SizedType::Vector(self.bracket1()?),
                Transformation::PositiveOrdered,
            ),
            Tok::Simplex => (SizedType::Vector(self.bracket1()?), Transformation::Simplex),
            Tok::UnitVector => (
                SizedType::Vector(self.bracket1()?),
                Transformation::UnitVector,
            ),
            Tok::SumToZeroVector => (
                SizedType::Vector(self.bracket1()?),
                Transformation::SumToZero,
            ),
            Tok::CholeskyFactorCorr => {
                let e = self.bracket1()?;
                (
                    SizedType::Matrix(e.clone(), e),
                    Transformation::CholeskyCorr,
                )
            }
            Tok::CholeskyFactorCov => {
                self.expect(Tok::LBrack, "\"[\" and a size")?;
                let a = self.expr()?;
                let b = if self.eat(&Tok::Comma) {
                    self.expr()?
                } else {
                    a.clone()
                };
                self.expect(Tok::RBrack, "\"]\"")?;
                (SizedType::Matrix(a, b), Transformation::CholeskyCov)
            }
            Tok::CorrMatrix => {
                let e = self.bracket1()?;
                (SizedType::Matrix(e.clone(), e), Transformation::Correlation)
            }
            Tok::CovMatrix => {
                let e = self.bracket1()?;
                (SizedType::Matrix(e.clone(), e), Transformation::Covariance)
            }
            Tok::SumToZeroMatrix => {
                let (a, b) = self.bracket2()?;
                (SizedType::Matrix(a, b), Transformation::SumToZero)
            }
            Tok::ColumnStochasticMatrix => {
                let (a, b) = self.bracket2()?;
                (SizedType::Matrix(a, b), Transformation::StochasticColumn)
            }
            Tok::RowStochasticMatrix => {
                let (a, b) = self.bracket2()?;
                (SizedType::Matrix(a, b), Transformation::StochasticRow)
            }
            _ => unreachable!("checked above"),
        })
    }

    fn constraint(&mut self, allow_offset: bool) -> PResult<Transformation> {
        self.expect(Tok::LAngle, "\"<\"")?;
        let first = self.peek().clone();
        let t = match first {
            Tok::Lower | Tok::Upper => {
                self.bump();
                self.expect(Tok::Assign, "\"=\" after \"lower\" or \"upper\"")?;
                let a = self.constr_expr()?;
                if self.eat(&Tok::Comma) {
                    let want = if first == Tok::Lower {
                        Tok::Upper
                    } else {
                        Tok::Lower
                    };
                    if self.peek() != &want {
                        return self.unexpected(if first == Tok::Lower {
                            "\"upper\""
                        } else {
                            "\"lower\""
                        });
                    }
                    self.bump();
                    self.expect(Tok::Assign, "\"=\"")?;
                    let b = self.constr_expr()?;
                    if first == Tok::Lower {
                        Transformation::LowerUpper(a, b)
                    } else {
                        Transformation::LowerUpper(b, a)
                    }
                } else if first == Tok::Lower {
                    Transformation::Lower(a)
                } else {
                    Transformation::Upper(a)
                }
            }
            Tok::Offset | Tok::Multiplier if allow_offset => {
                self.bump();
                self.expect(Tok::Assign, "\"=\" after \"offset\" or \"multiplier\"")?;
                let a = self.constr_expr()?;
                if self.eat(&Tok::Comma) {
                    let want = if first == Tok::Offset {
                        Tok::Multiplier
                    } else {
                        Tok::Offset
                    };
                    if self.peek() != &want {
                        return self.unexpected(if first == Tok::Offset {
                            "\"multiplier\""
                        } else {
                            "\"offset\""
                        });
                    }
                    self.bump();
                    self.expect(Tok::Assign, "\"=\"")?;
                    let b = self.constr_expr()?;
                    if first == Tok::Offset {
                        Transformation::OffsetMultiplier(a, b)
                    } else {
                        Transformation::OffsetMultiplier(b, a)
                    }
                } else if first == Tok::Offset {
                    Transformation::Offset(a)
                } else {
                    Transformation::Multiplier(a)
                }
            }
            _ => {
                return self.unexpected(if allow_offset {
                    "\"lower\", \"upper\", \"offset\" or \"multiplier\""
                } else {
                    "\"lower\" or \"upper\""
                });
            }
        };
        self.expect(Tok::RAngle, "\">\" to close the constraint")?;
        Ok(t)
    }

    // ----- Statements -----

    fn vardecl_or_statement(&mut self, ctx: DeclCtx) -> PResult<Stmt> {
        if self.peek().is_type_keyword() || self.peek() == &Tok::Data {
            self.var_decl(ctx)
        } else {
            self.statement(ctx)
        }
    }

    fn statement(&mut self, ctx: DeclCtx) -> PResult<Stmt> {
        // Nested statements always declare local variables.
        let _ = ctx;
        let start = self.span();
        let kind = match self.peek().clone() {
            Tok::LBrace => {
                // A brace starts a block, or an array expression in `{a, b} ~ normal(...)`.
                let save = self.pos;
                self.bump();
                match self.stmt_list() {
                    Ok(stmts) => StmtKind::Block(stmts),
                    Err(block_err) => {
                        self.pos = save;
                        return self.expr_statement(start).map_err(|_| block_err);
                    }
                }
            }
            Tok::If => {
                self.bump();
                self.expect(Tok::LParen, "\"(\" after \"if\"")?;
                let c = self.expr()?;
                self.expect(Tok::RParen, "\")\" after the condition")?;
                let then = self.vardecl_or_statement(DeclCtx::Local)?;
                let els = if self.eat(&Tok::Else) {
                    Some(Box::new(self.vardecl_or_statement(DeclCtx::Local)?))
                } else {
                    None
                };
                StmtKind::IfThenElse(c, Box::new(then), els)
            }
            Tok::While => {
                self.bump();
                self.expect(Tok::LParen, "\"(\" after \"while\"")?;
                let c = self.expr()?;
                self.expect(Tok::RParen, "\")\" after the condition")?;
                StmtKind::While(c, Box::new(self.vardecl_or_statement(DeclCtx::Local)?))
            }
            Tok::For => {
                self.bump();
                self.expect(Tok::LParen, "\"(\" after \"for\"")?;
                let var = self.identifier()?;
                self.expect(Tok::In, "\"in\"")?;
                let e = self.expr()?;
                if self.eat(&Tok::Colon) {
                    let upper = self.expr()?;
                    self.expect(Tok::RParen, "\")\"")?;
                    let body = self.vardecl_or_statement(DeclCtx::Local)?;
                    StmtKind::For {
                        var,
                        lower: e,
                        upper,
                        body: Box::new(body),
                    }
                } else {
                    self.expect(Tok::RParen, "\":\" or \")\"")?;
                    let body = self.vardecl_or_statement(DeclCtx::Local)?;
                    StmtKind::ForEach {
                        var,
                        iter: e,
                        body: Box::new(body),
                    }
                }
            }
            Tok::Profile => {
                self.bump();
                self.expect(Tok::LParen, "\"(\" after \"profile\"")?;
                let name = match self.peek().clone() {
                    Tok::StringLit(s) => {
                        self.bump();
                        s
                    }
                    _ => return self.unexpected("a string literal naming the profile"),
                };
                self.expect(Tok::RParen, "\")\"")?;
                self.expect(Tok::LBrace, "\"{\"")?;
                StmtKind::Profile(name, self.stmt_list()?)
            }
            Tok::Semi => {
                self.bump();
                StmtKind::Skip
            }
            Tok::Break => {
                self.bump();
                self.expect(Tok::Semi, "\";\"")?;
                StmtKind::Break
            }
            Tok::Continue => {
                self.bump();
                self.expect(Tok::Semi, "\";\"")?;
                StmtKind::Continue
            }
            Tok::Return => {
                self.bump();
                if self.eat(&Tok::Semi) {
                    StmtKind::Return(None)
                } else {
                    let e = self.expr()?;
                    self.expect(Tok::Semi, "\";\"")?;
                    StmtKind::Return(Some(e))
                }
            }
            Tok::Print | Tok::Reject | Tok::FatalError => {
                let t = self.bump().tok;
                self.expect(Tok::LParen, "\"(\"")?;
                let mut items = Vec::new();
                loop {
                    match self.peek().clone() {
                        Tok::StringLit(s) => {
                            self.bump();
                            items.push(Printable::Str(s));
                        }
                        _ => items.push(Printable::Expr(self.expr()?)),
                    }
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(Tok::RParen, "\",\" or \")\"")?;
                self.expect(Tok::Semi, "\";\"")?;
                match t {
                    Tok::Print => StmtKind::Print(items),
                    Tok::Reject => StmtKind::Reject(items),
                    _ => StmtKind::FatalError(items),
                }
            }
            Tok::Target if self.peek_at(1) == &Tok::PlusAssign => {
                self.bump();
                self.bump();
                let e = self.expr()?;
                self.expect(Tok::Semi, "\";\"")?;
                StmtKind::TargetPlusAssign(e)
            }
            Tok::Jacobian if self.peek_at(1) == &Tok::PlusAssign => {
                self.bump();
                self.bump();
                let e = self.expr()?;
                self.expect(Tok::Semi, "\";\"")?;
                StmtKind::JacobianPlusAssign(e)
            }
            _ => return self.expr_statement(start),
        };
        Ok(Stmt {
            kind,
            span: start.to(self.prev_span()),
        })
    }

    fn stmt_list(&mut self) -> PResult<Vec<Stmt>> {
        let mut stmts = Vec::new();
        while self.peek() != &Tok::RBrace {
            if self.peek() == &Tok::Eof {
                return self.unexpected("\"}\"");
            }
            stmts.push(self.vardecl_or_statement(DeclCtx::Local)?);
        }
        self.bump();
        Ok(stmts)
    }

    fn expr_statement(&mut self, start: Span) -> PResult<Stmt> {
        let e = self.expr()?;
        let op = match self.peek() {
            Tok::Assign => Some(AssignOp::Assign),
            Tok::PlusAssign => Some(AssignOp::Compound(BinOp::Plus)),
            Tok::MinusAssign => Some(AssignOp::Compound(BinOp::Minus)),
            Tok::TimesAssign => Some(AssignOp::Compound(BinOp::Times)),
            Tok::DivideAssign => Some(AssignOp::Compound(BinOp::Divide)),
            Tok::EltTimesAssign => Some(AssignOp::Compound(BinOp::EltTimes)),
            Tok::EltDivideAssign => Some(AssignOp::Compound(BinOp::EltDivide)),
            _ => None,
        };
        if let Some(op) = op {
            let Some(lhs) = lvalue_of_expr(&e) else {
                return Err(Diagnostic::new(
                    DiagnosticKind::Syntax,
                    e.span,
                    "Ill-formed assignment. Expected an assignable value but found a general expression.",
                ));
            };
            self.bump();
            let rhs = self.expr()?;
            self.expect(Tok::Semi, "\";\" after the assignment")?;
            return Ok(Stmt {
                kind: StmtKind::Assign { lhs, op, rhs },
                span: start.to(self.prev_span()),
            });
        }
        if self.eat(&Tok::Tilde) {
            let dist = self.identifier()?;
            self.expect(Tok::LParen, "\"(\" after the distribution name")?;
            let args = self.expr_list(Tok::RParen)?;
            let truncation = if self.peek() == &Tok::Truncate && self.peek_at(1) == &Tok::LBrack {
                self.bump();
                self.bump();
                let lo = if self.peek() == &Tok::Comma {
                    None
                } else {
                    Some(self.expr()?)
                };
                self.expect(Tok::Comma, "\",\" in the truncation")?;
                let hi = if self.peek() == &Tok::RBrack {
                    None
                } else {
                    Some(self.expr()?)
                };
                self.expect(Tok::RBrack, "\"]\" to close the truncation")?;
                match (lo, hi) {
                    (Some(a), Some(b)) => Truncation::Between(a, b),
                    (Some(a), None) => Truncation::UpFrom(a),
                    (None, Some(b)) => Truncation::DownFrom(b),
                    (None, None) => Truncation::None,
                }
            } else {
                Truncation::None
            };
            self.expect(Tok::Semi, "\";\" after the distribution statement")?;
            return Ok(Stmt {
                kind: StmtKind::Tilde {
                    arg: e,
                    dist,
                    args,
                    truncation,
                },
                span: start.to(self.prev_span()),
            });
        }
        if self.peek() == &Tok::Semi {
            if let ExprKind::FunApp(name, args) = &e.kind {
                self.bump();
                return Ok(Stmt {
                    kind: StmtKind::NRFunApp(name.clone(), args.clone()),
                    span: start.to(self.prev_span()),
                });
            }
            if let ExprKind::CondDistApp(name, args) = &e.kind
                && args.len() == 1
            {
                // `foo_lpdf(x);` with no bar parses as a plain call statement in stanc3.
                self.bump();
                return Ok(Stmt {
                    kind: StmtKind::NRFunApp(name.clone(), args.clone()),
                    span: start.to(self.prev_span()),
                });
            }
            return Err(Diagnostic::new(
                DiagnosticKind::Syntax,
                self.span(),
                "Ill-formed statement. Expressions cannot be used as statements; only function calls, assignments, and distribution statements are allowed.",
            ));
        }
        self.unexpected("an assignment operator, \"~\", or \";\" after the expression")
    }

    // ----- Expressions -----

    fn expr_list(&mut self, close: Tok) -> PResult<Vec<Expr>> {
        let mut v = Vec::new();
        if self.peek() != &close {
            loop {
                v.push(self.expr()?);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        let what = format!("\",\" or \"{}\"", close.describe());
        self.expect(close, &what)?;
        Ok(v)
    }

    fn expr(&mut self) -> PResult<Expr> {
        let cond = self.binary(0)?;
        if self.eat(&Tok::QMark) {
            let a = self.expr()?;
            self.expect(Tok::Colon, "\":\" in the conditional expression")?;
            let b = self.expr()?;
            let span = cond.span.to(b.span);
            return Ok(Expr {
                kind: ExprKind::TernaryIf(Box::new(cond), Box::new(a), Box::new(b)),
                span,
            });
        }
        Ok(cond)
    }

    fn constr_expr(&mut self) -> PResult<Expr> {
        self.binary(ARITH_LEVEL)
    }

    fn binary(&mut self, level: usize) -> PResult<Expr> {
        if level >= BINARY_LEVELS.len() {
            return self.prefix();
        }
        let mut lhs = self.binary(level + 1)?;
        while let Some(op) = BINARY_LEVELS[level]
            .iter()
            .find(|(t, _)| t == self.peek())
            .map(|(_, o)| *o)
        {
            self.bump();
            let rhs = self.binary(level + 1)?;
            let span = lhs.span.to(rhs.span);
            lhs = Expr {
                kind: ExprKind::BinOp(Box::new(lhs), op, Box::new(rhs)),
                span,
            };
        }
        Ok(lhs)
    }

    fn prefix(&mut self) -> PResult<Expr> {
        let op = match self.peek() {
            Tok::Bang => PrefixOp::Not,
            Tok::Minus => PrefixOp::Minus,
            Tok::Plus => PrefixOp::Plus,
            _ => return self.power(),
        };
        let start = self.bump().span;
        let e = self.prefix()?;
        let span = start.to(e.span);
        Ok(Expr {
            kind: ExprKind::PrefixOp(op, Box::new(e)),
            span,
        })
    }

    fn power(&mut self) -> PResult<Expr> {
        let base = self.postfix()?;
        let op = match self.peek() {
            Tok::Hat => BinOp::Pow,
            Tok::EltPow => BinOp::EltPow,
            _ => return Ok(base),
        };
        self.bump();
        let rhs = self.prefix()?;
        let span = base.span.to(rhs.span);
        Ok(Expr {
            kind: ExprKind::BinOp(Box::new(base), op, Box::new(rhs)),
            span,
        })
    }

    fn postfix(&mut self) -> PResult<Expr> {
        let mut e = self.common()?;
        while self.peek() == &Tok::Transpose {
            let end = self.bump().span;
            let span = e.span.to(end);
            e = Expr {
                kind: ExprKind::Transpose(Box::new(e)),
                span,
            };
        }
        Ok(e)
    }

    fn common(&mut self) -> PResult<Expr> {
        let start = self.span();
        let mut e = match self.peek().clone() {
            Tok::Ident(_) | Tok::Truncate => {
                let id = self.identifier()?;
                if self.peek() == &Tok::LParen {
                    self.bump();
                    if self.peek() == &Tok::RParen {
                        self.bump();
                        Expr {
                            kind: ExprKind::FunApp(id, vec![]),
                            span: start.to(self.prev_span()),
                        }
                    } else {
                        let first = self.expr()?;
                        if self.eat(&Tok::Bar) {
                            let mut args = vec![first];
                            args.extend(self.expr_list(Tok::RParen)?);
                            Expr {
                                kind: ExprKind::CondDistApp(id, args),
                                span: start.to(self.prev_span()),
                            }
                        } else {
                            let mut args = vec![first];
                            if self.eat(&Tok::Comma) {
                                args.extend(self.expr_list(Tok::RParen)?);
                            } else {
                                self.expect(
                                    Tok::RParen,
                                    "\",\", \"|\" or \")\" in the argument list",
                                )?;
                            }
                            let cond = args.len() == 1
                                && CONDITIONING_SUFFIXES.iter().any(|s| id.name.ends_with(s));
                            let kind = if cond {
                                ExprKind::CondDistApp(id, args)
                            } else {
                                ExprKind::FunApp(id, args)
                            };
                            Expr {
                                kind,
                                span: start.to(self.prev_span()),
                            }
                        }
                    }
                } else {
                    let span = id.span;
                    Expr {
                        kind: ExprKind::Variable(id),
                        span,
                    }
                }
            }
            Tok::IntLit(s) => {
                self.bump();
                Expr {
                    kind: ExprKind::IntNumeral(s),
                    span: start,
                }
            }
            Tok::RealLit(s) | Tok::DotNumeral(s) => {
                self.bump();
                Expr {
                    kind: ExprKind::RealNumeral(s),
                    span: start,
                }
            }
            Tok::ImagLit(s) => {
                self.bump();
                Expr {
                    kind: ExprKind::ImagNumeral(s),
                    span: start,
                }
            }
            Tok::LBrace => {
                self.bump();
                if self.peek() == &Tok::RBrace {
                    return self.error("Ill-formed expression. Array expressions must contain at least one element.");
                }
                let xs = self.expr_list(Tok::RBrace)?;
                Expr {
                    kind: ExprKind::ArrayExpr(xs),
                    span: start.to(self.prev_span()),
                }
            }
            Tok::LBrack => {
                self.bump();
                let xs = self.expr_list(Tok::RBrack)?;
                Expr {
                    kind: ExprKind::RowVectorExpr(xs),
                    span: start.to(self.prev_span()),
                }
            }
            Tok::Target => {
                self.bump();
                self.expect(Tok::LParen, "\"(\" after \"target\"")?;
                self.expect(Tok::RParen, "\")\"")?;
                Expr {
                    kind: ExprKind::GetTarget,
                    span: start.to(self.prev_span()),
                }
            }
            Tok::LParen => {
                self.bump();
                let first = self.expr()?;
                if self.eat(&Tok::Comma) {
                    let mut xs = vec![first];
                    xs.extend(self.expr_list(Tok::RParen)?);
                    Expr {
                        kind: ExprKind::TupleExpr(xs),
                        span: start.to(self.prev_span()),
                    }
                } else {
                    self.expect(Tok::RParen, "\")\" or \",\"")?;
                    Expr {
                        kind: ExprKind::Paren(Box::new(first)),
                        span: start.to(self.prev_span()),
                    }
                }
            }
            t => {
                if let Some(name) = t.reserved_name() {
                    return self.error(format!(
                        "Ill-formed expression. Found reserved keyword \"{name}\" where an expression was expected."
                    ));
                }
                return self.unexpected("an expression");
            }
        };
        loop {
            match self.peek().clone() {
                Tok::LBrack => {
                    self.bump();
                    let idx = self.indexes()?;
                    e = Expr {
                        kind: ExprKind::Indexed(Box::new(e), idx),
                        span: start.to(self.prev_span()),
                    };
                }
                Tok::DotNumeral(s) => {
                    let slot_span = self.span();
                    let digits: String = s[1..].chars().filter(|c| *c != '_').collect();
                    // stanc3 parses with OCaml's 63-bit int.
                    let Some(slot) = digits
                        .parse::<u64>()
                        .ok()
                        .filter(|v| *v < (1 << 62))
                        .map(|v| v as usize)
                    else {
                        return Err(Diagnostic::new(
                            DiagnosticKind::Syntax,
                            slot_span,
                            format!(
                                "Ill-formed index. Failed to parse integer from string \"{}\" in tuple index.",
                                &s[1..]
                            ),
                        ));
                    };
                    self.bump();
                    e = Expr {
                        kind: ExprKind::TupleProjection(Box::new(e), slot),
                        span: start.to(self.prev_span()),
                    };
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn indexes(&mut self) -> PResult<Vec<Index>> {
        let mut out = Vec::new();
        loop {
            let idx = match self.peek() {
                Tok::Comma | Tok::RBrack => Index::All,
                Tok::Colon => {
                    self.bump();
                    if matches!(self.peek(), Tok::Comma | Tok::RBrack) {
                        Index::All
                    } else {
                        Index::Downfrom(self.expr()?)
                    }
                }
                _ => {
                    let a = self.expr()?;
                    if self.eat(&Tok::Colon) {
                        if matches!(self.peek(), Tok::Comma | Tok::RBrack) {
                            Index::Upfrom(a)
                        } else {
                            Index::Between(a, self.expr()?)
                        }
                    } else {
                        Index::Single(a)
                    }
                }
            };
            out.push(idx);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(Tok::RBrack, "\",\" or \"]\" in the index")?;
        Ok(out)
    }
}

/// Converts an expression to an assignable value, as stanc3's `lvalue_of_expr_opt`.
pub fn lvalue_of_expr(e: &Expr) -> Option<LValue> {
    let kind = match &e.kind {
        ExprKind::Variable(id) => LValueKind::Variable(id.clone()),
        ExprKind::Indexed(inner, idx) => {
            let l = lvalue_of_expr(inner)?;
            if matches!(l.kind, LValueKind::Tuple(_)) {
                return None;
            }
            LValueKind::Indexed(Box::new(l), idx.clone())
        }
        ExprKind::TupleProjection(inner, k) => {
            let l = lvalue_of_expr(inner)?;
            if matches!(l.kind, LValueKind::Tuple(_)) {
                return None;
            }
            LValueKind::TupleProjection(Box::new(l), *k)
        }
        ExprKind::TupleExpr(xs) => {
            LValueKind::Tuple(xs.iter().map(lvalue_of_expr).collect::<Option<_>>()?)
        }
        _ => return None,
    };
    Some(LValue { kind, span: e.span })
}
