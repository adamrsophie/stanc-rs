//! Statements.

use stanc_rs_semantic::typed::{
    FunKind, TExpr, TExprKind, TIndex, TLValue, TLValueKind, TPrintable, TStmt, TStmtKind,
    TTruncation,
};
use stanc_rs_semantic::types::{Ad, Promotion};
use stanc_rs_syntax::ast::{AssignOp, UnsizedType as U};

use crate::expr::{is_copy, rust_type};
use crate::{Gen, R, mangle, unsupported};

impl Gen {
    /// Emits a list of statements.
    pub fn stmts(&mut self, ss: &[TStmt]) -> R<()> {
        for s in ss {
            self.stmt(s)?;
        }
        Ok(())
    }

    fn body(&mut self, s: &TStmt) -> R<()> {
        match &s.kind {
            TStmtKind::Block(ss) => self.stmts(ss),
            _ => self.stmt(s),
        }
    }

    /// Emits one statement.
    pub fn stmt(&mut self, s: &TStmt) -> R<()> {
        match &s.kind {
            TStmtKind::Assign { lhs, op, rhs } => {
                let value = match op {
                    AssignOp::Assign => self.expr_as(rhs, &lhs.ty, &lhs.ad)?,
                    AssignOp::Compound(bop) => {
                        let lhs_e = Gen::lvalue_expr(lhs);
                        let e = TExpr {
                            kind: TExprKind::BinOp(Box::new(lhs_e), *bop, Box::new(rhs.clone())),
                            ty: lhs.ty.clone(),
                            ad: Ad::lub_all([&lhs.ad, &rhs.ad])
                                .unwrap_or(Ad::AutoDiffable)
                                .fill_for(&lhs.ty),
                            span: s.span,
                        };
                        let c = self.expr(&e)?;
                        self.convert(c, &e.ty, &e.ad, &lhs.ty, &lhs.ad)
                    }
                };
                self.assign(lhs, value)
            }
            TStmtKind::NRFunApp { kind, name, args } => {
                let c = match kind {
                    FunKind::UserDefined => self.call(None, kind, name, args)?,
                    FunKind::StanLib => self.lib_call(None, name, args)?,
                };
                self.line(&format!("{c};"));
                Ok(())
            }
            TStmtKind::TargetPlusAssign(e) => self.add_target(e),
            TStmtKind::JacobianPlusAssign(e) => {
                self.line("if jacobian__ {");
                self.indent += 1;
                self.add_target(e)?;
                self.indent -= 1;
                self.line("}");
                Ok(())
            }
            TStmtKind::Tilde {
                arg,
                dist,
                dist_kind,
                args,
                truncation,
            } => self.tilde(s, arg, dist, dist_kind, args, truncation),
            TStmtKind::Break => {
                self.line("break;");
                Ok(())
            }
            TStmtKind::Continue => {
                self.line("continue;");
                Ok(())
            }
            TStmtKind::Return(None) => {
                self.line("return Ok(());");
                Ok(())
            }
            TStmtKind::Return(Some(e)) => {
                let Some(ret) = self.ret.clone() else {
                    return Err(unsupported(
                        s.span,
                        "return with a value outside a function",
                    ));
                };
                let ad = self.ret_ad();
                let c = self.expr_as(e, &ret, &ad.fill_for(&ret))?;
                self.line(&format!("return Ok({c});"));
                Ok(())
            }
            TStmtKind::Print(ps) => {
                let m = self.printables(ps)?;
                self.line(&format!("rt::stan::print({m});"));
                Ok(())
            }
            TStmtKind::Reject(ps) => {
                let m = self.printables(ps)?;
                self.line(&format!("return rt::reject({m});"));
                Ok(())
            }
            TStmtKind::FatalError(ps) => {
                let m = self.printables(ps)?;
                self.line(&format!("return rt::fatal_error({m});"));
                Ok(())
            }
            TStmtKind::Skip => Ok(()),
            TStmtKind::IfThenElse(c, a, b) => {
                let cc = self.truthy(c)?;
                self.line(&format!("if {cc} {{"));
                self.indent += 1;
                self.body(a)?;
                self.indent -= 1;
                if let Some(b) = b {
                    self.line("} else {");
                    self.indent += 1;
                    self.body(b)?;
                    self.indent -= 1;
                }
                self.line("}");
                Ok(())
            }
            TStmtKind::While(c, b) => {
                let cc = self.truthy(c)?;
                self.line(&format!("while {cc} {{"));
                self.indent += 1;
                self.body(b)?;
                self.indent -= 1;
                self.line("}");
                Ok(())
            }
            TStmtKind::For {
                var,
                lower,
                upper,
                body,
            } => {
                let lo = self.expr(lower)?;
                let hi = self.expr(upper)?;
                let h = self.fresh("hi");
                self.line(&format!("let {h} = {hi};"));
                self.line(&format!("for {} in ({lo})..=({h}) {{", mangle(var)));
                self.indent += 1;
                self.body(body)?;
                self.indent -= 1;
                self.line("}");
                Ok(())
            }
            TStmtKind::ForEach {
                var, iter, body, ..
            } => {
                let c = self.expr(iter)?;
                let c = match iter.ty {
                    U::Matrix | U::ComplexMatrix => format!("{c}.into_vec()"),
                    _ => c,
                };
                self.line(&format!("for {} in {c} {{", mangle(var)));
                self.indent += 1;
                self.body(body)?;
                self.indent -= 1;
                self.line("}");
                Ok(())
            }
            TStmtKind::Profile(_, ss) | TStmtKind::Block(ss) => {
                self.line("{");
                self.indent += 1;
                self.stmts(ss)?;
                self.indent -= 1;
                self.line("}");
                Ok(())
            }
            TStmtKind::VarDecl { ty, vars, ad, .. } => {
                let uty = ty.unsized_type();
                let ad = ad.clone().fill_for(&uty);
                let rty = rust_type(&uty, &ad);
                for v in vars {
                    let init = match &v.init {
                        Some(e) => self.expr_as(e, &uty, &ad)?,
                        None => self.default_value(ty, &ad)?,
                    };
                    self.line(&format!("let mut {}: {rty} = {init};", mangle(&v.name)));
                }
                Ok(())
            }
            TStmtKind::FunDef { .. } => Err(unsupported(s.span, "nested function definition")),
        }
    }

    fn ret_ad(&self) -> Ad {
        Ad::AutoDiffable
    }

    fn add_target(&mut self, e: &TExpr) -> R<()> {
        match &e.ty {
            U::Int => {
                let c = self.expr(e)?;
                self.line(&format!("*target__ += ({c}) as f64;"));
            }
            U::Real => {
                let c = self.expr(e)?;
                self.line(&format!("*target__ += {c};"));
            }
            U::Vector | U::RowVector | U::Matrix | U::Array(_) => {
                let c = self.expr(e)?;
                let v = self.fresh("x");
                self.line(&format!("let {v} = {c};"));
                self.sum_into_target(&v, &e.ty)?;
            }
            _ => return Err(unsupported(e.span, "target += of this type")),
        }
        Ok(())
    }

    fn sum_into_target(&mut self, v: &str, ty: &U) -> R<()> {
        match ty {
            U::Int => self.line(&format!("*target__ += *{v} as f64;")),
            U::Real => self.line(&format!("*target__ += *{v};")),
            U::Vector | U::RowVector => {
                self.line(&format!("for x__ in {v}.iter() {{ *target__ += *x__; }}"))
            }
            U::Matrix => self.line(&format!(
                "for x__ in {v}.as_slice().iter() {{ *target__ += *x__; }}"
            )),
            U::Array(inner) => {
                let e = self.fresh("e");
                self.line(&format!("for {e} in {v}.iter() {{"));
                self.indent += 1;
                self.sum_into_target(&e, inner)?;
                self.indent -= 1;
                self.line("}");
            }
            _ => return Err(self.internal("target += of this type")),
        }
        Ok(())
    }

    fn printables(&mut self, ps: &[TPrintable]) -> R<String> {
        let mut out = String::from("{ let mut s__ = String::new();");
        for p in ps {
            match p {
                TPrintable::Str(s) => out.push_str(&format!(" s__.push_str({});", rust_string(s))),
                TPrintable::Expr(e) => {
                    let c = if e.ad.has_autodiff() {
                        let v = self.expr(e)?;
                        format!("&({})", self.to_f64(v, &e.ty, &e.ad))
                    } else {
                        self.expr_ref(e)?
                    };
                    out.push_str(&format!(" rt::stan::StanDisplay::stan_fmt({c}, &mut s__);"));
                }
            }
        }
        out.push_str(" s__ }");
        Ok(out)
    }

    fn tilde(
        &mut self,
        s: &TStmt,
        arg: &TExpr,
        dist: &str,
        kind: &FunKind,
        args: &[TExpr],
        truncation: &TTruncation,
    ) -> R<()> {
        let mut all = vec![arg.clone()];
        all.extend(args.iter().cloned());
        let ad = Ad::lub_all(all.iter().map(|a| &a.ad)).unwrap_or(Ad::AutoDiffable);
        let lupdf = dist.replace("_lpdf", "_lupdf").replace("_lpmf", "_lupmf");
        let call_expr = |name: &str| TExpr {
            kind: TExprKind::FunApp {
                kind: kind.clone(),
                name: name.to_string(),
                args: all.clone(),
                cond: true,
            },
            ty: U::Real,
            ad: ad.clone(),
            span: s.span,
        };
        let c = self.expr(&call_expr(&lupdf))?;
        self.line(&format!("*target__ += {c};"));
        if matches!(truncation, TTruncation::None) {
            return Ok(());
        }
        if matches!(arg.ty, U::Int | U::Real) {
            return self.truncate(s, arg, dist, kind, args, truncation, &ad);
        }
        // A container outcome is truncated element by element: bind every container operand to a
        // local and index it in a loop.
        let elem = |t: &U| match t {
            U::Vector | U::RowVector => Some(U::Real),
            U::Array(t) if matches!(**t, U::Int | U::Real) => Some((**t).clone()),
            _ => None,
        };
        let mut ops = vec![arg.clone()];
        ops.extend(args.iter().cloned());
        let bounds: Vec<TExpr> = match truncation {
            TTruncation::UpFrom(l) => vec![l.clone()],
            TTruncation::DownFrom(u) => vec![u.clone()],
            TTruncation::Between(l, u) => vec![l.clone(), u.clone()],
            TTruncation::None => vec![],
        };
        ops.extend(bounds);
        let i = self.fresh("i");
        let mut lens = Vec::new();
        let mut scalars = Vec::new();
        self.line("{");
        self.indent += 1;
        for op in &ops {
            if matches!(op.ty, U::Int | U::Real) {
                scalars.push(op.clone());
                continue;
            }
            let Some(et) = elem(&op.ty) else {
                return Err(unsupported(
                    s.span,
                    "truncation of this outcome type is not supported by stanc-rs yet",
                ));
            };
            let t = self.fresh("t");
            let c = self.expr(op)?;
            self.line(&format!("let {t} = {c};"));
            lens.push(format!("{t}.len()"));
            let var = TExpr {
                kind: TExprKind::Variable {
                    name: t,
                    origin: stanc_rs_semantic::typed::Origin::Model,
                    global: false,
                },
                ..op.clone()
            };
            let idx = TExpr {
                kind: TExprKind::Variable {
                    name: i.clone(),
                    origin: stanc_rs_semantic::typed::Origin::Model,
                    global: false,
                },
                ty: U::Int,
                ad: Ad::DataOnly,
                span: op.span,
            };
            scalars.push(TExpr {
                kind: TExprKind::Indexed(Box::new(var), vec![TIndex::Single(idx)]),
                ty: et,
                ..op.clone()
            });
        }
        self.line(&format!(
            "for {i} in 1..=(rt::stan::common_len(\"truncation\", &[{}])? as i64) {{",
            lens.join(", ")
        ));
        self.indent += 1;
        let n_args = args.len();
        let (y, rest) = scalars.split_first().expect("outcome");
        let (eargs, ebounds) = rest.split_at(n_args);
        let etrunc = match truncation {
            TTruncation::UpFrom(_) => TTruncation::UpFrom(ebounds[0].clone()),
            TTruncation::DownFrom(_) => TTruncation::DownFrom(ebounds[0].clone()),
            _ => TTruncation::Between(ebounds[0].clone(), ebounds[1].clone()),
        };
        self.truncate(s, y, dist, kind, eargs, &etrunc, &ad)?;
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    /// Emits the truncation term of `arg ~ dist(args) T[...]` for a scalar `arg`.
    #[allow(clippy::too_many_arguments)]
    fn truncate(
        &mut self,
        s: &TStmt,
        arg: &TExpr,
        dist: &str,
        kind: &FunKind,
        args: &[TExpr],
        truncation: &TTruncation,
        ad: &Ad,
    ) -> R<()> {
        let base = dist.trim_end_matches("_lpdf").trim_end_matches("_lpmf");
        let discrete = dist.ends_with("_lpmf");
        let cdf = |g: &mut Gen, which: &str, bound: &TExpr| -> R<String> {
            // For a discrete distribution the lower truncation point is `L - 1`.
            let b = if discrete && which == "lccdf_lower" {
                TExpr {
                    kind: TExprKind::BinOp(
                        Box::new(bound.clone()),
                        stanc_rs_syntax::ast::BinOp::Minus,
                        Box::new(TExpr {
                            kind: TExprKind::IntLit(1),
                            ty: U::Int,
                            ad: Ad::DataOnly,
                            span: s.span,
                        }),
                    ),
                    ..bound.clone()
                }
            } else {
                bound.clone()
            };
            let fname = match which {
                "lccdf_lower" => format!("{base}_lccdf"),
                "lcdf_lower" => format!("{base}_lcdf"),
                _ => format!("{base}_lcdf"),
            };
            let b = if discrete && which == "lcdf_lower" {
                TExpr {
                    kind: TExprKind::BinOp(
                        Box::new(bound.clone()),
                        stanc_rs_syntax::ast::BinOp::Minus,
                        Box::new(TExpr {
                            kind: TExprKind::IntLit(1),
                            ty: U::Int,
                            ad: Ad::DataOnly,
                            span: s.span,
                        }),
                    ),
                    ..bound.clone()
                }
            } else {
                b
            };
            // A continuous distribution takes a real bound.
            let b = if !discrete && b.ty == U::Int {
                TExpr {
                    kind: TExprKind::Promotion(Box::new(b.clone()), Promotion::IntToReal),
                    ty: U::Real,
                    ..b
                }
            } else {
                b
            };
            let mut cargs = vec![b];
            cargs.extend(args.iter().cloned());
            let e = TExpr {
                // A user-defined distribution brings its own `_lcdf` and `_lccdf`.
                kind: TExprKind::FunApp {
                    kind: kind.clone(),
                    name: fname,
                    args: cargs,
                    cond: true,
                },
                ty: U::Real,
                ad: ad.clone(),
                span: s.span,
            };
            g.expr(&e)
        };
        let y = self.expr_as(arg, &U::Real, &Ad::DataOnly)?;
        let yv = self.fresh("y");
        self.line(&format!("let {yv}: f64 = ({y}).value();"));
        match truncation {
            TTruncation::UpFrom(l) => {
                let lv = self.bound_value(l)?;
                let lc = cdf(self, "lccdf_lower", l)?;
                self.line(&format!("if {yv} < {lv} {{ *target__ += f64::NEG_INFINITY; }} else {{ *target__ -= {lc}; }}"));
            }
            TTruncation::DownFrom(u) => {
                let uv = self.bound_value(u)?;
                let uc = cdf(self, "lcdf_upper", u)?;
                self.line(&format!("if {yv} > {uv} {{ *target__ += f64::NEG_INFINITY; }} else {{ *target__ -= {uc}; }}"));
            }
            TTruncation::Between(l, u) => {
                let lv = self.bound_value(l)?;
                let uv = self.bound_value(u)?;
                let uc = cdf(self, "lcdf_upper", u)?;
                let lc = cdf(self, "lcdf_lower", l)?;
                self.line(&format!(
                    "if {yv} < {lv} || {yv} > {uv} {{ *target__ += f64::NEG_INFINITY; }} else {{ *target__ -= sm::log_diff_exp({uc}, {lc}); }}"
                ));
            }
            TTruncation::None => {}
        }
        Ok(())
    }

    fn bound_value(&mut self, e: &TExpr) -> R<String> {
        let c = self.expr(e)?;
        Ok(match e.ty {
            U::Int => format!("(({c}) as f64)"),
            _ => format!("({c}).value()"),
        })
    }

    /// Emits `lhs = value`.
    pub fn assign(&mut self, lhs: &TLValue, value: String) -> R<()> {
        let v = self.fresh("v");
        self.line("{");
        self.indent += 1;
        self.line(&format!("let {v} = {value};"));
        self.assign_to(lhs, &v)?;
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    fn assign_to(&mut self, lhs: &TLValue, v: &str) -> R<()> {
        // `x[i][j...]` with single indexes `i` is `x[i, j...]`.
        if let TLValueKind::Indexed(inner, outer) = &lhs.kind
            && let TLValueKind::Indexed(base, first) = &inner.kind
            && first.iter().all(|i| !i.is_multi())
            && !(matches!(inner.ty, U::Matrix | U::ComplexMatrix) && outer.len() == 1)
        {
            let mut idxs = first.clone();
            idxs.extend(outer.iter().cloned());
            let merged = TLValue {
                kind: TLValueKind::Indexed(base.clone(), idxs),
                ..lhs.clone()
            };
            return self.assign_to(&merged, v);
        }
        // `x[i..., r]` where `x[i...]` is a matrix and `r` selects rows: assign to the rows of
        // the element.
        if let TLValueKind::Indexed(base, idxs) = &lhs.kind
            && idxs.len() >= 2
            && idxs.iter().all(|i| !i.is_multi())
        {
            let mut ty = base.ty.clone();
            for (k, _) in idxs.iter().enumerate() {
                match ty {
                    U::Array(t) => ty = *t,
                    U::Matrix | U::ComplexMatrix if k + 1 == idxs.len() && k > 0 => {
                        let elem = TLValue {
                            kind: TLValueKind::Indexed(base.clone(), idxs[..k].to_vec()),
                            ty: ty.clone(),
                            ..lhs.clone()
                        };
                        let split = TLValue {
                            kind: TLValueKind::Indexed(Box::new(elem), idxs[k..].to_vec()),
                            ..lhs.clone()
                        };
                        return self.assign_to(&split, v);
                    }
                    _ => break,
                }
            }
        }
        match &lhs.kind {
            TLValueKind::Tuple(lvs) => {
                for (i, lv) in lvs.iter().enumerate() {
                    self.assign_to(lv, &format!("{v}.{i}"))?;
                }
                Ok(())
            }
            TLValueKind::Indexed(base, idxs)
                if idxs.iter().any(TIndex::is_multi)
                    || matches!((&base.ty, idxs.len()), (U::Matrix | U::ComplexMatrix, 1)) =>
            {
                let p = self.place_mut(base)?;
                let ty = &base.ty;
                match (ty, idxs.as_slice()) {
                    (
                        U::Array(_)
                        | U::Vector
                        | U::RowVector
                        | U::ComplexVector
                        | U::ComplexRowVector,
                        [i],
                    ) => {
                        let ix = self.idx_tmp(i)?;
                        self.line(&format!("rt::index::assign_multi(&mut {p}, &{ix}, {v})?;"));
                    }
                    (U::Matrix | U::ComplexMatrix, [_] | [_, _]) => {
                        self.assign_matrix(&p, idxs, v)?;
                    }
                    _ => {
                        let ty = ty.clone();
                        self.assign_general(&p, &ty, idxs, v)?;
                    }
                }
                Ok(())
            }
            _ => {
                let p = self.place_mut(lhs)?;
                self.line(&format!("{p} = {v};"));
                Ok(())
            }
        }
    }

    /// Assigns `v` to matrix place `p` indexed by `idxs` (one or two indexes).
    fn assign_matrix(&mut self, p: &str, idxs: &[TIndex], v: &str) -> R<()> {
        match idxs {
            [i] => {
                if let TIndex::Single(e) = i
                    && e.ty == U::Int
                {
                    let ic = self.tmp_expr(e)?;
                    self.line(&format!("rt::index::assign_row(&mut {p}, {ic}, {v})?;"));
                } else {
                    let ix = self.idx_tmp(i)?;
                    self.line(&format!(
                        "rt::index::assign_submatrix(&mut {p}, &{ix}, &rt::index::Idx::All, {v})?;"
                    ));
                }
            }
            [i, j] => {
                let si = matches!(i, TIndex::Single(e) if e.ty == U::Int);
                let sj = matches!(j, TIndex::Single(e) if e.ty == U::Int);
                match (si, sj, i, j) {
                    (true, true, TIndex::Single(a), TIndex::Single(b)) => {
                        let ic = self.tmp_expr(a)?;
                        let jc = self.tmp_expr(b)?;
                        self.line(&format!(
                            "*rt::index::mat_at_mut(&mut {p}, {ic}, {jc})? = {v};"
                        ));
                    }
                    (true, false, TIndex::Single(e), _) => {
                        let ic = self.tmp_expr(e)?;
                        let jx = self.idx_tmp(j)?;
                        self.line(&format!(
                            "rt::index::assign_row_multi(&mut {p}, {ic}, &{jx}, {v})?;"
                        ));
                    }
                    (false, true, _, TIndex::Single(e)) => {
                        let ix = self.idx_tmp(i)?;
                        let jc = self.tmp_expr(e)?;
                        self.line(&format!(
                            "rt::index::assign_col_multi(&mut {p}, &{ix}, {jc}, {v})?;"
                        ));
                    }
                    _ => {
                        let ix = self.idx_tmp(i)?;
                        let jx = self.idx_tmp(j)?;
                        self.line(&format!(
                            "rt::index::assign_submatrix(&mut {p}, &{ix}, &{jx}, {v})?;"
                        ));
                    }
                }
            }
            _ => return Err(self.internal("matrix index count")),
        }
        Ok(())
    }

    /// Assigns `v` to `p[idxs...]` (with `p` of type `ty`) one index at a time: a single index
    /// selects an element, a multi-index loops over its positions and the elements of `v`.
    fn assign_general(&mut self, p: &str, ty: &U, idxs: &[TIndex], v: &str) -> R<()> {
        let Some((first, rest)) = idxs.split_first() else {
            self.line(&format!("{p} = {v};"));
            return Ok(());
        };
        let inner = match ty {
            U::Array(t) => (**t).clone(),
            U::Vector | U::RowVector => U::Real,
            U::ComplexVector | U::ComplexRowVector => U::Complex,
            U::Matrix | U::ComplexMatrix if idxs.len() <= 2 => {
                return self.assign_matrix(p, idxs, v);
            }
            _ => return Err(self.internal("indexing a scalar")),
        };
        match first {
            TIndex::Single(e) if e.ty == U::Int => {
                let ic = self.tmp_expr(e)?;
                let t = self.fresh("t");
                self.line("{");
                self.indent += 1;
                self.line(&format!("let {t} = rt::index::at_mut(&mut {p}, {ic})?;"));
                self.assign_general(&format!("(*{t})"), &inner, rest, v)?;
                self.indent -= 1;
                self.line("}");
            }
            _ => {
                let ix = self.idx_tmp(first)?;
                let (pos, k, x) = (self.fresh("pos"), self.fresh("k"), self.fresh("x"));
                self.line(&format!(
                    "let {pos} = rt::index::positions(&{ix}, {p}.len())?;"
                ));
                self.line(&format!(
                    "rt::stan::common_len(\"assignment\", &[{pos}.len(), {v}.len()])?;"
                ));
                self.line(&format!("for ({k}, {x}) in {pos}.into_iter().zip({v}) {{"));
                self.indent += 1;
                self.assign_general(&format!("{p}[{k}]"), &inner, rest, &x)?;
                self.indent -= 1;
                self.line("}");
            }
        }
        Ok(())
    }

    fn tmp_expr(&mut self, e: &TExpr) -> R<String> {
        let c = self.expr(e)?;
        let t = self.fresh("i");
        self.line(&format!("let {t} = {c};"));
        Ok(t)
    }

    fn idx_tmp(&mut self, i: &TIndex) -> R<String> {
        let c = self.idx(i)?;
        let t = self.fresh("ix");
        self.line(&format!("let {t} = {c};"));
        Ok(t)
    }

    /// A mutable place for an lvalue, with index values bound to temporaries first.
    fn place_mut(&mut self, lv: &TLValue) -> R<String> {
        match &lv.kind {
            TLValueKind::Variable { name, .. } => Ok(self.var(name)),
            TLValueKind::TupleProjection(b, k) => Ok(format!("{}.{}", self.place_mut(b)?, k - 1)),
            TLValueKind::Indexed(base, idxs) => {
                let mut p = self.place_mut(base)?;
                let mut ty = base.ty.clone();
                let mut i = 0;
                while i < idxs.len() {
                    let TIndex::Single(ix) = &idxs[i] else {
                        return Err(unsupported(
                            lv.span,
                            "multi-indexes before the last index in an assignment",
                        ));
                    };
                    let ic = self.tmp_expr(ix)?;
                    match &ty {
                        U::Matrix | U::ComplexMatrix => {
                            let Some(TIndex::Single(jx)) = idxs.get(i + 1) else {
                                return Err(unsupported(lv.span, "this matrix assignment"));
                            };
                            let jc = self.tmp_expr(jx)?;
                            p = format!("(*rt::index::mat_at_mut(&mut {p}, {ic}, {jc})?)");
                            ty = if ty == U::Matrix { U::Real } else { U::Complex };
                            i += 2;
                            continue;
                        }
                        U::Array(inner) => {
                            p = format!("(*rt::index::at_mut(&mut {p}, {ic})?)");
                            ty = (**inner).clone();
                        }
                        U::Vector | U::RowVector => {
                            p = format!("(*rt::index::at_mut(&mut {p}, {ic})?)");
                            ty = U::Real;
                        }
                        U::ComplexVector | U::ComplexRowVector => {
                            p = format!("(*rt::index::at_mut(&mut {p}, {ic})?)");
                            ty = U::Complex;
                        }
                        _ => return Err(self.internal("indexing a scalar")),
                    }
                    i += 1;
                }
                let _ = is_copy(&ty);
                Ok(p)
            }
            TLValueKind::Tuple(_) => Err(self.internal("tuple pattern as a place")),
        }
    }
}

/// A Rust string literal with the contents of a Stan string literal (Stan escapes kept).
pub fn rust_string(s: &str) -> String {
    let mut out = String::from("\"");
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some('n') => out.push_str("\\n"),
                Some('t') => out.push_str("\\t"),
                Some('"') => out.push_str("\\\""),
                Some('\'') => out.push('\''),
                Some('\\') => out.push_str("\\\\"),
                Some(o) => {
                    out.push_str("\\\\");
                    out.push(o);
                }
                None => out.push_str("\\\\"),
            },
            '"' => out.push_str("\\\""),
            '{' | '}' => out.push(c),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
