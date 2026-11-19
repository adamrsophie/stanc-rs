//! Pedantic mode warnings: a port of stanc3's `Pedantic_analysis`, `Pedantic_dist_warnings`,
//! `Factor_graph`, and the dependence analysis of `Dependence_analysis`.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::dataflow::{Graph, Powerset, assigned_vars, expr_vars, mfp, stmt_rhs, stmt_rhs_vars};
use crate::mir::*;

/// The names of the variables a statement reads (`stmt_rhs_names_set`).
fn rhs_names(s: &Stmt) -> BTreeSet<String> {
    stmt_rhs_vars(s).into_iter().map(|(_, v)| v).collect()
}

fn var_names(e: &Expr) -> BTreeSet<String> {
    let mut v = Vec::new();
    expr_vars(e, &mut v);
    v.into_iter().map(|(_, n)| n).collect()
}

/// Reaching definitions and control flow of a statement (`build_dep_info_map`). Labels number
/// the statements in preorder from 0, the root; definitions from before the statement are
/// attributed to the root.
struct Deps<'a> {
    g: Graph<'a>,
    rd: Vec<BTreeSet<(String, usize)>>,
}

impl<'a> Deps<'a> {
    fn new(p: &Program, s: &'a Stmt) -> Deps<'a> {
        let g = Graph::build(s, false);
        let init: BTreeSet<(String, Option<usize>)> = p
            .input_vars
            .iter()
            .map(|(n, ..)| n)
            .chain(p.output_vars.iter().map(|(n, ..)| n))
            .map(|n| (n.clone(), None))
            .collect();
        let res = mfp(&g, &Powerset(init), |l, prop| {
            let node = g.nodes[l];
            let kill: Option<&str> = match &node.pattern {
                StmtP::Decl { id, .. } => Some(id),
                StmtP::Assignment(lv, ..)
                    if lv.idx.is_empty() && matches!(lv.base, LBase::Variable(_)) =>
                {
                    Some(lv.variable())
                }
                StmtP::For { loopvar, .. } => Some(loopvar),
                StmtP::TargetPE(_) | StmtP::JacobianPE(_) => Some("target"),
                StmtP::NRFunApp(
                    FunKind::UserDefined(_, Suffix::Target | Suffix::Jacobian)
                    | FunKind::StanLib(_, Suffix::Target | Suffix::Jacobian),
                    _,
                ) => Some("target"),
                _ => None,
            };
            let mut out: BTreeSet<(String, Option<usize>)> = prop
                .iter()
                .filter(|(y, _)| Some(y.as_str()) != kill)
                .cloned()
                .collect();
            if let Some(v) = assigned_vars(node) {
                out.insert((v, Some(l)));
            }
            if let StmtP::Decl { id, .. } = &node.pattern {
                out.insert((id.clone(), Some(l)));
            }
            out
        });
        let rd = res
            .into_iter()
            .map(|ee| {
                ee.entry
                    .into_iter()
                    .map(|(v, l)| (v, l.unwrap_or(0)))
                    .collect()
            })
            .collect();
        Deps { g, rd }
    }

    fn lookup(&self, label: usize, var: &str, out: &mut BTreeSet<usize>) {
        out.extend(
            self.rd[label]
                .iter()
                .filter(|(v, _)| v == var)
                .map(|(_, l)| *l),
        );
    }

    /// `node_immediate_dependencies`.
    fn immediate(&self, label: usize, blockers: &BTreeSet<String>) -> BTreeSet<usize> {
        let mut out = self.g.parents[label].clone();
        for v in rhs_names(self.g.nodes[label]).difference(blockers) {
            self.lookup(label, v, &mut out);
        }
        out
    }

    /// `node_dependencies_rec`.
    fn rec(&self, label: usize, visited: &mut BTreeSet<usize>, blockers: &BTreeSet<String>) {
        if !visited.insert(label) {
            return;
        }
        for d in self.immediate(label, blockers) {
            self.rec(d, visited, &BTreeSet::new());
        }
    }

    /// `node_dependencies`: the label and the labels it depends on.
    fn node_deps(&self, label: usize) -> BTreeSet<usize> {
        let mut visited = BTreeSet::new();
        self.rec(label, &mut visited, &BTreeSet::new());
        visited
    }

    /// `node_vars_dependencies`: the labels the variables depend on at a label.
    fn vars_deps(
        &self,
        vars: &BTreeSet<String>,
        label: usize,
        blockers: &BTreeSet<String>,
    ) -> BTreeSet<usize> {
        let mut starts = self.g.parents[label].clone();
        for v in vars.difference(blockers) {
            self.lookup(label, v, &mut starts);
        }
        let mut visited = BTreeSet::new();
        for s in starts {
            self.rec(s, &mut visited, blockers);
        }
        visited
    }

    /// `var_deps`: the targets a label (or an expression at a label) depends on.
    fn var_deps(
        &self,
        label: usize,
        expr: Option<&Expr>,
        targets: &BTreeSet<String>,
    ) -> BTreeSet<String> {
        let (labels, mut vars) = match expr {
            None => (self.node_deps(label), BTreeSet::new()),
            Some(e) => {
                let vs = var_names(e);
                (self.vars_deps(&vs, label, &BTreeSet::new()), vs)
            }
        };
        for l in labels {
            vars.extend(rhs_names(self.g.nodes[l]));
        }
        targets.intersection(&vars).cloned().collect()
    }

    /// `list_target_dependant_cf`: control flow statements depending on the targets.
    fn target_dependant_cf(&self, targets: &BTreeSet<String>) -> Vec<(Loc, BTreeSet<String>)> {
        (0..self.g.nodes.len())
            .filter(|&l| {
                matches!(
                    self.g.nodes[l].pattern,
                    StmtP::IfElse(..) | StmtP::While(..) | StmtP::For { .. }
                )
            })
            .filter_map(|l| {
                let d = self.var_deps(l, None, targets);
                (!d.is_empty()).then(|| (self.g.nodes[l].loc, d))
            })
            .collect()
    }
}

/// Declared variables at the top of a statement (`top_var_declarations`).
fn top_decls(s: &Stmt, out: &mut BTreeSet<String>) {
    match &s.pattern {
        StmtP::Decl { id, .. } => {
            out.insert(id.clone());
        }
        StmtP::SList(ss) => ss.iter().for_each(|c| top_decls(c, out)),
        _ => {}
    }
}

/// `data_set`: data and, unless excluded, transformed data.
fn data_set(p: &Program, exclude_transformed: bool, exclude_ints: bool) -> BTreeSet<String> {
    let mut out: BTreeSet<String> = p
        .input_vars
        .iter()
        .filter(|(_, _, st)| !(exclude_ints && *st == SizedType::Int))
        .map(|(n, ..)| n.clone())
        .collect();
    if !exclude_transformed {
        let mut td = BTreeSet::new();
        p.prepare_data.iter().for_each(|s| top_decls(s, &mut td));
        for v in td {
            if !p.input_vars.iter().any(|(n, ..)| *n == v) {
                out.insert(v);
            }
        }
    }
    out
}

/// `parameter_set`: the parameters with their transformations and locations.
fn parameters(p: &Program) -> Vec<(&str, &Transformation, Loc)> {
    p.output_vars
        .iter()
        .filter(|(_, _, o)| o.block == IoBlock::Parameters)
        .map(|(n, l, o)| (n.as_str(), &o.trans, *l))
        .collect()
}

fn parameter_names(p: &Program) -> BTreeSet<String> {
    parameters(p)
        .into_iter()
        .map(|(n, ..)| n.to_string())
        .collect()
}

/// Factors of the log density.
#[derive(Clone, PartialEq)]
enum Factor {
    Term(Expr),
    Reject,
    LPFunction(Vec<Expr>),
}

/// The terms of a sum (`summation_terms`).
fn summation_terms(e: &Expr, out: &mut Vec<Expr>) {
    match &e.pattern {
        ExprP::FunApp(FunKind::StanLib(n, Suffix::Plain), a) if n == "Plus__" && a.len() == 2 => {
            summation_terms(&a[0], out);
            summation_terms(&a[1], out);
        }
        _ => out.push(e.clone()),
    }
}

/// A factor graph: the variables of each factor, and the factors of each variable.
#[derive(Clone)]
struct FactorGraph {
    factors: BTreeMap<usize, BTreeSet<String>>,
    vars: BTreeMap<String, BTreeSet<usize>>,
}

impl FactorGraph {
    /// `prog_factor_graph`.
    fn new(p: &Program, deps: &Deps) -> FactorGraph {
        let mut fs: Vec<(usize, Factor)> = Vec::new();
        for (l, s) in deps.g.nodes.iter().enumerate() {
            let mut here = Vec::new();
            match &s.pattern {
                StmtP::TargetPE(e) | StmtP::JacobianPE(e) => {
                    let mut ts = Vec::new();
                    summation_terms(e, &mut ts);
                    here.extend(ts.into_iter().map(Factor::Term));
                }
                StmtP::NRFunApp(
                    FunKind::CompilerInternal(Internal::Reject | Internal::FatalError),
                    _,
                ) => here.push(Factor::Reject),
                StmtP::NRFunApp(
                    FunKind::UserDefined(_, Suffix::Target) | FunKind::StanLib(_, Suffix::Target),
                    args,
                ) => here.push(Factor::LPFunction(args.clone())),
                _ => {}
            }
            for f in here {
                if !fs.iter().any(|(l2, f2)| *l2 == l && *f2 == f) {
                    fs.push((l, f));
                }
            }
        }
        let mut vars = data_set(p, false, false);
        vars.extend(parameter_names(p));
        let mut g = FactorGraph {
            factors: BTreeMap::new(),
            vars: BTreeMap::new(),
        };
        for (i, (l, f)) in fs.iter().enumerate() {
            let rhs: BTreeSet<String> = match f {
                Factor::Term(e) => var_names(e),
                Factor::Reject => BTreeSet::new(),
                // stanc3 stops with an internal error when an argument is not a variable;
                // the variables of every argument are what the factor depends on.
                Factor::LPFunction(es) => es.iter().flat_map(var_names).collect(),
            };
            let mut all = rhs.clone();
            for d in deps.vars_deps(&rhs, *l, &vars) {
                all.extend(rhs_names(deps.g.nodes[d]));
            }
            let set: BTreeSet<String> = all.intersection(&vars).cloned().collect();
            for v in &set {
                g.vars.entry(v.clone()).or_default().insert(i);
            }
            g.factors.insert(i, set);
        }
        g
    }

    fn remove_var(&mut self, v: &str) {
        for s in self.factors.values_mut() {
            s.remove(v);
        }
        self.vars.remove(v);
    }

    /// `fg_reaches`: a breadth first search from the variables to any of the goals.
    fn reaches(&self, starts: &BTreeSet<String>, goals: &BTreeSet<String>) -> bool {
        let mut frontier: std::collections::VecDeque<String> = starts.iter().cloned().collect();
        let mut visited = BTreeSet::new();
        while let Some(next) = frontier.pop_front() {
            if !visited.insert(next.clone()) {
                continue;
            }
            let mut expansion = BTreeSet::new();
            for f in self.vars.get(&next).into_iter().flatten() {
                expansion.extend(self.factors[f].iter().cloned());
            }
            if expansion.iter().any(|v| goals.contains(v)) {
                return true;
            }
            frontier.extend(expansion);
        }
        false
    }

    /// `list_priors`: for each parameter, the neighboring factors that reach the data only
    /// through it.
    fn priors(&self, p: &Program) -> Vec<(String, Option<usize>, Loc)> {
        let data = data_set(p, false, false);
        let no_ints = data_set(p, false, true);
        let mut fg = self.clone();
        for v in data.difference(&no_ints) {
            fg.remove_var(v);
        }
        let mut params: Vec<(String, Loc)> = parameters(p)
            .into_iter()
            .map(|(n, _, l)| (n.to_string(), l))
            .collect();
        params.sort_by(|a, b| a.0.cmp(&b.0));
        params.dedup_by(|a, b| a.0 == b.0);
        params
            .into_iter()
            .map(|(v, loc)| {
                let n = fg.vars.get(&v).map(|facs| {
                    let mut without = fg.clone();
                    without.remove_var(&v);
                    facs.iter()
                        .filter(|f| !without.reaches(&without.factors[f], &data))
                        .count()
                });
                (v, n, loc)
            })
            .collect()
    }
}

/// `num_expr_value`: a numeric literal, possibly negated.
fn num_value(e: &Expr) -> Option<(f64, String)> {
    match &e.pattern {
        ExprP::Promotion(x, ..) => num_value(x),
        ExprP::Lit(LitType::Real | LitType::Int, s) => {
            Some((s.parse().unwrap_or(f64::NAN), s.clone()))
        }
        ExprP::FunApp(FunKind::StanLib(n, Suffix::Plain), a) if n == "PMinus__" && a.len() == 1 => {
            num_value(&a[0]).map(|(v, s)| (-v, format!("-{s}")))
        }
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Bound {
    None,
    Nonlit,
    Lit(f64),
}

/// `trans_bounds_values`.
fn bounds(t: &Transformation) -> (Bound, Bound) {
    let b = |e: &Expr| match num_value(e) {
        None => Bound::Nonlit,
        Some((f, _)) => Bound::Lit(f),
    };
    match t {
        Transformation::Lower(l) => (b(l), Bound::None),
        Transformation::Upper(u) => (Bound::None, b(u)),
        Transformation::LowerUpper(l, u) => (b(l), b(u)),
        Transformation::Simplex
        | Transformation::StochasticColumn
        | Transformation::StochasticRow => (Bound::Lit(0.0), Bound::Lit(1.0)),
        Transformation::PositiveOrdered => (Bound::Lit(0.0), Bound::None),
        Transformation::UnitVector => (Bound::Lit(-1.0), Bound::Lit(1.0)),
        _ => (Bound::None, Bound::None),
    }
}

/// What is known at compile time about a distribution argument.
#[derive(Clone, PartialEq)]
enum CVal {
    Opaque,
    Number(f64, String),
    Param(String, Box<Transformation>),
    Data,
}

#[derive(Clone, PartialEq)]
struct DistInfo {
    name: String,
    loc: Loc,
    args: Vec<(CVal, Loc)>,
}

/// Every expression in the statements, roots included (`fold_stmts`).
fn each_expr(ss: &[Stmt], f: &mut impl FnMut(&Expr)) {
    fn go_e(e: &Expr, f: &mut impl FnMut(&Expr)) {
        f(e);
        for c in e.children() {
            go_e(c, f);
        }
    }
    for s in ss {
        s.any_stmt(&mut |s: &Stmt| {
            for e in s.exprs() {
                go_e(e, f);
            }
            false
        });
    }
}

/// `list_distributions`: the distributions of tilde statements.
fn distributions(p: &Program) -> Vec<DistInfo> {
    let params = parameters(p);
    let data = data_set(p, false, false);
    let mut ss = p.log_prob.clone();
    ss.extend(p.functions.iter().filter_map(|f| f.body.clone()));
    let mut out: Vec<DistInfo> = Vec::new();
    each_expr(&ss, &mut |e| {
        if let ExprP::FunApp(FunKind::StanLib(n, Suffix::Lpdf(true) | Suffix::Lpmf(true)), args) =
            &e.pattern
        {
            let name = n
                .strip_suffix("_lpmf")
                .or_else(|| n.strip_suffix("_lpdf"))
                .unwrap_or(n)
                .to_string();
            let args = args
                .iter()
                .map(|a| {
                    let v = match &a.pattern {
                        ExprP::Var(v) => match params.iter().find(|(n, ..)| n == v) {
                            Some((n, t, _)) => CVal::Param(n.to_string(), Box::new((*t).clone())),
                            None if data.contains(v) => CVal::Data,
                            None => CVal::Opaque,
                        },
                        _ => match num_value(a) {
                            Some((v, s)) => CVal::Number(v, s),
                            None => CVal::Opaque,
                        },
                    };
                    (v, a.meta.loc)
                })
                .collect();
            let d = DistInfo {
                name,
                loc: e.meta.loc,
                args,
            };
            if !out.contains(&d) {
                out.push(d);
            }
        }
    });
    out
}

/// A range of values; the flags are true for inclusive bounds.
#[derive(Clone, Copy)]
struct Range {
    lower: Option<(f64, bool)>,
    upper: Option<(f64, bool)>,
}

#[derive(Clone, Copy)]
enum Constr {
    Range(Range),
    Ordered,
    Simplex,
    CholeskyCorr,
    CholeskyCov,
    Correlation,
    Covariance,
}

const UNIT: (&str, Constr) = (
    "[0,1]",
    Constr::Range(Range {
        lower: Some((0.0, true)),
        upper: Some((1.0, true)),
    }),
);
const EXCL_UNIT: (&str, Constr) = (
    "(0,1)",
    Constr::Range(Range {
        lower: Some((0.0, false)),
        upper: Some((1.0, false)),
    }),
);
const POSITIVE: (&str, Constr) = (
    "strictly positive",
    Constr::Range(Range {
        lower: Some((0.0, false)),
        upper: None,
    }),
);
const NONNEG: (&str, Constr) = (
    "non-negative",
    Constr::Range(Range {
        lower: Some((0.0, true)),
        upper: None,
    }),
);
const SIMPLEX: (&str, Constr) = ("simplex", Constr::Simplex);
const ORDERED: (&str, Constr) = ("ordered", Constr::Ordered);
const CORRELATION: (&str, Constr) = ("correlation", Constr::Correlation);
const CHOL_CORR: (&str, Constr) = ("Cholesky factor of correlation", Constr::CholeskyCorr);
const COVARIANCE: (&str, Constr) = ("covariance", Constr::Covariance);
const CHOL_COV: (&str, Constr) = ("Cholesky factor of covariance", Constr::CholeskyCov);

fn transform_mismatch(c: Constr, t: &Transformation) -> bool {
    use Transformation as T;
    match c {
        Constr::Range(r) => {
            let (lo, hi) = bounds(t);
            matches!((lo, r.lower), (Bound::None, Some(_)))
                || matches!((hi, r.upper), (Bound::None, Some(_)))
                || matches!((lo, r.lower), (Bound::Lit(l), Some((l2, _))) if l < l2)
                || matches!((hi, r.upper), (Bound::Lit(u), Some((u2, _))) if u > u2)
        }
        Constr::Ordered => *t != T::Ordered,
        Constr::Simplex => *t != T::Simplex,
        Constr::CholeskyCorr => *t != T::CholeskyCorr,
        Constr::CholeskyCov => *t != T::CholeskyCov && *t != T::CholeskyCorr,
        Constr::Correlation => *t != T::Correlation,
        Constr::Covariance => *t != T::Covariance && *t != T::Correlation,
    }
}

fn value_mismatch(c: Constr, v: f64) -> bool {
    match c {
        Constr::Range(r) => {
            let lo = match r.lower {
                Some((l, true)) => v < l,
                Some((l, false)) => v <= l,
                None => false,
            };
            let hi = match r.upper {
                Some((u, true)) => v > u,
                Some((u, false)) => v >= u,
                None => false,
            };
            lo || hi
        }
        _ => false,
    }
}

/// An argument of a distribution: its position and role, or `None` for the variate.
type Arg = Option<(usize, &'static str)>;

fn constr_mismatch((cname, c): (&str, Constr), arg: Arg, d: &DistInfo) -> Option<(Loc, String)> {
    let n = arg.map_or(0, |(n, _)| n);
    let (v, loc) = d.args.get(n)?;
    let dist = &d.name;
    match v {
        CVal::Param(p, t) if transform_mismatch(c, t) => Some((
            *loc,
            match arg {
                Some((n, an)) => format!(
                    "A {dist} distribution is given parameter {p} as {an} (argument {n}), but {p} was not constrained to be {cname}."
                ),
                None => format!(
                    "Parameter {p} is given a {dist} distribution, which has {cname} support, but {p} was not constrained to be {cname}."
                ),
            },
        )),
        CVal::Number(x, s) if value_mismatch(c, *x) => Some((
            *loc,
            match arg {
                Some((n, an)) => format!(
                    "A {dist} distribution is given value {s} as {an} (argument {n}), but {an} is not {cname}."
                ),
                None => format!(
                    "Value {s} is given a {dist} distribution, which has {cname} support, but {s} is not {cname}."
                ),
            },
        )),
        _ => None,
    }
}

const UNIFORM_MESSAGE: &str = "is given a uniform distribution. The uniform distribution is not recommended, for two reasons: (a) Except when there are logical or physical constraints, it is very unusual for you to be sure that a parameter will fall inside a specified range, and (b) The infinite gradient induced by a uniform density can cause difficulties for Stan's sampling algorithm. As a consequence, we recommend soft constraints rather than hard constraints; for example, instead of giving an elasticity parameter a uniform(0,1) distribution, try normal(0.5,0.5).";

fn uniform_warning(d: &DistInfo) -> Option<(Loc, String)> {
    let [(CVal::Param(p, t), _), (a1, _), (a2, _), ..] = d.args.as_slice() else {
        return None;
    };
    let warning = Some((d.loc, format!("Parameter {p} {UNIFORM_MESSAGE}")));
    let (lo, hi) = bounds(t);
    if lo == Bound::None || hi == Bound::None {
        return warning;
    }
    let pair = match (a1, lo) {
        (CVal::Number(u, _), Bound::Lit(b)) => Some((*u, b)),
        _ => match (a2, hi) {
            (CVal::Number(u, _), Bound::Lit(b)) => Some((*u, b)),
            _ => None,
        },
    };
    match pair {
        Some((u, b)) if u == b => None,
        Some(_) => warning,
        None => None,
    }
}

const LKJ_CORR_MESSAGE: &str = "It is suggested to reparameterize your model to replace lkj_corr with lkj_corr_cholesky, the Cholesky factor variant. lkj_corr tends to run slower, consume more memory, and has higher risk of numerical errors.";

const GAMMA_MESSAGE: &str = "There is a gamma or inverse-gamma distribution with parameters that are equal to each other and set to values less than 1. This is mathematically acceptable and can make sense in some problems, but typically we see this model used as an attempt to assign a noninformative prior distribution. In fact, priors such as inverse-gamma(.001,.001) can be very strong, as explained by Gelman (2006). Instead we recommend something like a normal(0,1) or student_t(4,0,1), with parameter constrained to be positive.";

fn gamma_warning(d: &DistInfo) -> Option<(Loc, String)> {
    match d.args.as_slice() {
        [_, (CVal::Number(a, _), loc), (CVal::Number(b, _), _)] if a == b && *a < 1.0 => {
            Some((*loc, GAMMA_MESSAGE.to_string()))
        }
        _ => None,
    }
}

/// `distribution_warning`: the warnings of one distribution.
fn distribution_warning(d: &DistInfo) -> Vec<(Loc, String)> {
    const SCALE: &str = "a scale parameter";
    const SCALE_MAT: &str = "a scale matrix";
    const INV_SCALE: &str = "an inverse scale parameter";
    const SHAPE: &str = "a shape parameter";
    const DOF: &str = "degrees of freedom";
    const COV: &str = "a covariance matrix";
    let cm = |c: (&'static str, Constr), arg: Arg| constr_mismatch(c, arg, d);
    let a = |n: usize, s: &'static str| Some((n, s));
    let v: Vec<Option<(Loc, String)>> = match d.name.as_str() {
        "bernoulli" => vec![cm(UNIT, a(1, "chance of success"))],
        "binomial" => vec![cm(UNIT, a(2, "chance of success"))],
        "beta_binomial" => vec![
            cm(POSITIVE, a(2, "a prior success count")),
            cm(POSITIVE, a(3, "a prior failure count")),
        ],
        "categorical" => vec![cm(SIMPLEX, a(1, "a vector of outcome probabilities"))],
        "ordered_logistic" | "ordered_probit" => vec![cm(ORDERED, a(2, "cutpoints"))],
        "neg_binomial" => vec![cm(POSITIVE, a(1, SHAPE)), cm(POSITIVE, a(2, INV_SCALE))],
        "neg_binomial_2" => vec![
            cm(POSITIVE, a(1, SHAPE)),
            cm(POSITIVE, a(2, "a precision parameter")),
        ],
        "neg_binomial_2_log" => vec![cm(
            POSITIVE,
            a(2, "an inverse overdispersion control parameter"),
        )],
        "neg_binomial_2_log_glm" => vec![cm(
            POSITIVE,
            a(4, "an inverse overdispersion control parameter"),
        )],
        "poisson" => vec![cm(POSITIVE, a(1, "a rate parameter"))],
        "multinomial" => vec![cm(SIMPLEX, a(1, "a distribution parameter"))],
        "normal" | "skew_normal" | "cauchy" | "double_exponential" | "logistic" | "gumbel"
        | "von_mises" => vec![cm(POSITIVE, a(2, SCALE))],
        "normal_id_glm" => vec![cm(POSITIVE, a(4, SCALE))],
        "exp_mod_normal" => vec![cm(POSITIVE, a(2, SCALE)), cm(POSITIVE, a(3, SHAPE))],
        "student_t" => vec![cm(POSITIVE, a(1, DOF)), cm(POSITIVE, a(3, SCALE))],
        "lognormal" => vec![cm(POSITIVE, None), cm(POSITIVE, a(2, SCALE))],
        "chi_square" | "inv_chi_square" => vec![cm(POSITIVE, None), cm(POSITIVE, a(1, DOF))],
        "scaled_inv_chi_square" => vec![
            cm(POSITIVE, None),
            cm(POSITIVE, a(1, DOF)),
            cm(POSITIVE, a(2, SCALE)),
        ],
        "exponential" => vec![cm(POSITIVE, None), cm(POSITIVE, a(1, SCALE))],
        "gamma" => vec![
            cm(POSITIVE, None),
            cm(POSITIVE, a(1, SHAPE)),
            cm(POSITIVE, a(2, INV_SCALE)),
            gamma_warning(d),
        ],
        "inv_gamma" => vec![
            cm(POSITIVE, None),
            cm(POSITIVE, a(1, SHAPE)),
            cm(POSITIVE, a(2, SCALE)),
            gamma_warning(d),
        ],
        "weibull" => vec![
            cm(NONNEG, None),
            cm(POSITIVE, a(1, SHAPE)),
            cm(POSITIVE, a(2, SCALE)),
        ],
        "frechet" => vec![
            cm(POSITIVE, None),
            cm(POSITIVE, a(1, SHAPE)),
            cm(POSITIVE, a(2, SCALE)),
        ],
        "rayleigh" => vec![cm(NONNEG, None), cm(POSITIVE, a(1, SCALE))],
        "wiener" => vec![
            cm(POSITIVE, None),
            cm(POSITIVE, a(1, "a boundary separation parameter")),
            cm(POSITIVE, a(2, "a non-decision time parameter")),
            cm(UNIT, a(3, "an a-priori bias parameter")),
        ],
        "pareto" => vec![
            cm(POSITIVE, None),
            cm(POSITIVE, a(1, "a positive minimum parameter")),
            cm(POSITIVE, a(2, SHAPE)),
        ],
        "pareto_type_2" => vec![cm(POSITIVE, a(2, SCALE)), cm(POSITIVE, a(3, SHAPE))],
        "beta" => vec![
            cm(EXCL_UNIT, None),
            cm(POSITIVE, a(1, "a count parameter")),
            cm(POSITIVE, a(2, "a count parameter")),
        ],
        "beta_proportion" => vec![
            cm(EXCL_UNIT, None),
            cm(EXCL_UNIT, a(1, "a unit mean parameter")),
            cm(POSITIVE, a(2, "a precision parameter")),
        ],
        "uniform" => vec![uniform_warning(d)],
        "multi_normal" => vec![cm(COVARIANCE, a(2, COV))],
        "multi_normal_prec" => vec![cm(COVARIANCE, a(2, "a precision matrix"))],
        "multi_normal_cholesky" => vec![cm(CHOL_COV, a(2, COV))],
        "multi_gp" => vec![cm(COVARIANCE, a(1, "a kernel matrix"))],
        "multi_gp_cholesky" => vec![cm(CHOL_COV, a(1, "Cholesky factor of the kernel matrix"))],
        "multi_student_t" => vec![cm(POSITIVE, a(1, DOF)), cm(COVARIANCE, a(3, SCALE_MAT))],
        "multi_student_t_cholesky" => {
            vec![cm(POSITIVE, a(1, DOF)), cm(CHOL_COV, a(3, COV))]
        }
        "gaussian_dlm_obs" => vec![
            cm(COVARIANCE, a(3, "observation covariance matrix")),
            cm(COVARIANCE, a(4, "system covariance matrix")),
        ],
        "dirichlet" => vec![cm(SIMPLEX, None), cm(POSITIVE, a(1, "a count parameter"))],
        "lkj_corr" => vec![
            Some((d.loc, LKJ_CORR_MESSAGE.to_string())),
            cm(CORRELATION, None),
            cm(POSITIVE, a(1, SHAPE)),
        ],
        "lkj_corr_cholesky" => vec![cm(CHOL_CORR, None), cm(POSITIVE, a(1, SHAPE))],
        "wishart_cholesky" | "inv_wishart_cholesky" => vec![
            cm(CHOL_COV, None),
            cm(POSITIVE, a(1, DOF)),
            cm(CHOL_COV, a(2, SCALE_MAT)),
        ],
        "wishart" | "inv_wishart" => vec![
            cm(COVARIANCE, None),
            cm(POSITIVE, a(1, DOF)),
            cm(COVARIANCE, a(2, SCALE_MAT)),
        ],
        _ => vec![],
    };
    v.into_iter().flatten().collect()
}

/// Whether an expression is linear in its variables (`is_linear`); with `allow_var` false,
/// it must be constant.
fn is_linear(allow_var: bool, e: &Expr) -> bool {
    match &e.pattern {
        ExprP::Var(_) => allow_var,
        ExprP::Lit(..) => true,
        ExprP::Indexed(x, _) | ExprP::Promotion(x, ..) => is_linear(allow_var, x),
        ExprP::TernaryIf(a, b, c) => {
            is_linear(allow_var, a) && is_linear(allow_var, b) && is_linear(allow_var, c)
        }
        ExprP::FunApp(FunKind::StanLib(name, _), args) => is_linear_function(allow_var, name, args),
        ExprP::FunApp(
            FunKind::CompilerInternal(Internal::MakeArray | Internal::MakeRowVec),
            args,
        ) => args.iter().all(|a| is_linear(allow_var, a)),
        _ => false,
    }
}

const LINEAR_FUNCTIONS: [&str; 34] = [
    "Plus__",
    "PPlus__",
    "Minus__",
    "PMinus__",
    "PNot__",
    "Transpose__",
    "add",
    "append_block",
    "append_row",
    "append_col",
    "block",
    "col",
    "cols",
    "row",
    "rows",
    "diagonal",
    "head",
    "tail",
    "minus",
    "negative_infinity",
    "not_a_number",
    "rep_matrix",
    "rep_vector",
    "rep_row_vector",
    "positive_infinity",
    "segment",
    "subtract",
    "sum",
    "to_vector",
    "to_row_vector",
    "to_matrix",
    "to_array_1d",
    "to_array_2d",
    "transpose",
];

fn is_linear_function(allow_var: bool, name: &str, args: &[Expr]) -> bool {
    let lin = is_linear;
    if LINEAR_FUNCTIONS.contains(&name) {
        return args.iter().all(|a| lin(allow_var, a));
    }
    if args.iter().all(|a| lin(false, a)) {
        return true;
    }
    match (name, args) {
        ("Times__" | "Divide__" | "IntDivide__", [a, b]) => {
            (lin(allow_var, a) && lin(false, b)) || (lin(false, a) && lin(allow_var, b))
        }
        ("fma", [a, b, c]) => {
            lin(allow_var, c)
                && ((lin(allow_var, a) && lin(false, b)) || (lin(false, a) && lin(allow_var, b)))
        }
        _ => false,
    }
}

const JACOBIAN_MESSAGE: &str = "Left-hand side of distribution statement (~) may contain a non-linear transform of a parameter or local variable. If it does, you need to include a target += statement with the log absolute determinant of the Jacobian of the transform. You could also consider defining a transformed parameter and using jacobian += in the transformed parameters block.";

const HARD_CONSTRAINT_MESSAGE: &str = "with a lower and upper bound in its declaration. These hard constraints are not recommended, for two reasons: (a) Except when there are logical or physical constraints, it is very unusual for you to be sure that a parameter will fall inside a specified range, and (b) The infinite gradient induced by a hard constraint can cause difficulties for Stan's sampling algorithm. As a consequence, we recommend soft constraints rather than hard constraints; for example, instead of constraining an elasticity parameter to fall between 0, and 1, leave it unconstrained and give it a normal(0.5,0.5) prior distribution.";

/// The pedantic warnings other than the uninitialized variable ones, of a program the
/// pedantic optimization suite has run on (`warn_pedantic`). `cmp` orders locations as stanc3
/// does and `pp` prints them.
pub fn warnings(
    mir: &Program,
    cmp: &dyn Fn(&Loc, &Loc) -> Ordering,
    pp: &dyn Fn(&Loc) -> String,
) -> Vec<(Loc, String)> {
    let mut out: Vec<(Loc, String)> = Vec::new();
    let dists = distributions(mir);
    let log_prob = Stmt::new(StmtP::SList(mir.log_prob.clone()), Loc::EMPTY);
    let deps = Deps::new(mir, &log_prob);
    let fg = FactorGraph::new(mir, &deps);

    // Unscaled constants.
    for d in &dists {
        for (v, loc) in &d.args {
            if let CVal::Number(x, s) = v {
                let m = x.abs();
                if (m < 0.1 || m > 10.0) && m != 0.0 {
                    out.push((*loc, format!("Argument {s} suggests there may be parameters that are not unit scale; consider rescaling with a multiplier, see: https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html#standardizing-predictors")));
                }
            }
        }
    }

    // Multiple tildes, and possibly nonlinear tildes.
    let mut tildes: BTreeMap<String, Vec<Loc>> = BTreeMap::new();
    for s in &mir.log_prob {
        s.any_stmt(&mut |s: &Stmt| {
            if let StmtP::TargetPE(e) = &s.pattern
                && let ExprP::FunApp(k, args) = &e.pattern
            {
                if let Some(Expr {
                    pattern: ExprP::Var(v),
                    ..
                }) = args.first()
                {
                    let locs = tildes.entry(v.clone()).or_default();
                    if !locs.contains(&s.loc) {
                        locs.push(s.loc);
                    }
                }
                let density = matches!(
                    k,
                    FunKind::StanLib(_, Suffix::Lpdf(_) | Suffix::Lpmf(_))
                        | FunKind::UserDefined(_, Suffix::Lpdf(_) | Suffix::Lpmf(_))
                );
                if density
                    && let Some(a) = args.first()
                    && !is_linear(true, a)
                {
                    out.push((s.loc, JACOBIAN_MESSAGE.to_string()));
                }
            }
            false
        });
    }
    for (v, locs) in tildes {
        if locs.len() != 1 {
            let first = *locs.iter().min_by(|a, b| cmp(a, b)).unwrap();
            out.push((
                first,
                format!(
                    "The parameter {v} is on the left-hand side of more than one tilde statement."
                ),
            ));
        }
    }

    // Hard constraints and unused parameters.
    let used: BTreeSet<&String> = fg.vars.keys().collect();
    for (name, t, loc) in parameters(mir) {
        match bounds(t) {
            (Bound::Lit(l), Bound::Lit(u)) if (l == 0.0 || l == -1.0) && u == 1.0 => {}
            (Bound::Lit(l), Bound::Lit(u)) if l >= u => out.push((
                loc,
                format!("Parameter {name} has constraints that don't make sense. The lower bound should be strictly less than the upper bound."),
            )),
            (Bound::Lit(_), Bound::Lit(_)) => out.push((
                loc,
                format!("Your Stan program has a parameter {name} {HARD_CONSTRAINT_MESSAGE}"),
            )),
            _ => {}
        }
        if !used.contains(&name.to_string()) {
            out.push((
                loc,
                format!(
                    "The parameter {name} was declared but was not used in the density calculation."
                ),
            ));
        }
    }

    // Control flow depending on parameters.
    let params = parameter_names(mir);
    let list = |s: &BTreeSet<String>| s.iter().cloned().collect::<Vec<_>>().join(", ");
    for (loc, ps) in deps.target_dependant_cf(&params) {
        out.push((
            loc,
            format!(
                "A control flow statement depends on parameter(s): {}.",
                list(&ps)
            ),
        ));
    }
    for f in &mir.functions {
        let Some(body) = &f.body else { continue };
        let args: Vec<&str> = f.args.iter().map(|(_, n, _)| n.as_str()).collect();
        let fdeps = Deps::new(mir, body);
        let arg_set: BTreeSet<String> = args.iter().map(|s| s.to_string()).collect();
        let mut dep_args: Vec<(Loc, usize, String)> = Vec::new();
        for (loc, names) in fdeps.target_dependant_cf(&arg_set) {
            for n in names {
                let ix = args.iter().position(|a| *a == n).unwrap();
                dep_args.push((loc, ix, n));
            }
        }
        if dep_args.is_empty() {
            continue;
        }
        let mut calls: Vec<(Expr, usize)> = Vec::new();
        for (l, s) in deps.g.nodes.iter().enumerate() {
            for e in stmt_rhs(s) {
                for c in e.children() {
                    c.any(&mut |x: &Expr| {
                        if let ExprP::FunApp(FunKind::UserDefined(n, _), _) = &x.pattern
                            && *n == f.name
                            && !calls.iter().any(|(y, l2)| *l2 == l && y == x)
                        {
                            calls.push((x.clone(), l));
                        }
                        false
                    });
                }
            }
        }
        for (call, l) in calls {
            let ExprP::FunApp(_, call_args) = &call.pattern else {
                unreachable!()
            };
            for (cf_loc, ix, name) in &dep_args {
                let arg = &call_args[*ix];
                let ps = deps.var_deps(l, Some(arg), &params);
                if !ps.is_empty() {
                    out.push((
                        *cf_loc,
                        format!(
                            "A control flow statement inside function {} depends on argument {name}. At {}, the value of {name} depends on parameter(s): {}.",
                            f.name,
                            pp(&arg.meta.loc),
                            list(&ps)
                        ),
                    ));
                }
            }
        }
    }

    // Parameters without exactly one prior.
    for (v, n, loc) in fg.priors(mir) {
        match n {
            Some(0) => out.push((
                loc,
                format!("The parameter {v} has no priors. This means either no prior is provided, or the prior(s) depend on data variables. In the later case, this may be a false positive."),
            )),
            Some(1) | None => {}
            Some(n) => out.push((loc, format!("The parameter {v} has {n} priors."))),
        }
    }

    // Distribution argument checks.
    for d in &dists {
        out.extend(distribution_warning(d));
    }
    out
}
