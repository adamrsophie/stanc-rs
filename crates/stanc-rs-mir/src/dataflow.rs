//! Dataflow analysis on the MIR: stanc3's control flow graphs (`Dataflow_utils`), its monotone
//! framework (`Monotone_framework`), and the analyses built on them.

use std::collections::{BTreeSet, HashSet};

use crate::mir::*;

/// The statements of a MIR statement tree, numbered in preorder, with the control flow between
/// them.
pub struct Graph<'a> {
    /// The statement of each label.
    pub nodes: Vec<&'a Stmt>,
    /// The labels of the statements directly inside each statement.
    pub children: Vec<Vec<usize>>,
    /// The labels that may execute just before each label.
    pub preds: Vec<BTreeSet<usize>>,
    /// The labels that may execute just after each label.
    pub succs: Vec<BTreeSet<usize>>,
    /// The labels that control whether each label executes.
    pub parents: Vec<BTreeSet<usize>>,
    /// The labels with no predecessors, where execution starts.
    pub initials: BTreeSet<usize>,
    /// The labels where execution of the whole statement ends.
    pub exits: BTreeSet<usize>,
}

#[derive(Clone, Default)]
struct CfState {
    breaks: BTreeSet<usize>,
    continues: BTreeSet<usize>,
    returns: BTreeSet<usize>,
    exits: BTreeSet<usize>,
}

fn union(a: &BTreeSet<usize>, b: &BTreeSet<usize>) -> BTreeSet<usize> {
    a.union(b).copied().collect()
}

fn diff(a: &BTreeSet<usize>, b: &BTreeSet<usize>) -> BTreeSet<usize> {
    a.difference(b).copied().collect()
}

fn join(a: &CfState, b: &CfState) -> CfState {
    CfState {
        breaks: union(&a.breaks, &b.breaks),
        continues: union(&a.continues, &b.continues),
        returns: union(&a.returns, &b.returns),
        exits: union(&a.exits, &b.exits),
    }
}

impl<'a> Graph<'a> {
    /// The forward flowgraph of a statement (`forward_flowgraph_of_stmt`). With
    /// `flatten_loops`, the end of a loop body is not followed by the loop again.
    pub fn build(root: &'a Stmt, flatten_loops: bool) -> Graph<'a> {
        let mut g = Graph {
            nodes: Vec::new(),
            children: Vec::new(),
            preds: Vec::new(),
            succs: Vec::new(),
            parents: Vec::new(),
            initials: BTreeSet::new(),
            exits: BTreeSet::new(),
        };
        g.label(root);
        let n = g.nodes.len();
        g.preds = vec![BTreeSet::new(); n];
        g.parents = vec![BTreeSet::new(); n];
        g.exits = g.cf(0, None, CfState::default(), flatten_loops).exits;
        g.succs = vec![BTreeSet::new(); n];
        for l in 0..n {
            if g.preds[l].is_empty() {
                g.initials.insert(l);
            }
            for &p in &g.preds[l] {
                g.succs[p].insert(l);
            }
        }
        g
    }

    fn label(&mut self, s: &'a Stmt) -> usize {
        let l = self.nodes.len();
        self.nodes.push(s);
        self.children.push(Vec::new());
        let cs: Vec<usize> = s.children().into_iter().map(|c| self.label(c)).collect();
        self.children[l] = cs;
        l
    }

    /// `build_cf_graphs`, with blocks visited after their bodies.
    fn cf(&mut self, l: usize, cf_parent: Option<usize>, ins: CfState, flatten: bool) -> CfState {
        let node: &'a Stmt = self.nodes[l];
        let p = &node.pattern;
        let is_loop = matches!(p, StmtP::While(..) | StmtP::For { .. });
        let is_block = matches!(p, StmtP::Block(_) | StmtP::Profile(..));
        let child_cf = if is_loop || matches!(p, StmtP::IfElse(..)) {
            Some(l)
        } else {
            cf_parent
        };
        let child_init = CfState {
            exits: if is_block {
                ins.exits.clone()
            } else {
                BTreeSet::from([l])
            },
            ..ins.clone()
        };
        let children = self.children[l].clone();
        let unlooped = if let StmtP::IfElse(..) = p {
            let then = self.cf(children[0], child_cf, child_init.clone(), flatten);
            match children.get(1) {
                Some(&e) => {
                    let els = self.cf(e, child_cf, child_init, flatten);
                    join(&then, &els)
                }
                None => join(&then, &child_init),
            }
        } else {
            children
                .iter()
                .fold(child_init, |st, &c| self.cf(c, child_cf, st, flatten))
        };
        let (sub, preds) = if is_loop {
            let mut preds = union(&ins.exits, &unlooped.exits);
            preds.extend(diff(&unlooped.continues, &ins.continues));
            let exits = if flatten {
                unlooped.exits.clone()
            } else {
                let mut e = diff(&unlooped.breaks, &ins.breaks);
                e.insert(l);
                e
            };
            (CfState { exits, ..unlooped }, preds)
        } else if is_block {
            let preds = unlooped.exits.clone();
            (
                CfState {
                    exits: BTreeSet::from([l]),
                    ..unlooped
                },
                preds,
            )
        } else {
            (unlooped, ins.exits.clone())
        };
        let mut out = sub.clone();
        let mut extra = BTreeSet::new();
        match p {
            StmtP::Break => {
                out.breaks.insert(l);
            }
            StmtP::Return(_) => {
                out.returns.insert(l);
            }
            StmtP::Continue => {
                out.continues.insert(l);
            }
            StmtP::While(..) | StmtP::For { .. } => {
                out.breaks = ins.breaks.clone();
                out.continues = ins.continues.clone();
                extra = union(&sub.breaks, &sub.returns);
            }
            _ => {}
        }
        let mut parents: BTreeSet<usize> = cf_parent.into_iter().collect();
        parents.extend(&ins.returns);
        parents.extend(&ins.continues);
        parents.extend(extra);
        self.preds[l] = preds;
        self.parents[l] = parents;
        out
    }

    /// The inverse flowgraph, for backward analyses (`inverse_flowgraph_of_stmt`).
    pub fn reversed(&self) -> Graph<'a> {
        Graph {
            nodes: self.nodes.clone(),
            children: self.children.clone(),
            preds: self.succs.clone(),
            succs: self.preds.clone(),
            parents: self.parents.clone(),
            initials: self.exits.clone(),
            exits: self.initials.clone(),
        }
    }

    /// The labels reachable from the initial ones (`all_labels`).
    pub fn reachable(&self) -> BTreeSet<usize> {
        let mut seen = self.initials.clone();
        let mut stack: Vec<usize> = seen.iter().copied().collect();
        while let Some(l) = stack.pop() {
            for &s in &self.succs[l] {
                if seen.insert(s) {
                    stack.push(s);
                }
            }
        }
        seen
    }
}

/// The properties at the entry and exit of a label.
pub struct EntryExit<P> {
    pub entry: P,
    pub exit: P,
}

/// A lattice of dataflow properties.
pub trait Lattice {
    type P: Clone;
    fn bottom(&self) -> Self::P;
    fn initial(&self) -> Self::P;
    fn lub(&self, a: &Self::P, b: &Self::P) -> Self::P;
    fn leq(&self, a: &Self::P, b: &Self::P) -> bool;
}

/// The maximal fixed point of a monotone framework (`monotone_framework`).
pub fn mfp<L: Lattice>(
    g: &Graph,
    lat: &L,
    transfer: impl Fn(usize, &L::P) -> L::P,
) -> Vec<EntryExit<L::P>> {
    let n = g.nodes.len();
    let mut work: Vec<(usize, usize)> = Vec::new();
    for (l, ss) in g.succs.iter().enumerate() {
        for &s in ss {
            work.push((l, s));
        }
    }
    let mut ins: Vec<L::P> = (0..n)
        .map(|l| {
            if g.initials.contains(&l) {
                lat.initial()
            } else {
                lat.bottom()
            }
        })
        .collect();
    while let Some((l, l2)) = work.pop() {
        let new = transfer(l, &ins[l]);
        if !lat.leq(&new, &ins[l2]) {
            ins[l2] = lat.lub(&ins[l2], &new);
            for &l3 in &g.succs[l2] {
                work.push((l2, l3));
            }
        }
    }
    ins.into_iter()
        .enumerate()
        .map(|(l, entry)| {
            let exit = transfer(l, &entry);
            EntryExit { entry, exit }
        })
        .collect()
}

/// Sets ordered by inclusion and joined by intersection, starting empty
/// (`dual_powerset_lattice_empty_initial`).
pub struct DualPowerset<T>(pub BTreeSet<T>);

impl<T: Ord + Clone> Lattice for DualPowerset<T> {
    type P = BTreeSet<T>;
    fn bottom(&self) -> BTreeSet<T> {
        self.0.clone()
    }
    fn initial(&self) -> BTreeSet<T> {
        BTreeSet::new()
    }
    fn lub(&self, a: &BTreeSet<T>, b: &BTreeSet<T>) -> BTreeSet<T> {
        a.intersection(b).cloned().collect()
    }
    fn leq(&self, a: &BTreeSet<T>, b: &BTreeSet<T>) -> bool {
        b.is_subset(a)
    }
}

/// Sets ordered by inclusion and joined by union (`powerset_lattice`).
pub struct Powerset<T>(pub BTreeSet<T>);

impl<T: Ord + Clone> Lattice for Powerset<T> {
    type P = BTreeSet<T>;
    fn bottom(&self) -> BTreeSet<T> {
        BTreeSet::new()
    }
    fn initial(&self) -> BTreeSet<T> {
        self.0.clone()
    }
    fn lub(&self, a: &BTreeSet<T>, b: &BTreeSet<T>) -> BTreeSet<T> {
        a.union(b).cloned().collect()
    }
    fn leq(&self, a: &BTreeSet<T>, b: &BTreeSet<T>) -> bool {
        a.is_subset(b)
    }
}

/// The variables of an expression with their locations (`expr_var_set`).
pub fn expr_vars(e: &Expr, out: &mut Vec<(Loc, String)>) {
    if let ExprP::Var(v) = &e.pattern {
        out.push((e.meta.loc, v.clone()));
    }
    for c in e.children() {
        expr_vars(c, out);
    }
}

/// The expressions a statement reads, not counting its sub-statements (`stmt_rhs`).
pub fn stmt_rhs(s: &Stmt) -> Vec<&Expr> {
    match &s.pattern {
        StmtP::For { lower, upper, .. } => vec![lower, upper],
        StmtP::NRFunApp(k, args) => {
            let mut v: Vec<&Expr> = args.iter().collect();
            if let FunKind::CompilerInternal(f) = k {
                v.extend(f.exprs());
            }
            v
        }
        StmtP::IfElse(e, ..)
        | StmtP::While(e, _)
        | StmtP::Assignment(_, _, e)
        | StmtP::TargetPE(e)
        | StmtP::JacobianPE(e)
        | StmtP::Return(Some(e)) => vec![e],
        _ => vec![],
    }
}

/// The variables a statement reads, with their locations (`stmt_rhs_var_set`).
pub fn stmt_rhs_vars(s: &Stmt) -> Vec<(Loc, String)> {
    let mut v = Vec::new();
    for e in stmt_rhs(s) {
        expr_vars(e, &mut v);
    }
    v
}

fn affects_target(k: &FunKind) -> bool {
    matches!(
        k,
        FunKind::UserDefined(_, Suffix::Target | Suffix::Jacobian)
            | FunKind::StanLib(_, Suffix::Target | Suffix::Jacobian)
    )
}

/// The variables a statement may assign (`assigned_vars_stmt`).
pub fn assigned_vars(s: &Stmt) -> Option<String> {
    match &s.pattern {
        StmtP::Assignment(lv, ..) => Some(lv.variable().to_string()),
        StmtP::Decl {
            id,
            init: DeclInit::Assign(_),
            ..
        } => Some(id.clone()),
        StmtP::TargetPE(_) | StmtP::JacobianPE(_) => Some("target".to_string()),
        StmtP::NRFunApp(k, _) if affects_target(k) => Some("target".to_string()),
        StmtP::For { loopvar, .. } => Some(loopvar.clone()),
        _ => None,
    }
}

/// Reads of variables that may happen before they are assigned
/// (`stmt_uninitialized_variables`).
fn stmt_uninitialized(exceptions: &HashSet<String>, s: &Stmt) -> HashSet<(Loc, String)> {
    let g = Graph::build(s, true);
    let total: BTreeSet<String> = g
        .reachable()
        .into_iter()
        .flat_map(|l| stmt_rhs_vars(g.nodes[l]).into_iter().map(|(_, v)| v))
        .collect();
    let res = mfp(&g, &DualPowerset(total), |l, p| {
        let mut p = p.clone();
        if let Some(v) = assigned_vars(g.nodes[l]) {
            p.insert(v);
        }
        p
    });
    let mut out = HashSet::new();
    for (l, inits) in res.iter().enumerate() {
        for (loc, v) in stmt_rhs_vars(g.nodes[l]) {
            if !inits.entry.contains(&v) && !exceptions.contains(&v) {
                out.insert((loc, v));
            }
        }
    }
    out
}

/// The flag variables the generated code reads (`Flag_vars`).
pub const FLAG_VARS: [&str; 2] = [
    "emit_generated_quantities__",
    "emit_transformed_parameters__",
];

/// Reads of variables that may happen before they are assigned, in every block of a program
/// (`mir_uninitialized_variables`). Unordered; compiler generated locations are left out.
pub fn uninitialized_variables(p: &Program) -> Vec<(Loc, String)> {
    let mut globals: HashSet<String> = p.functions.iter().map(|f| f.name.clone()).collect();
    globals.extend(FLAG_VARS.iter().map(|s| s.to_string()));
    let mut globals_data = globals.clone();
    globals_data.extend(p.input_vars.iter().map(|(n, ..)| n.clone()));
    let mut with_params = globals_data.clone();
    with_params.extend(
        p.output_vars
            .iter()
            .filter(|(_, _, o)| o.block == IoBlock::Parameters)
            .map(|(n, ..)| n.clone()),
    );
    let slist = |ss: Vec<Stmt>| Stmt::new(StmtP::SList(ss), Loc::EMPTY);
    let with_prep = |b: &[Stmt]| {
        let mut v = p.prepare_data.clone();
        v.extend(b.iter().cloned());
        slist(v)
    };
    let mut out = stmt_uninitialized(&globals_data, &slist(p.prepare_data.clone()));
    out.extend(stmt_uninitialized(&with_params, &with_prep(&p.log_prob)));
    out.extend(stmt_uninitialized(
        &with_params,
        &with_prep(&p.generate_quantities),
    ));
    for f in &p.functions {
        if let Some(b) = &f.body {
            let mut ex = globals.clone();
            ex.extend(f.args.iter().map(|(_, n, _)| n.clone()));
            out.extend(stmt_uninitialized(&ex, b));
        }
    }
    out.into_iter().filter(|(l, _)| !l.is_empty()).collect()
}
