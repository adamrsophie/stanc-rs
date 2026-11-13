//! Type helpers, autodiff levels, and promotions, following stanc3's `UnsizedType` and
//! `Promotion` modules.

use stanc_rs_syntax::ast::{AdLevel, FnSuffix, FunctionType, UnsizedType};

/// Whether an expression may carry derivatives. Tuples carry one level per slot.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ad {
    /// Data: plain `f64` values.
    DataOnly,
    /// May depend on parameters.
    AutoDiffable,
    /// Per-slot levels of a tuple.
    Tuple(Vec<Ad>),
}

impl From<AdLevel> for Ad {
    fn from(a: AdLevel) -> Ad {
        match a {
            AdLevel::DataOnly => Ad::DataOnly,
            AdLevel::AutoDiffable => Ad::AutoDiffable,
        }
    }
}

impl Ad {
    /// True if any part may carry derivatives.
    pub fn has_autodiff(&self) -> bool {
        match self {
            Ad::DataOnly => false,
            Ad::AutoDiffable => true,
            Ad::Tuple(xs) => xs.iter().any(Ad::has_autodiff),
        }
    }

    /// Least upper bound of two levels.
    pub fn lub(&self, other: &Ad) -> Option<Ad> {
        Some(match (self, other) {
            (Ad::DataOnly, a) | (a, Ad::DataOnly) => a.clone(),
            (Ad::AutoDiffable, Ad::AutoDiffable) => Ad::AutoDiffable,
            (Ad::Tuple(a), Ad::Tuple(b)) => {
                if a.len() != b.len() {
                    return None;
                }
                Ad::Tuple(
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| x.lub(y))
                        .collect::<Option<_>>()?,
                )
            }
            (Ad::Tuple(a), Ad::AutoDiffable) | (Ad::AutoDiffable, Ad::Tuple(a)) => Ad::Tuple(
                a.iter()
                    .map(|x| x.lub(&Ad::AutoDiffable))
                    .collect::<Option<_>>()?,
            ),
        })
    }

    /// Least upper bound of many levels.
    pub fn lub_all<'a>(xs: impl IntoIterator<Item = &'a Ad>) -> Option<Ad> {
        xs.into_iter().try_fold(Ad::DataOnly, |acc, x| acc.lub(x))
    }

    /// The level `self` applied to every slot of `ty` (tuples get per-slot levels).
    pub fn fill_for(self, ty: &UnsizedType) -> Ad {
        match (ty.array_dims().1, &self) {
            (UnsizedType::Tuple(ts), Ad::DataOnly | Ad::AutoDiffable) => {
                Ad::Tuple(ts.iter().map(|t| self.clone().fill_for(t)).collect())
            }
            _ => self,
        }
    }
}

/// True if a value of level `from` can be passed where `to` is expected.
pub fn ad_can_convert(to: &Ad, from: &Ad) -> bool {
    match (to, from) {
        (Ad::DataOnly, Ad::AutoDiffable) => false,
        (Ad::Tuple(a), Ad::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| ad_can_convert(x, y))
        }
        (Ad::DataOnly, Ad::Tuple(b)) => b.iter().all(|y| ad_can_convert(&Ad::DataOnly, y)),
        _ => true,
    }
}

/// A conversion applied to an argument so that it matches a signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Promotion {
    None,
    IntToReal,
    /// Data to autodiff (used in arrays and assignments).
    ToVar,
    ToComplexVar,
    IntToComplex,
    RealToComplex,
    Tuple(Vec<Promotion>),
}

impl Promotion {
    /// Cost used to choose among overloads.
    pub fn cost(&self) -> usize {
        match self {
            Promotion::None | Promotion::ToVar | Promotion::ToComplexVar => 0,
            Promotion::RealToComplex | Promotion::IntToReal => 1,
            Promotion::IntToComplex => 2,
            Promotion::Tuple(ps) => ps.iter().map(Promotion::cost).sum(),
        }
    }

    /// True for [`Promotion::None`] and tuples of it.
    pub fn is_none(&self) -> bool {
        match self {
            Promotion::None => true,
            Promotion::Tuple(ps) => ps.iter().all(Promotion::is_none),
            _ => false,
        }
    }
}

/// True for `int` and arrays and tuples made only of `int`.
pub fn is_discrete(t: &UnsizedType) -> bool {
    match t {
        UnsizedType::Int => true,
        UnsizedType::Array(t) => is_discrete(t),
        UnsizedType::Tuple(ts) => ts.iter().all(is_discrete),
        _ => false,
    }
}

/// True for `int` and arrays of `int` (not tuples).
pub fn is_int_type(t: &UnsizedType) -> bool {
    match t {
        UnsizedType::Int => true,
        UnsizedType::Array(t) => is_int_type(t),
        _ => false,
    }
}

/// True if any part of the type is an `int`.
pub fn contains_int(t: &UnsizedType) -> bool {
    match t {
        UnsizedType::Int => true,
        UnsizedType::Array(t) => contains_int(t),
        UnsizedType::Tuple(ts) => ts.iter().any(contains_int),
        _ => false,
    }
}

/// True for complex scalars and containers (and arrays of them).
pub fn is_complex_type(t: &UnsizedType) -> bool {
    match t {
        UnsizedType::Complex
        | UnsizedType::ComplexVector
        | UnsizedType::ComplexRowVector
        | UnsizedType::ComplexMatrix => true,
        UnsizedType::Array(t) => is_complex_type(t),
        _ => false,
    }
}

/// True for `int`, `real`, `complex`.
pub fn is_scalar(t: &UnsizedType) -> bool {
    matches!(
        t,
        UnsizedType::Int | UnsizedType::Real | UnsizedType::Complex
    )
}

/// True for function types.
pub fn is_fun_type(t: &UnsizedType) -> bool {
    matches!(
        t,
        UnsizedType::Function(_) | UnsizedType::MathLibraryFunction
    )
}

/// The minimal type both can convert to.
pub fn common_type(a: &UnsizedType, b: &UnsizedType) -> Option<UnsizedType> {
    use UnsizedType::*;
    Some(match (a, b) {
        (Real, Int) | (Int, Real) => Real,
        (Complex, Int) | (Int, Complex) | (Complex, Real) | (Real, Complex) => Complex,
        (ComplexVector, Vector) | (Vector, ComplexVector) => ComplexVector,
        (ComplexRowVector, RowVector) | (RowVector, ComplexRowVector) => ComplexRowVector,
        (ComplexMatrix, Matrix) | (Matrix, ComplexMatrix) => ComplexMatrix,
        (Array(x), Array(y)) => Array(Box::new(common_type(x, y)?)),
        (Tuple(xs), Tuple(ys)) => {
            if xs.len() != ys.len() {
                return None;
            }
            Tuple(
                xs.iter()
                    .zip(ys)
                    .map(|(x, y)| common_type(x, y))
                    .collect::<Option<_>>()?,
            )
        }
        (x, y) if x == y => x.clone(),
        _ => return None,
    })
}

/// Why an argument does not match a signature.
#[derive(Clone, Debug, PartialEq)]
pub enum Mismatch {
    /// Types differ.
    Type(UnsizedType, UnsizedType),
    /// A data-only argument received an autodiff value.
    DataOnly,
    /// Wrong number of arguments: expected, found.
    ArgCount(usize, usize),
    /// Function argument has the wrong suffix kind.
    Suffix,
    /// Function argument has the wrong return type.
    ReturnType,
    /// Function argument has incompatible arguments.
    Input(Box<Mismatch>),
    /// The argument at (one-based) position has a mismatch.
    Arg(usize, Box<Mismatch>),
}

fn without_propto(s: FnSuffix) -> FnSuffix {
    s
}

/// Can a value of type `actual` be used where `expected` is required (stanc3 `check_same_type`)?
pub fn check_same_type(
    depth: usize,
    expected: &UnsizedType,
    actual: &UnsizedType,
) -> Result<Promotion, Mismatch> {
    use UnsizedType::*;
    let mismatch = || Mismatch::Type(expected.clone(), actual.clone());
    match (expected, actual) {
        (a, b) if a == b => Ok(Promotion::None),
        (Real, Int) if depth < 1 => Ok(Promotion::IntToReal),
        (Complex, Int) if depth < 1 => Ok(Promotion::IntToComplex),
        (Complex, Real)
        | (ComplexMatrix, Matrix)
        | (ComplexVector, Vector)
        | (ComplexRowVector, RowVector)
            if depth < 1 =>
        {
            Ok(Promotion::RealToComplex)
        }
        (Array(a), Array(b)) => check_same_type(depth, a, b).map_err(|e| match e {
            Mismatch::Type(..) => mismatch(),
            e => e,
        }),
        (Tuple(xs), Tuple(ys)) => {
            if xs.len() != ys.len() {
                return Err(mismatch());
            }
            let mut ps = Vec::new();
            for (x, y) in xs.iter().zip(ys) {
                match check_same_type(depth, x, y) {
                    Ok(p) => ps.push(p),
                    Err(Mismatch::Type(..)) => return Err(mismatch()),
                    Err(e) => return Err(e),
                }
            }
            Ok(Promotion::Tuple(ps))
        }
        (Function(f1), Function(f2)) => {
            if without_propto(f1.suffix) != without_propto(f2.suffix) {
                return Err(Mismatch::Suffix);
            }
            if f1.ret != f2.ret {
                return Err(Mismatch::ReturnType);
            }
            // Contravariant: the supplied function must accept what the signature passes it.
            let a1: Vec<(Ad, UnsizedType)> = f1
                .args
                .iter()
                .map(|(a, t)| ((*a).into(), t.clone()))
                .collect();
            check_compatible_args(depth + 1, &f2.args, &a1)
                .map(|_| Promotion::None)
                .map_err(|e| Mismatch::Input(Box::new(e)))
        }
        _ => Err(mismatch()),
    }
}

/// Matches arguments against a signature's parameter list.
pub fn check_compatible_args(
    depth: usize,
    sig: &[(AdLevel, UnsizedType)],
    args: &[(Ad, UnsizedType)],
) -> Result<Vec<Promotion>, Mismatch> {
    if sig.len() != args.len() {
        return Err(Mismatch::ArgCount(sig.len(), args.len()));
    }
    let mut out = Vec::with_capacity(sig.len());
    for (i, ((sad, st), (aad, at))) in sig.iter().zip(args).enumerate() {
        let p = check_same_type(depth, st, at).map_err(|e| Mismatch::Arg(i + 1, Box::new(e)))?;
        let sad: Ad = (*sad).into();
        if sad != *aad && !(depth < 2 && ad_can_convert(&sad, aad)) {
            return Err(Mismatch::Arg(i + 1, Box::new(Mismatch::DataOnly)));
        }
        out.push(p);
    }
    Ok(out)
}

/// The promotion that turns a value of `(ad_current, current)` into `(ad_requested, requested)`.
/// The types must already be known to be compatible.
pub fn type_promotion(
    ad_requested: &Ad,
    requested: &UnsizedType,
    ad_current: &Ad,
    current: &UnsizedType,
) -> Promotion {
    use UnsizedType::*;
    let differs =
        ad_current != ad_requested && ad_requested.has_autodiff() && !ad_current.has_autodiff();
    match (requested, current) {
        (Real, Real | Int) | (Vector, Vector) | (RowVector, RowVector) | (Matrix, Matrix)
            if differs =>
        {
            Promotion::ToVar
        }
        (Complex, Real | Int | Complex)
        | (ComplexMatrix, Matrix | ComplexMatrix)
        | (ComplexVector, Vector | ComplexVector)
        | (ComplexRowVector, RowVector | ComplexRowVector)
            if differs =>
        {
            Promotion::ToComplexVar
        }
        (Real, Int) => Promotion::IntToReal,
        (Complex, Int) => Promotion::IntToComplex,
        (Complex, Real)
        | (ComplexMatrix, Matrix)
        | (ComplexVector, Vector)
        | (ComplexRowVector, RowVector) => Promotion::RealToComplex,
        (Array(a), Array(b)) => type_promotion(ad_requested, a, ad_current, b),
        (Tuple(xs), Tuple(ys)) => {
            let get = |ad: &Ad, i: usize| match ad {
                Ad::Tuple(v) => v.get(i).cloned().unwrap_or(Ad::DataOnly),
                a => a.clone(),
            };
            Promotion::Tuple(
                xs.iter()
                    .zip(ys)
                    .enumerate()
                    .map(|(i, (x, y))| {
                        type_promotion(&get(ad_requested, i), x, &get(ad_current, i), y)
                    })
                    .collect(),
            )
        }
        _ => Promotion::None,
    }
}

/// The type after applying a promotion.
pub fn promoted_type(t: &UnsizedType, p: &Promotion) -> UnsizedType {
    use UnsizedType::*;
    match (p, t) {
        (Promotion::None, t) => t.clone(),
        (_, Array(inner)) => Array(Box::new(promoted_type(inner, p))),
        (Promotion::IntToReal | Promotion::ToVar, Int) => Real,
        (
            Promotion::IntToComplex | Promotion::RealToComplex | Promotion::ToComplexVar,
            Int | Real,
        ) => Complex,
        (Promotion::RealToComplex | Promotion::ToComplexVar, Vector) => ComplexVector,
        (Promotion::RealToComplex | Promotion::ToComplexVar, RowVector) => ComplexRowVector,
        (Promotion::RealToComplex | Promotion::ToComplexVar, Matrix) => ComplexMatrix,
        (Promotion::Tuple(ps), Tuple(ts)) => Tuple(
            ts.iter()
                .zip(ps)
                .map(|(t, p)| promoted_type(t, p))
                .collect(),
        ),
        (_, t) => t.clone(),
    }
}

/// Function type helper.
pub fn fun_type(
    args: Vec<(AdLevel, UnsizedType)>,
    ret: Option<UnsizedType>,
    suffix: FnSuffix,
) -> UnsizedType {
    UnsizedType::Function(Box::new(FunctionType { ret, args, suffix }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use UnsizedType::*;

    #[test]
    fn promotions() {
        assert_eq!(check_same_type(0, &Real, &Int), Ok(Promotion::IntToReal));
        assert!(check_same_type(1, &Real, &Int).is_err());
        assert_eq!(
            check_same_type(0, &Array(Box::new(Real)), &Array(Box::new(Int))),
            Ok(Promotion::IntToReal)
        );
        assert!(check_same_type(0, &Vector, &RowVector).is_err());
        assert_eq!(Promotion::IntToComplex.cost(), 2);
        let sig = [(AdLevel::DataOnly, Real)];
        assert!(check_compatible_args(0, &sig, &[(Ad::AutoDiffable, Real)]).is_err());
        assert!(
            check_compatible_args(0, &[(AdLevel::AutoDiffable, Real)], &[(Ad::DataOnly, Int)])
                .is_ok()
        );
        assert_eq!(common_type(&Int, &Real), Some(Real));
        assert_eq!(common_type(&Vector, &RowVector), None);
    }
}
