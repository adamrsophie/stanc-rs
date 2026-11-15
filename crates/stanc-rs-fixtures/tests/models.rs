//! Numerical tests of the generated models against independent reference computations.
//!
//! Every log density is checked against a closed form written out here; gradients are checked
//! against analytic derivatives where those are short, and against central finite differences
//! otherwise (all test points are well conditioned).

use stanc_rs_fixtures as fx;
use stanc_rs_runtime::{
    DensityOptions, ErrorKind, Model, data::Data, rand::SeedableRng, rand::rngs::StdRng,
};

const LN_2PI: f64 = 1.8378770664093453;

const FULL: DensityOptions = DensityOptions {
    jacobian: true,
    propto: false,
};
const NO_JAC: DensityOptions = DensityOptions {
    jacobian: false,
    propto: false,
};
const PROPTO: DensityOptions = DensityOptions {
    jacobian: true,
    propto: true,
};

fn data(json: &str) -> Data {
    Data::from_json_str(json).unwrap()
}

fn normal_lpdf(y: f64, mu: f64, sigma: f64) -> f64 {
    -0.5 * LN_2PI - sigma.ln() - 0.5 * ((y - mu) / sigma).powi(2)
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

#[track_caller]
fn assert_close(a: f64, b: f64, tol: f64) {
    assert!(close(a, b, tol), "{a} != {b} (tolerance {tol})");
}

/// Checks the value and gradient against central finite differences of `log_density`.
#[track_caller]
fn check_gradient(m: &dyn Model, theta: &[f64], opts: DensityOptions) {
    let (lp, g) = m.log_density_gradient(theta, opts).unwrap();
    assert_close(lp, m.log_density(theta, opts).unwrap(), 1e-12);
    let h = 1e-6;
    for i in 0..theta.len() {
        let mut up = theta.to_vec();
        let mut down = theta.to_vec();
        up[i] += h;
        down[i] -= h;
        let fd =
            (m.log_density(&up, opts).unwrap() - m.log_density(&down, opts).unwrap()) / (2.0 * h);
        assert_close(g[i], fd, 1e-6);
    }
}

fn rng() -> StdRng {
    StdRng::seed_from_u64(20261113)
}

// Section 23: the normal model.

#[test]
fn normal_scalar_density_and_gradient() {
    for y in [-3.0, 0.0, 0.7, 12.5] {
        let m = fx::normal_scalar::Model::new(&data(&format!("{{\"y\": {y}}}")), 0).unwrap();
        assert_eq!(m.num_unconstrained(), 1);
        for mu in [-10.0, -1.0, 0.0, 0.3, 4.0, 100.0] {
            let expected = normal_lpdf(mu, 0.0, 5.0) + normal_lpdf(y, mu, 1.0);
            let (lp, g) = m.log_density_gradient(&[mu], FULL).unwrap();
            assert_close(lp, expected, 1e-14);
            assert_close(m.log_density(&[mu], FULL).unwrap(), expected, 1e-14);
            assert_close(g[0], -mu / 25.0 + (y - mu), 1e-14);
            // Dropping constants keeps only the terms that depend on mu.
            let propto = -mu * mu / 50.0 - 0.5 * (y - mu).powi(2);
            assert_close(m.log_density(&[mu], PROPTO).unwrap(), propto, 1e-14);
            let (lp, g) = m.log_density_gradient(&[mu], PROPTO).unwrap();
            assert_close(lp, propto, 1e-14);
            assert_close(g[0], -mu / 25.0 + (y - mu), 1e-14);
        }
    }
}

#[test]
fn normal_scalar_errors() {
    let missing = fx::normal_scalar::Model::new(&data("{}"), 0).err().unwrap();
    assert_eq!(missing.kind, ErrorKind::Data);
    assert!(missing.message.contains('y'), "{}", missing.message);
    let wrong_type = fx::normal_scalar::Model::new(&data("{\"y\": [1, 2]}"), 0)
        .err()
        .unwrap();
    assert_eq!(wrong_type.kind, ErrorKind::Data);
    let m = fx::normal_scalar::Model::new(&data("{\"y\": 1}"), 0).unwrap();
    assert_eq!(
        m.log_density(&[1.0, 2.0], FULL).err().unwrap().kind,
        ErrorKind::Parameters
    );
    assert_eq!(m.param_names(true, true), vec!["mu"]);
}

// Section 24: a constrained parameter.

#[test]
fn constrained_scale_jacobian() {
    for y in [0.0, 0.5, 2.0, 30.0] {
        let m = fx::constrained_scale::Model::new(&data(&format!("{{\"y\": {y}}}")), 0).unwrap();
        for z in [-5.0, -1.0, 0.0, 0.4, 3.0] {
            let sigma = f64::exp(z);
            let base = -sigma + normal_lpdf(y, 0.0, sigma);
            let (lp, g) = m.log_density_gradient(&[z], FULL).unwrap();
            assert_close(lp, base + z, 1e-13);
            assert_close(g[0], -sigma + y * y / (sigma * sigma), 1e-12);
            let (lp, g) = m.log_density_gradient(&[z], NO_JAC).unwrap();
            assert_close(lp, base, 1e-13);
            assert_close(g[0], -sigma - 1.0 + y * y / (sigma * sigma), 1e-12);
            assert_close(m.log_density(&[z], NO_JAC).unwrap(), base, 1e-13);
            let out = m.write_array(&[z], true, true, &mut rng()).unwrap();
            assert_close(out[0], sigma, 1e-15);
            let back = m
                .unconstrain(&data(&format!("{{\"sigma\": {sigma:e}}}")))
                .unwrap();
            assert_close(back[0], z, 1e-14);
        }
    }
}

#[test]
fn constrained_scale_extremes_and_invalid_input() {
    let bad = fx::constrained_scale::Model::new(&data("{\"y\": -1}"), 0)
        .err()
        .unwrap();
    assert_eq!(bad.kind, ErrorKind::Data);
    let m = fx::constrained_scale::Model::new(&data("{\"y\": 1}"), 0).unwrap();
    // sigma = exp(-40) underflows nothing but makes the density tiny; sigma = exp(40) is huge.
    for z in [-40.0, 40.0] {
        let sigma = f64::exp(z);
        let expected = -sigma + normal_lpdf(1.0, 0.0, sigma) + z;
        assert_close(m.log_density(&[z], FULL).unwrap(), expected, 1e-12);
    }
    // A non-finite parameter is not a panic.
    let r = m.log_density(&[f64::NAN], FULL);
    assert!(r.is_err() || r.unwrap().is_nan());
    // A negative constrained value cannot be unconstrained.
    assert_eq!(
        m.unconstrain(&data("{\"sigma\": -2}")).err().unwrap().kind,
        ErrorKind::Domain
    );
}

// Section 25: larger models.

#[test]
fn linear_regression() {
    let json = r#"{"N": 3, "K": 2, "x": [[1, 0.5], [0, -1], [2, 2]], "y": [1.5, -0.2, 3.1]}"#;
    let m = fx::linear_regression::Model::new(&data(json), 0).unwrap();
    assert_eq!(m.num_unconstrained(), 4);
    let x = [[1.0, 0.5], [0.0, -1.0], [2.0, 2.0]];
    let y = [1.5, -0.2, 3.1];
    let theta = [0.3, 0.8, -0.4, 0.2];
    let (alpha, beta, sigma) = (theta[0], [theta[1], theta[2]], f64::exp(theta[3]));
    let mu: Vec<f64> = x
        .iter()
        .map(|r| alpha + r[0] * beta[0] + r[1] * beta[1])
        .collect();
    let student_t = |v: f64, nu: f64, s: f64| {
        use stanc_rs_runtime::stanmath_rs::prelude::lgamma;
        lgamma((nu + 1.0) / 2.0)
            - lgamma(nu / 2.0)
            - 0.5 * (nu * std::f64::consts::PI).ln()
            - s.ln()
            - (nu + 1.0) / 2.0 * (1.0 + (v / s).powi(2) / nu).ln()
    };
    let log_lik: f64 = (0..3).map(|n| normal_lpdf(y[n], mu[n], sigma)).sum();
    let expected = normal_lpdf(alpha, 0.0, 10.0)
        + normal_lpdf(beta[0], 0.0, 2.0)
        + normal_lpdf(beta[1], 0.0, 2.0)
        + student_t(sigma, 3.0, 2.0)
        + log_lik
        + theta[3];
    assert_close(m.log_density(&theta, FULL).unwrap(), expected, 1e-13);
    check_gradient(&m, &theta, FULL);
    check_gradient(&m, &theta, PROPTO);

    let names = m.param_names(true, true);
    assert_eq!(
        names,
        [
            "alpha", "beta.1", "beta.2", "sigma", "mu.1", "mu.2", "mu.3", "y_rep.1", "y_rep.2",
            "y_rep.3", "log_lik"
        ]
    );
    let out = m.write_array(&theta, true, true, &mut rng()).unwrap();
    assert_eq!(out.len(), names.len());
    for n in 0..3 {
        assert_close(out[4 + n], mu[n], 1e-14);
        assert!(out[7 + n].is_finite());
    }
    assert_close(out[10], log_lik, 1e-13);
    assert_eq!(
        m.write_array(&theta, false, false, &mut rng())
            .unwrap()
            .len(),
        4
    );
    // Mismatched data sizes are reported, not a panic.
    let bad = r#"{"N": 3, "K": 2, "x": [[1, 0.5], [0, -1]], "y": [1.5, -0.2, 3.1]}"#;
    assert_eq!(
        fx::linear_regression::Model::new(&data(bad), 0)
            .err()
            .unwrap()
            .kind,
        ErrorKind::Data
    );
}

#[test]
fn logistic_regression() {
    let json = r#"{"N": 4, "K": 1, "x": [[-1], [0.5], [2], [0]], "y": [0, 1, 1, 0]}"#;
    let m = fx::logistic_regression::Model::new(&data(json), 0).unwrap();
    let theta = [0.2, -0.7];
    let xs = [-1.0, 0.5, 2.0, 0.0];
    let ys = [0.0, 1.0, 1.0, 0.0];
    let mut expected = normal_lpdf(theta[0], 0.0, 2.5) + normal_lpdf(theta[1], 0.0, 1.0);
    for n in 0..4 {
        let eta = theta[0] + theta[1] * xs[n];
        let p = 1.0 / (1.0 + (-eta).exp());
        expected += ys[n] * p.ln() + (1.0 - ys[n]) * (1.0 - p).ln();
    }
    assert_close(m.log_density(&theta, FULL).unwrap(), expected, 1e-13);
    check_gradient(&m, &theta, FULL);
    let out = m.write_array(&theta, true, true, &mut rng()).unwrap();
    assert_close(
        out[2 + 2],
        1.0 / (1.0 + (-(theta[0] + theta[1] * 2.0)).exp()),
        1e-14,
    );
    let bad = r#"{"N": 1, "K": 1, "x": [[0]], "y": [2]}"#;
    assert_eq!(
        fx::logistic_regression::Model::new(&data(bad), 0)
            .err()
            .unwrap()
            .kind,
        ErrorKind::Data
    );
}

#[test]
fn poisson_regression() {
    let json = r#"{"N": 3, "x": [0, 1, 2.5], "y": [1, 0, 7]}"#;
    let m = fx::poisson_regression::Model::new(&data(json), 0).unwrap();
    let theta = [0.1, 0.6];
    let mut expected = normal_lpdf(theta[0], 0.0, 1.0) + normal_lpdf(theta[1], 0.0, 1.0);
    for (x, y) in [(0.0, 1.0), (1.0, 0.0), (2.5, 7.0f64)] {
        let eta = theta[0] + theta[1] * x;
        let lfact: f64 = (1..=y as u32).map(|k| f64::from(k).ln()).sum();
        expected += y * eta - eta.exp() - lfact;
    }
    assert_close(m.log_density(&theta, FULL).unwrap(), expected, 1e-13);
    check_gradient(&m, &theta, FULL);
}

#[test]
fn eight_schools_noncentered() {
    let json =
        r#"{"J": 8, "y": [28, 8, -3, 7, -1, 1, 18, 12], "sigma": [15, 10, 16, 11, 9, 11, 10, 18]}"#;
    let m = fx::eight_schools::Model::new(&data(json), 0).unwrap();
    assert_eq!(m.num_unconstrained(), 10);
    let ys = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
    let ss = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0];
    let theta: Vec<f64> = (0..10).map(|i| 0.1 * i as f64 - 0.3).collect();
    let (mu, log_tau) = (theta[0], theta[1]);
    let tau = log_tau.exp();
    let cauchy = |x: f64, loc: f64, s: f64| {
        -(std::f64::consts::PI * s).ln() - (1.0 + ((x - loc) / s).powi(2)).ln()
    };
    let mut expected = normal_lpdf(mu, 0.0, 5.0) + cauchy(tau, 0.0, 5.0) + log_tau;
    for j in 0..8 {
        let t = theta[2 + j];
        expected += normal_lpdf(t, 0.0, 1.0) + normal_lpdf(ys[j], mu + tau * t, ss[j]);
    }
    assert_close(m.log_density(&theta, FULL).unwrap(), expected, 1e-13);
    check_gradient(&m, &theta, FULL);
    let out = m.write_array(&theta, true, false, &mut rng()).unwrap();
    assert_eq!(out.len(), 2 + 8 + 8);
    assert_close(out[10 + 3], mu + tau * theta[2 + 3], 1e-14);
}

#[test]
fn simplex_ordered_bounded() {
    let m =
        fx::simplex_ordered::Model::new(&data(r#"{"K": 4, "alpha": [1, 2, 3, 0.5]}"#), 0).unwrap();
    assert_eq!(m.num_unconstrained(), 3 + 3 + 1);
    let theta = [0.3, -0.2, 0.5, -1.0, 0.4, -0.3, 0.8];
    let out = m.write_array(&theta, true, true, &mut rng()).unwrap();
    let simplex = &out[0..4];
    assert!(simplex.iter().all(|v| *v > 0.0));
    assert_close(simplex.iter().sum::<f64>(), 1.0, 1e-14);
    assert!(out[4] < out[5] && out[5] < out[6]);
    assert!(out[7] > -1.0 && out[7] < 2.0);
    assert_close(out[8], 1.0, 1e-14);
    // Unconstraining the constrained values gives back the unconstrained vector.
    let json = format!(
        r#"{{"theta": [{}, {}, {}, {}], "cut": [{}, {}, {}], "u": {}}}"#,
        out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]
    );
    let back = m.unconstrain(&data(&json)).unwrap();
    for (a, b) in back.iter().zip(&theta) {
        assert_close(*a, *b, 1e-10);
    }
    check_gradient(&m, &theta, FULL);
    check_gradient(&m, &theta, NO_JAC);
    // The bounded scalar's Jacobian: u = -1 + 3 inv_logit(z), so log |du/dz| = log 3 + log p (1 - p).
    let lp = m.log_density(&theta, FULL).unwrap() - m.log_density(&theta, NO_JAC).unwrap();
    let p = 1.0 / (1.0 + (-theta[6]).exp());
    let ordered_jac = theta[4] + theta[5];
    let simplex_jac = lp - ordered_jac - (3.0f64.ln() + p.ln() + (1.0 - p).ln());
    assert!(simplex_jac.is_finite() && simplex_jac < 0.0);
    // Data constraints are checked.
    let bad = r#"{"K": 1, "alpha": [1]}"#;
    assert_eq!(
        fx::simplex_ordered::Model::new(&data(bad), 0)
            .err()
            .unwrap()
            .kind,
        ErrorKind::Data
    );
}

#[test]
fn correlation_cholesky() {
    let json = r#"{"D": 3, "N": 2, "y": [[0.1, -0.3, 1.2], [0.5, 0.0, -0.4]]}"#;
    let m = fx::correlated::Model::new(&data(json), 0).unwrap();
    assert_eq!(m.num_unconstrained(), 3 + 3 + 3);
    let theta = [0.1, -0.2, 0.3, 0.4, -0.5, 0.2, 0.0, 0.3, -0.1];
    check_gradient(&m, &theta, FULL);
    let names = m.param_names(true, true);
    let out = m.write_array(&theta, true, true, &mut rng()).unwrap();
    assert_eq!(out.len(), names.len());
    let omega = names.iter().position(|n| n == "Omega.1.1").unwrap();
    for i in 0..3 {
        assert_close(out[omega + i * 3 + i], 1.0, 1e-14);
    }
}

#[test]
fn user_defined_functions() {
    let m = fx::udf::Model::new(&data(r#"{"y": 1.3, "z": 0.5}"#), 0).unwrap();
    for mu in [-2.0, 0.0, 0.9] {
        // `z ~ my_normal(mu, 2) T[0, ]` subtracts the user's `my_normal_lccdf(0 | mu, 2)`.
        let expected =
            -0.5 * mu * mu + normal_lpdf(1.3, mu, 2.0) + normal_lpdf(0.5, mu, 2.0) + 0.1 * mu * mu;
        let (lp, g) = m.log_density_gradient(&[mu], FULL).unwrap();
        assert_close(lp, expected, 1e-14);
        assert_close(
            g[0],
            -mu + (1.3 - mu) / 4.0 + (0.5 - mu) / 4.0 + 0.2 * mu,
            1e-14,
        );
        let out = m.write_array(&[mu], true, true, &mut rng()).unwrap();
        assert_eq!(
            out,
            vec![mu, mu * mu + 1.0, if mu > 0.0 { 1.0 } else { 0.0 }]
        );
    }
}

#[test]
fn control_flow() {
    let m = fx::control_flow::Model::new(&data(r#"{"N": 4, "x": [1.5, -2, 0.5, 3]}"#), 0).unwrap();
    let out = m.write_array(&[0.2], true, true, &mut rng()).unwrap();
    assert_eq!(out, vec![0.2, 5.0, 3.0, 3.0]);
    let acc: f64 = [1.5f64, -2.0, 0.5, 3.0].iter().map(|v| v * v).sum();
    let m0 = 0.2;
    let expected = normal_lpdf(m0, 5.0 / 3.0, 1.0) - 0.001 * acc * m0 * m0;
    assert_close(m.log_density(&[m0], FULL).unwrap(), expected, 1e-14);
    check_gradient(&m, &[m0], FULL);
}

#[test]
fn ode_solutions_and_gradient() {
    let m = fx::ode_decay::Model::new(&data(r#"{"T": 3, "ts": [0.5, 1, 2], "y0": [1, 2]}"#), 0)
        .unwrap();
    for z in [-0.5, 0.2] {
        let k = f64::exp(z);
        let out = m.write_array(&[z], true, true, &mut rng()).unwrap();
        // k, then y[3, 2] and y_bdf[3, 2], each column-major (time index fastest).
        for (block, tol) in [(1, 1e-8), (7, 1e-7)] {
            for (j, t) in [0.5, 1.0, 2.0].iter().enumerate() {
                for (c, y0) in [1.0, 2.0].iter().enumerate() {
                    assert_close(out[block + c * 3 + j], y0 * (-k * t).exp(), tol);
                }
            }
        }
        let y_end = (-k * 2.0).exp();
        let expected = -0.5 * LN_2PI - z - 0.5 * z * z + normal_lpdf(y_end, 0.3, 0.1) + z;
        let (lp, g) = m.log_density_gradient(&[z], FULL).unwrap();
        assert_close(lp, expected, 1e-7);
        // d/dz of the likelihood term through y = exp(-2 k), dk/dz = k.
        let dy = -2.0 * k * y_end;
        let analytic = -z - (y_end - 0.3) / 0.01 * dy;
        assert_close(g[0], analytic, 1e-6);
    }
}

#[test]
fn algebra_integration_and_functionals() {
    let ys = [0.5, -1.0, 2.0, 0.1];
    let m = fx::solvers::Model::new(&data(r#"{"ys": [0.5, -1.0, 2.0, 0.1]}"#), 0).unwrap();
    let theta = [0.3, -0.4];
    let a = 0.5 + f64::exp(theta[0]);
    let mu = theta[1];
    let out = m.write_array(&theta, true, false, &mut rng()).unwrap();
    // a, mu, root, root2, area, rs, mr[2]
    assert_close(out[0], a, 1e-15);
    assert_close(out[2], a.sqrt(), 1e-10);
    assert_close(out[3], 2.0 * a, 1e-10);
    assert_close(out[4], std::f64::consts::PI.sqrt(), 1e-8);
    let rs: f64 = ys.iter().map(|y| normal_lpdf(*y, mu, 1.0)).sum();
    assert_close(out[5], rs, 1e-14);
    assert_close(out[6], a + 1.0 + 10.0, 1e-14);
    assert_close(out[7], 2.0 * a + 2.0 + 20.0, 1e-14);
    check_gradient(&m, &theta, FULL);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn constraint_functions() {
    let m = fx::transforms::Model::new(&data("{}"), 0).unwrap();
    let theta = [0.3, -0.2, 0.5, 0.7];
    let (y, z) = (&theta[0..3], theta[3]);
    let names = m.param_names(true, true);
    let out = m.write_array(&theta, true, true, &mut rng()).unwrap();
    assert_eq!(out.len(), names.len());
    let at = |n: &str| out[names.iter().position(|x| x == n).unwrap()];
    // The constrained values agree with the parameter transforms of the runtime.
    let s = stanc_rs_runtime::transform::simplex(y, None).unwrap();
    for (k, sk) in s.iter().enumerate() {
        assert_close(at(&format!("s.{}", k + 1)), *sk, 1e-14);
        assert_close(at(&format!("s2.{}", k + 1)), *sk, 1e-14);
    }
    assert_close(at("p"), 1.0 + z.exp(), 1e-14);
    // Unconstraining inverts constraining.
    for k in 0..3 {
        assert_close(at(&format!("y_back.{}", k + 1)), y[k], 1e-10);
    }
    assert_close(at("z_back"), z, 1e-12);
    // ordered_constrain is applied to each array element: x[1] = y[1], x[k] = x[k-1] + exp(y[k]).
    for (i, scale) in [1.0, 2.0].iter().enumerate() {
        let mut x = scale * y[0];
        assert_close(at(&format!("o.{}.1", i + 1)), x, 1e-14);
        for k in 1..3 {
            x += (scale * y[k]).exp();
            assert_close(at(&format!("o.{}.{}", i + 1, k + 1)), x, 1e-14);
            assert_close(
                at(&format!("o_back.{}.{}", i + 1, k + 1)),
                scale * y[k],
                1e-10,
            );
        }
    }
    // Tuple outputs are named `t:1`, `t:2.1`, ... and written slot by slot.
    assert_close(at("t:1"), z, 0.0);
    assert_close(at("t:2.1"), 1.0, 0.0);
    assert_close(at("t:2.2"), 2.0, 0.0);
    // Vectorized pow over two vectors.
    for k in 0..3 {
        assert_close(
            at(&format!("w.{}", k + 1)),
            y[k].exp().powf(y[k] + 1.0),
            1e-14,
        );
    }
    // Assignments through several indexes, multi-indexes first or last.
    let a = [[0.0, 5.0, 6.0], [0.0, 2.0, 0.0]];
    for (i, row) in a.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            assert_close(at(&format!("a.{}.{}", i + 1, j + 1)), *v, 0.0);
        }
    }
    // `b[{2, 1}, 1] = {7, 8}` sets b[2][1] = 7 and b[1][1] = 8.
    assert_close(at("b.2.1"), 7.0, 0.0);
    assert_close(at("b.1.1"), 8.0, 0.0);
    assert_close(at("b.1.2"), 0.0, 0.0);
    // Rows and elements of matrices inside arrays.
    for (n, v) in [
        ("mm.1.1.1", 0.0),
        ("mm.1.2.1", 6.0),
        ("mm.1.2.2", 4.0),
        ("mm.2.1.2", 5.0),
        ("mm.2.2.1", 7.0),
        ("mm.2.2.2", 0.0),
    ] {
        assert_close(at(n), v, 0.0);
    }
    // The `_jacobian` calls add the log Jacobians only when the Jacobian is requested.
    let base: f64 = theta.iter().map(|v| normal_lpdf(*v, 0.0, 1.0)).sum();
    assert_close(m.log_density(&theta, NO_JAC).unwrap(), base, 1e-12);
    let mut simplex_jac = 0.0;
    stanc_rs_runtime::transform::simplex(y, Some(&mut simplex_jac)).unwrap();
    assert_close(
        m.log_density(&theta, FULL).unwrap(),
        base + simplex_jac + z,
        1e-12,
    );
    check_gradient(&m, &theta, FULL);
    check_gradient(&m, &theta, NO_JAC);
}

#[test]
fn parameter_times_dae_adjoint_quadrature() {
    let m = fx::solvers2::Model::new(&data("{}"), 0).unwrap();
    let theta = [0.2_f64, -0.3];
    let (k, t) = (theta[0].exp(), theta[1].exp());
    let out = m.write_array(&theta, true, true, &mut rng()).unwrap();
    // k, T, y_end, y_adj_t, y_dae[2], y_adj[2], gk, de
    let y_end = (-k * t).exp();
    assert_close(out[2], y_end, 1e-8);
    assert_close(out[3], y_end, 1e-7);
    for (j, s) in [0.5, 1.0].iter().enumerate() {
        assert_close(out[4 + j], (-k * s).exp(), 1e-7);
        assert_close(out[6 + j], (-k * s).exp(), 1e-7);
    }
    let gauss = (2.0 * std::f64::consts::PI).sqrt() * k;
    assert_close(out[8], gauss, 1e-7);
    assert_close(out[9], gauss / 2.0, 1e-7);
    let lognormal = |z: f64| -0.5 * LN_2PI - z - 0.5 * z * z;
    let expected = lognormal(theta[0])
        + lognormal(theta[1])
        + theta[0]
        + theta[1]
        + 2.0 * normal_lpdf(y_end, 0.5, 0.2);
    let (lp, g) = m.log_density_gradient(&theta, FULL).unwrap();
    assert_close(lp, expected, 1e-7);
    // The derivative with respect to the output time: d y_end / dz_T = -k T y_end; both
    // solutions contribute.
    let dlik = -2.0 * (y_end - 0.5) / 0.04;
    assert_close(g[1], -theta[1] + dlik * (-k * t * y_end), 1e-6);
    assert_close(g[0], -theta[0] + dlik * (-k * t * y_end), 1e-6);
}

#[test]
fn laplace_approximation() {
    let m = fx::laplace::Model::new(&data(r#"{"y": [0.4, -1.1], "c": [3, 0]}"#), 0).unwrap();
    let theta = [-0.2_f64, 0.3];
    let (sigma, alpha) = (theta[0].exp(), theta[1].exp());
    let ys = [0.4, -1.1];
    // With a Gaussian likelihood the approximation is exact: y ~ normal(0, sqrt(alpha^2 + sigma^2)).
    let sd = (alpha * alpha + sigma * sigma).sqrt();
    let marginal: f64 = ys.iter().map(|y| normal_lpdf(*y, 0.0, sd)).sum();
    assert_close(m.log_density(&theta, NO_JAC).unwrap(), marginal, 1e-8);
    check_gradient(&m, &theta, FULL);
    let names = m.param_names(true, true);
    let out = m.write_array(&theta, true, true, &mut rng()).unwrap();
    assert_eq!(out.len(), names.len());
    let at = |n: &str| out[names.iter().position(|x| x == n).unwrap()];
    // Posterior of the latent values: mean shrunk by a^2 / (a^2 + s^2), variance a^2 s^2 / (a^2 + s^2).
    let w = alpha * alpha / (sd * sd);
    let chol = (w * sigma * sigma).sqrt();
    for (i, y) in ys.iter().enumerate() {
        assert_close(at(&format!("fit:1.{}", i + 1)), w * y, 1e-8);
        assert_close(at(&format!("fit:2.{}.{}", i + 1, i + 1)), chol, 1e-8);
        assert!(at(&format!("draw.{}", i + 1)).is_finite());
        assert!(at(&format!("pdraw.{}", i + 1)).is_finite());
    }
    assert_close(at("fit:2.2.1"), 0.0, 1e-12);
    // The built in Poisson likelihood agrees with the same likelihood written as a function.
    assert_close(at("builtin"), at("user"), 1e-8);
    // A covariance taking nested tuples with data and parameter parts.
    assert_close(at("nested"), at("user"), 1e-12);
}

// Tuple parameters, alone and in arrays, with constrained components.
#[test]
fn tuple_parameters() {
    let m = fx::tuple_params::Model::new(&data("{}"), 0).unwrap();
    assert_eq!(m.num_unconstrained(), 8);
    let theta: [f64; 8] = [0.4, -0.3, 0.2, 1.1, -0.5, -0.7, 0.3, 0.9];
    let inv_logit = |x: f64| 1.0 / (1.0 + (-x).exp());
    let (a, b) = (theta[0], theta[1]);
    let p2 = b.exp();
    let mut expected = normal_lpdf(a, 0.0, 1.0) - p2 + b;
    let mut q1 = Vec::new();
    for (c, v) in [(theta[2], &theta[3..5]), (theta[5], &theta[6..8])] {
        let u = inv_logit(c);
        let x = -1.0 + 2.0 * u;
        q1.push(x);
        expected += normal_lpdf(x, 0.0, 1.0) + (2.0 * u * (1.0 - u)).ln();
        expected += normal_lpdf(v[0], 0.0, 1.0) + normal_lpdf(v[1], 0.0, 1.0);
    }
    let (lp, _) = m.log_density_gradient(&theta, FULL).unwrap();
    assert_close(lp, expected, 1e-13);
    check_gradient(&m, &theta, FULL);
    assert_eq!(
        m.param_names(true, true),
        [
            "p:1", "p:2", "q.1:1", "q.1:2.1", "q.1:2.2", "q.2:1", "q.2:2.1", "q.2:2.2"
        ]
    );
    let out = m.write_array(&theta, true, true, &mut rng()).unwrap();
    let want = [a, p2, q1[0], theta[3], theta[4], q1[1], theta[6], theta[7]];
    for (o, w) in out.iter().zip(want) {
        assert_close(*o, w, 1e-14);
    }
    let json = format!(
        r#"{{"p": {{"1": {a}, "2": {p2:e}}}, "q": [{{"1": {:e}, "2": [{}, {}]}}, {{"1": {:e}, "2": [{}, {}]}}]}}"#,
        q1[0], theta[3], theta[4], q1[1], theta[6], theta[7]
    );
    let back = m.unconstrain(&data(&json)).unwrap();
    for (x, t) in back.iter().zip(theta) {
        assert_close(*x, t, 1e-12);
    }
}

// A truncated container outcome subtracts each element's normalizing term.
#[test]
fn vector_truncation() {
    let m = fx::vector_truncate::Model::new(&data(r#"{"y": [0.5, -0.2, 1.4]}"#), 0).unwrap();
    // The standard normal CDF from the Taylor series of erf, accurate on |x| < 3.
    fn erf(x: f64) -> f64 {
        let mut sum = x;
        let mut term = x;
        for n in 1..200 {
            term *= -x * x / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    }
    let phi = |z: f64| 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
    let y = [0.5, -0.2, 1.4];
    let mu: [f64; 3] = [0.1, -0.4, 0.8];
    let mut expected = 0.0;
    for k in 0..3 {
        let lp = normal_lpdf(y[k], mu[k], 1.0);
        let (lo, hi) = (phi(-1.0 - mu[k]), phi(2.0 - mu[k]));
        expected += 3.0 * lp - (1.0 - lo).ln() - hi.ln() - (hi - lo).ln();
    }
    let (lp, _) = m.log_density_gradient(&mu, FULL).unwrap();
    assert_close(lp, expected, 1e-12);
    check_gradient(&m, &mu, FULL);
}

// Library functions that stanc-rs implements itself rather than through stanmath-rs.
#[test]
fn library_functions() {
    let m = fx::library::Model::new(&data("{}"), 0).unwrap();
    let names = m.param_names(true, true);
    let out = m.write_array(&[], true, true, &mut rng()).unwrap();
    assert_eq!(out.len(), names.len());
    let at = |n: &str| out[names.iter().position(|x| x == n).unwrap()];
    // (x * A) / A recovers x.
    for (n, v) in [
        ("d.1.real", 0.7),
        ("d.1.imag", -0.1),
        ("d.2.real", 1.2),
        ("d.2.imag", 0.4),
    ] {
        assert_close(at(n), v, 1e-14);
    }
    // append_row puts x under A.
    assert_close(at("ar.1.2.real"), -1.0, 0.0);
    assert_close(at("ar.3.1.real"), 0.7, 0.0);
    assert_close(at("ar.3.2.imag"), 0.4, 0.0);
    assert_close(at("s.real"), 4.0, 0.0);
    assert_close(at("s.imag"), 1.0, 0.0);
    assert_close(at("ab.1"), 5.0, 1e-15);
    assert_close(at("ab.2"), 2.0, 1e-15);
    assert_close(at("tr.real"), 3.0, 0.0);
    assert_close(at("tr.imag"), -1.0, 0.0);
    // i * (2 + i) = -1 + 2i; 2 * (1 - 2i) = 2 - 4i.
    assert_close(at("pre.1.1.real"), -1.0, 1e-15);
    assert_close(at("pre.1.1.imag"), 2.0, 1e-15);
    assert_close(at("pre.2.2.real"), 2.0, 1e-15);
    assert_close(at("pre.2.2.imag"), -4.0, 1e-15);
    assert_close(at("ad.1.1.imag"), 2.0, 0.0);
    assert_close(at("ad.1.2.real"), -1.0, 0.0);
    for (k, v) in [1.0, 4.0, 9.0].into_iter().enumerate() {
        assert_close(at(&format!("rds.{}", k + 1)), v, 0.0);
    }
    for (k, v) in [0.5, 1.0, 2.0, 3.0].into_iter().enumerate() {
        assert_close(at(&format!("apr.{}", k + 1)), v, 0.0);
    }
    assert_close(at("l10"), std::f64::consts::LN_10, 0.0);
    // Rows: [5, 0, 6] and [0, 7, 0].
    let dense = [[5.0, 0.0, 6.0], [0.0, 7.0, 0.0]];
    for (i, row) in dense.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            assert_close(at(&format!("dense.{}.{}", i + 1, j + 1)), *v, 0.0);
        }
    }
    assert_close(at("mv.1"), 5.0 + 18.0, 0.0);
    assert_close(at("mv.2"), 14.0, 0.0);
    assert_close(at("sq"), 4.0, 0.0);
    // i^2 = -1.
    assert_close(at("pw.real"), -1.0, 1e-15);
    assert_close(at("pw.imag"), 0.0, 1e-15);
    // [1 2] [[2, 1], [0, 3]] [1 2]' = 2 + 2 + 12.
    assert_close(at("tqf"), 16.0, 0.0);
    // Column-major elements 1, 4, 2, 5, 3, 6 filled by rows.
    for (k, v) in [1.0, 4.0, 2.0, 5.0, 3.0, 6.0].into_iter().enumerate() {
        assert_close(at(&format!("rs.{}.{}", k / 2 + 1, k % 2 + 1)), v, 0.0);
    }
    assert_close(at("a1.2.imag"), 0.4, 0.0);
}

// Data assigned to transformed parameters and model locals, which are autodiff values.
#[test]
fn data_assigned_to_parameter_variables() {
    let m = fx::data_into_params::Model::new(&data(r#"{"a": 0.5, "v": [1.0, -2.0]}"#), 0).unwrap();
    let mu = 0.3;
    let expected = normal_lpdf(1.0 + mu, 0.5, 1.0)
        + normal_lpdf(-2.0, 0.5, 1.0)
        + normal_lpdf(1.0, 0.0, 1.0)
        + normal_lpdf(mu, 0.0, 1.0);
    let (lp, g) = m.log_density_gradient(&[mu], FULL).unwrap();
    assert_close(lp, expected, 1e-14);
    assert_close(g[0], -(1.0 + mu - 0.5) - mu, 1e-14);
}

// A bound that depends on an earlier parameter, and a `~` inside an `_lp` function.
#[test]
fn dependent_bounds_and_lp_tilde() {
    let m = fx::dependent_bounds::Model::new(&data("{}"), 0).unwrap();
    let back = m.unconstrain(&data(r#"{"a": 0.25, "b": 0.5}"#)).unwrap();
    let out = m.write_array(&back, true, false, &mut rng()).unwrap();
    for (o, e) in out.iter().zip([0.25, 0.5, 0.75, 6.0, 1.0]) {
        assert_close(*o, e, 1e-14);
    }
    // Dropping constants removes the normal normalizers of both `~` statements.
    let full = m.log_density(&back, FULL).unwrap();
    let propto = m.log_density(&back, PROPTO).unwrap();
    assert_close(full - propto, -LN_2PI, 1e-12);
    // Without Jacobians: the two normal priors and the -3 from `shift_lp` called on data, but not
    // the 0.5 from `twice_jacobian`.
    let no_jac = m.log_density(&back, NO_JAC).unwrap();
    assert_close(no_jac, -LN_2PI - 0.5 * (0.0625 + 0.25) - 3.0, 1e-12);
}

// Bounds that are arrays give each element its own bound.
#[test]
fn array_valued_bounds() {
    let m = fx::array_bounds::Model::new(
        &data(r#"{"lo": [1.0, -2.0], "hi": [[0.0, 1.0], [2.0, 3.0]]}"#),
        0,
    )
    .unwrap();
    let values = r#"{"a": [1.5, -1.0], "b": [[-1.0, 0.5], [1.0, 2.5]]}"#;
    let z = m.unconstrain(&data(values)).unwrap();
    assert_close(z[0], 0.5_f64.ln(), 1e-14);
    assert_close(z[1], 1.0_f64.ln(), 1e-14);
    assert_close(z[2], 1.0_f64.ln(), 1e-14);
    let out = m.write_array(&z, true, false, &mut rng()).unwrap();
    let expected = [1.5, -1.0, -1.0, 1.0, 0.5, 2.5, 1.5, -1.0];
    for (o, e) in out.iter().zip(expected) {
        assert_close(*o, e, 1e-13);
    }
    // The density is the log Jacobian alone: the sum of the unconstrained values.
    let lp = m.log_density(&z, FULL).unwrap();
    assert_close(lp, z.iter().sum(), 1e-13);
}

// dims, the one-based CSR indices, and appends mixing parameters with data.
#[test]
fn structural_functions() {
    let m = fx::structure::Model::new(
        &data(r#"{"ms": [[[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0]]], "sp": [[0, 5], [6, 0]]}"#),
        0,
    )
    .unwrap();
    let out = m
        .write_array(&[0.5, -0.5], false, true, &mut rng())
        .unwrap();
    // After the 17 original values: exp(i pi); i^2 and 2^3 as (re, im) pairs;
    // the row-major 2 by 3 reshape, column by column; and the softmax of each
    // vector, as s[1][1], s[2][1], s[1][2], s[2][2]. Then cv = (1 + 2i, 3i), the squared
    // norm 14 of cv, the unconjugated products (1 + 2i)^2 and (3i)^2, and 2 / cv. Last,
    // cm = [[1, 2], [3, 4 + i]] by column, its columns (1, 3) and (2, 4 + i) from
    // to_vector_array, and its rows from to_array_2d, both with the first index fastest.
    let third = 2.0 / 3.0;
    let expected = [
        0.5, -0.5, 2.0, 3.0, 4.0, 2.0, 1.0, 1.0, 2.0, 3.0, 0.5, -0.5, 1.0, 2.0, 5.0, 3.0, 4.0,
        -1.0, -1.0, 0.0, 8.0, 0.0, 1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 0.5, 0.25, 0.5, 0.75, 1.0, 2.0,
        0.0, 3.0, 14.0, 0.0, -3.0, 4.0, -9.0, 0.0, 0.4, -0.8, 0.0, -third, 1.0, 0.0, 3.0, 0.0, 2.0,
        0.0, 4.0, 1.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 1.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0,
        1.0,
    ];
    assert_eq!(out.len(), expected.len());
    for (o, e) in out.iter().zip(expected) {
        assert_close(*o, e, 1e-14);
    }
}

/// log |det A| by Gaussian elimination with partial pivoting.
fn log_abs_det(mut a: Vec<Vec<f64>>) -> f64 {
    let n = a.len();
    let mut s = 0.0;
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .unwrap();
        a.swap(c, p);
        s += a[c][c].abs().ln();
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..n {
                a[r][k] -= f * a[c][k];
            }
        }
    }
    s
}

/// log |det| of the central-difference Jacobian of constrained outputs `outs` with respect to
/// unconstrained coordinates `ins`.
fn fd_log_jacobian(m: &dyn Model, theta: &[f64], ins: &[usize], outs: &[usize]) -> f64 {
    let h = 1e-6;
    let mut jac = vec![vec![0.0; ins.len()]; outs.len()];
    for (j, &i) in ins.iter().enumerate() {
        let (mut up, mut down) = (theta.to_vec(), theta.to_vec());
        up[i] += h;
        down[i] -= h;
        let fu = m.write_array(&up, false, false, &mut rng()).unwrap();
        let fd = m.write_array(&down, false, false, &mut rng()).unwrap();
        for (r, &o) in outs.iter().enumerate() {
            jac[r][j] = (fu[o] - fd[o]) / (2.0 * h);
        }
    }
    log_abs_det(jac)
}

// The remaining constraint transforms: values satisfy their constraints, unconstraining inverts
// constraining, and the Jacobian adjustment matches finite differences over the free
// coordinates (or the closed form where Stan uses one that is not a determinant).
#[test]
fn constraint_transforms() {
    let (mu, tau) = (1.5, 0.5);
    let m = fx::constraints::Model::new(&data(r#"{"mu": 1.5, "tau": 0.5}"#), 0).unwrap();
    assert_eq!(m.num_unconstrained(), 3 + 3 + 2 + 1 + 3 + 1 + 5 + 2 + 4);
    // The unit vector part is already of norm one, so it survives a round trip.
    let theta = [
        0.2, -0.4, 0.1, 0.6, 0.0, 0.8, 0.3, -0.7, 0.9, 0.2, -0.3, 0.4, 0.5, -0.2, 0.1, 0.3, 0.4,
        -0.6, 0.25, -0.5, 0.1, 0.2, -0.3, 0.4,
    ];
    let o = m.write_array(&theta, false, false, &mut rng()).unwrap();
    assert_eq!(o.len(), 34);
    // positive_ordered
    assert!(0.0 < o[0] && o[0] < o[1] && o[1] < o[2]);
    // unit_vector
    assert_close(o[3] * o[3] + o[4] * o[4] + o[5] * o[5], 1.0, 1e-14);
    // sum_to_zero_vector
    assert_close(o[6] + o[7] + o[8], 0.0, 1e-14);
    // offset and multiplier
    assert_close(o[9], mu + tau * theta[8], 1e-14);
    // cov_matrix, column major
    assert_close(o[11], o[12], 1e-14);
    assert!(o[10] > 0.0 && o[10] * o[13] - o[11] * o[12] > 0.0);
    // corr_matrix
    assert_close(o[14], 1.0, 1e-14);
    assert_close(o[17], 1.0, 1e-14);
    assert_close(o[15], o[16], 1e-14);
    assert!(o[15].abs() < 1.0);
    // cholesky_factor_cov[3, 2]: L[1, 2] is zero and the diagonal is positive
    assert_eq!(o[21], 0.0);
    assert!(o[18] > 0.0 && o[22] > 0.0);
    // column_stochastic_matrix[2, 2] and row_stochastic_matrix[2, 3]
    assert_close(o[24] + o[25], 1.0, 1e-14);
    assert_close(o[26] + o[27], 1.0, 1e-14);
    assert_close(o[28] + o[30] + o[32], 1.0, 1e-14);
    assert_close(o[29] + o[31] + o[33], 1.0, 1e-14);
    assert!(o[24..34].iter().all(|v| *v > 0.0));

    let names = ["po", "uv", "sz", "om", "S", "R", "L", "C", "Q"];
    let shapes: [&[usize]; 9] = [
        &[3],
        &[3],
        &[3],
        &[],
        &[2, 2],
        &[2, 2],
        &[3, 2],
        &[2, 2],
        &[2, 3],
    ];
    let mut json = String::from("{");
    let mut at = 0;
    for (k, (n, sh)) in names.iter().zip(shapes).enumerate() {
        let len: usize = sh.iter().product();
        let vals = &o[at..at + len];
        at += len;
        // Column-major output to nested JSON arrays.
        let v = match sh {
            [] => format!("{}", vals[0]),
            [_] => format!("{vals:?}"),
            [r, c] => {
                let rows: Vec<String> = (0..*r)
                    .map(|i| format!("{:?}", (0..*c).map(|j| vals[i + j * r]).collect::<Vec<_>>()))
                    .collect();
                format!("[{}]", rows.join(", "))
            }
            _ => unreachable!(),
        };
        if k > 0 {
            json.push_str(", ");
        }
        json.push_str(&format!("\"{n}\": {v}"));
    }
    json.push('}');
    let back = m.unconstrain(&data(&json)).unwrap();
    for (a, b) in back.iter().zip(&theta) {
        assert_close(*a, *b, 1e-10);
    }

    let jac = m.log_density(&theta, FULL).unwrap() - m.log_density(&theta, NO_JAC).unwrap();
    let fd = |ins: std::ops::Range<usize>, outs: &[usize]| {
        fd_log_jacobian(&m, &theta, &ins.collect::<Vec<_>>(), outs)
    };
    let expected = fd(0..3, &[0, 1, 2])
        // unit_vector: Stan adds the standard normal kernel of the unconstrained vector.
        - 0.5 * (theta[3] * theta[3] + theta[4] * theta[4] + theta[5] * theta[5])
        // sum_to_zero_vector: the isometric transform has no adjustment.
        + tau.ln()
        + fd(9..12, &[10, 11, 13])
        + fd(12..13, &[15])
        + fd(13..18, &[18, 19, 20, 22, 23])
        + fd(18..20, &[24, 26])
        + fd(20..24, &[28, 29, 30, 31]);
    assert_close(jac, expected, 1e-6);
    check_gradient(&m, &theta, FULL);
}
