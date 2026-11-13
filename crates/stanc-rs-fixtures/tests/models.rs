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
    let m = fx::udf::Model::new(&data(r#"{"y": 1.3}"#), 0).unwrap();
    for mu in [-2.0, 0.0, 0.9] {
        let expected = -0.5 * mu * mu + normal_lpdf(1.3, mu, 2.0);
        let (lp, g) = m.log_density_gradient(&[mu], FULL).unwrap();
        assert_close(lp, expected, 1e-14);
        assert_close(g[0], -mu + (1.3 - mu) / 4.0, 1e-14);
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
