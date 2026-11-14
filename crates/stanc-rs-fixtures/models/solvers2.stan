functions {
  vector decay(real t, vector y, real k) {
    return -k * y;
  }
  vector residual(real t, vector y, vector yp, real k) {
    return yp + k * y;
  }
  real gauss(real x, real xc, real s) {
    return exp(-square(x / s) / 2);
  }
}
parameters {
  real<lower=0> k;
  real<lower=0> T;
}
transformed parameters {
  // The output time is a parameter.
  array[1] vector[1] y_end = ode_rk45_tol(decay, [1.0]', 0.0, {T}, 1e-10, 1e-10, 100000, k);
  array[1] vector[1] y_adj_t = ode_adjoint_tol_ctl(decay, [1.0]', 0.0, {T}, 1e-10, [1e-10]',
                                                   1e-10, [1e-10]', 1e-10, 1e-10, 100000, 150, 1, 2, 2, k);
}
model {
  k ~ lognormal(0, 1);
  T ~ lognormal(0, 1);
  y_end[1, 1] ~ normal(0.5, 0.2);
  y_adj_t[1, 1] ~ normal(0.5, 0.2);
}
generated quantities {
  array[2] vector[1] y_dae = dae_tol(residual, [1.0]', [-k]', 0.0, {0.5, 1.0}, 1e-10, 1e-10, 100000, k);
  array[2] vector[1] y_adj = ode_adjoint_tol_ctl(decay, [1.0]', 0.0, {0.5, 1.0}, 1e-10, [1e-10]',
                                                 1e-10, [1e-10]', 1e-10, 1e-10, 100000, 150, 1, 2, 2, k);
  real gk = integrate_1d_gauss_kronrod(gauss, negative_infinity(), positive_infinity(), k);
  real de = integrate_1d_double_exponential(gauss, 0, positive_infinity(), k);
}
