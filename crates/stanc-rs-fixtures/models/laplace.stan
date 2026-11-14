functions {
  real gaussian(vector theta, real sigma, vector y) {
    return normal_lpdf(y | theta, sigma);
  }
  real counts(vector theta, array[] int c, real scale) {
    return scale * poisson_log_lpmf(c | theta);
  }
  matrix kernel(int n, real alpha) {
    return diag_matrix(rep_vector(square(alpha), n));
  }
}
data {
  vector[2] y;
  array[2] int c;
}
parameters {
  real<lower=0> sigma;
  real<lower=0> alpha;
}
model {
  target += laplace_marginal(gaussian, (sigma, y), 1, kernel, (2, alpha));
}
generated quantities {
  tuple(vector[2], matrix[2, 2]) fit = laplace_latent_solve(gaussian, (sigma, y), 1, kernel, (2, alpha));
  vector[2] draw = laplace_latent_rng(gaussian, (sigma, y), 1, kernel, (2, alpha));
  real builtin = laplace_marginal_poisson_log_lpmf(c | {1, 2}, rep_vector(0, 2), 1, kernel, (2, alpha));
  real user = laplace_marginal(counts, (c, 1.0), 1, kernel, (2, alpha));
  vector[2] pdraw = laplace_latent_poisson_log_rng(c, {1, 2}, rep_vector(0, 2), 1, kernel, (2, alpha));
}
