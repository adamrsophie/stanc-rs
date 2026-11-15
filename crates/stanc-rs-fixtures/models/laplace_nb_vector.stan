functions {
  matrix K(int n, real s) {
    return diag_matrix(rep_vector(s, n));
  }
}
data {
  int n;
  int k;
  array[n] int y;
  array[n] int g;
  int n_eta;
  vector[n_eta] eta;
}
parameters {
  real<lower=0> s;
}
model {
  target += laplace_marginal_neg_binomial_2_log_lpmf(y | g, eta, rep_vector(0.0, k), 1, K, (k, s));
}
generated quantities {
  vector[k] theta = laplace_latent_neg_binomial_2_log_rng(y, g, eta, rep_vector(0.0, k), 1, K, (k, s));
}
