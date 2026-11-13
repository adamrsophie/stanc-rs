data {
  int<lower=1> D;
  int<lower=0> N;
  array[N] vector[D] y;
}
parameters {
  vector[D] mu;
  cholesky_factor_corr[D] L;
  vector<lower=0>[D] tau;
}
model {
  mu ~ normal(0, 5);
  L ~ lkj_corr_cholesky(2);
  tau ~ cauchy(0, 2.5);
  y ~ multi_normal_cholesky(mu, diag_pre_multiply(tau, L));
}
generated quantities {
  matrix[D, D] Omega = multiply_lower_tri_self_transpose(L);
}
