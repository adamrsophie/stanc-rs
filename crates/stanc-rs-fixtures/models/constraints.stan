data {
  real mu;
  real<lower=0> tau;
}
parameters {
  positive_ordered[3] po;
  unit_vector[3] uv;
  sum_to_zero_vector[3] sz;
  real<offset=mu, multiplier=tau> om;
  cov_matrix[2] S;
  corr_matrix[2] R;
  cholesky_factor_cov[3, 2] L;
  column_stochastic_matrix[2, 2] C;
  row_stochastic_matrix[2, 3] Q;
  sum_to_zero_matrix[2, 3] Z;
  simplex[3] s;
  cholesky_factor_corr[3] Lc;
}
