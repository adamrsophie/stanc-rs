data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] x;
  vector[N] y;
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> sigma;
}
transformed parameters {
  vector[N] mu = alpha + x * beta;
}
model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 2);
  sigma ~ student_t(3, 0, 2);
  y ~ normal(mu, sigma);
}
generated quantities {
  array[N] real y_rep = normal_rng(mu, sigma);
  real log_lik = normal_lpdf(y | mu, sigma);
}
