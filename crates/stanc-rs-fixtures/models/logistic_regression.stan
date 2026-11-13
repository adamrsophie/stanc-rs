data {
  int<lower=0> N;
  int<lower=1> K;
  matrix[N, K] x;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  real alpha;
  vector[K] beta;
}
model {
  alpha ~ normal(0, 2.5);
  beta ~ normal(0, 1);
  y ~ bernoulli_logit(alpha + x * beta);
}
generated quantities {
  vector[N] p = inv_logit(alpha + x * beta);
}
