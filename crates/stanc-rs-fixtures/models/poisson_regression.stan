data {
  int<lower=0> N;
  vector[N] x;
  array[N] int<lower=0> y;
}
parameters {
  real a;
  real b;
}
model {
  a ~ normal(0, 1);
  b ~ normal(0, 1);
  for (n in 1:N) {
    y[n] ~ poisson_log(a + b * x[n]);
  }
}
