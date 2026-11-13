data {
  real<lower=0> y;
}
parameters {
  real<lower=0> sigma;
}
model {
  sigma ~ exponential(1);
  y ~ normal(0, sigma);
}
