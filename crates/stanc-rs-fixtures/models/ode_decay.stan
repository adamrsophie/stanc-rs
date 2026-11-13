functions {
  vector decay(real t, vector y, real k, data real scale) {
    return -k * y * scale;
  }
}
data {
  int<lower=1> T;
  array[T] real ts;
  vector[2] y0;
}
parameters {
  real<lower=0> k;
}
transformed parameters {
  array[T] vector[2] y = ode_rk45_tol(decay, y0, 0.0, ts, 1e-10, 1e-10, 100000, k, 1.0);
}
model {
  k ~ lognormal(0, 1);
  y[T, 1] ~ normal(0.3, 0.1);
}
generated quantities {
  array[T] vector[2] y_bdf = ode_bdf(decay, y0, 0.0, ts, k, 1.0);
}
