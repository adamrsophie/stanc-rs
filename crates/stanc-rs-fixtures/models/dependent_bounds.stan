functions {
  void prior_lp(real x) {
    x ~ normal(0, 1);
  }
}
parameters {
  real<lower=0, upper=1> a;
  real<lower=0, upper=1 - a> b;
}
transformed parameters {
  real s = a + b;
}
model {
  prior_lp(a);
  prior_lp(b);
}
