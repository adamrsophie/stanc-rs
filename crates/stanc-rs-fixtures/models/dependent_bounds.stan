functions {
  void prior_lp(real x) {
    x ~ normal(0, 1);
  }
  real shift_lp(real r) {
    target += -r;
    return 2 * r;
  }
  real twice_jacobian(real x) {
    jacobian += x;
    return 2 * x;
  }
}
parameters {
  real<lower=0, upper=1> a;
  real<lower=0, upper=1 - a> b;
}
transformed parameters {
  real s = a + b;
  real c = shift_lp(3.0);
  real d = twice_jacobian(0.5);
}
model {
  prior_lp(a);
  prior_lp(b);
}
