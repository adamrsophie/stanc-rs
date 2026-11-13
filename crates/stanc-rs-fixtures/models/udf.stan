functions {
  real square_plus(real x, real c) {
    return x * x + c;
  }
  real my_normal_lpdf(real y, real mu, real s) {
    return -0.5 * square((y - mu) / s) - log(s) - 0.5 * log(2 * pi());
  }
  void add_prior_lp(real x) {
    target += -0.5 * x * x;
  }
}
data {
  real y;
}
parameters {
  real mu;
}
model {
  add_prior_lp(mu);
  y ~ my_normal(mu, 2);
}
generated quantities {
  real sq = square_plus(mu, 1);
  int flag = mu > 0 ? 1 : 0;
}
