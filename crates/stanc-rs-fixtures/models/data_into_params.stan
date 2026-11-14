data {
  real a;
  vector[2] v;
}
parameters {
  real mu;
}
transformed parameters {
  real t = a;
  vector[2] w;
  w = v;
  w[1] += mu;
}
model {
  vector[2] m = v;
  m[2] = mu;
  target += normal_lpdf(w | t, 1) + normal_lpdf(m | 0, 1);
}
