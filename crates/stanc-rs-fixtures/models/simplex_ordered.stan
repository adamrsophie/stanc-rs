data {
  int<lower=2> K;
  vector<lower=0>[K] alpha;
}
parameters {
  simplex[K] theta;
  ordered[3] cut;
  real<lower=-1, upper=2> u;
}
model {
  theta ~ dirichlet(alpha);
  cut ~ normal(0, 3);
  u ~ uniform(-1, 2);
}
generated quantities {
  real total = sum(theta);
}
