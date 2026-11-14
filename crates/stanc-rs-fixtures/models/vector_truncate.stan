data {
  array[3] real y;
}
parameters {
  vector[3] mu;
}
model {
  y ~ normal(mu, 1) T[-1, ];
  y ~ normal(mu, 1) T[, 2];
  y ~ normal(mu, 1) T[-1, 2];
}
