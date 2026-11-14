parameters {
  tuple(real, real<lower=0>) p;
  array[2] tuple(real<lower=-1, upper=1>, vector[2]) q;
}
model {
  p.1 ~ normal(0, 1);
  p.2 ~ exponential(1);
  for (i in 1:2) {
    q[i].1 ~ normal(0, 1);
    q[i].2 ~ normal(0, 1);
  }
}
