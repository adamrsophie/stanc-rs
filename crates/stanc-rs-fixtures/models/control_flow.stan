data {
  int<lower=0> N;
  array[N] real x;
}
transformed data {
  real total = 0;
  int count = 0;
  for (n in 1:N) {
    if (x[n] > 0) {
      total += x[n];
      count += 1;
    } else {
      continue;
    }
  }
  array[N] real x2;
  for (i in 1:N) x2[i] = x[i] ^ 2;
  int k = 0;
  while (k < 3) {
    k += 1;
  }
}
parameters {
  real m;
}
model {
  real acc = 0;
  for (v in x2) {
    acc += v;
  }
  m ~ normal(total / max(count, 1), 1);
  target += -0.001 * acc * m^2;
}
generated quantities {
  real tot = total;
  int cnt = count;
  int kk = k;
}
