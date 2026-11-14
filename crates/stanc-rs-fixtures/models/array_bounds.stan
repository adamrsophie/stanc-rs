data {
  array[2] real lo;
  array[2] vector[2] hi;
}
parameters {
  array[2] real<lower=lo> a;
  array[2] vector<upper=hi>[2] b;
}
transformed parameters {
  array[2] real<lower=lo> ta = a;
}
