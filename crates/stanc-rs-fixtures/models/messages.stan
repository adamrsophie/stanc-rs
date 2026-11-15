parameters {
  real x;
}
model {
  if (x > 10) {
    reject("x is ", x, ", with ", [1, 2.5]', " and ", {1, 2});
  }
  x ~ normal(0, 1);
}
generated quantities {
  if (x < -10) {
    fatal_error("x below ", -10);
  }
}
