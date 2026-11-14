parameters {
  vector[3] y;
  real z;
}
transformed parameters {
  simplex[4] s = simplex_jacobian(y);
  real<lower=1> p = lower_bound_jacobian(z, 1);
}
model {
  y ~ normal(0, 1);
  z ~ normal(0, 1);
}
generated quantities {
  vector[3] y_back = simplex_unconstrain(s);
  real z_back = lower_bound_unconstrain(p, 1);
  vector[4] s2 = simplex_constrain(y);
  array[2] vector[3] o = ordered_constrain({y, 2 * y});
  array[2] vector[3] o_back = ordered_unconstrain(o);
  tuple(real, vector[2]) t = (z, [1, 2]');
  vector[3] w = pow(exp(y), y + 1);
  array[2, 3] real a = rep_array(0.0, 2, 3);
  a[:, 2] = {1.0, 2.0};
  a[1, 2:3] = {5.0, 6.0};
  array[2] vector[3] b = rep_array(rep_vector(0, 3), 2);
  b[{2, 1}, 1] = {7.0, 8.0};
  array[2] matrix[2, 2] mm = rep_array(rep_matrix(0, 2, 2), 2);
  mm[1, 2] = [3, 4];
  mm[2][1, 2] = 5;
  mm[1:2, 2, 1] = {6.0, 7.0};
}
