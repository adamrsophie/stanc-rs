functions {
  vector sqrt_system(vector y, vector theta, data array[] real x_r, array[] int x_i) {
    return [y[1] * y[1] - theta[1]]';
  }
  vector shifted(vector y, real a, real b) {
    return [y[1] - a * b]';
  }
  real gauss(real x, real xc, array[] real theta, array[] real x_r, array[] int x_i) {
    return exp(-square(x - theta[1]));
  }
  real partial(array[] real slice, int start, int end, real mu) {
    return normal_lpdf(slice | mu, 1);
  }
  vector job(vector phi, vector theta, data array[] real x_r, data array[] int x_i) {
    return [phi[1] * theta[1] + x_r[1] + x_i[1]]';
  }
}
data {
  array[4] real ys;
}
transformed data {
  array[0] real x_r;
  array[0] int x_i;
  array[2, 1] real job_r = {{1.0}, {2.0}};
  array[2, 1] int job_i = {{10}, {20}};
}
parameters {
  real<lower=0.5> a;
  real mu;
}
transformed parameters {
  vector[1] root = algebra_solver(sqrt_system, [1.0]', [a]', x_r, x_i);
  vector[1] root2 = solve_newton(shifted, [0.0]', a, 2.0);
  real area = integrate_1d(gauss, negative_infinity(), positive_infinity(), {mu}, x_r, x_i);
  real rs = reduce_sum(partial, ys, 1, mu);
  vector[2] mr = map_rect(job, [a]', {[1.0]', [2.0]'}, job_r, job_i);
}
model {
  a ~ normal(1, 1);
  mu ~ normal(0, 1);
  target += 0.1 * (root[1] + root2[1] + area * mu + sum(mr)) + rs;
}
