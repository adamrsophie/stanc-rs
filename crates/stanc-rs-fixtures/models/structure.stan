data {
  array[2] matrix[3, 4] ms;
  matrix[2, 2] sp;
}
parameters {
  vector[2] p;
}
model {
  p ~ normal(0, 1);
}
generated quantities {
  array[3] int d = dims(ms);
  array[2] int v = csr_extract_v(sp);
  array[3] int u = csr_extract_u(sp);
  vector[4] a = append_row(p, [1, 2]');
  row_vector[3] c = append_col(5, [3, 4]);
}
