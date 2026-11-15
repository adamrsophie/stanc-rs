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
  real e = get_real(exp(to_complex(0, pi())));
  array[2] complex w = pow({to_complex(0, 1), to_complex(2, 0)}, {to_complex(2, 0), to_complex(3, 0)});
  matrix[2, 3] r = to_matrix({1, 2, 3, 4, 5, 6}, 2, 3, 0);
  array[2] vector[2] s = softmax({[0, 0]', [0, log(3)]'});
}
