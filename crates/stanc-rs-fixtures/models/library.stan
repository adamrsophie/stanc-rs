transformed data {
  complex_matrix[2, 2] A = [[to_complex(2, 1), to_complex(-1, 0.5)],
                            [to_complex(0.5, 0), to_complex(1, -2)]];
  complex_row_vector[2] x = [to_complex(0.7, -0.1), to_complex(1.2, 0.4)];
  vector[3] w = [1, 2, 3]';
}
generated quantities {
  complex_row_vector[2] d = (x * A) / A;
  complex_matrix[3, 2] ar = append_row(A, x);
  complex s = sum({to_complex(1, 2), to_complex(3, -1)});
  array[2] real ab = abs({to_complex(3, 4), to_complex(0, -2)});
  complex tr = trace(A);
  complex_matrix[2, 2] pre = diag_pre_multiply([to_complex(0, 1), 2]', A);
  complex_matrix[2, 2] ad = add_diag(A, to_complex(1, 1));
  vector[3] rds = rows_dot_self(w);
  vector[4] apr = append_row(0.5, w);
  real l10 = log10();
  matrix[2, 3] dense = csr_to_dense_matrix(2, 3, [5, 6, 7]', {1, 3, 2}, {1, 3, 4});
  vector[2] mv = csr_matrix_times_vector(2, 3, [5, 6, 7]', {1, 3, 2}, {1, 3, 4}, w);
  real sq = squared_distance(1.5, -0.5);
  complex pw = pow(to_complex(0, 1), to_complex(2, 0));
  real tqf = trace_quad_form([[2, 1], [0, 3]], [1, 2]');
  matrix[3, 2] rs = to_matrix([[1, 2, 3], [4, 5, 6]], 3, 2, 0);
  array[2] complex a1 = to_array_1d(x);
}
