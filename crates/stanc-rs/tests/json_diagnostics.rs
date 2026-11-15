//! `check_json`: errors and warnings as JSON.

#[test]
fn errors_and_warnings_as_json() {
    let opts = stanc_rs::Options::default();
    let (ok, out) = stanc_rs::check_json("w.stan", "transformed data { int x = 3 / 2; }", &opts);
    assert!(ok);
    assert!(out.starts_with("{\"ok\": true, \"diagnostics\": [{\"severity\": \"warning\""));
    assert!(out.contains("\"begin\": {\"line\": 1, \"column\": 27}"));
    assert!(out.contains("\"included_from\": []"));

    let (ok, out) = stanc_rs::check_json("e.stan", "model {\n  real y = z;\n}", &opts);
    assert!(!ok);
    assert!(out.contains("\"severity\": \"error\", \"kind\": \"semantic\""));
    assert!(out.contains("\"message\": \"Identifier \\\"z\\\" not in scope.\""));
    assert!(out.contains("\"begin\": {\"line\": 2, \"column\": 11}"));

    let (ok, out) = stanc_rs::check_json("s.stan", "model { real }", &opts);
    assert!(!ok);
    assert!(out.contains("\"kind\": \"parsing\""));
}
