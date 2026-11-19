//! Single optimization passes against stanc3's unit tests.
//!
//! `data/optimize_cases.txt` is extracted from stanc3's `test/unit/Optimize.ml` by
//! `tools/optimize_cases.py`. Each case compiles a program to the MIR, applies passes, and
//! compares the printed MIR with stanc3's expected text, ignoring blank lines and the common
//! indentation.

use stanc_rs::mir::{mir::Program, optimize};

struct Case {
    name: String,
    passes: Vec<String>,
    program: String,
    expected: String,
}

fn cases() -> Vec<Case> {
    let text = include_str!("data/optimize_cases.txt");
    let mut out: Vec<Case> = Vec::new();
    let mut section = "";
    for line in text.lines() {
        if let Some(name) = line.strip_prefix("@@@ case ") {
            out.push(Case {
                name: name.to_string(),
                passes: vec![],
                program: String::new(),
                expected: String::new(),
            });
        } else if let Some(p) = line.strip_prefix("@@@ passes ") {
            out.last_mut().unwrap().passes = p.split(' ').map(str::to_string).collect();
        } else if line == "@@@ program" || line == "@@@ expected" {
            section = if line == "@@@ program" {
                "program"
            } else {
                "expected"
            };
        } else {
            let c = out.last_mut().unwrap();
            let buf = if section == "program" {
                &mut c.program
            } else {
                &mut c.expected
            };
            buf.push_str(line);
            buf.push('\n');
        }
    }
    out
}

/// Applies a pass by its stanc3 name; `None` if stanc-rs does not have it.
fn apply(pass: &str, p: &mut Program) -> Option<()> {
    match pass {
        "constant_propagation" => optimize::constant_propagation(p, false),
        "copy_propagation" => optimize::copy_propagation(p),
        "partial_evaluation" => optimize::partial_evaluation(p),
        _ => return None,
    }
    Some(())
}

/// Drops blank lines and the common indentation. stanc3's tests print the program inside a
/// vertical box, which puts the closing brace of each block on its own line.
fn normalize(s: &str) -> String {
    let mut split: Vec<String> = Vec::new();
    for l in s.lines().map(str::trim_end).filter(|l| !l.is_empty()) {
        match l.strip_suffix(" }") {
            Some(body) if !body.trim().is_empty() => {
                let indent = l.len() - l.trim_start().len();
                split.push(body.to_string());
                split.push(format!("{}}}", " ".repeat(indent.saturating_sub(2))));
            }
            _ => split.push(l.to_string()),
        }
    }
    let lines: Vec<&str> = split.iter().map(String::as_str).collect();
    let indent = lines
        .iter()
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    lines
        .iter()
        .map(|l| &l[indent..])
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn passes_match_stanc3_unit_tests() {
    let opts = stanc_rs::Options {
        allow_undefined: true,
        model_name: Some(String::new()),
        ..Default::default()
    };
    let mut failures = Vec::new();
    let mut run = 0;
    for c in cases() {
        let checked = stanc_rs::check_source("test.stan", &c.program, &opts)
            .unwrap_or_else(|e| panic!("{}: {e}", c.name));
        let mut mir = checked.mir();
        if c.passes.iter().any(|p| apply(p, &mut mir).is_none()) {
            continue;
        }
        run += 1;
        let got = normalize(&stanc_rs::mir::pretty::program(&mir));
        let want = normalize(&c.expected);
        if got != want {
            failures.push(format!(
                "=== {}\n--- stanc3\n{want}\n--- stanc-rs\n{got}\n",
                c.name
            ));
        }
    }
    assert!(run > 0);
    assert!(
        failures.is_empty(),
        "{} of {run} cases differ:\n{}",
        failures.len(),
        failures.concat()
    );
}
