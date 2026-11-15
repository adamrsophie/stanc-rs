//! Prints a fixture model's log density and gradient at an unconstrained point, for comparison
//! with CmdStan's `log_prob` method (see `tools/cmdstan_diff.py`).
//!
//! Usage: `log_prob <model> <data.json> [<theta>]`, where `<theta>` is a comma separated list of
//! unconstrained values. Without it, prints the number of unconstrained parameters. Otherwise
//! prints one JSON object with the log density up to a constant and its gradient, with and
//! without the Jacobian adjustment, as CmdStan's `log_prob` computes them.

use stanc_rs_runtime::DensityOptions;
use stanc_rs_runtime::data::Data;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [name, data, rest @ ..] = args.as_slice() else {
        return Err("usage: log_prob <model> <data.json> [<theta>]".into());
    };
    let text = std::fs::read_to_string(data).map_err(|e| e.to_string())?;
    let data = Data::from_json_str(&text).map_err(|e| e.message)?;
    let m = stanc_rs_fixtures::model(name, &data, 0)
        .ok_or_else(|| format!("no fixture model '{name}'"))?
        .map_err(|e| e.message)?;
    let Some(theta) = rest.first() else {
        println!("{}", m.num_unconstrained());
        return Ok(());
    };
    let theta: Vec<f64> = theta
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f64>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let mut parts = Vec::new();
    for (key, jacobian) in [("jacobian", true), ("no_jacobian", false)] {
        let opts = DensityOptions {
            jacobian,
            propto: true,
        };
        let (lp, g) = m
            .log_density_gradient(&theta, opts)
            .map_err(|e| e.message)?;
        let g: Vec<String> = g.iter().map(|v| format!("{v:e}")).collect();
        parts.push(format!(
            "\"{key}\": {{\"lp\": {lp:e}, \"gradient\": [{}]}}",
            g.join(", ")
        ));
    }
    println!("{{{}}}", parts.join(", "));
    Ok(())
}
