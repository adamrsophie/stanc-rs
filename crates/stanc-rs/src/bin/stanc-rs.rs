//! The `stanc-rs` command line interface.

use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "\
stanc-rs: compile Stan programs to Rust

USAGE:
    stanc-rs check   <file.stan> [options]
    stanc-rs compile <file.stan> [--output <file.rs>] [options]
    stanc-rs ast     <file.stan> [options]
    stanc-rs ir      <file.stan> [options]
    stanc-rs info    <file.stan> [options]   (variables and calls as JSON, as stanc3 --info)
    stanc-rs format  <file.stan> [options]

OPTIONS:
    --output, -o <path>       Where to write the generated Rust (compile; default: stdout)
    --name <name>             Model name (default: <file stem>_model)
    --include-paths <dirs>    Comma-separated directories searched for #include files
    --allow-undefined         Accept function declarations without definitions
    --max-line-length <n>     Line length for format (default: 78)
    --canonicalize <list>     Rewrites for format: deprecations, parentheses, braces,
                              includes, strip-comments (comma-separated)
    --print-canonical         For format: --canonicalize deprecations,parentheses,braces,includes
    --json-diagnostics        For check: print errors and warnings to stdout as JSON
    --warn-uninitialized      Warn about variables that may be read before they are assigned
    --warn-pedantic           Warn about likely modeling mistakes, as stanc3's pedantic mode does
    --debug-mir-pretty        For check: print the MIR as stanc3 --debug-mir-pretty does
    --help, -h                Print this message
    --version, -V             Print the version
";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            // Rendered diagnostics end with a newline; other messages do not.
            if msg.is_empty() {
                // Already reported (JSON diagnostics).
            } else if msg.ends_with('\n') {
                eprint!("{msg}");
            } else {
                eprintln!("{msg}");
            }
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let mut cmd = None;
    let mut file = None;
    let mut output = None;
    let mut opts = stanc_rs::Options::default();
    let mut fopts = stanc_rs::syntax::autoformat::Options::default();
    let mut canon = stanc_rs::syntax::canonicalize::Settings::default();
    let mut print_canonical = false;
    let mut json = false;
    let mut debug_mir = false;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        let mut value = |flag: &str| it.next().ok_or_else(|| format!("missing value for {flag}"));
        match a.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            "--version" | "-V" => {
                println!("stanc-rs {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--output" | "-o" => output = Some(PathBuf::from(value(&a)?)),
            "--name" => opts.model_name = Some(value(&a)?),
            "--include-paths" => {
                opts.include_paths.extend(
                    value(&a)?
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(PathBuf::from),
                );
            }
            "--allow-undefined" => opts.allow_undefined = true,
            "--warn-uninitialized" => opts.warn_uninitialized = true,
            "--warn-pedantic" => opts.warn_pedantic = true,
            "--canonicalize" => {
                let v = stanc_rs::syntax::canonicalize::Settings::parse(&value(&a)?)?;
                // Repeated options accumulate, as in stanc3.
                canon.deprecations |= v.deprecations;
                canon.parentheses |= v.parentheses;
                canon.braces |= v.braces;
                canon.inline_includes |= v.inline_includes;
                canon.strip_comments |= v.strip_comments;
            }
            "--print-canonical" => print_canonical = true,
            "--json-diagnostics" => json = true,
            "--debug-mir-pretty" => debug_mir = true,
            "--max-line-length" => {
                let v = value(&a)?;
                fopts.line_length = v
                    .parse()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| format!("invalid line length '{v}'"))?;
            }
            s if s.starts_with('-') => return Err(format!("unknown option '{s}'\n\n{USAGE}")),
            _ if cmd.is_none() => cmd = Some(a),
            _ if file.is_none() => file = Some(a),
            _ => return Err(format!("unexpected argument '{a}'\n\n{USAGE}")),
        }
    }
    // As in stanc3, --print-canonical overrides --canonicalize.
    if print_canonical {
        canon = stanc_rs::syntax::canonicalize::Settings::LEGACY;
    }
    let (Some(cmd), Some(file)) = (cmd, file) else {
        return Err(USAGE.to_string());
    };
    let text = std::fs::read_to_string(&file).map_err(|e| format!("cannot read '{file}': {e}"))?;
    let print_warnings = |ws: &[String]| ws.iter().for_each(|w| eprint!("{w}"));
    match cmd.as_str() {
        "check" if json => {
            let (ok, out) = stanc_rs::check_json(&file, &text, &opts);
            print!("{out}");
            if !ok {
                return Err(String::new());
            }
        }
        "check" => {
            let c = stanc_rs::check_source(&file, &text, &opts)?;
            print_warnings(&c.warnings);
            if debug_mir {
                println!("{}", stanc_rs::mir::pretty::program(&c.mir()));
            }
        }
        "compile" => {
            let c = stanc_rs::compile_source(&file, &text, &opts)?;
            print_warnings(&c.warnings);
            match output {
                Some(p) => std::fs::write(&p, c.rust)
                    .map_err(|e| format!("cannot write '{}': {e}", p.display()))?,
                None => print!("{}", c.rust),
            }
        }
        "ast" => {
            let (p, _) = stanc_rs::parse(&file, &text, &opts)?;
            println!("{p:#?}");
        }
        "ir" => {
            let c = stanc_rs::check_source(&file, &text, &opts)?;
            print_warnings(&c.warnings);
            println!("{:#?}", c.typed);
        }
        "info" => {
            let c = stanc_rs::check_source(&file, &text, &opts)?;
            print_warnings(&c.warnings);
            print!("{}", c.info());
        }
        "format" => {
            let (out, warnings) = stanc_rs::format_source(&file, &text, &opts, &fopts, &canon)?;
            print_warnings(&warnings);
            println!("{out}");
        }
        other => return Err(format!("unknown command '{other}'\n\n{USAGE}")),
    }
    Ok(())
}
