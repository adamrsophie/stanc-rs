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
    --help, -h                Print this message
    --version, -V             Print the version
";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let mut cmd = None;
    let mut file = None;
    let mut output = None;
    let mut opts = stanc_rs::Options::default();
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
            s if s.starts_with('-') => return Err(format!("unknown option '{s}'\n\n{USAGE}")),
            _ if cmd.is_none() => cmd = Some(a),
            _ if file.is_none() => file = Some(a),
            _ => return Err(format!("unexpected argument '{a}'\n\n{USAGE}")),
        }
    }
    let (Some(cmd), Some(file)) = (cmd, file) else {
        return Err(USAGE.to_string());
    };
    let text = std::fs::read_to_string(&file).map_err(|e| format!("cannot read '{file}': {e}"))?;
    if let Some(dir) = std::path::Path::new(&file).parent() {
        opts.include_paths.push(if dir.as_os_str().is_empty() {
            PathBuf::from(".")
        } else {
            dir.to_path_buf()
        });
    }
    let print_warnings = |ws: &[String]| ws.iter().for_each(|w| eprintln!("{w}"));
    match cmd.as_str() {
        "check" => {
            let c = stanc_rs::check_source(&file, &text, &opts)?;
            print_warnings(&c.warnings);
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
            let (p, _) = stanc_rs::parse(&file, &text, &opts)?;
            print!("{}", stanc_rs::syntax::pretty::program(&p));
        }
        other => return Err(format!("unknown command '{other}'\n\n{USAGE}")),
    }
    Ok(())
}
