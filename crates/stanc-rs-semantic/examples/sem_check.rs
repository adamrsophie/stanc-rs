//! Parses and type checks each file named on the command line, printing one line per file:
//! `OK path` or `ERR kind line:col path | message` (column zero-based, like stanc3).
use stanc_rs_semantic::{CheckOptions, check_program};
use stanc_rs_syntax::{IncludeResolver, parse_with};
use std::path::{Path, PathBuf};

struct Dirs(Vec<PathBuf>);
impl IncludeResolver for Dirs {
    fn resolve(&self, name: &str) -> Option<(String, String)> {
        self.0.iter().find_map(|d| {
            let p = d.join(name);
            std::fs::read_to_string(&p)
                .ok()
                .map(|t| (p.display().to_string(), t))
        })
    }
}

fn main() {
    for arg in std::env::args().skip(1) {
        let path = Path::new(&arg);
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let fo = arg.ends_with(".stanfunctions");
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        let opts = CheckOptions {
            allow_undefined: fo,
            model_name: format!("{stem}_model"),
            ..Default::default()
        };
        let res = parse_with(&arg, &text, &Dirs(vec![dir]), fo).and_then(|(p, sm)| {
            match check_program(&p, &opts) {
                Ok(_) => Ok(()),
                Err(d) => Err((d, sm)),
            }
        });
        match res {
            Ok(()) => println!("OK {arg}"),
            Err((d, sm)) => {
                let (l, c) = d.span.and_then(|s| sm.line_col(s)).unwrap_or((0, 0));
                println!(
                    "ERR {:?} {}:{} {} | {}",
                    d.kind,
                    l,
                    c.saturating_sub(1),
                    arg,
                    d.message.lines().next().unwrap_or("")
                );
            }
        }
    }
}
