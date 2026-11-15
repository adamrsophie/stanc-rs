//! stanc-rs: a Stan to Rust compiler.
//!
//! This crate ties the pipeline together: [`stanc_rs_syntax`] parses a Stan program,
//! [`stanc_rs_semantic`] type checks it, and [`stanc_rs_codegen`] emits a Rust module that
//! implements `stanc_rs_runtime::GeneratedModel` on top of stanmath-rs.
//!
//! stanc-rs is an independent, unofficial implementation of the Stan language (stanc3 v2.40.0).
//!
//! ```
//! let rust = stanc_rs::compile_source(
//!     "normal.stan",
//!     "data { real y; } parameters { real mu; } model { y ~ normal(mu, 5); }",
//!     &stanc_rs::Options::default(),
//! )
//! .unwrap()
//! .rust;
//! assert!(rust.contains("impl rt::GeneratedModel for Model"));
//! ```

use std::path::{Path, PathBuf};

mod info;

pub use stanc_rs_codegen as codegen;
pub use stanc_rs_semantic as semantic;
pub use stanc_rs_syntax as syntax;

use stanc_rs_semantic::typed::TProgram;
use stanc_rs_syntax::{IncludeResolver, SourceMap, ast::Program};

/// Compiler options.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Model name; defaults to the file stem followed by `_model`, as in stanc3.
    pub model_name: Option<String>,
    /// Directories searched for `#include` files.
    pub include_paths: Vec<PathBuf>,
    /// Accept functions that are declared without a definition.
    pub allow_undefined: bool,
}

/// A successfully checked program.
pub struct Checked {
    /// The parsed program.
    pub ast: Program,
    /// The typed program.
    pub typed: TProgram,
    /// Rendered warnings.
    pub warnings: Vec<String>,
    /// The files pulled in by `#include`, in the order they were read.
    pub included_files: Vec<String>,
}

impl Checked {
    /// Model information as JSON, with the content of stanc3's `--info`.
    pub fn info(&self) -> String {
        info::info_json(&self.typed, &self.included_files)
    }
}

/// A successfully compiled program.
pub struct Compiled {
    /// The generated Rust module.
    pub rust: String,
    /// Rendered warnings.
    pub warnings: Vec<String>,
}

/// Finds `#include` files in the include paths, then in the directory of the main file. As in
/// stanc3, messages list only the include paths.
struct FileIncludes<'a> {
    paths: &'a [PathBuf],
    main_dir: PathBuf,
}

impl<'a> FileIncludes<'a> {
    fn new(name: &str, opts: &'a Options) -> Self {
        let dir = Path::new(name).parent().unwrap_or(Path::new(""));
        FileIncludes {
            paths: &opts.include_paths,
            main_dir: if dir.as_os_str().is_empty() {
                PathBuf::from(".")
            } else {
                dir.to_path_buf()
            },
        }
    }
}

impl IncludeResolver for FileIncludes<'_> {
    fn resolve(&self, name: &str) -> Option<(String, String)> {
        self.paths
            .iter()
            .chain(std::iter::once(&self.main_dir))
            .find_map(|dir| {
                // OCaml's `Filename.concat`, which stanc3 uses: a name starting with a slash
                // is still looked up inside the directory.
                let dir = dir.display().to_string();
                let p = if dir.ends_with('/') {
                    format!("{dir}{name}")
                } else {
                    format!("{dir}/{name}")
                };
                std::fs::read_to_string(&p).ok().map(|text| (p, text))
            })
    }

    fn search_paths(&self) -> Vec<String> {
        self.paths.iter().map(|p| p.display().to_string()).collect()
    }
}

/// The default model name for a file name: `<stem>_model`.
pub fn default_model_name(file_name: &str) -> String {
    let stem = Path::new(file_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let mut name: String = stem
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if name.starts_with(|c: char| c.is_ascii_digit()) {
        name.insert(0, '_');
    }
    format!("{name}_model")
}

/// Parses a program; errors are rendered with source context.
pub fn parse(name: &str, text: &str, opts: &Options) -> Result<(Program, SourceMap), String> {
    let functions_only = name.ends_with(".stanfunctions");
    stanc_rs_syntax::parse_with(name, text, &FileIncludes::new(name, opts), functions_only)
        .map_err(|(d, sources)| d.render(&sources))
}

/// Warnings stanc3 issues while parsing: only the one for a program without blocks.
fn parser_warnings(ast: &Program, sources: &SourceMap) -> Vec<stanc_rs_syntax::Diagnostic> {
    if ast.blocks().iter().all(|(_, b)| b.is_none()) {
        let end = sources.files[0].text.len();
        vec![stanc_rs_syntax::Diagnostic::new(
            stanc_rs_syntax::DiagnosticKind::Warning,
            stanc_rs_syntax::Span::new(0, 0, end),
            "Empty model detected; this is a valid Stan model but likely unintended!",
        )]
    } else {
        Vec::new()
    }
}

/// Parses and type checks a program.
pub fn check_source(name: &str, text: &str, opts: &Options) -> Result<Checked, String> {
    let (ast, sources) = parse(name, text, opts)?;
    let model_name = opts
        .model_name
        .clone()
        .unwrap_or_else(|| default_model_name(name));
    let copts = stanc_rs_semantic::CheckOptions {
        allow_undefined: opts.allow_undefined,
        model_name,
    };
    let checked = stanc_rs_semantic::check_program(&ast, &copts).map_err(|d| d.render(&sources))?;
    let warnings = parser_warnings(&ast, &sources)
        .iter()
        .chain(&checked.warnings)
        .map(|w| w.render(&sources))
        .collect();
    let included_files = sources.files[1..].iter().map(|f| f.name.clone()).collect();
    Ok(Checked {
        ast,
        typed: checked.program,
        warnings,
        included_files,
    })
}

/// Parses and type checks a program, canonicalizes it with `canon`, and formats it as
/// `stanc --auto-format` does. Returns the formatted program and the rendered warnings.
pub fn format_source(
    name: &str,
    text: &str,
    opts: &Options,
    format: &stanc_rs_syntax::autoformat::Options,
    canon: &stanc_rs_syntax::canonicalize::Settings,
) -> Result<(String, Vec<String>), String> {
    let (ast, sources) = parse(name, text, opts)?;
    let model_name = opts
        .model_name
        .clone()
        .unwrap_or_else(|| default_model_name(name));
    let copts = stanc_rs_semantic::CheckOptions {
        allow_undefined: opts.allow_undefined,
        model_name,
    };
    let checked = stanc_rs_semantic::check_program(&ast, &copts).map_err(|d| d.render(&sources))?;
    let keep = checked.warnings.len()
        - if canon.deprecations {
            checked.deprecations
        } else {
            0
        };
    let warnings = parser_warnings(&ast, &sources)
        .iter()
        .chain(&checked.warnings[..keep])
        .map(|w| w.render(&sources))
        .collect();
    let ast = stanc_rs_syntax::canonicalize::canonicalize(&ast, canon);
    let format = stanc_rs_syntax::autoformat::Options {
        bare_functions: name.ends_with(".stanfunctions"),
        inline_includes: format.inline_includes || canon.inline_includes,
        strip_comments: format.strip_comments || canon.strip_comments,
        ..format.clone()
    };
    let out = stanc_rs_syntax::autoformat::format_program(&ast, &sources, &format)?;
    // As stanc3 does, check that the output parses back to the same program. The canonical
    // printer ignores locations, so equal renderings mean equal syntax trees.
    let same = stanc_rs_syntax::parse_with(
        name,
        &out,
        &FileIncludes::new(name, opts),
        format.bare_functions,
    )
    .is_ok_and(|(again, _)| {
        stanc_rs_syntax::pretty::program(&again) == stanc_rs_syntax::pretty::program(&ast)
    });
    if !same {
        return Err(format!(
            "Cannot format '{name}': an #include or comment sits where the formatter cannot \
             keep it, and the formatted program would differ from the original."
        ));
    }
    Ok((out, warnings))
}

/// Parses and type checks a program, and reports every diagnostic, warnings first, as a JSON
/// object `{"ok": bool, "diagnostics": [...]}`; see [`syntax::Diagnostic::to_json`] for the
/// fields of each diagnostic. `ok` is false when the program has an error.
pub fn check_json(name: &str, text: &str, opts: &Options) -> (bool, String) {
    let functions_only = name.ends_with(".stanfunctions");
    let (diags, sources) = match stanc_rs_syntax::parse_with(
        name,
        text,
        &FileIncludes::new(name, opts),
        functions_only,
    ) {
        Err((d, sources)) => (vec![d], sources),
        Ok((ast, sources)) => {
            let copts = stanc_rs_semantic::CheckOptions {
                allow_undefined: opts.allow_undefined,
                model_name: opts
                    .model_name
                    .clone()
                    .unwrap_or_else(|| default_model_name(name)),
            };
            let mut diags = parser_warnings(&ast, &sources);
            match stanc_rs_semantic::check_program(&ast, &copts) {
                Ok(c) => diags.extend(c.warnings),
                Err(d) => diags.push(d),
            }
            (diags, sources)
        }
    };
    let ok = !diags.iter().any(|d| d.is_error());
    let items: Vec<String> = diags.iter().map(|d| d.to_json(&sources)).collect();
    (
        ok,
        format!(
            "{{\"ok\": {ok}, \"diagnostics\": [{}]}}\n",
            items.join(", ")
        ),
    )
}

/// Parses, type checks, and translates a program to Rust.
pub fn compile_source(name: &str, text: &str, opts: &Options) -> Result<Compiled, String> {
    let (ast, sources) = parse(name, text, opts)?;
    let model_name = opts
        .model_name
        .clone()
        .unwrap_or_else(|| default_model_name(name));
    let copts = stanc_rs_semantic::CheckOptions {
        allow_undefined: opts.allow_undefined,
        model_name,
    };
    let checked = stanc_rs_semantic::check_program(&ast, &copts).map_err(|d| d.render(&sources))?;
    let warnings = parser_warnings(&ast, &sources)
        .iter()
        .chain(&checked.warnings)
        .map(|w| w.render(&sources))
        .collect();
    let rust = stanc_rs_codegen::generate(&checked.program).map_err(|d| d.render(&sources))?;
    Ok(Compiled { rust, warnings })
}
