//! Compiles every `models/*.stan` file to `$OUT_DIR/<stem>.rs` and writes `$OUT_DIR/models.rs`,
//! which declares one module per model.

use std::fmt::Write as _;
use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("models");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo::rerun-if-changed={}", dir.display());
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "stan"))
        .collect();
    files.sort();
    let mut index = String::new();
    let mut stems = Vec::new();
    for f in files {
        println!("cargo::rerun-if-changed={}", f.display());
        let stem = f.file_stem().unwrap().to_str().unwrap().to_string();
        let text = std::fs::read_to_string(&f).unwrap();
        let opts = stanc_rs::Options {
            include_paths: vec![dir.clone()],
            ..Default::default()
        };
        let compiled = stanc_rs::compile_source(f.to_str().unwrap(), &text, &opts)
            .unwrap_or_else(|e| panic!("stanc-rs failed on {}:\n{e}", f.display()));
        std::fs::write(out.join(format!("{stem}.rs")), compiled.rust).unwrap();
        writeln!(
            index,
            "pub mod {stem} {{ include!(concat!(env!(\"OUT_DIR\"), \"/{stem}.rs\")); }}"
        )
        .unwrap();
        stems.push(stem);
    }
    index.push_str(
        "/// Builds the fixture model called `name` from `data`, or `None` if there is no such model.\n\
         pub fn model(name: &str, data: &stanc_rs_runtime::data::Data, seed: u64) \
         -> Option<stanc_rs_runtime::Result<Box<dyn stanc_rs_runtime::Model>>> {\n    match name {\n",
    );
    for stem in &stems {
        writeln!(
            index,
            "        {stem:?} => Some({stem}::Model::new(data, seed).map(|m| Box::new(m) as Box<dyn stanc_rs_runtime::Model>)),"
        )
        .unwrap();
    }
    index.push_str("        _ => None,\n    }\n}\n");
    std::fs::write(out.join("models.rs"), index).unwrap();
}
