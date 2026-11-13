//! Stan models from `models/`, compiled to Rust by stanc-rs when this crate is built.
//!
//! Each `models/<name>.stan` becomes module `<name>` with a `Model` type implementing
//! `stanc_rs_runtime::GeneratedModel`.

#![allow(missing_docs)]

include!(concat!(env!("OUT_DIR"), "/models.rs"));
