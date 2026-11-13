//! Semantic analysis for stanc-rs: Stan Math signatures, the type checker, and the typed program
//! that code generation consumes.

#![allow(missing_docs)]

pub mod check;
pub mod signatures;
pub mod typed;
pub mod types;

pub use check::{CheckOptions, Checked, check_program};
