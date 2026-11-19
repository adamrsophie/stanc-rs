//! The middle intermediate representation of stanc-rs, a port of stanc3 v2.40.0's MIR: lowering
//! from the typed program, the MIR printer, dataflow analysis, optimization passes, and the
//! pedantic mode warnings.

pub mod dataflow;
pub mod inline;
pub mod lower;
pub mod mir;
pub mod optimize;
pub mod pedantic;
pub mod pretty;
pub mod soa;
pub mod transform;

pub use lower::lower_program;
