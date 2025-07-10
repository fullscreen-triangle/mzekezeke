//! # mzekezeke-core
//!
//! Core cryptographic algorithms for Multi-Dimensional Temporal Ephemeral Cryptography (MDTEC).
//! 
//! This crate provides the fundamental cryptographic primitives for environmental cryptography,
//! including challenge generation, response validation, and secure channel establishment.

#![deny(missing_docs)]
#![deny(unsafe_code)]

pub mod crypto;
pub mod dimensions;
pub mod oscillatory;
pub mod security;
pub mod utils;
pub mod error;
pub mod types;

pub use error::{Error, Result};
pub use types::*;

/// Version of the MDTEC protocol
pub const PROTOCOL_VERSION: u32 = 1;

/// Maximum number of environmental dimensions supported
pub const MAX_DIMENSIONS: usize = 12;

/// Default challenge timeout in seconds
pub const DEFAULT_CHALLENGE_TIMEOUT: u64 = 30;

/// Default key size in bytes
pub const DEFAULT_KEY_SIZE: usize = 32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_version() {
        assert_eq!(PROTOCOL_VERSION, 1);
    }

    #[test]
    fn test_max_dimensions() {
        assert_eq!(MAX_DIMENSIONS, 12);
    }
}
