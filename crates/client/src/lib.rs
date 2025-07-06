//! # mzekezeke-client
//!
//! Client library for Multi-Dimensional Temporal Ephemeral Cryptography (MDTEC).
//!
//! This crate provides the client-side implementation for connecting to MDTEC servers
//! and performing environmental cryptographic operations.

#![deny(missing_docs)]
#![deny(unsafe_code)]

pub mod client;
pub mod error;
pub mod types;

pub use client::Client;
pub use error::{Error, Result};
pub use types::*;

/// Default server URL
pub const DEFAULT_SERVER_URL: &str = "http://localhost:8080";

/// Default timeout for operations in seconds
pub const DEFAULT_TIMEOUT: u64 = 30;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_server_url() {
        assert_eq!(DEFAULT_SERVER_URL, "http://localhost:8080");
    }

    #[test]
    fn test_default_timeout() {
        assert_eq!(DEFAULT_TIMEOUT, 30);
    }
} 