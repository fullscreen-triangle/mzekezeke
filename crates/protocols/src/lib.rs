//! # mzekezeke-protocols
//!
//! Network protocols for Multi-Dimensional Temporal Ephemeral Cryptography (MDTEC).
//!
//! This crate provides the network communication protocols used by the MDTEC system,
//! including gRPC, WebSocket, and HTTP protocols.

#![deny(missing_docs)]
#![deny(unsafe_code)]

pub mod grpc;
pub mod websocket;
pub mod http;
pub mod error;
pub mod types;

pub use error::{Error, Result};
pub use types::*;

/// Protocol version
pub const PROTOCOL_VERSION: &str = "1.0.0";

/// Default gRPC port
pub const DEFAULT_GRPC_PORT: u16 = 50051;

/// Default WebSocket port
pub const DEFAULT_WEBSOCKET_PORT: u16 = 8080;

/// Default HTTP port
pub const DEFAULT_HTTP_PORT: u16 = 8080;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_version() {
        assert_eq!(PROTOCOL_VERSION, "1.0.0");
    }

    #[test]
    fn test_default_ports() {
        assert_eq!(DEFAULT_GRPC_PORT, 50051);
        assert_eq!(DEFAULT_WEBSOCKET_PORT, 8080);
        assert_eq!(DEFAULT_HTTP_PORT, 8080);
    }
} 