//! # mzekezeke-wasm
//!
//! WebAssembly bindings for Multi-Dimensional Temporal Ephemeral Cryptography (MDTEC).
//!
//! This crate provides WebAssembly bindings for the MDTEC system, allowing it to run
//! in web browsers and other WebAssembly environments.

#![deny(missing_docs)]

use wasm_bindgen::prelude::*;

mod client;
mod sensors;
mod utils;

pub use client::*;
pub use sensors::*;
pub use utils::*;

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    tracing_wasm::set_as_global_default();
}

/// Get the version of the WASM module
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Initialize logging for the WASM module
#[wasm_bindgen]
pub fn init_logging() {
    tracing_wasm::set_as_global_default();
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_version() {
        let version = version();
        assert!(!version.is_empty());
    }
} 