//! # mzekezeke-mobile
//!
//! Mobile platform bindings for Multi-Dimensional Temporal Ephemeral Cryptography (MDTEC).
//!
//! This crate provides bindings for Android and iOS platforms, allowing the MDTEC system
//! to run on mobile devices.

#![deny(missing_docs)]
#![deny(unsafe_code)]

#[cfg(target_os = "android")]
pub mod android;

#[cfg(target_os = "ios")]
pub mod ios;

pub mod client;
pub mod sensors;
pub mod error;
pub mod types;

pub use client::*;
pub use error::{Error, Result};
pub use types::*;

/// Mobile platform types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// Android platform
    Android,
    /// iOS platform
    Ios,
}

/// Get the current platform
pub fn current_platform() -> Platform {
    #[cfg(target_os = "android")]
    return Platform::Android;
    
    #[cfg(target_os = "ios")]
    return Platform::Ios;
    
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    compile_error!("Mobile crate only supports Android and iOS platforms");
}

/// Initialize the mobile platform
pub fn init_platform() -> Result<()> {
    match current_platform() {
        #[cfg(target_os = "android")]
        Platform::Android => android::init(),
        
        #[cfg(target_os = "ios")]
        Platform::Ios => ios::init(),
        
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let platform = current_platform();
        assert!(matches!(platform, Platform::Android | Platform::Ios));
    }
} 