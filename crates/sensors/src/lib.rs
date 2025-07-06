//! # mzekezeke-sensors
//!
//! Environmental sensor implementations for Multi-Dimensional Temporal Ephemeral Cryptography (MDTEC).
//!
//! This crate provides implementations for various environmental sensors used in the
//! MDTEC system, including GPS, cellular, WiFi, hardware oscillations, and more.

#![deny(missing_docs)]
#![deny(unsafe_code)]

pub mod gps;
pub mod cellular;
pub mod wifi;
pub mod hardware;
pub mod atmospheric;
pub mod electromagnetic;
pub mod thermal;
pub mod acoustic;
pub mod network;
pub mod power;
pub mod quantum;
pub mod error;
pub mod types;

pub use error::{Error, Result};
pub use types::*;

/// Number of sensor types available
pub const SENSOR_COUNT: usize = 12;

/// Sensor type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorType {
    /// GPS positioning sensor
    Gps,
    /// Cellular network sensor
    Cellular,
    /// WiFi network sensor
    Wifi,
    /// Hardware oscillation sensor
    Hardware,
    /// Atmospheric pressure sensor
    Atmospheric,
    /// Electromagnetic field sensor
    Electromagnetic,
    /// Thermal gradient sensor
    Thermal,
    /// Acoustic environment sensor
    Acoustic,
    /// Network latency sensor
    Network,
    /// Power oscillation sensor
    Power,
    /// System load sensor
    System,
    /// Quantum noise sensor
    Quantum,
}

impl SensorType {
    /// Get all sensor types
    pub fn all() -> [SensorType; SENSOR_COUNT] {
        [
            SensorType::Gps,
            SensorType::Cellular,
            SensorType::Wifi,
            SensorType::Hardware,
            SensorType::Atmospheric,
            SensorType::Electromagnetic,
            SensorType::Thermal,
            SensorType::Acoustic,
            SensorType::Network,
            SensorType::Power,
            SensorType::System,
            SensorType::Quantum,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_count() {
        assert_eq!(SENSOR_COUNT, 12);
        assert_eq!(SensorType::all().len(), SENSOR_COUNT);
    }
}
