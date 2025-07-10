//! Core types for Multi-Dimensional Temporal Ephemeral Cryptography (MDTEC)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Unique identifier for cryptographic challenges
pub type ChallengeId = u64;

/// Unique identifier for client sessions
pub type SessionId = u64;

/// Cryptographic key material
pub type KeyMaterial = Vec<u8>;

/// Environmental measurement value
pub type MeasurementValue = f64;

/// Timestamp in milliseconds since UNIX epoch
pub type Timestamp = u64;

/// Environmental challenge containing multi-dimensional requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Challenge {
    /// Unique challenge identifier
    pub id: ChallengeId,
    /// Timestamp when challenge was created
    pub created_at: Timestamp,
    /// Expiration timestamp
    pub expires_at: Timestamp,
    /// Required environmental dimensions
    pub dimensions: Vec<DimensionRequirement>,
    /// Server-side state for validation
    pub server_state: Vec<u8>,
}

/// Requirement for a specific environmental dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionRequirement {
    /// Type of dimension
    pub dimension_type: DimensionType,
    /// Required measurement parameters
    pub parameters: DimensionParameters,
    /// Tolerance for measurement accuracy
    pub tolerance: f64,
}

/// Type of environmental dimension
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DimensionType {
    /// GPS spatial coordinates
    Spatial,
    /// Temporal synchronization
    Temporal,
    /// Atmospheric pressure and conditions
    Atmospheric,
    /// Electromagnetic field measurements
    Electromagnetic,
    /// Acoustic environment fingerprint
    Acoustic,
    /// Thermal gradient analysis
    Thermal,
    /// Network latency characteristics
    Network,
    /// Hardware oscillation patterns
    Hardware,
    /// Quantum noise measurements
    Quantum,
    /// Cellular network characteristics
    Cellular,
    /// WiFi network environment
    Wifi,
    /// System load and performance
    System,
}

/// Parameters for environmental dimension measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionParameters {
    /// Parameter values as key-value pairs
    pub values: HashMap<String, MeasurementValue>,
    /// Metadata for measurement context
    pub metadata: HashMap<String, String>,
}

/// Response to an environmental challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeResponse {
    /// Challenge ID this response is for
    pub challenge_id: ChallengeId,
    /// Timestamp when response was created
    pub created_at: Timestamp,
    /// Environmental measurements for each dimension
    pub measurements: Vec<DimensionMeasurement>,
    /// Derived cryptographic material
    pub key_material: KeyMaterial,
}

/// Measurement for a specific environmental dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionMeasurement {
    /// Type of dimension measured
    pub dimension_type: DimensionType,
    /// Measured values
    pub values: HashMap<String, MeasurementValue>,
    /// Measurement confidence/quality
    pub confidence: f64,
    /// Measurement timestamp
    pub timestamp: Timestamp,
}

/// Validation result for environmental measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// Validation score (0.0 to 1.0)
    pub score: f64,
    /// Detailed validation per dimension
    pub dimension_results: Vec<DimensionValidation>,
    /// Overall validation message
    pub message: String,
}

/// Validation result for a specific dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionValidation {
    /// Dimension type
    pub dimension_type: DimensionType,
    /// Whether this dimension passed validation
    pub valid: bool,
    /// Validation score for this dimension
    pub score: f64,
    /// Validation details
    pub details: String,
}

/// Cryptographic session between client and server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoSession {
    /// Unique session identifier
    pub id: SessionId,
    /// Session creation timestamp
    pub created_at: Timestamp,
    /// Session expiration timestamp
    pub expires_at: Timestamp,
    /// Shared secret key material
    pub key_material: KeyMaterial,
    /// Session state
    pub state: SessionState,
}

/// State of a cryptographic session
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionState {
    /// Session is being established
    Establishing,
    /// Session is active and ready for use
    Active,
    /// Session is expired
    Expired,
    /// Session was terminated
    Terminated,
}

/// Configuration for environmental dimension sensing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionConfig {
    /// Whether this dimension is enabled
    pub enabled: bool,
    /// Sampling rate in Hz
    pub sampling_rate: f64,
    /// Measurement precision
    pub precision: f64,
    /// Timeout for measurements in seconds
    pub timeout: Duration,
    /// Dimension-specific configuration
    pub parameters: HashMap<String, String>,
}

/// Oscillatory field analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryField {
    /// Field frequency in Hz
    pub frequency: f64,
    /// Field amplitude
    pub amplitude: f64,
    /// Phase offset in radians
    pub phase: f64,
    /// Entropy measure
    pub entropy: f64,
    /// Thermodynamic energy
    pub energy: f64,
}

/// Thermodynamic properties of environmental state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicState {
    /// Total system entropy
    pub entropy: f64,
    /// Energy required for decryption
    pub energy_requirement: f64,
    /// Temperature of information
    pub temperature: f64,
    /// Pressure of dimensional constraints
    pub pressure: f64,
}

impl Challenge {
    /// Create a new challenge with given parameters
    pub fn new(
        id: ChallengeId,
        dimensions: Vec<DimensionRequirement>,
        server_state: Vec<u8>,
        duration: Duration,
    ) -> Self {
        let now = current_timestamp();
        Self {
            id,
            created_at: now,
            expires_at: now + duration.as_millis() as u64,
            dimensions,
            server_state,
        }
    }

    /// Check if the challenge has expired
    pub fn is_expired(&self) -> bool {
        current_timestamp() > self.expires_at
    }

    /// Get the remaining time until expiration
    pub fn time_remaining(&self) -> Option<Duration> {
        let now = current_timestamp();
        if now < self.expires_at {
            Some(Duration::from_millis(self.expires_at - now))
        } else {
            None
        }
    }
}

impl DimensionParameters {
    /// Create new dimension parameters
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a measurement value
    pub fn add_value(&mut self, key: String, value: MeasurementValue) {
        self.values.insert(key, value);
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

impl Default for DimensionParameters {
    fn default() -> Self {
        Self::new()
    }
}

impl DimensionType {
    /// Get all available dimension types
    pub fn all() -> Vec<Self> {
        vec![
            Self::Spatial,
            Self::Temporal,
            Self::Atmospheric,
            Self::Electromagnetic,
            Self::Acoustic,
            Self::Thermal,
            Self::Network,
            Self::Hardware,
            Self::Quantum,
            Self::Cellular,
            Self::Wifi,
            Self::System,
        ]
    }

    /// Get human-readable name for the dimension
    pub fn name(&self) -> &'static str {
        match self {
            Self::Spatial => "Spatial",
            Self::Temporal => "Temporal",
            Self::Atmospheric => "Atmospheric",
            Self::Electromagnetic => "Electromagnetic",
            Self::Acoustic => "Acoustic",
            Self::Thermal => "Thermal",
            Self::Network => "Network",
            Self::Hardware => "Hardware",
            Self::Quantum => "Quantum",
            Self::Cellular => "Cellular",
            Self::Wifi => "WiFi",
            Self::System => "System",
        }
    }
}

/// Get current timestamp in milliseconds since UNIX epoch
pub fn current_timestamp() -> Timestamp {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_challenge_creation() {
        let challenge = Challenge::new(
            1,
            vec![],
            vec![1, 2, 3],
            Duration::from_secs(60),
        );
        assert_eq!(challenge.id, 1);
        assert!(!challenge.is_expired());
    }

    #[test]
    fn test_dimension_types() {
        let types = DimensionType::all();
        assert_eq!(types.len(), 12);
        assert!(types.contains(&DimensionType::Spatial));
        assert!(types.contains(&DimensionType::Quantum));
    }

    #[test]
    fn test_dimension_parameters() {
        let mut params = DimensionParameters::new();
        params.add_value("test".to_string(), 42.0);
        params.add_metadata("source".to_string(), "test".to_string());
        
        assert_eq!(params.values.get("test"), Some(&42.0));
        assert_eq!(params.metadata.get("source"), Some(&"test".to_string()));
    }
} 