//! Error types for Multi-Dimensional Temporal Ephemeral Cryptography (MDTEC)

use thiserror::Error;

/// Result type for MDTEC operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for MDTEC operations
#[derive(Error, Debug)]
pub enum Error {
    /// Cryptographic operation failed
    #[error("Cryptographic error: {0}")]
    Crypto(String),

    /// Environmental measurement failed
    #[error("Environmental measurement error: {0}")]
    Environmental(String),

    /// Challenge validation failed
    #[error("Challenge validation error: {0}")]
    Validation(String),

    /// Temporal synchronization failed
    #[error("Temporal synchronization error: {0}")]
    Temporal(String),

    /// Dimension analysis failed
    #[error("Dimension analysis error: {0}")]
    Dimension(String),

    /// Oscillatory field analysis failed
    #[error("Oscillatory field error: {0}")]
    Oscillatory(String),

    /// Thermodynamic calculation failed
    #[error("Thermodynamic error: {0}")]
    Thermodynamic(String),

    /// Security constraint violation
    #[error("Security error: {0}")]
    Security(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Network communication error
    #[error("Network error: {0}")]
    Network(String),

    /// Timeout occurred
    #[error("Timeout error: {0}")]
    Timeout(String),

    /// Invalid input provided
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Resource not found
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Operation not supported
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// Internal system error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Other error
    #[error("Other error: {0}")]
    Other(String),
}

impl Error {
    /// Create a new cryptographic error
    pub fn crypto(msg: impl Into<String>) -> Self {
        Self::Crypto(msg.into())
    }

    /// Create a new environmental error
    pub fn environmental(msg: impl Into<String>) -> Self {
        Self::Environmental(msg.into())
    }

    /// Create a new validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }

    /// Create a new temporal error
    pub fn temporal(msg: impl Into<String>) -> Self {
        Self::Temporal(msg.into())
    }

    /// Create a new dimension error
    pub fn dimension(msg: impl Into<String>) -> Self {
        Self::Dimension(msg.into())
    }

    /// Create a new oscillatory error
    pub fn oscillatory(msg: impl Into<String>) -> Self {
        Self::Oscillatory(msg.into())
    }

    /// Create a new thermodynamic error
    pub fn thermodynamic(msg: impl Into<String>) -> Self {
        Self::Thermodynamic(msg.into())
    }

    /// Create a new security error
    pub fn security(msg: impl Into<String>) -> Self {
        Self::Security(msg.into())
    }

    /// Create a new configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a new network error
    pub fn network(msg: impl Into<String>) -> Self {
        Self::Network(msg.into())
    }

    /// Create a new timeout error
    pub fn timeout(msg: impl Into<String>) -> Self {
        Self::Timeout(msg.into())
    }

    /// Create a new invalid input error
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create a new not found error
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }

    /// Create a new not supported error
    pub fn not_supported(msg: impl Into<String>) -> Self {
        Self::NotSupported(msg.into())
    }

    /// Create a new internal error
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    /// Create a new serialization error
    pub fn serialization(msg: impl Into<String>) -> Self {
        Self::Serialization(msg.into())
    }

    /// Create a new other error
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }

    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Crypto(_) => false,
            Self::Environmental(_) => true,
            Self::Validation(_) => true,
            Self::Temporal(_) => true,
            Self::Dimension(_) => true,
            Self::Oscillatory(_) => true,
            Self::Thermodynamic(_) => true,
            Self::Security(_) => false,
            Self::Config(_) => false,
            Self::Network(_) => true,
            Self::Timeout(_) => true,
            Self::InvalidInput(_) => false,
            Self::NotFound(_) => false,
            Self::NotSupported(_) => false,
            Self::Internal(_) => false,
            Self::Serialization(_) => false,
            Self::Io(_) => true,
            Self::Json(_) => false,
            Self::Other(_) => false,
        }
    }

    /// Get the error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::Crypto(_) => ErrorCategory::Cryptographic,
            Self::Environmental(_) => ErrorCategory::Environmental,
            Self::Validation(_) => ErrorCategory::Validation,
            Self::Temporal(_) => ErrorCategory::Temporal,
            Self::Dimension(_) => ErrorCategory::Dimensional,
            Self::Oscillatory(_) => ErrorCategory::Oscillatory,
            Self::Thermodynamic(_) => ErrorCategory::Thermodynamic,
            Self::Security(_) => ErrorCategory::Security,
            Self::Config(_) => ErrorCategory::Configuration,
            Self::Network(_) => ErrorCategory::Network,
            Self::Timeout(_) => ErrorCategory::Timeout,
            Self::InvalidInput(_) => ErrorCategory::Input,
            Self::NotFound(_) => ErrorCategory::Resource,
            Self::NotSupported(_) => ErrorCategory::Support,
            Self::Internal(_) => ErrorCategory::Internal,
            Self::Serialization(_) => ErrorCategory::Serialization,
            Self::Io(_) => ErrorCategory::Io,
            Self::Json(_) => ErrorCategory::Serialization,
            Self::Other(_) => ErrorCategory::Other,
        }
    }
}

/// Error category for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Cryptographic errors
    Cryptographic,
    /// Environmental sensing errors
    Environmental,
    /// Validation errors
    Validation,
    /// Temporal synchronization errors
    Temporal,
    /// Dimensional analysis errors
    Dimensional,
    /// Oscillatory field errors
    Oscillatory,
    /// Thermodynamic calculation errors
    Thermodynamic,
    /// Security constraint errors
    Security,
    /// Configuration errors
    Configuration,
    /// Network communication errors
    Network,
    /// Timeout errors
    Timeout,
    /// Input validation errors
    Input,
    /// Resource access errors
    Resource,
    /// Feature support errors
    Support,
    /// Internal system errors
    Internal,
    /// Serialization errors
    Serialization,
    /// I/O errors
    Io,
    /// Other errors
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = Error::crypto("test error");
        assert_eq!(err.category(), ErrorCategory::Cryptographic);
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_error_recoverability() {
        assert!(!Error::crypto("test").is_recoverable());
        assert!(Error::environmental("test").is_recoverable());
        assert!(Error::network("test").is_recoverable());
        assert!(!Error::security("test").is_recoverable());
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(Error::crypto("test").category(), ErrorCategory::Cryptographic);
        assert_eq!(Error::environmental("test").category(), ErrorCategory::Environmental);
        assert_eq!(Error::validation("test").category(), ErrorCategory::Validation);
        assert_eq!(Error::temporal("test").category(), ErrorCategory::Temporal);
    }
} 