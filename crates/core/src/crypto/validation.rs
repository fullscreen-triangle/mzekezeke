//! Cryptographic validation for MDTEC
//!
//! This module implements validation of environmental measurements, temporal proofs,
//! and cryptographic keys within the MDTEC system.

use crate::types::*;
use crate::error::{Error, Result};
use crate::crypto::environmental_key::EnvironmentalKey;
use crate::crypto::temporal_encryption::{TemporalCiphertext, TemporalProof};
use crate::crypto::dimensional_synthesis::{DimensionalSynthesisResult, TemporalCoordinationProof};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;

/// Cryptographic validator for MDTEC components
pub struct MdtecValidator {
    /// Configuration for validation
    config: ValidationConfig,
    /// Validation state tracking
    validation_state: ValidationState,
}

/// Configuration for MDTEC validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Minimum validation confidence required
    pub min_confidence: f64,
    /// Maximum allowed time skew for temporal validation
    pub max_time_skew: u64,
    /// Validation strictness level
    pub strictness: ValidationStrictness,
    /// Required validation criteria
    pub required_criteria: Vec<ValidationCriterion>,
    /// Validation timeouts
    pub timeouts: ValidationTimeouts,
}

/// Validation strictness levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationStrictness {
    /// Strict validation - all criteria must pass
    Strict,
    /// Moderate validation - most criteria must pass
    Moderate,
    /// Lenient validation - basic criteria must pass
    Lenient,
}

/// Validation criteria
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationCriterion {
    /// Temporal consistency validation
    TemporalConsistency,
    /// Dimensional coherence validation
    DimensionalCoherence,
    /// Cryptographic integrity validation
    CryptographicIntegrity,
    /// Environmental authenticity validation
    EnvironmentalAuthenticity,
    /// Coordinate transformation validation
    CoordinateTransformation,
    /// Masunda memorial validation
    MasundaMemorial,
}

/// Validation timeouts
#[derive(Debug, Clone)]
pub struct ValidationTimeouts {
    /// Timeout for temporal validation
    pub temporal: u64,
    /// Timeout for dimensional validation
    pub dimensional: u64,
    /// Timeout for cryptographic validation
    pub cryptographic: u64,
    /// Timeout for environmental validation
    pub environmental: u64,
}

/// Validation state tracking
#[derive(Debug, Clone)]
pub struct ValidationState {
    /// Cache of recent validation results
    pub validation_cache: HashMap<String, CachedValidationResult>,
    /// Validation statistics
    pub statistics: ValidationStatistics,
    /// Active validations
    pub active_validations: Vec<ActiveValidation>,
}

/// Cached validation result
#[derive(Debug, Clone)]
pub struct CachedValidationResult {
    /// Validation result
    pub result: ValidationResult,
    /// Cache timestamp
    pub cached_at: Timestamp,
    /// Cache expiry
    pub expires_at: Timestamp,
}

/// Validation statistics
#[derive(Debug, Clone)]
pub struct ValidationStatistics {
    /// Total validations performed
    pub total_validations: u64,
    /// Successful validations
    pub successful_validations: u64,
    /// Failed validations
    pub failed_validations: u64,
    /// Average validation time
    pub avg_validation_time: f64,
    /// Validation success rate
    pub success_rate: f64,
}

/// Active validation tracking
#[derive(Debug, Clone)]
pub struct ActiveValidation {
    /// Validation ID
    pub id: String,
    /// Validation type
    pub validation_type: ValidationType,
    /// Started timestamp
    pub started_at: Timestamp,
    /// Current progress
    pub progress: f64,
}

/// Type of validation being performed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationType {
    /// Environmental key validation
    EnvironmentalKey,
    /// Temporal encryption validation
    TemporalEncryption,
    /// Dimensional synthesis validation
    DimensionalSynthesis,
    /// Challenge response validation
    ChallengeResponse,
    /// Measurement validation
    Measurement,
}

/// Comprehensive validation result
#[derive(Debug, Clone)]
pub struct ComprehensiveValidationResult {
    /// Overall validation result
    pub overall_result: ValidationResult,
    /// Individual validation results
    pub individual_results: HashMap<ValidationCriterion, ValidationResult>,
    /// Validation metadata
    pub metadata: ValidationMetadata,
}

/// Validation metadata
#[derive(Debug, Clone)]
pub struct ValidationMetadata {
    /// Validation performed at
    pub validated_at: Timestamp,
    /// Validation duration
    pub duration: u64,
    /// Validator version
    pub validator_version: String,
    /// Validation method used
    pub method: String,
    /// Additional context
    pub context: HashMap<String, String>,
}

impl MdtecValidator {
    /// Create new MDTEC validator
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            validation_state: ValidationState::new(),
        }
    }

    /// Create validator with default configuration
    pub fn default() -> Self {
        Self::new(ValidationConfig::default())
    }

    /// Validate environmental key
    pub fn validate_environmental_key(&mut self, key: &EnvironmentalKey) -> Result<ComprehensiveValidationResult> {
        let start_time = current_timestamp();
        let validation_id = format!("env_key_{}", start_time);
        
        // Track active validation
        self.track_validation(validation_id.clone(), ValidationType::EnvironmentalKey)?;
        
        let mut individual_results = HashMap::new();
        let mut overall_valid = true;
        let mut overall_score = 0.0;
        let mut criteria_count = 0;

        // Validate temporal consistency
        if self.config.required_criteria.contains(&ValidationCriterion::TemporalConsistency) {
            let result = self.validate_temporal_consistency_key(key)?;
            overall_valid &= result.valid;
            overall_score += result.score;
            criteria_count += 1;
            individual_results.insert(ValidationCriterion::TemporalConsistency, result);
        }

        // Validate dimensional coherence
        if self.config.required_criteria.contains(&ValidationCriterion::DimensionalCoherence) {
            let result = self.validate_dimensional_coherence_key(key)?;
            overall_valid &= result.valid;
            overall_score += result.score;
            criteria_count += 1;
            individual_results.insert(ValidationCriterion::DimensionalCoherence, result);
        }

        // Validate cryptographic integrity
        if self.config.required_criteria.contains(&ValidationCriterion::CryptographicIntegrity) {
            let result = self.validate_cryptographic_integrity_key(key)?;
            overall_valid &= result.valid;
            overall_score += result.score;
            criteria_count += 1;
            individual_results.insert(ValidationCriterion::CryptographicIntegrity, result);
        }

        // Validate environmental authenticity
        if self.config.required_criteria.contains(&ValidationCriterion::EnvironmentalAuthenticity) {
            let result = self.validate_environmental_authenticity_key(key)?;
            overall_valid &= result.valid;
            overall_score += result.score;
            criteria_count += 1;
            individual_results.insert(ValidationCriterion::EnvironmentalAuthenticity, result);
        }

        // Calculate overall score
        let final_score = if criteria_count > 0 {
            overall_score / criteria_count as f64
        } else {
            0.0
        };

        // Apply strictness requirements
        let final_valid = self.apply_strictness_requirements(overall_valid, final_score)?;

        // Create overall result
        let overall_result = ValidationResult {
            valid: final_valid,
            score: final_score,
            dimension_results: vec![], // Not applicable for key validation
            message: format!("Environmental key validation completed with score {:.2}", final_score),
        };

        // Create metadata
        let end_time = current_timestamp();
        let metadata = ValidationMetadata {
            validated_at: start_time,
            duration: end_time - start_time,
            validator_version: "MDTEC_1.0".to_string(),
            method: "ComprehensiveValidation".to_string(),
            context: HashMap::new(),
        };

        // Update statistics
        self.update_statistics(final_valid, end_time - start_time)?;

        // Remove from active validations
        self.complete_validation(&validation_id)?;

        Ok(ComprehensiveValidationResult {
            overall_result,
            individual_results,
            metadata,
        })
    }

    /// Validate temporal encryption
    pub fn validate_temporal_encryption(&mut self, ciphertext: &TemporalCiphertext) -> Result<ValidationResult> {
        let start_time = current_timestamp();
        
        // Validate temporal proof
        let temporal_valid = self.validate_temporal_proof(&ciphertext.temporal_proof)?;
        
        // Validate encryption timestamp
        let time_valid = self.validate_encryption_timestamp(ciphertext.encrypted_at, start_time)?;
        
        // Validate window ID
        let window_valid = self.validate_window_id(ciphertext.window_id, ciphertext.encrypted_at)?;
        
        // Calculate overall validation
        let valid = temporal_valid && time_valid && window_valid;
        let score = if valid {
            0.9 // High confidence for temporal validation
        } else {
            0.1 // Low confidence for failed validation
        };

        let message = if valid {
            "Temporal encryption validation passed".to_string()
        } else {
            "Temporal encryption validation failed".to_string()
        };

        Ok(ValidationResult {
            valid,
            score,
            dimension_results: vec![],
            message,
        })
    }

    /// Validate dimensional synthesis result
    pub fn validate_dimensional_synthesis(&mut self, synthesis: &DimensionalSynthesisResult) -> Result<ValidationResult> {
        let start_time = current_timestamp();
        
        // Validate quality metrics
        let quality_valid = synthesis.quality.overall_quality >= self.config.min_confidence;
        
        // Validate temporal coordination proof
        let temporal_valid = self.validate_temporal_coordination_proof(&synthesis.temporal_proof)?;
        
        // Validate Masunda transformation
        let masunda_valid = self.validate_masunda_transformation(&synthesis.masunda_metadata)?;
        
        // Calculate overall validation
        let valid = quality_valid && temporal_valid && masunda_valid;
        let score = if valid {
            synthesis.quality.overall_quality
        } else {
            synthesis.quality.overall_quality * 0.5
        };

        let message = if valid {
            format!("Dimensional synthesis validation passed (Memorial: {})", 
                   synthesis.masunda_metadata.memorial_dedication)
        } else {
            "Dimensional synthesis validation failed".to_string()
        };

        Ok(ValidationResult {
            valid,
            score,
            dimension_results: vec![],
            message,
        })
    }

    /// Validate challenge response
    pub fn validate_challenge_response(&mut self, challenge: &Challenge, response: &ChallengeResponse) -> Result<ValidationResult> {
        let start_time = current_timestamp();
        
        // Validate challenge ID match
        if challenge.id != response.challenge_id {
            return Ok(ValidationResult {
                valid: false,
                score: 0.0,
                dimension_results: vec![],
                message: "Challenge ID mismatch".to_string(),
            });
        }

        // Validate response timing
        if response.created_at < challenge.created_at || response.created_at > challenge.expires_at {
            return Ok(ValidationResult {
                valid: false,
                score: 0.0,
                dimension_results: vec![],
                message: "Response timing invalid".to_string(),
            });
        }

        // Validate dimensions
        let mut dimension_results = Vec::new();
        let mut total_score = 0.0;
        let mut valid_count = 0;

        for dimension_req in &challenge.dimensions {
            if let Some(measurement) = response.measurements.iter()
                .find(|m| m.dimension_type == dimension_req.dimension_type) {
                
                let dimension_result = self.validate_dimension_measurement(dimension_req, measurement)?;
                total_score += dimension_result.score;
                if dimension_result.valid {
                    valid_count += 1;
                }
                dimension_results.push(dimension_result);
            } else {
                // Missing dimension
                dimension_results.push(DimensionValidation {
                    dimension_type: dimension_req.dimension_type,
                    valid: false,
                    score: 0.0,
                    details: "Missing dimension measurement".to_string(),
                });
            }
        }

        // Calculate overall validation
        let required_valid = match self.config.strictness {
            ValidationStrictness::Strict => valid_count == challenge.dimensions.len(),
            ValidationStrictness::Moderate => valid_count >= (challenge.dimensions.len() * 2) / 3,
            ValidationStrictness::Lenient => valid_count >= challenge.dimensions.len() / 2,
        };

        let average_score = if !challenge.dimensions.is_empty() {
            total_score / challenge.dimensions.len() as f64
        } else {
            0.0
        };

        let valid = required_valid && average_score >= self.config.min_confidence;

        Ok(ValidationResult {
            valid,
            score: average_score,
            dimension_results,
            message: format!("Challenge response validation: {}/{} dimensions valid", 
                           valid_count, challenge.dimensions.len()),
        })
    }

    /// Validate temporal consistency of environmental key
    fn validate_temporal_consistency_key(&self, key: &EnvironmentalKey) -> Result<ValidationResult> {
        let now = current_timestamp();
        let key_age = now - key.generated_at;
        
        // Check if key is too old
        let max_age = 24 * 60 * 60 * 1000; // 24 hours in milliseconds
        if key_age > max_age {
            return Ok(ValidationResult {
                valid: false,
                score: 0.0,
                dimension_results: vec![],
                message: "Environmental key too old".to_string(),
            });
        }

        // Check temporal stability from metadata
        let temporal_score = key.metadata.quality_metrics.temporal_stability;
        let valid = temporal_score >= self.config.min_confidence;

        Ok(ValidationResult {
            valid,
            score: temporal_score,
            dimension_results: vec![],
            message: format!("Temporal consistency score: {:.2}", temporal_score),
        })
    }

    /// Validate dimensional coherence of environmental key
    fn validate_dimensional_coherence_key(&self, key: &EnvironmentalKey) -> Result<ValidationResult> {
        let dimensional_score = key.metadata.quality_metrics.dimensional_diversity;
        let valid = dimensional_score >= self.config.min_confidence && key.dimensions.len() >= 4;

        Ok(ValidationResult {
            valid,
            score: dimensional_score,
            dimension_results: vec![],
            message: format!("Dimensional coherence: {} dimensions, score: {:.2}", 
                           key.dimensions.len(), dimensional_score),
        })
    }

    /// Validate cryptographic integrity of environmental key
    fn validate_cryptographic_integrity_key(&self, key: &EnvironmentalKey) -> Result<ValidationResult> {
        // Check key material length
        if key.key_material.len() < 32 {
            return Ok(ValidationResult {
                valid: false,
                score: 0.0,
                dimension_results: vec![],
                message: "Key material too short".to_string(),
            });
        }

        // Check entropy
        if key.entropy < 128.0 {
            return Ok(ValidationResult {
                valid: false,
                score: 0.2,
                dimension_results: vec![],
                message: "Insufficient entropy".to_string(),
            });
        }

        // Validate key material entropy
        let entropy_score = self.calculate_key_entropy(&key.key_material)?;
        let valid = entropy_score >= 0.7; // Minimum entropy threshold

        Ok(ValidationResult {
            valid,
            score: entropy_score,
            dimension_results: vec![],
            message: format!("Cryptographic integrity score: {:.2}", entropy_score),
        })
    }

    /// Validate environmental authenticity of key
    fn validate_environmental_authenticity_key(&self, key: &EnvironmentalKey) -> Result<ValidationResult> {
        // Check validation data integrity
        let validation_valid = self.verify_validation_data(key)?;
        
        // Check sample count
        let sample_count_valid = key.metadata.sample_count >= 10;
        
        // Check time span
        let time_span_valid = key.metadata.time_span >= 1000; // At least 1 second
        
        let valid = validation_valid && sample_count_valid && time_span_valid;
        let score = if valid { 0.8 } else { 0.3 };

        Ok(ValidationResult {
            valid,
            score,
            dimension_results: vec![],
            message: format!("Environmental authenticity: {} samples over {}ms", 
                           key.metadata.sample_count, key.metadata.time_span),
        })
    }

    /// Validate temporal proof
    fn validate_temporal_proof(&self, proof: &TemporalProof) -> Result<bool> {
        // Check proof structure
        if proof.window_proof.len() != 32 || 
           proof.sequence_proof.len() != 32 || 
           proof.sync_data.len() != 32 {
            return Ok(false);
        }

        // Additional temporal proof validation would go here
        // For now, we validate the structure
        Ok(true)
    }

    /// Validate encryption timestamp
    fn validate_encryption_timestamp(&self, encrypted_at: Timestamp, current_time: Timestamp) -> Result<bool> {
        let time_diff = if current_time > encrypted_at {
            current_time - encrypted_at
        } else {
            encrypted_at - current_time
        };

        // Check if within acceptable time skew
        Ok(time_diff <= self.config.max_time_skew * 1000)
    }

    /// Validate window ID
    fn validate_window_id(&self, window_id: u64, encrypted_at: Timestamp) -> Result<bool> {
        // Window ID should be based on timestamp
        let expected_window_id = encrypted_at / 1000;
        let window_diff = if window_id > expected_window_id {
            window_id - expected_window_id
        } else {
            expected_window_id - window_id
        };

        // Allow some tolerance for window boundaries
        Ok(window_diff <= 1)
    }

    /// Validate temporal coordination proof
    fn validate_temporal_coordination_proof(&self, proof: &TemporalCoordinationProof) -> Result<bool> {
        // Check proof structure
        if proof.synchronization_proof.len() != 32 || 
           proof.ordering_proof.len() != 32 || 
           proof.causality_proof.len() != 32 {
            return Ok(false);
        }

        // Additional coordination proof validation would go here
        Ok(true)
    }

    /// Validate Masunda transformation
    fn validate_masunda_transformation(&self, metadata: &crate::crypto::dimensional_synthesis::MasundaMetadata) -> Result<bool> {
        // Check transformation ID
        if metadata.transformation_id != "MasundaTransform_v1.0" {
            return Ok(false);
        }

        // Check memorial dedication
        if !metadata.memorial_dedication.contains("Stella-Lorraine Masunda") {
            return Ok(false);
        }

        // Check precision
        if metadata.precision_achieved < 0.5 {
            return Ok(false);
        }

        Ok(true)
    }

    /// Validate dimension measurement
    fn validate_dimension_measurement(&self, requirement: &DimensionRequirement, measurement: &DimensionMeasurement) -> Result<DimensionValidation> {
        let mut valid = true;
        let mut score = measurement.confidence;
        let mut details = Vec::new();

        // Check confidence threshold
        if measurement.confidence < 0.5 {
            valid = false;
            details.push("Low confidence");
        }

        // Check dimension-specific requirements
        match requirement.dimension_type {
            DimensionType::Spatial => {
                if !measurement.values.contains_key("latitude") || !measurement.values.contains_key("longitude") {
                    valid = false;
                    details.push("Missing spatial coordinates");
                }
            }
            DimensionType::Temporal => {
                if measurement.timestamp == 0 {
                    valid = false;
                    details.push("Invalid timestamp");
                }
            }
            _ => {
                // Generic validation for other dimensions
                if measurement.values.is_empty() {
                    valid = false;
                    details.push("No measurement values");
                }
            }
        }

        // Apply tolerance
        if score < requirement.tolerance {
            valid = false;
            details.push("Below tolerance threshold");
        }

        Ok(DimensionValidation {
            dimension_type: requirement.dimension_type,
            valid,
            score,
            details: details.join("; "),
        })
    }

    /// Calculate entropy of key material
    fn calculate_key_entropy(&self, key_material: &[u8]) -> Result<f64> {
        if key_material.is_empty() {
            return Ok(0.0);
        }

        let mut counts = [0u32; 256];
        for &byte in key_material {
            counts[byte as usize] += 1;
        }

        let len = key_material.len() as f64;
        let mut entropy = 0.0;

        for count in counts.iter() {
            if *count > 0 {
                let p = *count as f64 / len;
                entropy -= p * p.log2();
            }
        }

        // Normalize entropy to 0-1 range
        Ok(entropy / 8.0)
    }

    /// Verify validation data integrity
    fn verify_validation_data(&self, key: &EnvironmentalKey) -> Result<bool> {
        // Check if validation data exists
        if key.metadata.validation_data.is_empty() {
            return Ok(false);
        }

        // Check validation data length
        if key.metadata.validation_data.len() != 32 {
            return Ok(false);
        }

        // Additional validation data checks would go here
        Ok(true)
    }

    /// Apply strictness requirements
    fn apply_strictness_requirements(&self, base_valid: bool, score: f64) -> Result<bool> {
        match self.config.strictness {
            ValidationStrictness::Strict => Ok(base_valid && score >= 0.9),
            ValidationStrictness::Moderate => Ok(base_valid && score >= 0.7),
            ValidationStrictness::Lenient => Ok(base_valid && score >= 0.5),
        }
    }

    /// Track active validation
    fn track_validation(&mut self, id: String, validation_type: ValidationType) -> Result<()> {
        let active_validation = ActiveValidation {
            id,
            validation_type,
            started_at: current_timestamp(),
            progress: 0.0,
        };

        self.validation_state.active_validations.push(active_validation);
        Ok(())
    }

    /// Complete validation tracking
    fn complete_validation(&mut self, id: &str) -> Result<()> {
        self.validation_state.active_validations.retain(|v| v.id != id);
        Ok(())
    }

    /// Update validation statistics
    fn update_statistics(&mut self, valid: bool, duration: u64) -> Result<()> {
        let stats = &mut self.validation_state.statistics;
        
        stats.total_validations += 1;
        if valid {
            stats.successful_validations += 1;
        } else {
            stats.failed_validations += 1;
        }

        // Update average validation time
        stats.avg_validation_time = (stats.avg_validation_time * (stats.total_validations - 1) as f64 + duration as f64) / stats.total_validations as f64;

        // Update success rate
        stats.success_rate = stats.successful_validations as f64 / stats.total_validations as f64;

        Ok(())
    }
}

impl ValidationConfig {
    /// Create default validation configuration
    pub fn default() -> Self {
        Self {
            min_confidence: 0.7,
            max_time_skew: 60, // 1 minute
            strictness: ValidationStrictness::Moderate,
            required_criteria: vec![
                ValidationCriterion::TemporalConsistency,
                ValidationCriterion::DimensionalCoherence,
                ValidationCriterion::CryptographicIntegrity,
                ValidationCriterion::EnvironmentalAuthenticity,
            ],
            timeouts: ValidationTimeouts {
                temporal: 30,
                dimensional: 60,
                cryptographic: 45,
                environmental: 90,
            },
        }
    }

    /// Create strict validation configuration
    pub fn strict() -> Self {
        Self {
            min_confidence: 0.9,
            max_time_skew: 10, // 10 seconds
            strictness: ValidationStrictness::Strict,
            required_criteria: vec![
                ValidationCriterion::TemporalConsistency,
                ValidationCriterion::DimensionalCoherence,
                ValidationCriterion::CryptographicIntegrity,
                ValidationCriterion::EnvironmentalAuthenticity,
                ValidationCriterion::CoordinateTransformation,
                ValidationCriterion::MasundaMemorial,
            ],
            timeouts: ValidationTimeouts {
                temporal: 15,
                dimensional: 30,
                cryptographic: 30,
                environmental: 60,
            },
        }
    }
}

impl ValidationState {
    /// Create new validation state
    pub fn new() -> Self {
        Self {
            validation_cache: HashMap::new(),
            statistics: ValidationStatistics {
                total_validations: 0,
                successful_validations: 0,
                failed_validations: 0,
                avg_validation_time: 0.0,
                success_rate: 0.0,
            },
            active_validations: Vec::new(),
        }
    }
}

impl Default for ValidationState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let config = ValidationConfig::default();
        let validator = MdtecValidator::new(config);
        assert_eq!(validator.config.min_confidence, 0.7);
        assert_eq!(validator.config.strictness, ValidationStrictness::Moderate);
    }

    #[test]
    fn test_validation_strictness() {
        let strict_config = ValidationConfig::strict();
        let default_config = ValidationConfig::default();
        
        assert_eq!(strict_config.strictness, ValidationStrictness::Strict);
        assert_eq!(default_config.strictness, ValidationStrictness::Moderate);
        assert!(strict_config.min_confidence > default_config.min_confidence);
    }

    #[test]
    fn test_key_entropy_calculation() {
        let validator = MdtecValidator::default();
        
        // Test with uniform distribution
        let uniform_key = vec![0u8; 32];
        let entropy = validator.calculate_key_entropy(&uniform_key).unwrap();
        assert!(entropy < 0.1); // Should be low entropy
        
        // Test with random-like distribution
        let random_key: Vec<u8> = (0..32).map(|i| i as u8).collect();
        let entropy = validator.calculate_key_entropy(&random_key).unwrap();
        assert!(entropy > 0.5); // Should be higher entropy
    }

    #[test]
    fn test_validation_statistics() {
        let mut validator = MdtecValidator::default();
        
        // Initial statistics
        assert_eq!(validator.validation_state.statistics.total_validations, 0);
        assert_eq!(validator.validation_state.statistics.success_rate, 0.0);
        
        // Update statistics
        validator.update_statistics(true, 100).unwrap();
        assert_eq!(validator.validation_state.statistics.total_validations, 1);
        assert_eq!(validator.validation_state.statistics.success_rate, 1.0);
        
        validator.update_statistics(false, 200).unwrap();
        assert_eq!(validator.validation_state.statistics.total_validations, 2);
        assert_eq!(validator.validation_state.statistics.success_rate, 0.5);
    }
}
