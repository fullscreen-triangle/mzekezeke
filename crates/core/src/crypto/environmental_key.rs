//! Environmental key generation for MDTEC
//! 
//! This module implements the core environmental key generation algorithm that
//! combines multiple environmental dimensions to create cryptographic keys.

use crate::types::*;
use crate::error::{Error, Result};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;

/// Environmental key generator
pub struct EnvironmentalKeyGenerator {
    /// Configuration for key generation
    config: KeyGenerationConfig,
    /// Entropy accumulator for environmental randomness
    entropy_accumulator: EntropyAccumulator,
}

/// Configuration for environmental key generation
#[derive(Debug, Clone)]
pub struct KeyGenerationConfig {
    /// Target key size in bytes
    pub key_size: usize,
    /// Minimum entropy requirement in bits
    pub min_entropy: f64,
    /// Maximum age of environmental data in seconds
    pub max_data_age: u64,
    /// Required number of dimensions
    pub required_dimensions: usize,
    /// Dimension weights for key generation
    pub dimension_weights: HashMap<DimensionType, f64>,
}

/// Entropy accumulator for environmental randomness
#[derive(Debug, Clone)]
pub struct EntropyAccumulator {
    /// Accumulated entropy samples
    samples: Vec<EntropySample>,
    /// Total entropy collected
    total_entropy: f64,
}

/// Individual entropy sample from environmental dimension
#[derive(Debug, Clone)]
pub struct EntropySample {
    /// Dimension type
    pub dimension_type: DimensionType,
    /// Entropy value in bits
    pub entropy: f64,
    /// Raw measurement data
    pub data: Vec<u8>,
    /// Timestamp of measurement
    pub timestamp: Timestamp,
    /// Confidence in measurement
    pub confidence: f64,
}

/// Generated environmental key
#[derive(Debug, Clone)]
pub struct EnvironmentalKey {
    /// Key material
    pub key_material: KeyMaterial,
    /// Entropy used in key generation
    pub entropy: f64,
    /// Dimensions used in key generation
    pub dimensions: Vec<DimensionType>,
    /// Generation timestamp
    pub generated_at: Timestamp,
    /// Key derivation metadata
    pub metadata: KeyMetadata,
}

/// Metadata for key derivation
#[derive(Debug, Clone)]
pub struct KeyMetadata {
    /// Number of environmental samples used
    pub sample_count: usize,
    /// Total measurement time span
    pub time_span: u64,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    /// Validation data
    pub validation_data: Vec<u8>,
}

/// Quality metrics for key generation
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Entropy density (bits per byte)
    pub entropy_density: f64,
    /// Temporal stability score
    pub temporal_stability: f64,
    /// Dimensional diversity score
    pub dimensional_diversity: f64,
    /// Overall quality score (0.0 to 1.0)
    pub overall_quality: f64,
}

impl EnvironmentalKeyGenerator {
    /// Create a new environmental key generator
    pub fn new(config: KeyGenerationConfig) -> Self {
        Self {
            config,
            entropy_accumulator: EntropyAccumulator::new(),
        }
    }

    /// Create a new key generator with default configuration
    pub fn default() -> Self {
        Self::new(KeyGenerationConfig::default())
    }

    /// Add environmental measurement to entropy accumulator
    pub fn add_measurement(&mut self, measurement: &DimensionMeasurement) -> Result<()> {
        // Validate measurement age
        let now = current_timestamp();
        if now - measurement.timestamp > self.config.max_data_age * 1000 {
            return Err(Error::temporal("Environmental measurement too old"));
        }

        // Calculate entropy from measurement
        let entropy = self.calculate_entropy(measurement)?;
        
        // Convert measurement to entropy sample
        let sample = EntropySample {
            dimension_type: measurement.dimension_type,
            entropy,
            data: self.serialize_measurement(measurement)?,
            timestamp: measurement.timestamp,
            confidence: measurement.confidence,
        };

        // Add to accumulator
        self.entropy_accumulator.add_sample(sample);
        
        Ok(())
    }

    /// Generate environmental key from accumulated entropy
    pub fn generate_key(&mut self) -> Result<EnvironmentalKey> {
        // Check if we have enough entropy
        if self.entropy_accumulator.total_entropy < self.config.min_entropy {
            return Err(Error::crypto(format!(
                "Insufficient entropy: {} < {}",
                self.entropy_accumulator.total_entropy,
                self.config.min_entropy
            )));
        }

        // Check if we have enough dimensions
        let dimensions = self.entropy_accumulator.get_dimensions();
        if dimensions.len() < self.config.required_dimensions {
            return Err(Error::crypto(format!(
                "Insufficient dimensions: {} < {}",
                dimensions.len(),
                self.config.required_dimensions
            )));
        }

        // Generate key material
        let key_material = self.derive_key_material()?;
        
        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics()?;
        
        // Create key metadata
        let metadata = KeyMetadata {
            sample_count: self.entropy_accumulator.samples.len(),
            time_span: self.entropy_accumulator.get_time_span(),
            quality_metrics,
            validation_data: self.generate_validation_data()?,
        };

        // Create environmental key
        let key = EnvironmentalKey {
            key_material,
            entropy: self.entropy_accumulator.total_entropy,
            dimensions,
            generated_at: current_timestamp(),
            metadata,
        };

        // Reset accumulator for next key generation
        self.entropy_accumulator.reset();

        Ok(key)
    }

    /// Calculate entropy from environmental measurement
    fn calculate_entropy(&self, measurement: &DimensionMeasurement) -> Result<f64> {
        let mut entropy = 0.0;
        
        // Calculate entropy from measurement values
        for (key, value) in &measurement.values {
            // Use Shannon entropy calculation
            let normalized_value = self.normalize_value(key, *value)?;
            entropy += self.shannon_entropy(normalized_value);
        }

        // Apply dimension weight
        if let Some(weight) = self.config.dimension_weights.get(&measurement.dimension_type) {
            entropy *= weight;
        }

        // Apply confidence factor
        entropy *= measurement.confidence;

        Ok(entropy)
    }

    /// Normalize measurement value for entropy calculation
    fn normalize_value(&self, key: &str, value: f64) -> Result<f64> {
        // Normalize based on expected ranges for different measurement types
        match key {
            "latitude" | "longitude" => Ok(value / 180.0),
            "altitude" => Ok(value / 10000.0),
            "frequency" => Ok(value / 1000000.0),
            "amplitude" => Ok(value.clamp(0.0, 1.0)),
            "temperature" => Ok((value + 273.15) / 373.15),
            "pressure" => Ok(value / 101325.0),
            "humidity" => Ok(value / 100.0),
            _ => Ok(value.clamp(0.0, 1.0)),
        }
    }

    /// Calculate Shannon entropy for a normalized value
    fn shannon_entropy(&self, value: f64) -> f64 {
        if value <= 0.0 || value >= 1.0 {
            return 0.0;
        }
        
        -value * value.log2() - (1.0 - value) * (1.0 - value).log2()
    }

    /// Serialize measurement for key derivation
    fn serialize_measurement(&self, measurement: &DimensionMeasurement) -> Result<Vec<u8>> {
        // Create a deterministic serialization of the measurement
        let mut data = Vec::new();
        
        // Add dimension type
        data.extend_from_slice(&(measurement.dimension_type as u8).to_be_bytes());
        
        // Add timestamp
        data.extend_from_slice(&measurement.timestamp.to_be_bytes());
        
        // Add sorted measurement values
        let mut sorted_values: Vec<_> = measurement.values.iter().collect();
        sorted_values.sort_by_key(|(k, _)| *k);
        
        for (key, value) in sorted_values {
            data.extend_from_slice(key.as_bytes());
            data.extend_from_slice(&value.to_be_bytes());
        }
        
        // Add confidence
        data.extend_from_slice(&measurement.confidence.to_be_bytes());
        
        Ok(data)
    }

    /// Derive key material from accumulated entropy
    fn derive_key_material(&self) -> Result<KeyMaterial> {
        let mut hasher = Sha3_256::new();
        
        // Sort samples by timestamp for deterministic key generation
        let mut samples = self.entropy_accumulator.samples.clone();
        samples.sort_by_key(|s| s.timestamp);
        
        // Hash all entropy samples
        for sample in &samples {
            hasher.update(&sample.data);
        }
        
        // Generate key material
        let hash = hasher.finalize();
        let mut key_material = hash.to_vec();
        
        // Extend key material if needed
        while key_material.len() < self.config.key_size {
            let mut hasher = Sha3_256::new();
            hasher.update(&key_material);
            hasher.update(&self.entropy_accumulator.total_entropy.to_be_bytes());
            let additional = hasher.finalize();
            key_material.extend_from_slice(&additional);
        }
        
        // Truncate to desired size
        key_material.truncate(self.config.key_size);
        
        Ok(key_material)
    }

    /// Calculate quality metrics for generated key
    fn calculate_quality_metrics(&self) -> Result<QualityMetrics> {
        let sample_count = self.entropy_accumulator.samples.len();
        let time_span = self.entropy_accumulator.get_time_span();
        
        // Calculate entropy density
        let entropy_density = if sample_count > 0 {
            self.entropy_accumulator.total_entropy / sample_count as f64
        } else {
            0.0
        };

        // Calculate temporal stability
        let temporal_stability = self.calculate_temporal_stability()?;
        
        // Calculate dimensional diversity
        let dimensional_diversity = self.calculate_dimensional_diversity()?;
        
        // Calculate overall quality score
        let overall_quality = (entropy_density / 10.0 + temporal_stability + dimensional_diversity) / 3.0;
        
        Ok(QualityMetrics {
            entropy_density,
            temporal_stability,
            dimensional_diversity,
            overall_quality: overall_quality.clamp(0.0, 1.0),
        })
    }

    /// Calculate temporal stability score
    fn calculate_temporal_stability(&self) -> Result<f64> {
        if self.entropy_accumulator.samples.len() < 2 {
            return Ok(0.0);
        }

        let mut stability_sum = 0.0;
        let mut comparison_count = 0;

        // Compare entropy across time windows
        for i in 1..self.entropy_accumulator.samples.len() {
            let prev_entropy = self.entropy_accumulator.samples[i-1].entropy;
            let curr_entropy = self.entropy_accumulator.samples[i].entropy;
            
            if prev_entropy > 0.0 && curr_entropy > 0.0 {
                let stability = 1.0 - (prev_entropy - curr_entropy).abs() / prev_entropy.max(curr_entropy);
                stability_sum += stability;
                comparison_count += 1;
            }
        }

        Ok(if comparison_count > 0 {
            stability_sum / comparison_count as f64
        } else {
            0.0
        })
    }

    /// Calculate dimensional diversity score
    fn calculate_dimensional_diversity(&self) -> Result<f64> {
        let dimensions = self.entropy_accumulator.get_dimensions();
        let total_dimensions = DimensionType::all().len();
        
        // Base diversity score
        let mut diversity = dimensions.len() as f64 / total_dimensions as f64;
        
        // Bonus for even distribution across dimensions
        let mut dimension_counts = HashMap::new();
        for sample in &self.entropy_accumulator.samples {
            *dimension_counts.entry(sample.dimension_type).or_insert(0) += 1;
        }
        
        // Calculate evenness using Shannon diversity index
        let total_samples = self.entropy_accumulator.samples.len();
        let mut evenness = 0.0;
        for count in dimension_counts.values() {
            let proportion = *count as f64 / total_samples as f64;
            evenness -= proportion * proportion.log2();
        }
        evenness /= (dimensions.len() as f64).log2();
        
        // Combine diversity and evenness
        diversity *= evenness;
        
        Ok(diversity.clamp(0.0, 1.0))
    }

    /// Generate validation data for key verification
    fn generate_validation_data(&self) -> Result<Vec<u8>> {
        let mut hasher = Sha3_256::new();
        
        // Hash metadata about the key generation process
        hasher.update(&self.entropy_accumulator.total_entropy.to_be_bytes());
        hasher.update(&(self.entropy_accumulator.samples.len() as u64).to_be_bytes());
        hasher.update(&self.entropy_accumulator.get_time_span().to_be_bytes());
        
        // Hash dimension types used
        let mut dimensions = self.entropy_accumulator.get_dimensions();
        dimensions.sort_by_key(|d| *d as u8);
        for dimension in dimensions {
            hasher.update(&(dimension as u8).to_be_bytes());
        }
        
        Ok(hasher.finalize().to_vec())
    }
}

impl KeyGenerationConfig {
    /// Create default configuration
    pub fn default() -> Self {
        let mut dimension_weights = HashMap::new();
        
        // Set default weights for different dimensions
        dimension_weights.insert(DimensionType::Spatial, 1.0);
        dimension_weights.insert(DimensionType::Temporal, 1.2);
        dimension_weights.insert(DimensionType::Atmospheric, 0.8);
        dimension_weights.insert(DimensionType::Electromagnetic, 1.1);
        dimension_weights.insert(DimensionType::Acoustic, 0.9);
        dimension_weights.insert(DimensionType::Thermal, 0.7);
        dimension_weights.insert(DimensionType::Network, 0.6);
        dimension_weights.insert(DimensionType::Hardware, 0.8);
        dimension_weights.insert(DimensionType::Quantum, 1.5);
        dimension_weights.insert(DimensionType::Cellular, 0.7);
        dimension_weights.insert(DimensionType::Wifi, 0.6);
        dimension_weights.insert(DimensionType::System, 0.5);
        
        Self {
            key_size: 32,
            min_entropy: 128.0,
            max_data_age: 60,
            required_dimensions: 6,
            dimension_weights,
        }
    }
}

impl EntropyAccumulator {
    /// Create new entropy accumulator
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            total_entropy: 0.0,
        }
    }

    /// Add entropy sample
    pub fn add_sample(&mut self, sample: EntropySample) {
        self.total_entropy += sample.entropy;
        self.samples.push(sample);
    }

    /// Get unique dimensions represented in samples
    pub fn get_dimensions(&self) -> Vec<DimensionType> {
        let mut dimensions: Vec<_> = self.samples
            .iter()
            .map(|s| s.dimension_type)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        dimensions.sort_by_key(|d| *d as u8);
        dimensions
    }

    /// Get time span of samples
    pub fn get_time_span(&self) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }
        
        let min_time = self.samples.iter().map(|s| s.timestamp).min().unwrap_or(0);
        let max_time = self.samples.iter().map(|s| s.timestamp).max().unwrap_or(0);
        max_time - min_time
    }

    /// Reset accumulator
    pub fn reset(&mut self) {
        self.samples.clear();
        self.total_entropy = 0.0;
    }
}

impl Default for EntropyAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_key_generator_creation() {
        let config = KeyGenerationConfig::default();
        let generator = EnvironmentalKeyGenerator::new(config);
        assert_eq!(generator.config.key_size, 32);
        assert_eq!(generator.config.min_entropy, 128.0);
    }

    #[test]
    fn test_entropy_calculation() {
        let mut generator = EnvironmentalKeyGenerator::default();
        let mut values = HashMap::new();
        values.insert("latitude".to_string(), 45.0);
        values.insert("longitude".to_string(), -122.0);
        
        let measurement = DimensionMeasurement {
            dimension_type: DimensionType::Spatial,
            values,
            confidence: 0.9,
            timestamp: current_timestamp(),
        };
        
        let entropy = generator.calculate_entropy(&measurement).unwrap();
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_key_generation_insufficient_entropy() {
        let mut generator = EnvironmentalKeyGenerator::default();
        let result = generator.generate_key();
        assert!(result.is_err());
    }

    #[test]
    fn test_entropy_accumulator() {
        let mut accumulator = EntropyAccumulator::new();
        assert_eq!(accumulator.total_entropy, 0.0);
        assert_eq!(accumulator.samples.len(), 0);
        
        let sample = EntropySample {
            dimension_type: DimensionType::Spatial,
            entropy: 10.0,
            data: vec![1, 2, 3],
            timestamp: current_timestamp(),
            confidence: 0.9,
        };
        
        accumulator.add_sample(sample);
        assert_eq!(accumulator.total_entropy, 10.0);
        assert_eq!(accumulator.samples.len(), 1);
    }
}
