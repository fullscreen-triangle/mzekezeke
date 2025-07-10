//! Serialization utilities for MDTEC
//!
//! This module provides secure serialization, compression, and integrity
//! verification for MDTEC data structures.

use crate::types::*;
use crate::error::{Error, Result};
use serde::{Serialize, Deserialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;

/// Secure serialization wrapper
pub struct SecureSerializer {
    /// Configuration for serialization
    config: SerializationConfig,
    /// Integrity verification state
    verification_state: VerificationState,
}

/// Configuration for serialization
#[derive(Debug, Clone)]
pub struct SerializationConfig {
    /// Enable compression
    pub compression: bool,
    /// Compression level (1-9)
    pub compression_level: u32,
    /// Enable integrity verification
    pub verify_integrity: bool,
    /// Maximum serialized size allowed
    pub max_size: usize,
    /// Serialization format
    pub format: SerializationFormat,
    /// Security options
    pub security: SecurityOptions,
}

/// Available serialization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// Binary format (bincode)
    Binary,
    /// JSON format
    Json,
    /// MessagePack format
    MessagePack,
    /// Custom MDTEC format
    MdtecCustom,
}

/// Security options for serialization
#[derive(Debug, Clone)]
pub struct SecurityOptions {
    /// Include timestamp in serialized data
    pub include_timestamp: bool,
    /// Include checksum for integrity
    pub include_checksum: bool,
    /// Include version information
    pub include_version: bool,
    /// Obfuscate sensitive data
    pub obfuscate_sensitive: bool,
}

/// Verification state for integrity checking
#[derive(Debug, Clone)]
pub struct VerificationState {
    /// Cache of verified hashes
    pub verified_hashes: HashMap<String, VerificationEntry>,
    /// Maximum cache size
    pub max_cache_size: usize,
}

/// Verification cache entry
#[derive(Debug, Clone)]
pub struct VerificationEntry {
    /// Hash of the data
    pub hash: Vec<u8>,
    /// Verification timestamp
    pub verified_at: Timestamp,
    /// Verification confidence
    pub confidence: f64,
}

/// Serialized data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedData {
    /// The actual serialized payload
    pub data: Vec<u8>,
    /// Format used for serialization
    pub format: SerializationFormat,
    /// Metadata about the serialization
    pub metadata: SerializationMetadata,
    /// Integrity verification data
    pub verification: Option<IntegrityData>,
}

/// Metadata about serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializationMetadata {
    /// Original data size before compression
    pub original_size: usize,
    /// Compressed size
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Serialization timestamp
    pub timestamp: Timestamp,
    /// MDTEC version
    pub version: String,
    /// Data type identifier
    pub data_type: String,
}

/// Integrity verification data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityData {
    /// SHA3-256 hash of the data
    pub hash: Vec<u8>,
    /// Additional checksum
    pub checksum: u32,
    /// Verification salt
    pub salt: Vec<u8>,
    /// Verification timestamp
    pub verification_time: Timestamp,
}

/// Compression utilities
pub struct Compression;

/// Data integrity utilities
pub struct Integrity;

impl SecureSerializer {
    /// Create new secure serializer
    pub fn new(config: SerializationConfig) -> Self {
        Self {
            config,
            verification_state: VerificationState::new(),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(SerializationConfig::default())
    }

    /// Serialize data with security and compression
    pub fn serialize<T>(&mut self, data: &T) -> Result<SerializedData>
    where
        T: Serialize + ?Sized,
    {
        // Get data type name
        let data_type = std::any::type_name::<T>().to_string();
        
        // Serialize to intermediate format
        let raw_data = match self.config.format {
            SerializationFormat::Binary => {
                bincode::serialize(data)
                    .map_err(|e| Error::serialization(format!("Binary serialization failed: {}", e)))?
            }
            SerializationFormat::Json => {
                serde_json::to_vec(data)
                    .map_err(|e| Error::serialization(format!("JSON serialization failed: {}", e)))?
            }
            SerializationFormat::MessagePack => {
                rmp_serde::to_vec(data)
                    .map_err(|e| Error::serialization(format!("MessagePack serialization failed: {}", e)))?
            }
            SerializationFormat::MdtecCustom => {
                // Custom MDTEC format would be implemented here
                bincode::serialize(data)
                    .map_err(|e| Error::serialization(format!("MDTEC serialization failed: {}", e)))?
            }
        };

        let original_size = raw_data.len();
        
        // Check size limits
        if original_size > self.config.max_size {
            return Err(Error::serialization(format!(
                "Data too large: {} > {}",
                original_size,
                self.config.max_size
            )));
        }

        // Apply compression if enabled
        let compressed_data = if self.config.compression {
            Compression::compress(&raw_data, self.config.compression_level)?
        } else {
            raw_data
        };

        let compressed_size = compressed_data.len();
        let compression_ratio = if original_size > 0 {
            compressed_size as f64 / original_size as f64
        } else {
            1.0
        };

        // Apply obfuscation if enabled
        let final_data = if self.config.security.obfuscate_sensitive {
            self.obfuscate_data(&compressed_data)?
        } else {
            compressed_data
        };

        // Create metadata
        let metadata = SerializationMetadata {
            original_size,
            compressed_size,
            compression_ratio,
            timestamp: crate::utils::timing::current_timestamp(),
            version: "MDTEC_1.0".to_string(),
            data_type,
        };

        // Create integrity verification if enabled
        let verification = if self.config.verify_integrity {
            Some(self.create_integrity_data(&final_data)?)
        } else {
            None
        };

        Ok(SerializedData {
            data: final_data,
            format: self.config.format,
            metadata,
            verification,
        })
    }

    /// Deserialize data with verification
    pub fn deserialize<T>(&mut self, serialized: &SerializedData) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        // Verify integrity if present
        if let Some(ref verification) = serialized.verification {
            self.verify_integrity(&serialized.data, verification)?;
        }

        // Remove obfuscation if applied
        let deobfuscated_data = if self.config.security.obfuscate_sensitive {
            self.deobfuscate_data(&serialized.data)?
        } else {
            serialized.data.clone()
        };

        // Decompress if needed
        let raw_data = if self.config.compression {
            Compression::decompress(&deobfuscated_data)?
        } else {
            deobfuscated_data
        };

        // Deserialize based on format
        let result = match serialized.format {
            SerializationFormat::Binary => {
                bincode::deserialize(&raw_data)
                    .map_err(|e| Error::serialization(format!("Binary deserialization failed: {}", e)))?
            }
            SerializationFormat::Json => {
                serde_json::from_slice(&raw_data)
                    .map_err(|e| Error::serialization(format!("JSON deserialization failed: {}", e)))?
            }
            SerializationFormat::MessagePack => {
                rmp_serde::from_slice(&raw_data)
                    .map_err(|e| Error::serialization(format!("MessagePack deserialization failed: {}", e)))?
            }
            SerializationFormat::MdtecCustom => {
                bincode::deserialize(&raw_data)
                    .map_err(|e| Error::serialization(format!("MDTEC deserialization failed: {}", e)))?
            }
        };

        Ok(result)
    }

    /// Create integrity verification data
    fn create_integrity_data(&self, data: &[u8]) -> Result<IntegrityData> {
        let mut hasher = Sha3_256::new();
        let salt = self.generate_salt();
        
        hasher.update(data);
        hasher.update(&salt);
        let hash = hasher.finalize().to_vec();
        
        let checksum = Integrity::crc32(data);
        
        Ok(IntegrityData {
            hash,
            checksum,
            salt,
            verification_time: crate::utils::timing::current_timestamp(),
        })
    }

    /// Verify data integrity
    fn verify_integrity(&mut self, data: &[u8], verification: &IntegrityData) -> Result<()> {
        // Verify hash
        let mut hasher = Sha3_256::new();
        hasher.update(data);
        hasher.update(&verification.salt);
        let computed_hash = hasher.finalize().to_vec();
        
        if computed_hash != verification.hash {
            return Err(Error::verification("Hash verification failed"));
        }

        // Verify checksum
        let computed_checksum = Integrity::crc32(data);
        if computed_checksum != verification.checksum {
            return Err(Error::verification("Checksum verification failed"));
        }

        // Cache successful verification
        let cache_key = hex::encode(&verification.hash);
        let entry = VerificationEntry {
            hash: verification.hash.clone(),
            verified_at: crate::utils::timing::current_timestamp(),
            confidence: 1.0,
        };
        
        self.verification_state.add_entry(cache_key, entry);
        
        Ok(())
    }

    /// Generate cryptographic salt
    fn generate_salt(&self) -> Vec<u8> {
        use sha3::Digest;
        let mut hasher = Sha3_256::new();
        hasher.update(&crate::utils::timing::current_timestamp().to_be_bytes());
        hasher.update(b"MDTEC_SALT");
        hasher.finalize().to_vec()
    }

    /// Obfuscate sensitive data
    fn obfuscate_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simple XOR obfuscation with generated key
        let key = self.generate_obfuscation_key(data.len());
        let obfuscated: Vec<u8> = data.iter()
            .zip(key.iter().cycle())
            .map(|(d, k)| d ^ k)
            .collect();
        
        Ok(obfuscated)
    }

    /// Remove obfuscation from data
    fn deobfuscate_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // XOR is its own inverse
        self.obfuscate_data(data)
    }

    /// Generate obfuscation key
    fn generate_obfuscation_key(&self, length: usize) -> Vec<u8> {
        let mut key = Vec::with_capacity(length);
        let mut hasher = Sha3_256::new();
        hasher.update(b"MDTEC_OBFUSCATION");
        
        let mut seed = hasher.finalize().to_vec();
        for i in 0..length {
            key.push(seed[i % seed.len()]);
            if i % seed.len() == 0 && i > 0 {
                // Regenerate seed periodically
                let mut hasher = Sha3_256::new();
                hasher.update(&seed);
                hasher.update(&(i as u64).to_be_bytes());
                seed = hasher.finalize().to_vec();
            }
        }
        
        key
    }
}

impl Compression {
    /// Compress data using the specified level
    pub fn compress(data: &[u8], level: u32) -> Result<Vec<u8>> {
        use flate2::{Compression as FlateCompression, write::GzEncoder};
        use std::io::Write;

        let compression_level = FlateCompression::new(level.min(9));
        let mut encoder = GzEncoder::new(Vec::new(), compression_level);
        
        encoder.write_all(data)
            .map_err(|e| Error::serialization(format!("Compression failed: {}", e)))?;
        
        encoder.finish()
            .map_err(|e| Error::serialization(format!("Compression finalization failed: {}", e)))
    }

    /// Decompress data
    pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| Error::serialization(format!("Decompression failed: {}", e)))?;
        
        Ok(decompressed)
    }

    /// Estimate compression ratio
    pub fn estimate_ratio(data: &[u8]) -> Result<f64> {
        // Sample compression for estimation
        let sample_size = data.len().min(1024);
        let sample = &data[..sample_size];
        
        let compressed = Self::compress(sample, 6)?; // Medium compression
        let ratio = compressed.len() as f64 / sample.len() as f64;
        
        Ok(ratio)
    }
}

impl Integrity {
    /// Calculate CRC32 checksum
    pub fn crc32(data: &[u8]) -> u32 {
        crc32fast::hash(data)
    }

    /// Calculate SHA3-256 hash
    pub fn sha3_256(data: &[u8]) -> Vec<u8> {
        let mut hasher = Sha3_256::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }

    /// Verify data integrity using multiple methods
    pub fn verify_data(data: &[u8], expected_hash: &[u8], expected_checksum: u32) -> Result<bool> {
        // Verify hash
        let computed_hash = Self::sha3_256(data);
        if computed_hash != expected_hash {
            return Ok(false);
        }

        // Verify checksum
        let computed_checksum = Self::crc32(data);
        if computed_checksum != expected_checksum {
            return Ok(false);
        }

        Ok(true)
    }

    /// Calculate integrity score based on multiple factors
    pub fn integrity_score(data: &[u8], metadata: &SerializationMetadata) -> f64 {
        let mut score = 1.0;

        // Check for suspicious compression ratios
        if metadata.compression_ratio > 1.0 {
            score *= 0.5; // Data expanded instead of compressed
        }

        // Check for reasonable size bounds
        if metadata.original_size == 0 || metadata.compressed_size == 0 {
            score *= 0.1;
        }

        // Check data entropy
        let entropy = Self::calculate_entropy(data);
        if entropy < 1.0 {
            score *= 0.8; // Low entropy might indicate issues
        }

        score.clamp(0.0, 1.0)
    }

    /// Calculate entropy of data
    fn calculate_entropy(data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mut counts = [0u32; 256];
        for &byte in data {
            counts[byte as usize] += 1;
        }

        let len = data.len() as f64;
        let mut entropy = 0.0;

        for count in counts.iter() {
            if *count > 0 {
                let p = *count as f64 / len;
                entropy -= p * p.log2();
            }
        }

        entropy / 8.0 // Normalize to 0-1 range
    }
}

impl SerializationConfig {
    /// Create default configuration
    pub fn default() -> Self {
        Self {
            compression: true,
            compression_level: 6,
            verify_integrity: true,
            max_size: 10 * 1024 * 1024, // 10 MB
            format: SerializationFormat::Binary,
            security: SecurityOptions {
                include_timestamp: true,
                include_checksum: true,
                include_version: true,
                obfuscate_sensitive: false,
            },
        }
    }

    /// Create high-security configuration
    pub fn high_security() -> Self {
        Self {
            compression: true,
            compression_level: 9,
            verify_integrity: true,
            max_size: 1024 * 1024, // 1 MB
            format: SerializationFormat::MdtecCustom,
            security: SecurityOptions {
                include_timestamp: true,
                include_checksum: true,
                include_version: true,
                obfuscate_sensitive: true,
            },
        }
    }

    /// Create fast configuration (minimal security)
    pub fn fast() -> Self {
        Self {
            compression: false,
            compression_level: 1,
            verify_integrity: false,
            max_size: 100 * 1024 * 1024, // 100 MB
            format: SerializationFormat::Binary,
            security: SecurityOptions {
                include_timestamp: false,
                include_checksum: false,
                include_version: false,
                obfuscate_sensitive: false,
            },
        }
    }
}

impl VerificationState {
    /// Create new verification state
    pub fn new() -> Self {
        Self {
            verified_hashes: HashMap::new(),
            max_cache_size: 1000,
        }
    }

    /// Add verification entry to cache
    pub fn add_entry(&mut self, key: String, entry: VerificationEntry) {
        if self.verified_hashes.len() >= self.max_cache_size {
            // Remove oldest entries (simple LRU)
            let mut entries: Vec<_> = self.verified_hashes.iter().collect();
            entries.sort_by_key(|(_, entry)| entry.verified_at);
            
            if let Some((oldest_key, _)) = entries.first() {
                let oldest_key = oldest_key.clone();
                self.verified_hashes.remove(&oldest_key);
            }
        }

        self.verified_hashes.insert(key, entry);
    }

    /// Check if hash was recently verified
    pub fn is_verified(&self, hash: &[u8]) -> bool {
        let key = hex::encode(hash);
        self.verified_hashes.contains_key(&key)
    }

    /// Get verification confidence for hash
    pub fn get_confidence(&self, hash: &[u8]) -> Option<f64> {
        let key = hex::encode(hash);
        self.verified_hashes.get(&key).map(|entry| entry.confidence)
    }
}

impl Default for VerificationState {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for quick serialization
pub mod quick {
    use super::*;

    /// Quick binary serialization
    pub fn serialize_binary<T>(data: &T) -> Result<Vec<u8>>
    where
        T: Serialize,
    {
        bincode::serialize(data)
            .map_err(|e| Error::serialization(format!("Quick binary serialization failed: {}", e)))
    }

    /// Quick binary deserialization
    pub fn deserialize_binary<T>(data: &[u8]) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        bincode::deserialize(data)
            .map_err(|e| Error::serialization(format!("Quick binary deserialization failed: {}", e)))
    }

    /// Quick JSON serialization
    pub fn serialize_json<T>(data: &T) -> Result<String>
    where
        T: Serialize,
    {
        serde_json::to_string(data)
            .map_err(|e| Error::serialization(format!("Quick JSON serialization failed: {}", e)))
    }

    /// Quick JSON deserialization
    pub fn deserialize_json<T>(data: &str) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(data)
            .map_err(|e| Error::serialization(format!("Quick JSON deserialization failed: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestData {
        id: u64,
        name: String,
        values: Vec<f64>,
    }

    #[test]
    fn test_secure_serialization() {
        let mut serializer = SecureSerializer::default();
        
        let test_data = TestData {
            id: 123,
            name: "test".to_string(),
            values: vec![1.0, 2.0, 3.0],
        };

        let serialized = serializer.serialize(&test_data).unwrap();
        assert!(serialized.verification.is_some());
        
        let deserialized: TestData = serializer.deserialize(&serialized).unwrap();
        assert_eq!(test_data, deserialized);
    }

    #[test]
    fn test_compression() {
        let data = vec![0u8; 1000]; // Highly compressible data
        let compressed = Compression::compress(&data, 6).unwrap();
        assert!(compressed.len() < data.len());
        
        let decompressed = Compression::decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_integrity() {
        let data = b"Hello, MDTEC!";
        let hash = Integrity::sha3_256(data);
        let checksum = Integrity::crc32(data);
        
        assert!(Integrity::verify_data(data, &hash, checksum).unwrap());
        
        // Test with wrong data
        let wrong_data = b"Hello, world!";
        assert!(!Integrity::verify_data(wrong_data, &hash, checksum).unwrap());
    }

    #[test]
    fn test_serialization_formats() {
        let config = SerializationConfig {
            format: SerializationFormat::Json,
            ..SerializationConfig::default()
        };
        
        let mut serializer = SecureSerializer::new(config);
        let test_data = TestData {
            id: 456,
            name: "json_test".to_string(),
            values: vec![4.0, 5.0, 6.0],
        };

        let serialized = serializer.serialize(&test_data).unwrap();
        let deserialized: TestData = serializer.deserialize(&serialized).unwrap();
        assert_eq!(test_data, deserialized);
    }

    #[test]
    fn test_quick_serialization() {
        let test_data = TestData {
            id: 789,
            name: "quick_test".to_string(),
            values: vec![7.0, 8.0, 9.0],
        };

        let binary_data = quick::serialize_binary(&test_data).unwrap();
        let deserialized: TestData = quick::deserialize_binary(&binary_data).unwrap();
        assert_eq!(test_data, deserialized);

        let json_data = quick::serialize_json(&test_data).unwrap();
        let deserialized: TestData = quick::deserialize_json(&json_data).unwrap();
        assert_eq!(test_data, deserialized);
    }
}
