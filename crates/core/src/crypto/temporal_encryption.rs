//! Temporal encryption for MDTEC
//!
//! This module implements temporal encryption where cryptographic operations
//! are bound to specific time windows, ensuring temporal ephemeral properties.

use crate::types::*;
use crate::error::{Error, Result};
use crate::crypto::environmental_key::EnvironmentalKey;
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce, aead::{Aead, AeadCore, KeyInit, OsRng}};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Temporal encryption engine
pub struct TemporalEncryptor {
    /// Configuration for temporal encryption
    config: TemporalConfig,
    /// Current time-based key schedule
    key_schedule: TimeKeySchedule,
}

/// Configuration for temporal encryption
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// Time window duration in seconds
    pub time_window: u64,
    /// Maximum allowed clock skew in seconds
    pub max_clock_skew: u64,
    /// Number of time windows to cache
    pub cache_size: usize,
    /// Temporal validation strictness
    pub strictness: TemporalStrictness,
}

/// Temporal validation strictness levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalStrictness {
    /// Strict temporal validation
    Strict,
    /// Moderate temporal validation with some tolerance
    Moderate,
    /// Loose temporal validation for testing
    Loose,
}

/// Time-based key schedule for temporal encryption
#[derive(Debug, Clone)]
pub struct TimeKeySchedule {
    /// Current active time window
    current_window: TimeWindow,
    /// Previous time windows (for decryption)
    previous_windows: Vec<TimeWindow>,
    /// Next time window (for pre-computation)
    next_window: Option<TimeWindow>,
}

/// Time window with associated cryptographic material
#[derive(Debug, Clone)]
pub struct TimeWindow {
    /// Start time of the window
    pub start_time: Timestamp,
    /// End time of the window
    pub end_time: Timestamp,
    /// Cryptographic key for this window
    pub key: Vec<u8>,
    /// Nonce for this window
    pub nonce: Vec<u8>,
    /// Window identifier
    pub window_id: u64,
}

/// Temporal encryption result
#[derive(Debug, Clone)]
pub struct TemporalCiphertext {
    /// Encrypted data
    pub ciphertext: Vec<u8>,
    /// Time window used for encryption
    pub window_id: u64,
    /// Encryption timestamp
    pub encrypted_at: Timestamp,
    /// Temporal validation data
    pub temporal_proof: TemporalProof,
}

/// Temporal proof for validation
#[derive(Debug, Clone)]
pub struct TemporalProof {
    /// Proof that encryption occurred within valid time window
    pub window_proof: Vec<u8>,
    /// Chronological sequence proof
    pub sequence_proof: Vec<u8>,
    /// Environmental time synchronization data
    pub sync_data: Vec<u8>,
}

/// Temporal decryption result
#[derive(Debug, Clone)]
pub struct TemporalPlaintext {
    /// Decrypted data
    pub plaintext: Vec<u8>,
    /// Original encryption time
    pub encrypted_at: Timestamp,
    /// Decryption timestamp
    pub decrypted_at: Timestamp,
    /// Temporal validation result
    pub temporal_validation: TemporalValidation,
}

/// Temporal validation result
#[derive(Debug, Clone)]
pub struct TemporalValidation {
    /// Whether temporal validation passed
    pub valid: bool,
    /// Validation confidence score
    pub confidence: f64,
    /// Time window match result
    pub window_match: bool,
    /// Clock skew detected
    pub clock_skew: Option<Duration>,
    /// Validation details
    pub details: String,
}

impl TemporalEncryptor {
    /// Create new temporal encryptor
    pub fn new(config: TemporalConfig) -> Result<Self> {
        let key_schedule = TimeKeySchedule::new(config.time_window)?;
        
        Ok(Self {
            config,
            key_schedule,
        })
    }

    /// Create temporal encryptor with default configuration
    pub fn default() -> Result<Self> {
        Self::new(TemporalConfig::default())
    }

    /// Initialize temporal encryptor with environmental key
    pub fn initialize_with_key(&mut self, env_key: &EnvironmentalKey) -> Result<()> {
        self.key_schedule.initialize_from_environmental_key(env_key, self.config.time_window)?;
        Ok(())
    }

    /// Encrypt data with temporal binding
    pub fn encrypt(&mut self, plaintext: &[u8]) -> Result<TemporalCiphertext> {
        let now = current_timestamp();
        
        // Update key schedule if needed
        self.update_key_schedule(now)?;
        
        // Get current time window
        let current_window = &self.key_schedule.current_window;
        
        // Validate current time is within active window
        if now < current_window.start_time || now > current_window.end_time {
            return Err(Error::temporal("Current time outside active time window"));
        }

        // Perform encryption
        let ciphertext = self.encrypt_with_window(plaintext, current_window)?;
        
        // Generate temporal proof
        let temporal_proof = self.generate_temporal_proof(current_window, now)?;
        
        Ok(TemporalCiphertext {
            ciphertext,
            window_id: current_window.window_id,
            encrypted_at: now,
            temporal_proof,
        })
    }

    /// Decrypt temporally encrypted data
    pub fn decrypt(&mut self, ciphertext: &TemporalCiphertext) -> Result<TemporalPlaintext> {
        let now = current_timestamp();
        
        // Find the appropriate time window for decryption
        let window = self.find_decryption_window(ciphertext.window_id)?;
        
        // Validate temporal constraints
        let temporal_validation = self.validate_temporal_constraints(ciphertext, &window, now)?;
        
        // Decrypt the data
        let plaintext = self.decrypt_with_window(&ciphertext.ciphertext, &window)?;
        
        Ok(TemporalPlaintext {
            plaintext,
            encrypted_at: ciphertext.encrypted_at,
            decrypted_at: now,
            temporal_validation,
        })
    }

    /// Update key schedule based on current time
    fn update_key_schedule(&mut self, current_time: Timestamp) -> Result<()> {
        // Check if we need to advance to next time window
        if current_time >= self.key_schedule.current_window.end_time {
            self.key_schedule.advance_window(self.config.time_window)?;
        }
        
        // Pre-compute next window if needed
        if self.key_schedule.next_window.is_none() {
            self.key_schedule.prepare_next_window(self.config.time_window)?;
        }
        
        // Clean up old windows
        self.key_schedule.cleanup_old_windows(self.config.cache_size);
        
        Ok(())
    }

    /// Encrypt data with specific time window
    fn encrypt_with_window(&self, plaintext: &[u8], window: &TimeWindow) -> Result<Vec<u8>> {
        // Create ChaCha20Poly1305 cipher
        let key = Key::from_slice(&window.key[..32]);
        let cipher = ChaCha20Poly1305::new(key);
        
        // Use window nonce
        let nonce = Nonce::from_slice(&window.nonce[..12]);
        
        // Encrypt the data
        cipher.encrypt(nonce, plaintext)
            .map_err(|e| Error::crypto(format!("Encryption failed: {}", e)))
    }

    /// Decrypt data with specific time window
    fn decrypt_with_window(&self, ciphertext: &[u8], window: &TimeWindow) -> Result<Vec<u8>> {
        // Create ChaCha20Poly1305 cipher
        let key = Key::from_slice(&window.key[..32]);
        let cipher = ChaCha20Poly1305::new(key);
        
        // Use window nonce
        let nonce = Nonce::from_slice(&window.nonce[..12]);
        
        // Decrypt the data
        cipher.decrypt(nonce, ciphertext)
            .map_err(|e| Error::crypto(format!("Decryption failed: {}", e)))
    }

    /// Find appropriate time window for decryption
    fn find_decryption_window(&self, window_id: u64) -> Result<TimeWindow> {
        // Check current window
        if self.key_schedule.current_window.window_id == window_id {
            return Ok(self.key_schedule.current_window.clone());
        }

        // Check previous windows
        for window in &self.key_schedule.previous_windows {
            if window.window_id == window_id {
                return Ok(window.clone());
            }
        }

        // Check next window
        if let Some(ref next_window) = self.key_schedule.next_window {
            if next_window.window_id == window_id {
                return Ok(next_window.clone());
            }
        }

        Err(Error::temporal(format!("Time window {} not found", window_id)))
    }

    /// Validate temporal constraints for decryption
    fn validate_temporal_constraints(
        &self,
        ciphertext: &TemporalCiphertext,
        window: &TimeWindow,
        current_time: Timestamp,
    ) -> Result<TemporalValidation> {
        let mut validation = TemporalValidation {
            valid: true,
            confidence: 1.0,
            window_match: true,
            clock_skew: None,
            details: String::new(),
        };

        // Check if encryption time was within the window
        if ciphertext.encrypted_at < window.start_time || ciphertext.encrypted_at > window.end_time {
            validation.valid = false;
            validation.confidence *= 0.5;
            validation.details.push_str("Encryption time outside window bounds; ");
        }

        // Check clock skew
        let time_diff = if current_time > ciphertext.encrypted_at {
            current_time - ciphertext.encrypted_at
        } else {
            ciphertext.encrypted_at - current_time
        };

        if time_diff > self.config.max_clock_skew * 1000 {
            let skew = Duration::from_millis(time_diff);
            validation.clock_skew = Some(skew);
            
            match self.config.strictness {
                TemporalStrictness::Strict => {
                    validation.valid = false;
                    validation.confidence *= 0.1;
                    validation.details.push_str("Clock skew exceeds strict limits; ");
                }
                TemporalStrictness::Moderate => {
                    validation.confidence *= 0.7;
                    validation.details.push_str("Clock skew detected; ");
                }
                TemporalStrictness::Loose => {
                    validation.confidence *= 0.9;
                }
            }
        }

        // Validate temporal proof
        let proof_valid = self.validate_temporal_proof(&ciphertext.temporal_proof, window)?;
        if !proof_valid {
            validation.valid = false;
            validation.confidence *= 0.3;
            validation.details.push_str("Temporal proof validation failed; ");
        }

        Ok(validation)
    }

    /// Generate temporal proof for encryption
    fn generate_temporal_proof(&self, window: &TimeWindow, timestamp: Timestamp) -> Result<TemporalProof> {
        use sha3::{Digest, Sha3_256};
        
        // Generate window proof
        let mut window_hasher = Sha3_256::new();
        window_hasher.update(&window.window_id.to_be_bytes());
        window_hasher.update(&window.start_time.to_be_bytes());
        window_hasher.update(&window.end_time.to_be_bytes());
        window_hasher.update(&timestamp.to_be_bytes());
        let window_proof = window_hasher.finalize().to_vec();

        // Generate sequence proof
        let mut sequence_hasher = Sha3_256::new();
        sequence_hasher.update(&window.window_id.to_be_bytes());
        sequence_hasher.update(&timestamp.to_be_bytes());
        if let Some(prev_window) = self.key_schedule.previous_windows.last() {
            sequence_hasher.update(&prev_window.window_id.to_be_bytes());
        }
        let sequence_proof = sequence_hasher.finalize().to_vec();

        // Generate sync data
        let mut sync_hasher = Sha3_256::new();
        sync_hasher.update(&timestamp.to_be_bytes());
        sync_hasher.update(&window.key);
        let sync_data = sync_hasher.finalize().to_vec();

        Ok(TemporalProof {
            window_proof,
            sequence_proof,
            sync_data,
        })
    }

    /// Validate temporal proof
    fn validate_temporal_proof(&self, proof: &TemporalProof, window: &TimeWindow) -> Result<bool> {
        // This is a simplified validation - in practice, this would involve
        // more sophisticated cryptographic proofs
        
        // Validate window proof structure
        if proof.window_proof.len() != 32 || proof.sequence_proof.len() != 32 || proof.sync_data.len() != 32 {
            return Ok(false);
        }

        // Additional validation logic would go here
        // For now, we assume the proof is valid if it has the right structure
        Ok(true)
    }
}

impl TimeKeySchedule {
    /// Create new time key schedule
    pub fn new(window_duration: u64) -> Result<Self> {
        let current_time = current_timestamp();
        let current_window = TimeWindow::new(current_time, window_duration)?;
        
        Ok(Self {
            current_window,
            previous_windows: Vec::new(),
            next_window: None,
        })
    }

    /// Initialize from environmental key
    pub fn initialize_from_environmental_key(
        &mut self,
        env_key: &EnvironmentalKey,
        window_duration: u64,
    ) -> Result<()> {
        use sha3::{Digest, Sha3_256};
        
        // Derive time-based key from environmental key
        let mut hasher = Sha3_256::new();
        hasher.update(&env_key.key_material);
        hasher.update(&env_key.generated_at.to_be_bytes());
        let base_key = hasher.finalize();

        // Initialize current window with derived key
        let current_time = current_timestamp();
        self.current_window = TimeWindow::new_with_key(current_time, window_duration, base_key.to_vec())?;
        
        Ok(())
    }

    /// Advance to next time window
    pub fn advance_window(&mut self, window_duration: u64) -> Result<()> {
        // Move current window to previous windows
        self.previous_windows.push(self.current_window.clone());
        
        // Use next window if available, otherwise create new one
        if let Some(next_window) = self.next_window.take() {
            self.current_window = next_window;
        } else {
            let next_start = self.current_window.end_time;
            self.current_window = TimeWindow::new(next_start, window_duration)?;
        }
        
        Ok(())
    }

    /// Prepare next time window
    pub fn prepare_next_window(&mut self, window_duration: u64) -> Result<()> {
        let next_start = self.current_window.end_time;
        self.next_window = Some(TimeWindow::new(next_start, window_duration)?);
        Ok(())
    }

    /// Clean up old windows
    pub fn cleanup_old_windows(&mut self, max_cache_size: usize) {
        if self.previous_windows.len() > max_cache_size {
            let excess = self.previous_windows.len() - max_cache_size;
            self.previous_windows.drain(0..excess);
        }
    }
}

impl TimeWindow {
    /// Create new time window
    pub fn new(start_time: Timestamp, duration: u64) -> Result<Self> {
        use sha3::{Digest, Sha3_256};
        
        let end_time = start_time + duration * 1000;
        let window_id = start_time / 1000; // Use second-based window ID
        
        // Generate key and nonce from window parameters
        let mut key_hasher = Sha3_256::new();
        key_hasher.update(&window_id.to_be_bytes());
        key_hasher.update(&start_time.to_be_bytes());
        key_hasher.update(b"MDTEC_TEMPORAL_KEY");
        let key = key_hasher.finalize().to_vec();
        
        let mut nonce_hasher = Sha3_256::new();
        nonce_hasher.update(&window_id.to_be_bytes());
        nonce_hasher.update(&start_time.to_be_bytes());
        nonce_hasher.update(b"MDTEC_TEMPORAL_NONCE");
        let nonce = nonce_hasher.finalize()[..12].to_vec();
        
        Ok(Self {
            start_time,
            end_time,
            key,
            nonce,
            window_id,
        })
    }

    /// Create new time window with specific key
    pub fn new_with_key(start_time: Timestamp, duration: u64, base_key: Vec<u8>) -> Result<Self> {
        use sha3::{Digest, Sha3_256};
        
        let end_time = start_time + duration * 1000;
        let window_id = start_time / 1000;
        
        // Derive key and nonce from base key
        let mut key_hasher = Sha3_256::new();
        key_hasher.update(&base_key);
        key_hasher.update(&window_id.to_be_bytes());
        key_hasher.update(b"MDTEC_TEMPORAL_KEY");
        let key = key_hasher.finalize().to_vec();
        
        let mut nonce_hasher = Sha3_256::new();
        nonce_hasher.update(&base_key);
        nonce_hasher.update(&window_id.to_be_bytes());
        nonce_hasher.update(b"MDTEC_TEMPORAL_NONCE");
        let nonce = nonce_hasher.finalize()[..12].to_vec();
        
        Ok(Self {
            start_time,
            end_time,
            key,
            nonce,
            window_id,
        })
    }

    /// Check if timestamp is within this window
    pub fn contains_time(&self, timestamp: Timestamp) -> bool {
        timestamp >= self.start_time && timestamp <= self.end_time
    }

    /// Get window duration
    pub fn duration(&self) -> u64 {
        self.end_time - self.start_time
    }
}

impl TemporalConfig {
    /// Create default temporal configuration
    pub fn default() -> Self {
        Self {
            time_window: 300, // 5 minutes
            max_clock_skew: 60, // 1 minute
            cache_size: 10,
            strictness: TemporalStrictness::Moderate,
        }
    }

    /// Create strict temporal configuration
    pub fn strict() -> Self {
        Self {
            time_window: 60, // 1 minute
            max_clock_skew: 5, // 5 seconds
            cache_size: 5,
            strictness: TemporalStrictness::Strict,
        }
    }

    /// Create loose temporal configuration for testing
    pub fn loose() -> Self {
        Self {
            time_window: 3600, // 1 hour
            max_clock_skew: 300, // 5 minutes
            cache_size: 20,
            strictness: TemporalStrictness::Loose,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_encryptor_creation() {
        let config = TemporalConfig::default();
        let encryptor = TemporalEncryptor::new(config);
        assert!(encryptor.is_ok());
    }

    #[test]
    fn test_time_window_creation() {
        let now = current_timestamp();
        let window = TimeWindow::new(now, 300).unwrap();
        assert_eq!(window.start_time, now);
        assert_eq!(window.end_time, now + 300000);
        assert!(window.contains_time(now + 150000));
        assert!(!window.contains_time(now + 400000));
    }

    #[test]
    fn test_temporal_encryption_decryption() {
        let mut encryptor = TemporalEncryptor::default().unwrap();
        let plaintext = b"Hello, temporal world!";
        
        let ciphertext = encryptor.encrypt(plaintext).unwrap();
        assert_ne!(ciphertext.ciphertext, plaintext);
        
        let decrypted = encryptor.decrypt(&ciphertext).unwrap();
        assert_eq!(decrypted.plaintext, plaintext);
        assert!(decrypted.temporal_validation.valid);
    }

    #[test]
    fn test_temporal_validation_strictness() {
        let strict_config = TemporalConfig::strict();
        let loose_config = TemporalConfig::loose();
        
        assert_eq!(strict_config.strictness, TemporalStrictness::Strict);
        assert_eq!(loose_config.strictness, TemporalStrictness::Loose);
        assert_eq!(strict_config.max_clock_skew, 5);
        assert_eq!(loose_config.max_clock_skew, 300);
    }
}
