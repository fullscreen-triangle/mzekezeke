use crate::types::{DimensionMeasurement, DimensionType, MdtecError, MdtecResult};
use crate::utils::math::{Statistics, VectorOps};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

/// Acoustic measurement
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AcousticMeasurement {
    /// Sound pressure level in dB
    pub spl_db: f64,
    /// Frequency in Hz
    pub frequency: f64,
    /// Sound intensity in W/m²
    pub intensity: f64,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

/// Acoustic frequency spectrum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticSpectrum {
    /// Frequency bins in Hz
    pub frequencies: Vec<f64>,
    /// Amplitude levels in dB
    pub amplitudes: Vec<f64>,
    /// Spectral centroid
    pub spectral_centroid: f64,
    /// Spectral bandwidth
    pub spectral_bandwidth: f64,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

/// Acoustic pattern types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AcousticPatternType {
    /// Constant tone
    Tone,
    /// Periodic pattern
    Periodic,
    /// Noise
    Noise,
    /// Transient
    Transient,
    /// Speech-like
    Speech,
    /// Music-like
    Music,
}

/// Detected acoustic pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticPattern {
    /// Pattern type
    pub pattern_type: AcousticPatternType,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Dominant frequency
    pub dominant_frequency: f64,
    /// Pattern duration
    pub duration: Duration,
    /// Timestamp when detected
    pub timestamp: Instant,
}

/// Acoustic dimension configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticConfig {
    /// Sample window size for acoustic analysis
    pub sample_window_size: usize,
    /// Minimum SPL threshold (dB)
    pub min_spl_threshold: f64,
    /// Maximum SPL threshold (dB)
    pub max_spl_threshold: f64,
    /// Enable frequency spectrum analysis
    pub enable_spectrum_analysis: bool,
    /// Enable pattern recognition
    pub enable_pattern_recognition: bool,
    /// Frequency bands for analysis
    pub frequency_bands: Vec<(f64, f64)>,
    /// Spectrum analysis window size
    pub spectrum_window_size: usize,
}

impl Default for AcousticConfig {
    fn default() -> Self {
        Self {
            sample_window_size: 1000,
            min_spl_threshold: 20.0,  // 20 dB
            max_spl_threshold: 120.0, // 120 dB
            enable_spectrum_analysis: true,
            enable_pattern_recognition: true,
            frequency_bands: vec![
                (20.0, 200.0),     // Sub-bass
                (200.0, 500.0),    // Bass
                (500.0, 2000.0),   // Midrange
                (2000.0, 5000.0),  // Upper midrange
                (5000.0, 20000.0), // Treble
            ],
            spectrum_window_size: 128,
        }
    }
}

/// Acoustic dimension analyzer
pub struct AcousticAnalyzer {
    config: AcousticConfig,
    acoustic_readings: VecDeque<AcousticMeasurement>,
    spectrum_history: VecDeque<AcousticSpectrum>,
    detected_patterns: Vec<AcousticPattern>,
    baseline_noise_level: Option<f64>,
    last_pattern_time: Option<Instant>,
}

impl AcousticAnalyzer {
    pub fn new(config: AcousticConfig) -> Self {
        Self {
            config,
            acoustic_readings: VecDeque::new(),
            spectrum_history: VecDeque::new(),
            detected_patterns: Vec::new(),
            baseline_noise_level: None,
            last_pattern_time: None,
        }
    }

    /// Add acoustic measurement
    pub fn add_measurement(&mut self, measurement: AcousticMeasurement) -> MdtecResult<()> {
        // Validate measurement
        if measurement.spl_db < 0.0 || measurement.spl_db > 140.0 {
            return Err(MdtecError::InvalidInput(format!(
                "SPL {} dB is out of valid range (0-140 dB)",
                measurement.spl_db
            )));
        }

        if measurement.frequency < 1.0 || measurement.frequency > 50000.0 {
            return Err(MdtecError::InvalidInput(format!(
                "Frequency {} Hz is out of valid range (1-50000 Hz)",
                measurement.frequency
            )));
        }

        self.acoustic_readings.push_back(measurement);

        // Maintain window size
        while self.acoustic_readings.len() > self.config.sample_window_size {
            self.acoustic_readings.pop_front();
        }

        // Update baseline noise level
        self.update_baseline_noise_level();

        // Pattern recognition
        if self.config.enable_pattern_recognition {
            self.detect_patterns(&measurement)?;
        }

        Ok(())
    }

    /// Add acoustic spectrum
    pub fn add_spectrum(&mut self, spectrum: AcousticSpectrum) -> MdtecResult<()> {
        if !self.config.enable_spectrum_analysis {
            return Ok(());
        }

        if spectrum.frequencies.len() != spectrum.amplitudes.len() {
            return Err(MdtecError::InvalidInput(
                "Frequency and amplitude arrays must have same length".to_string(),
            ));
        }

        self.spectrum_history.push_back(spectrum);

        // Maintain window size
        while self.spectrum_history.len() > self.config.spectrum_window_size {
            self.spectrum_history.pop_front();
        }

        Ok(())
    }

    /// Update baseline noise level
    fn update_baseline_noise_level(&mut self) {
        if self.acoustic_readings.len() < 50 {
            return;
        }

        let recent_readings: Vec<_> = self.acoustic_readings.iter().rev().take(50).collect();
        let spl_values: Vec<f64> = recent_readings.iter().map(|r| r.spl_db).collect();
        
        // Use 10th percentile as baseline noise
        let mut sorted_spl = spl_values.clone();
        sorted_spl.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let baseline_index = (sorted_spl.len() as f64 * 0.1) as usize;
        
        if baseline_index < sorted_spl.len() {
            self.baseline_noise_level = Some(sorted_spl[baseline_index]);
        }
    }

    /// Detect acoustic patterns
    fn detect_patterns(&mut self, measurement: &AcousticMeasurement) -> MdtecResult<()> {
        if self.acoustic_readings.len() < 20 {
            return Ok(());
        }

        let recent_readings: Vec<_> = self.acoustic_readings.iter().rev().take(20).collect();
        
        // Detect tone patterns
        if let Some(pattern) = self.detect_tone_pattern(&recent_readings) {
            self.detected_patterns.push(pattern);
            self.last_pattern_time = Some(measurement.timestamp);
        }

        // Detect periodic patterns
        if let Some(pattern) = self.detect_periodic_pattern(&recent_readings) {
            self.detected_patterns.push(pattern);
            self.last_pattern_time = Some(measurement.timestamp);
        }

        // Detect transient patterns
        if let Some(pattern) = self.detect_transient_pattern(&recent_readings) {
            self.detected_patterns.push(pattern);
            self.last_pattern_time = Some(measurement.timestamp);
        }

        // Limit pattern history
        if self.detected_patterns.len() > 100 {
            self.detected_patterns.drain(0..50);
        }

        Ok(())
    }

    /// Detect tone patterns
    fn detect_tone_pattern(&self, readings: &[&AcousticMeasurement]) -> Option<AcousticPattern> {
        if readings.len() < 10 {
            return None;
        }

        let frequencies: Vec<f64> = readings.iter().map(|r| r.frequency).collect();
        let freq_std = Statistics::std_dev(&frequencies).unwrap_or(0.0);
        let freq_mean = Statistics::mean(&frequencies);

        // Check for stable frequency (low standard deviation)
        if freq_std < freq_mean * 0.05 && freq_mean > 100.0 {
            return Some(AcousticPattern {
                pattern_type: AcousticPatternType::Tone,
                confidence: 1.0 - (freq_std / freq_mean).min(1.0),
                dominant_frequency: freq_mean,
                duration: Duration::from_millis(readings.len() as u64 * 10),
                timestamp: readings[0].timestamp,
            });
        }

        None
    }

    /// Detect periodic patterns
    fn detect_periodic_pattern(&self, readings: &[&AcousticMeasurement]) -> Option<AcousticPattern> {
        if readings.len() < 15 {
            return None;
        }

        let spl_values: Vec<f64> = readings.iter().map(|r| r.spl_db).collect();
        
        // Simple autocorrelation-based periodicity detection
        let autocorr = self.calculate_autocorrelation(&spl_values);
        let max_autocorr = autocorr.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);

        if *max_autocorr > 0.7 {
            let freq_mean = Statistics::mean(&readings.iter().map(|r| r.frequency).collect::<Vec<_>>());
            return Some(AcousticPattern {
                pattern_type: AcousticPatternType::Periodic,
                confidence: *max_autocorr,
                dominant_frequency: freq_mean,
                duration: Duration::from_millis(readings.len() as u64 * 10),
                timestamp: readings[0].timestamp,
            });
        }

        None
    }

    /// Detect transient patterns
    fn detect_transient_pattern(&self, readings: &[&AcousticMeasurement]) -> Option<AcousticPattern> {
        if readings.len() < 5 {
            return None;
        }

        let spl_values: Vec<f64> = readings.iter().map(|r| r.spl_db).collect();
        let spl_mean = Statistics::mean(&spl_values);
        let spl_std = Statistics::std_dev(&spl_values).unwrap_or(0.0);

        // Check for high variability (indicating transient)
        if spl_std > 10.0 && spl_mean > 40.0 {
            let freq_mean = Statistics::mean(&readings.iter().map(|r| r.frequency).collect::<Vec<_>>());
            return Some(AcousticPattern {
                pattern_type: AcousticPatternType::Transient,
                confidence: (spl_std / 20.0).min(1.0),
                dominant_frequency: freq_mean,
                duration: Duration::from_millis(readings.len() as u64 * 10),
                timestamp: readings[0].timestamp,
            });
        }

        None
    }

    /// Calculate autocorrelation
    fn calculate_autocorrelation(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        let mut autocorr = vec![0.0; n / 2];

        for lag in 0..n / 2 {
            let mut sum = 0.0;
            let mut count = 0;

            for i in 0..n - lag {
                sum += signal[i] * signal[i + lag];
                count += 1;
            }

            if count > 0 {
                autocorr[lag] = sum / count as f64;
            }
        }

        autocorr
    }

    /// Calculate acoustic entropy
    pub fn calculate_acoustic_entropy(&self) -> MdtecResult<f64> {
        if self.acoustic_readings.len() < 10 {
            return Err(MdtecError::InsufficientData("Not enough acoustic readings".to_string()));
        }

        let mut entropy = 0.0;

        // SPL entropy (40% weight)
        let spl_entropy = self.calculate_spl_entropy()?;
        entropy += spl_entropy * 0.4;

        // Frequency entropy (35% weight)
        let freq_entropy = self.calculate_frequency_entropy()?;
        entropy += freq_entropy * 0.35;

        // Spectral entropy (25% weight)
        if self.config.enable_spectrum_analysis {
            if let Some(spec_entropy) = self.calculate_spectral_entropy()? {
                entropy += spec_entropy * 0.25;
            }
        }

        Ok(entropy.min(1.0))
    }

    /// Calculate SPL entropy
    fn calculate_spl_entropy(&self) -> MdtecResult<f64> {
        let spl_values: Vec<f64> = self.acoustic_readings.iter().map(|r| r.spl_db).collect();
        
        if spl_values.is_empty() {
            return Ok(0.0);
        }

        let spl_variations: Vec<f64> = spl_values
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .collect();

        if spl_variations.is_empty() {
            return Ok(0.0);
        }

        let entropy = Statistics::shannon_entropy(&spl_variations);
        Ok(entropy)
    }

    /// Calculate frequency entropy
    fn calculate_frequency_entropy(&self) -> MdtecResult<f64> {
        let frequencies: Vec<f64> = self.acoustic_readings.iter().map(|r| r.frequency).collect();
        
        if frequencies.is_empty() {
            return Ok(0.0);
        }

        let entropy = Statistics::shannon_entropy(&frequencies);
        Ok(entropy)
    }

    /// Calculate spectral entropy
    fn calculate_spectral_entropy(&self) -> MdtecResult<Option<f64>> {
        if self.spectrum_history.is_empty() {
            return Ok(None);
        }

        let mut combined_amplitudes = Vec::new();
        for spectrum in &self.spectrum_history {
            combined_amplitudes.extend(&spectrum.amplitudes);
        }

        if combined_amplitudes.is_empty() {
            return Ok(None);
        }

        let entropy = Statistics::shannon_entropy(&combined_amplitudes);
        Ok(Some(entropy))
    }

    /// Analyze frequency bands
    pub fn analyze_frequency_bands(&self) -> Vec<(String, f64)> {
        let mut band_powers = Vec::new();

        for (i, (low_freq, high_freq)) in self.config.frequency_bands.iter().enumerate() {
            let band_name = format!("band_{}_{}_{}Hz", i, low_freq as u32, high_freq as u32);
            
            let band_readings: Vec<_> = self.acoustic_readings
                .iter()
                .filter(|r| r.frequency >= *low_freq && r.frequency <= *high_freq)
                .collect();

            if !band_readings.is_empty() {
                let avg_spl = Statistics::mean(&band_readings.iter().map(|r| r.spl_db).collect::<Vec<_>>());
                band_powers.push((band_name, avg_spl));
            }
        }

        band_powers
    }

    /// Get current acoustic measurement
    pub fn get_measurement(&self) -> MdtecResult<DimensionMeasurement> {
        let entropy = self.calculate_acoustic_entropy()?;
        let quality = self.calculate_measurement_quality();

        let mut metadata = std::collections::HashMap::new();
        
        if let Some(measurement) = self.acoustic_readings.back() {
            metadata.insert("spl_db".to_string(), measurement.spl_db.to_string());
            metadata.insert("frequency".to_string(), measurement.frequency.to_string());
            metadata.insert("intensity".to_string(), measurement.intensity.to_string());
        }

        metadata.insert("sample_count".to_string(), self.acoustic_readings.len().to_string());
        metadata.insert("spectrum_count".to_string(), self.spectrum_history.len().to_string());
        metadata.insert("detected_patterns".to_string(), self.detected_patterns.len().to_string());

        if let Some(baseline) = self.baseline_noise_level {
            metadata.insert("baseline_noise_level".to_string(), baseline.to_string());
        }

        if let Some(last_pattern) = self.last_pattern_time {
            metadata.insert("last_pattern_age".to_string(), last_pattern.elapsed().as_secs().to_string());
        }

        // Add frequency band analysis
        let band_analysis = self.analyze_frequency_bands();
        for (band_name, avg_spl) in band_analysis {
            metadata.insert(band_name, avg_spl.to_string());
        }

        Ok(DimensionMeasurement {
            dimension_type: DimensionType::Acoustic,
            value: entropy,
            quality,
            timestamp: SystemTime::now(),
            metadata,
        })
    }

    /// Calculate measurement quality
    fn calculate_measurement_quality(&self) -> f64 {
        let mut quality = 0.0;

        // Sample size quality
        let sample_coverage = (self.acoustic_readings.len() as f64 / self.config.sample_window_size as f64).min(1.0);
        quality += sample_coverage * 0.4;

        // Baseline establishment quality
        if self.baseline_noise_level.is_some() {
            quality += 0.3;
        }

        // Spectrum analysis quality
        if self.config.enable_spectrum_analysis {
            let spectrum_coverage = (self.spectrum_history.len() as f64 / self.config.spectrum_window_size as f64).min(1.0);
            quality += spectrum_coverage * 0.2;
        } else {
            quality += 0.2;
        }

        // Pattern recognition quality
        if self.config.enable_pattern_recognition {
            quality += 0.1;
        }

        quality.min(1.0)
    }

    /// Get acoustic statistics
    pub fn get_statistics(&self) -> AcousticStatistics {
        let spl_values: Vec<f64> = self.acoustic_readings.iter().map(|r| r.spl_db).collect();
        let frequencies: Vec<f64> = self.acoustic_readings.iter().map(|r| r.frequency).collect();
        let intensities: Vec<f64> = self.acoustic_readings.iter().map(|r| r.intensity).collect();

        AcousticStatistics {
            sample_count: self.acoustic_readings.len(),
            spectrum_count: self.spectrum_history.len(),
            spl_mean: Statistics::mean(&spl_values),
            spl_std: Statistics::std_dev(&spl_values).unwrap_or(0.0),
            frequency_mean: Statistics::mean(&frequencies),
            frequency_std: Statistics::std_dev(&frequencies).unwrap_or(0.0),
            intensity_mean: Statistics::mean(&intensities),
            intensity_std: Statistics::std_dev(&intensities).unwrap_or(0.0),
            detected_patterns: self.detected_patterns.len(),
            baseline_noise_level: self.baseline_noise_level,
            current_entropy: self.calculate_acoustic_entropy().unwrap_or(0.0),
            measurement_quality: self.calculate_measurement_quality(),
        }
    }

    /// Reset acoustic analyzer
    pub fn reset(&mut self) {
        self.acoustic_readings.clear();
        self.spectrum_history.clear();
        self.detected_patterns.clear();
        self.baseline_noise_level = None;
        self.last_pattern_time = None;
    }
}

/// Acoustic measurement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticStatistics {
    pub sample_count: usize,
    pub spectrum_count: usize,
    pub spl_mean: f64,
    pub spl_std: f64,
    pub frequency_mean: f64,
    pub frequency_std: f64,
    pub intensity_mean: f64,
    pub intensity_std: f64,
    pub detected_patterns: usize,
    pub baseline_noise_level: Option<f64>,
    pub current_entropy: f64,
    pub measurement_quality: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acoustic_measurement_creation() {
        let measurement = AcousticMeasurement {
            spl_db: 65.0,
            frequency: 1000.0,
            intensity: 0.001,
            timestamp: Instant::now(),
        };

        assert_eq!(measurement.spl_db, 65.0);
        assert_eq!(measurement.frequency, 1000.0);
        assert_eq!(measurement.intensity, 0.001);
    }

    #[test]
    fn test_acoustic_analyzer_add_measurement() {
        let mut analyzer = AcousticAnalyzer::new(AcousticConfig::default());

        let measurement = AcousticMeasurement {
            spl_db: 65.0,
            frequency: 1000.0,
            intensity: 0.001,
            timestamp: Instant::now(),
        };

        assert!(analyzer.add_measurement(measurement).is_ok());
        assert_eq!(analyzer.acoustic_readings.len(), 1);
    }

    #[test]
    fn test_autocorrelation_calculation() {
        let analyzer = AcousticAnalyzer::new(AcousticConfig::default());
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
        let autocorr = analyzer.calculate_autocorrelation(&signal);
        
        assert!(!autocorr.is_empty());
        assert!(autocorr[0] > 0.0); // Self-correlation should be positive
    }

    #[test]
    fn test_frequency_band_analysis() {
        let mut analyzer = AcousticAnalyzer::new(AcousticConfig::default());

        // Add measurements in different frequency bands
        let measurements = vec![
            AcousticMeasurement {
                spl_db: 60.0,
                frequency: 100.0, // Sub-bass
                intensity: 0.001,
                timestamp: Instant::now(),
            },
            AcousticMeasurement {
                spl_db: 65.0,
                frequency: 1000.0, // Midrange
                intensity: 0.002,
                timestamp: Instant::now(),
            },
            AcousticMeasurement {
                spl_db: 70.0,
                frequency: 10000.0, // Treble
                intensity: 0.003,
                timestamp: Instant::now(),
            },
        ];

        for measurement in measurements {
            analyzer.add_measurement(measurement).unwrap();
        }

        let band_analysis = analyzer.analyze_frequency_bands();
        assert!(!band_analysis.is_empty());
    }
} 