use crate::types::{DimensionMeasurement, DimensionType, MdtecError, MdtecResult};
use crate::utils::math::{Statistics, VectorOps};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

/// Electromagnetic field measurement
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ElectromagneticField {
    /// Electric field strength in V/m
    pub electric_field: f64,
    /// Magnetic field strength in Tesla
    pub magnetic_field: f64,
    /// Frequency in Hz
    pub frequency: f64,
    /// Power density in W/m²
    pub power_density: f64,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

/// Radio frequency spectrum measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RfSpectrum {
    /// Frequency bins in Hz
    pub frequencies: Vec<f64>,
    /// Power levels in dBm
    pub power_levels: Vec<f64>,
    /// Measurement bandwidth in Hz
    pub bandwidth: f64,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

/// Electromagnetic anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmAnomalyDetection {
    /// Anomaly threshold multiplier
    pub threshold_multiplier: f64,
    /// Minimum anomaly duration in seconds
    pub min_duration: f64,
    /// Detection sensitivity (0.0 to 1.0)
    pub sensitivity: f64,
}

/// Electromagnetic dimension configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectromagneticConfig {
    /// Sample window size for EM analysis
    pub sample_window_size: usize,
    /// Frequency bands to monitor (Hz)
    pub frequency_bands: Vec<(f64, f64)>,
    /// Enable RF spectrum analysis
    pub enable_rf_spectrum: bool,
    /// Enable magnetic field monitoring
    pub enable_magnetic_field: bool,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Anomaly detection configuration
    pub anomaly_detection: EmAnomalyDetection,
    /// Minimum signal strength threshold (dBm)
    pub min_signal_strength: f64,
}

impl Default for ElectromagneticConfig {
    fn default() -> Self {
        Self {
            sample_window_size: 500,
            frequency_bands: vec![
                (87.5e6, 108.0e6),    // FM radio
                (470.0e6, 862.0e6),   // UHF TV
                (2.4e9, 2.485e9),     // WiFi 2.4GHz
                (5.15e9, 5.825e9),    // WiFi 5GHz
                (0.7e9, 2.6e9),       // Cellular
            ],
            enable_rf_spectrum: true,
            enable_magnetic_field: true,
            enable_anomaly_detection: true,
            anomaly_detection: EmAnomalyDetection {
                threshold_multiplier: 3.0,
                min_duration: 0.1,
                sensitivity: 0.8,
            },
            min_signal_strength: -100.0,
        }
    }
}

/// Electromagnetic dimension analyzer
pub struct ElectromagneticAnalyzer {
    config: ElectromagneticConfig,
    em_readings: VecDeque<ElectromagneticField>,
    rf_spectrum_history: VecDeque<RfSpectrum>,
    baseline_levels: HashMap<String, f64>,
    detected_anomalies: Vec<EmAnomaly>,
    last_anomaly_time: Option<Instant>,
}

/// Electromagnetic anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmAnomaly {
    /// Anomaly type
    pub anomaly_type: EmAnomalyType,
    /// Strength of anomaly (0.0 to 1.0)
    pub strength: f64,
    /// Frequency associated with anomaly
    pub frequency: f64,
    /// Duration of anomaly
    pub duration: Duration,
    /// Timestamp when detected
    pub timestamp: Instant,
}

/// Types of electromagnetic anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmAnomalyType {
    /// Sudden spike in power
    PowerSpike,
    /// Unusual frequency pattern
    FrequencyAnomaly,
    /// Magnetic field disturbance
    MagneticDisturbance,
    /// Broadband interference
    BroadbandInterference,
    /// Signal jamming
    SignalJamming,
}

impl ElectromagneticAnalyzer {
    pub fn new(config: ElectromagneticConfig) -> Self {
        Self {
            config,
            em_readings: VecDeque::new(),
            rf_spectrum_history: VecDeque::new(),
            baseline_levels: HashMap::new(),
            detected_anomalies: Vec::new(),
            last_anomaly_time: None,
        }
    }

    /// Add electromagnetic field measurement
    pub fn add_em_reading(&mut self, reading: ElectromagneticField) -> MdtecResult<()> {
        // Validate reading
        if reading.electric_field < 0.0 || reading.electric_field > 1000.0 {
            return Err(MdtecError::InvalidInput(format!(
                "Electric field {} V/m is out of valid range",
                reading.electric_field
            )));
        }

        if reading.magnetic_field < 0.0 || reading.magnetic_field > 1.0 {
            return Err(MdtecError::InvalidInput(format!(
                "Magnetic field {} T is out of valid range",
                reading.magnetic_field
            )));
        }

        self.em_readings.push_back(reading);

        // Maintain window size
        while self.em_readings.len() > self.config.sample_window_size {
            self.em_readings.pop_front();
        }

        // Update baseline levels
        self.update_baseline_levels();

        // Detect anomalies
        if self.config.enable_anomaly_detection {
            self.detect_anomalies(&reading)?;
        }

        Ok(())
    }

    /// Add RF spectrum measurement
    pub fn add_rf_spectrum(&mut self, spectrum: RfSpectrum) -> MdtecResult<()> {
        if !self.config.enable_rf_spectrum {
            return Ok(());
        }

        if spectrum.frequencies.len() != spectrum.power_levels.len() {
            return Err(MdtecError::InvalidInput(
                "Frequency and power level arrays must have same length".to_string(),
            ));
        }

        self.rf_spectrum_history.push_back(spectrum);

        // Maintain window size
        while self.rf_spectrum_history.len() > self.config.sample_window_size / 10 {
            self.rf_spectrum_history.pop_front();
        }

        Ok(())
    }

    /// Update baseline electromagnetic levels
    fn update_baseline_levels(&mut self) {
        if self.em_readings.len() < 10 {
            return;
        }

        let recent_readings: Vec<_> = self.em_readings.iter().rev().take(50).collect();
        
        let electric_fields: Vec<f64> = recent_readings.iter().map(|r| r.electric_field).collect();
        let magnetic_fields: Vec<f64> = recent_readings.iter().map(|r| r.magnetic_field).collect();
        let power_densities: Vec<f64> = recent_readings.iter().map(|r| r.power_density).collect();

        self.baseline_levels.insert("electric_field".to_string(), Statistics::mean(&electric_fields));
        self.baseline_levels.insert("magnetic_field".to_string(), Statistics::mean(&magnetic_fields));
        self.baseline_levels.insert("power_density".to_string(), Statistics::mean(&power_densities));
    }

    /// Detect electromagnetic anomalies
    fn detect_anomalies(&mut self, reading: &ElectromagneticField) -> MdtecResult<()> {
        let threshold = self.config.anomaly_detection.threshold_multiplier;
        
        // Check for electric field anomalies
        if let Some(baseline_electric) = self.baseline_levels.get("electric_field") {
            if reading.electric_field > baseline_electric * threshold {
                let anomaly = EmAnomaly {
                    anomaly_type: EmAnomalyType::PowerSpike,
                    strength: (reading.electric_field / baseline_electric - 1.0).min(1.0),
                    frequency: reading.frequency,
                    duration: Duration::from_millis(100), // Initial duration
                    timestamp: reading.timestamp,
                };
                self.detected_anomalies.push(anomaly);
                self.last_anomaly_time = Some(reading.timestamp);
            }
        }

        // Check for magnetic field anomalies
        if let Some(baseline_magnetic) = self.baseline_levels.get("magnetic_field") {
            if reading.magnetic_field > baseline_magnetic * threshold {
                let anomaly = EmAnomaly {
                    anomaly_type: EmAnomalyType::MagneticDisturbance,
                    strength: (reading.magnetic_field / baseline_magnetic - 1.0).min(1.0),
                    frequency: reading.frequency,
                    duration: Duration::from_millis(100),
                    timestamp: reading.timestamp,
                };
                self.detected_anomalies.push(anomaly);
                self.last_anomaly_time = Some(reading.timestamp);
            }
        }

        // Limit anomaly history
        if self.detected_anomalies.len() > 100 {
            self.detected_anomalies.drain(0..50);
        }

        Ok(())
    }

    /// Calculate electromagnetic entropy
    pub fn calculate_em_entropy(&self) -> MdtecResult<f64> {
        if self.em_readings.len() < 10 {
            return Err(MdtecError::InsufficientData("Not enough EM readings".to_string()));
        }

        let mut entropy = 0.0;

        // Electric field entropy (30% weight)
        let electric_entropy = self.calculate_electric_field_entropy()?;
        entropy += electric_entropy * 0.3;

        // Magnetic field entropy (30% weight)
        let magnetic_entropy = self.calculate_magnetic_field_entropy()?;
        entropy += magnetic_entropy * 0.3;

        // Frequency entropy (25% weight)
        let frequency_entropy = self.calculate_frequency_entropy()?;
        entropy += frequency_entropy * 0.25;

        // RF spectrum entropy (15% weight)
        if self.config.enable_rf_spectrum {
            if let Some(spectrum_entropy) = self.calculate_rf_spectrum_entropy()? {
                entropy += spectrum_entropy * 0.15;
            }
        }

        Ok(entropy.min(1.0))
    }

    /// Calculate electric field entropy
    fn calculate_electric_field_entropy(&self) -> MdtecResult<f64> {
        let electric_fields: Vec<f64> = self.em_readings.iter().map(|r| r.electric_field).collect();
        
        if electric_fields.is_empty() {
            return Ok(0.0);
        }

        let field_variations: Vec<f64> = electric_fields
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .collect();

        if field_variations.is_empty() {
            return Ok(0.0);
        }

        let entropy = Statistics::shannon_entropy(&field_variations);
        Ok(entropy)
    }

    /// Calculate magnetic field entropy
    fn calculate_magnetic_field_entropy(&self) -> MdtecResult<f64> {
        let magnetic_fields: Vec<f64> = self.em_readings.iter().map(|r| r.magnetic_field).collect();
        
        if magnetic_fields.is_empty() {
            return Ok(0.0);
        }

        let field_variations: Vec<f64> = magnetic_fields
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .collect();

        if field_variations.is_empty() {
            return Ok(0.0);
        }

        let entropy = Statistics::shannon_entropy(&field_variations);
        Ok(entropy)
    }

    /// Calculate frequency entropy
    fn calculate_frequency_entropy(&self) -> MdtecResult<f64> {
        let frequencies: Vec<f64> = self.em_readings.iter().map(|r| r.frequency).collect();
        
        if frequencies.is_empty() {
            return Ok(0.0);
        }

        let entropy = Statistics::shannon_entropy(&frequencies);
        Ok(entropy)
    }

    /// Calculate RF spectrum entropy
    fn calculate_rf_spectrum_entropy(&self) -> MdtecResult<Option<f64>> {
        if self.rf_spectrum_history.is_empty() {
            return Ok(None);
        }

        let mut combined_power_levels = Vec::new();
        for spectrum in &self.rf_spectrum_history {
            combined_power_levels.extend(&spectrum.power_levels);
        }

        if combined_power_levels.is_empty() {
            return Ok(None);
        }

        let entropy = Statistics::shannon_entropy(&combined_power_levels);
        Ok(Some(entropy))
    }

    /// Analyze frequency bands
    pub fn analyze_frequency_bands(&self) -> HashMap<String, f64> {
        let mut band_analysis = HashMap::new();

        for (i, (low_freq, high_freq)) in self.config.frequency_bands.iter().enumerate() {
            let band_name = format!("band_{}", i);
            let band_power = self.em_readings
                .iter()
                .filter(|r| r.frequency >= *low_freq && r.frequency <= *high_freq)
                .map(|r| r.power_density)
                .collect::<Vec<_>>();

            if !band_power.is_empty() {
                let avg_power = Statistics::mean(&band_power);
                band_analysis.insert(band_name, avg_power);
            }
        }

        band_analysis
    }

    /// Get current electromagnetic measurement
    pub fn get_measurement(&self) -> MdtecResult<DimensionMeasurement> {
        let entropy = self.calculate_em_entropy()?;
        let quality = self.calculate_measurement_quality();

        let mut metadata = std::collections::HashMap::new();
        
        if let Some(reading) = self.em_readings.back() {
            metadata.insert("electric_field".to_string(), reading.electric_field.to_string());
            metadata.insert("magnetic_field".to_string(), reading.magnetic_field.to_string());
            metadata.insert("frequency".to_string(), reading.frequency.to_string());
            metadata.insert("power_density".to_string(), reading.power_density.to_string());
        }

        metadata.insert("em_sample_count".to_string(), self.em_readings.len().to_string());
        metadata.insert("rf_spectrum_count".to_string(), self.rf_spectrum_history.len().to_string());
        metadata.insert("detected_anomalies".to_string(), self.detected_anomalies.len().to_string());

        if let Some(last_anomaly) = self.last_anomaly_time {
            metadata.insert("last_anomaly_age".to_string(), last_anomaly.elapsed().as_secs().to_string());
        }

        // Add frequency band analysis
        let band_analysis = self.analyze_frequency_bands();
        for (band, power) in band_analysis {
            metadata.insert(format!("power_{}", band), power.to_string());
        }

        Ok(DimensionMeasurement {
            dimension_type: DimensionType::Electromagnetic,
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
        let sample_coverage = (self.em_readings.len() as f64 / self.config.sample_window_size as f64).min(1.0);
        quality += sample_coverage * 0.4;

        // Baseline establishment quality
        let baseline_coverage = (self.baseline_levels.len() as f64 / 3.0).min(1.0);
        quality += baseline_coverage * 0.3;

        // RF spectrum quality
        if self.config.enable_rf_spectrum {
            let spectrum_coverage = (self.rf_spectrum_history.len() as f64 / (self.config.sample_window_size / 10) as f64).min(1.0);
            quality += spectrum_coverage * 0.2;
        } else {
            quality += 0.2;
        }

        // Anomaly detection quality
        if self.config.enable_anomaly_detection {
            quality += 0.1;
        }

        quality.min(1.0)
    }

    /// Get electromagnetic statistics
    pub fn get_statistics(&self) -> ElectromagneticStatistics {
        let electric_fields: Vec<f64> = self.em_readings.iter().map(|r| r.electric_field).collect();
        let magnetic_fields: Vec<f64> = self.em_readings.iter().map(|r| r.magnetic_field).collect();
        let frequencies: Vec<f64> = self.em_readings.iter().map(|r| r.frequency).collect();

        ElectromagneticStatistics {
            em_sample_count: self.em_readings.len(),
            rf_spectrum_count: self.rf_spectrum_history.len(),
            electric_field_mean: Statistics::mean(&electric_fields),
            electric_field_std: Statistics::std_dev(&electric_fields).unwrap_or(0.0),
            magnetic_field_mean: Statistics::mean(&magnetic_fields),
            magnetic_field_std: Statistics::std_dev(&magnetic_fields).unwrap_or(0.0),
            frequency_mean: Statistics::mean(&frequencies),
            frequency_std: Statistics::std_dev(&frequencies).unwrap_or(0.0),
            detected_anomalies: self.detected_anomalies.len(),
            current_entropy: self.calculate_em_entropy().unwrap_or(0.0),
            measurement_quality: self.calculate_measurement_quality(),
        }
    }

    /// Reset electromagnetic analyzer
    pub fn reset(&mut self) {
        self.em_readings.clear();
        self.rf_spectrum_history.clear();
        self.baseline_levels.clear();
        self.detected_anomalies.clear();
        self.last_anomaly_time = None;
    }
}

/// Electromagnetic measurement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectromagneticStatistics {
    pub em_sample_count: usize,
    pub rf_spectrum_count: usize,
    pub electric_field_mean: f64,
    pub electric_field_std: f64,
    pub magnetic_field_mean: f64,
    pub magnetic_field_std: f64,
    pub frequency_mean: f64,
    pub frequency_std: f64,
    pub detected_anomalies: usize,
    pub current_entropy: f64,
    pub measurement_quality: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_electromagnetic_field_creation() {
        let field = ElectromagneticField {
            electric_field: 1.5,
            magnetic_field: 0.0001,
            frequency: 2.4e9,
            power_density: 0.001,
            timestamp: Instant::now(),
        };

        assert_eq!(field.electric_field, 1.5);
        assert_eq!(field.magnetic_field, 0.0001);
        assert_eq!(field.frequency, 2.4e9);
        assert_eq!(field.power_density, 0.001);
    }

    #[test]
    fn test_electromagnetic_analyzer_add_reading() {
        let mut analyzer = ElectromagneticAnalyzer::new(ElectromagneticConfig::default());

        let reading = ElectromagneticField {
            electric_field: 1.5,
            magnetic_field: 0.0001,
            frequency: 2.4e9,
            power_density: 0.001,
            timestamp: Instant::now(),
        };

        assert!(analyzer.add_em_reading(reading).is_ok());
        assert_eq!(analyzer.em_readings.len(), 1);
    }

    #[test]
    fn test_rf_spectrum_analysis() {
        let mut analyzer = ElectromagneticAnalyzer::new(ElectromagneticConfig::default());

        let spectrum = RfSpectrum {
            frequencies: vec![2.4e9, 2.41e9, 2.42e9],
            power_levels: vec![-30.0, -35.0, -40.0],
            bandwidth: 10e6,
            timestamp: Instant::now(),
        };

        assert!(analyzer.add_rf_spectrum(spectrum).is_ok());
        assert_eq!(analyzer.rf_spectrum_history.len(), 1);
    }

    #[test]
    fn test_frequency_band_analysis() {
        let mut analyzer = ElectromagneticAnalyzer::new(ElectromagneticConfig::default());

        // Add readings in different frequency bands
        let readings = vec![
            ElectromagneticField {
                electric_field: 1.0,
                magnetic_field: 0.0001,
                frequency: 2.4e9, // WiFi band
                power_density: 0.001,
                timestamp: Instant::now(),
            },
            ElectromagneticField {
                electric_field: 1.5,
                magnetic_field: 0.0002,
                frequency: 900e6, // Cellular band
                power_density: 0.002,
                timestamp: Instant::now(),
            },
        ];

        for reading in readings {
            analyzer.add_em_reading(reading).unwrap();
        }

        let band_analysis = analyzer.analyze_frequency_bands();
        assert!(!band_analysis.is_empty());
    }
} 