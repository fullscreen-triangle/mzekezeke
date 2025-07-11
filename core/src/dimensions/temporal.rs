use crate::types::{DimensionMeasurement, DimensionType, MdtecError, MdtecResult};
use crate::utils::timing::{PrecisionTimer, TimeSync, TemporalAnalysis};
use crate::utils::math::Statistics;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

/// Temporal measurement point
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TemporalPoint {
    /// System timestamp
    pub system_time: SystemTime,
    /// Monotonic instant
    pub instant: Instant,
    /// Nanosecond precision offset
    pub nano_offset: u64,
    /// Clock drift compensation
    pub drift_compensation: f64,
}

/// Temporal rhythm pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRhythm {
    /// Pattern frequency in Hz
    pub frequency: f64,
    /// Pattern amplitude
    pub amplitude: f64,
    /// Pattern phase offset
    pub phase: f64,
    /// Pattern confidence score
    pub confidence: f64,
}

/// Temporal dimension configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    /// Sample window size for temporal analysis
    pub sample_window_size: usize,
    /// Minimum precision required (nanoseconds)
    pub min_precision: u64,
    /// Maximum clock drift tolerance (milliseconds)
    pub max_drift_tolerance: f64,
    /// Enable high-precision timing
    pub enable_high_precision: bool,
    /// Enable temporal rhythm detection
    pub enable_rhythm_detection: bool,
    /// Temporal entropy calculation method
    pub entropy_method: TemporalEntropyMethod,
}

/// Temporal entropy calculation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalEntropyMethod {
    /// Simple interval-based entropy
    Interval,
    /// Frequency domain entropy
    Frequency,
    /// Multi-scale entropy
    MultiScale,
    /// Approximate entropy
    Approximate,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            sample_window_size: 1000,
            min_precision: 1_000, // 1 microsecond
            max_drift_tolerance: 50.0, // 50ms
            enable_high_precision: true,
            enable_rhythm_detection: true,
            entropy_method: TemporalEntropyMethod::MultiScale,
        }
    }
}

/// Temporal dimension analyzer
pub struct TemporalAnalyzer {
    config: TemporalConfig,
    temporal_points: VecDeque<TemporalPoint>,
    precision_timer: PrecisionTimer,
    time_sync: TimeSync,
    temporal_analysis: TemporalAnalysis,
    detected_rhythms: Vec<TemporalRhythm>,
    last_sync_time: Option<Instant>,
}

impl TemporalAnalyzer {
    pub fn new(config: TemporalConfig) -> Self {
        Self {
            config,
            temporal_points: VecDeque::new(),
            precision_timer: PrecisionTimer::new(),
            time_sync: TimeSync::new(),
            temporal_analysis: TemporalAnalysis::new(),
            detected_rhythms: Vec::new(),
            last_sync_time: None,
        }
    }

    /// Add temporal measurement point
    pub fn add_temporal_point(&mut self) -> MdtecResult<()> {
        let system_time = SystemTime::now();
        let instant = Instant::now();
        let nano_offset = if self.config.enable_high_precision {
            self.precision_timer.elapsed_nanos()
        } else {
            0
        };
        
        let drift_compensation = self.time_sync.get_drift_compensation()
            .unwrap_or(0.0);

        let point = TemporalPoint {
            system_time,
            instant,
            nano_offset,
            drift_compensation,
        };

        self.temporal_points.push_back(point);

        // Maintain window size
        while self.temporal_points.len() > self.config.sample_window_size {
            self.temporal_points.pop_front();
        }

        // Update rhythm detection
        if self.config.enable_rhythm_detection {
            self.update_rhythm_detection()?;
        }

        Ok(())
    }

    /// Calculate temporal entropy
    pub fn calculate_temporal_entropy(&self) -> MdtecResult<f64> {
        if self.temporal_points.len() < 10 {
            return Err(MdtecError::InsufficientData("Not enough temporal points".to_string()));
        }

        match self.config.entropy_method {
            TemporalEntropyMethod::Interval => self.calculate_interval_entropy(),
            TemporalEntropyMethod::Frequency => self.calculate_frequency_entropy(),
            TemporalEntropyMethod::MultiScale => self.calculate_multiscale_entropy(),
            TemporalEntropyMethod::Approximate => self.calculate_approximate_entropy(),
        }
    }

    /// Calculate interval-based entropy
    fn calculate_interval_entropy(&self) -> MdtecResult<f64> {
        let intervals: Vec<f64> = self.temporal_points
            .windows(2)
            .map(|pair| {
                let duration = pair[1].instant.duration_since(pair[0].instant);
                duration.as_nanos() as f64
            })
            .collect();

        if intervals.is_empty() {
            return Ok(0.0);
        }

        let entropy = Statistics::shannon_entropy(&intervals);
        Ok(entropy)
    }

    /// Calculate frequency domain entropy
    fn calculate_frequency_entropy(&self) -> MdtecResult<f64> {
        let intervals: Vec<f64> = self.temporal_points
            .windows(2)
            .map(|pair| {
                let duration = pair[1].instant.duration_since(pair[0].instant);
                duration.as_secs_f64()
            })
            .collect();

        if intervals.len() < 8 {
            return Ok(0.0);
        }

        // Simple frequency analysis - in practice would use FFT
        let frequencies = self.temporal_analysis.detect_frequencies(&intervals)?;
        let entropy = Statistics::shannon_entropy(&frequencies);
        Ok(entropy)
    }

    /// Calculate multi-scale entropy
    fn calculate_multiscale_entropy(&self) -> MdtecResult<f64> {
        let intervals: Vec<f64> = self.temporal_points
            .windows(2)
            .map(|pair| {
                let duration = pair[1].instant.duration_since(pair[0].instant);
                duration.as_nanos() as f64
            })
            .collect();

        if intervals.len() < 20 {
            return Ok(0.0);
        }

        let mut total_entropy = 0.0;
        let scales = [1, 2, 4, 8, 16];

        for &scale in &scales {
            let coarse_grained = self.coarse_grain(&intervals, scale);
            if coarse_grained.len() > 2 {
                let entropy = Statistics::shannon_entropy(&coarse_grained);
                total_entropy += entropy / scale as f64;
            }
        }

        Ok(total_entropy / scales.len() as f64)
    }

    /// Calculate approximate entropy
    fn calculate_approximate_entropy(&self) -> MdtecResult<f64> {
        let intervals: Vec<f64> = self.temporal_points
            .windows(2)
            .map(|pair| {
                let duration = pair[1].instant.duration_since(pair[0].instant);
                duration.as_nanos() as f64
            })
            .collect();

        if intervals.len() < 10 {
            return Ok(0.0);
        }

        // Simplified approximate entropy calculation
        let m = 2; // Pattern length
        let r = Statistics::std_dev(&intervals)? * 0.2; // Tolerance

        let mut phi_m = 0.0;
        let mut phi_m_plus_1 = 0.0;

        let n = intervals.len();
        for i in 0..=n - m {
            let mut count_m = 0;
            let mut count_m_plus_1 = 0;

            for j in 0..=n - m {
                if self.max_distance(&intervals[i..i + m], &intervals[j..j + m]) <= r {
                    count_m += 1;
                    if i < n - m && j < n - m {
                        if self.max_distance(&intervals[i..i + m + 1], &intervals[j..j + m + 1]) <= r {
                            count_m_plus_1 += 1;
                        }
                    }
                }
            }

            if count_m > 0 {
                phi_m += (count_m as f64 / (n - m + 1) as f64).ln();
            }
            if count_m_plus_1 > 0 {
                phi_m_plus_1 += (count_m_plus_1 as f64 / (n - m) as f64).ln();
            }
        }

        phi_m /= (n - m + 1) as f64;
        phi_m_plus_1 /= (n - m) as f64;

        Ok(phi_m - phi_m_plus_1)
    }

    /// Coarse grain time series for multi-scale entropy
    fn coarse_grain(&self, series: &[f64], scale: usize) -> Vec<f64> {
        let mut coarse_grained = Vec::new();
        for i in (0..series.len()).step_by(scale) {
            let end = (i + scale).min(series.len());
            let mean = series[i..end].iter().sum::<f64>() / (end - i) as f64;
            coarse_grained.push(mean);
        }
        coarse_grained
    }

    /// Calculate maximum distance between two sequences
    fn max_distance(&self, seq1: &[f64], seq2: &[f64]) -> f64 {
        seq1.iter()
            .zip(seq2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    /// Update rhythm detection
    fn update_rhythm_detection(&mut self) -> MdtecResult<()> {
        if self.temporal_points.len() < 50 {
            return Ok(());
        }

        let intervals: Vec<f64> = self.temporal_points
            .windows(2)
            .map(|pair| {
                let duration = pair[1].instant.duration_since(pair[0].instant);
                duration.as_secs_f64()
            })
            .collect();

        // Detect rhythmic patterns
        self.detected_rhythms = self.temporal_analysis.detect_rhythms(&intervals)?;

        Ok(())
    }

    /// Get current temporal measurement
    pub fn get_measurement(&self) -> MdtecResult<DimensionMeasurement> {
        let entropy = self.calculate_temporal_entropy()?;
        let quality = self.calculate_measurement_quality();

        let mut metadata = std::collections::HashMap::new();
        
        metadata.insert("sample_count".to_string(), self.temporal_points.len().to_string());
        metadata.insert("entropy_method".to_string(), format!("{:?}", self.config.entropy_method));
        
        if let Some(point) = self.temporal_points.back() {
            metadata.insert("drift_compensation".to_string(), point.drift_compensation.to_string());
            metadata.insert("nano_precision".to_string(), point.nano_offset.to_string());
        }

        if !self.detected_rhythms.is_empty() {
            metadata.insert("detected_rhythms".to_string(), self.detected_rhythms.len().to_string());
            if let Some(rhythm) = self.detected_rhythms.first() {
                metadata.insert("dominant_frequency".to_string(), rhythm.frequency.to_string());
                metadata.insert("rhythm_confidence".to_string(), rhythm.confidence.to_string());
            }
        }

        if let Some(sync_time) = self.last_sync_time {
            metadata.insert("last_sync_age".to_string(), sync_time.elapsed().as_secs().to_string());
        }

        Ok(DimensionMeasurement {
            dimension_type: DimensionType::Temporal,
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
        let sample_coverage = (self.temporal_points.len() as f64 / self.config.sample_window_size as f64).min(1.0);
        quality += sample_coverage * 0.4;

        // Precision quality
        if self.config.enable_high_precision {
            quality += 0.3;
        }

        // Drift tolerance quality
        if let Some(point) = self.temporal_points.back() {
            let drift_quality = 1.0 - (point.drift_compensation.abs() / self.config.max_drift_tolerance).min(1.0);
            quality += drift_quality * 0.2;
        }

        // Rhythm detection quality
        if !self.detected_rhythms.is_empty() {
            let avg_confidence = self.detected_rhythms.iter()
                .map(|r| r.confidence)
                .sum::<f64>() / self.detected_rhythms.len() as f64;
            quality += avg_confidence * 0.1;
        }

        quality.min(1.0)
    }

    /// Synchronize with external time source
    pub fn sync_time(&mut self) -> MdtecResult<()> {
        self.time_sync.sync_with_external_source()?;
        self.last_sync_time = Some(Instant::now());
        Ok(())
    }

    /// Reset temporal analyzer
    pub fn reset(&mut self) {
        self.temporal_points.clear();
        self.detected_rhythms.clear();
        self.last_sync_time = None;
        self.precision_timer.reset();
    }

    /// Get temporal statistics
    pub fn get_statistics(&self) -> TemporalStatistics {
        TemporalStatistics {
            total_points: self.temporal_points.len(),
            current_entropy: self.calculate_temporal_entropy().unwrap_or(0.0),
            measurement_quality: self.calculate_measurement_quality(),
            detected_rhythms: self.detected_rhythms.len(),
            high_precision_enabled: self.config.enable_high_precision,
            last_sync_age: self.last_sync_time.map(|t| t.elapsed()),
        }
    }
}

/// Temporal measurement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalStatistics {
    pub total_points: usize,
    pub current_entropy: f64,
    pub measurement_quality: f64,
    pub detected_rhythms: usize,
    pub high_precision_enabled: bool,
    pub last_sync_age: Option<Duration>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_temporal_point_creation() {
        let point = TemporalPoint {
            system_time: SystemTime::now(),
            instant: Instant::now(),
            nano_offset: 1000,
            drift_compensation: 0.5,
        };

        assert_eq!(point.nano_offset, 1000);
        assert_eq!(point.drift_compensation, 0.5);
    }

    #[test]
    fn test_temporal_analyzer_basic() {
        let mut analyzer = TemporalAnalyzer::new(TemporalConfig::default());

        // Add some temporal points
        for _ in 0..10 {
            analyzer.add_temporal_point().unwrap();
            thread::sleep(Duration::from_millis(1));
        }

        assert_eq!(analyzer.temporal_points.len(), 10);
    }

    #[test]
    fn test_interval_entropy() {
        let mut analyzer = TemporalAnalyzer::new(TemporalConfig {
            entropy_method: TemporalEntropyMethod::Interval,
            ..Default::default()
        });

        // Add temporal points with varying intervals
        for i in 0..20 {
            analyzer.add_temporal_point().unwrap();
            thread::sleep(Duration::from_millis(i % 5 + 1));
        }

        let entropy = analyzer.calculate_temporal_entropy().unwrap();
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_coarse_graining() {
        let analyzer = TemporalAnalyzer::new(TemporalConfig::default());
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let coarse_grained = analyzer.coarse_grain(&series, 2);
        
        assert_eq!(coarse_grained.len(), 3);
        assert_eq!(coarse_grained[0], 1.5); // (1+2)/2
        assert_eq!(coarse_grained[1], 3.5); // (3+4)/2
        assert_eq!(coarse_grained[2], 5.5); // (5+6)/2
    }
} 