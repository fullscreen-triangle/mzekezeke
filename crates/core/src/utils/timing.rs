//! Timing utilities for MDTEC
//!
//! This module provides high-precision timing functions, time synchronization,
//! and temporal analysis utilities for the MDTEC system.

use crate::types::Timestamp;
use crate::error::{Error, Result};
use std::time::{Duration, SystemTime, UNIX_EPOCH, Instant};
use std::collections::VecDeque;

/// High-precision timer for MDTEC operations
pub struct PrecisionTimer {
    /// Start time reference
    start_time: Instant,
    /// Accumulated elapsed time
    accumulated_time: Duration,
    /// Whether timer is currently running
    is_running: bool,
}

/// Time synchronization utilities
pub struct TimeSync;

/// Temporal analysis utilities
pub struct TemporalAnalysis;

/// Time window utilities
pub struct TimeWindow;

/// Clock drift detector and compensator
pub struct ClockDriftDetector {
    /// Reference timestamps from external sources
    reference_points: VecDeque<TimestampReference>,
    /// Maximum number of reference points to keep
    max_references: usize,
    /// Detected drift rate (seconds per second)
    drift_rate: f64,
    /// Last calibration time
    last_calibration: Instant,
}

/// Reference timestamp for drift detection
#[derive(Debug, Clone)]
pub struct TimestampReference {
    /// Local timestamp when reference was obtained
    pub local_time: Instant,
    /// External reference timestamp
    pub reference_time: Timestamp,
    /// Confidence in this reference
    pub confidence: f64,
}

/// Time synchronization result
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Synchronization offset in milliseconds
    pub offset_ms: i64,
    /// Round-trip time in milliseconds
    pub rtt_ms: u64,
    /// Synchronization accuracy estimate
    pub accuracy_ms: f64,
    /// Sync quality score (0.0 to 1.0)
    pub quality: f64,
}

/// Temporal statistics
#[derive(Debug, Clone)]
pub struct TemporalStats {
    /// Mean time interval
    pub mean_interval: f64,
    /// Standard deviation of intervals
    pub std_deviation: f64,
    /// Minimum interval
    pub min_interval: f64,
    /// Maximum interval
    pub max_interval: f64,
    /// Jitter (variation in intervals)
    pub jitter: f64,
    /// Temporal stability score
    pub stability: f64,
}

impl PrecisionTimer {
    /// Create a new precision timer
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            accumulated_time: Duration::ZERO,
            is_running: false,
        }
    }

    /// Start the timer
    pub fn start(&mut self) {
        if !self.is_running {
            self.start_time = Instant::now();
            self.is_running = true;
        }
    }

    /// Stop the timer and accumulate elapsed time
    pub fn stop(&mut self) -> Duration {
        if self.is_running {
            let elapsed = self.start_time.elapsed();
            self.accumulated_time += elapsed;
            self.is_running = false;
            elapsed
        } else {
            Duration::ZERO
        }
    }

    /// Reset the timer
    pub fn reset(&mut self) {
        self.accumulated_time = Duration::ZERO;
        self.is_running = false;
    }

    /// Get elapsed time (including accumulated time)
    pub fn elapsed(&self) -> Duration {
        if self.is_running {
            self.accumulated_time + self.start_time.elapsed()
        } else {
            self.accumulated_time
        }
    }

    /// Get elapsed time in nanoseconds
    pub fn elapsed_nanos(&self) -> u128 {
        self.elapsed().as_nanos()
    }

    /// Get elapsed time in microseconds
    pub fn elapsed_micros(&self) -> u128 {
        self.elapsed().as_micros()
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_millis(&self) -> u128 {
        self.elapsed().as_millis()
    }

    /// Get elapsed time in seconds as f64
    pub fn elapsed_secs_f64(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }

    /// Check if timer is running
    pub fn is_running(&self) -> bool {
        self.is_running
    }
}

impl TimeSync {
    /// Get current system timestamp in milliseconds since UNIX epoch
    pub fn current_timestamp() -> Timestamp {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as Timestamp
    }

    /// Get high-precision timestamp in nanoseconds
    pub fn current_timestamp_nanos() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos()
    }

    /// Convert timestamp to system time
    pub fn timestamp_to_system_time(timestamp: Timestamp) -> SystemTime {
        UNIX_EPOCH + Duration::from_millis(timestamp)
    }

    /// Calculate time difference in milliseconds
    pub fn time_diff_ms(time1: Timestamp, time2: Timestamp) -> i64 {
        time1 as i64 - time2 as i64
    }

    /// Check if timestamp is within tolerance of current time
    pub fn is_time_valid(timestamp: Timestamp, tolerance_ms: u64) -> bool {
        let now = Self::current_timestamp();
        let diff = Self::time_diff_ms(now, timestamp).unsigned_abs();
        diff <= tolerance_ms
    }

    /// Synchronize with external time source
    pub fn sync_with_external(external_timestamp: Timestamp) -> SyncResult {
        let local_time = Self::current_timestamp();
        let offset = Self::time_diff_ms(external_timestamp, local_time);
        
        // Simple sync result - in practice this would involve more sophisticated NTP-like protocol
        SyncResult {
            offset_ms: offset,
            rtt_ms: 0, // Would be measured in real implementation
            accuracy_ms: 10.0, // Estimated accuracy
            quality: if offset.abs() < 1000 { 0.9 } else { 0.5 },
        }
    }

    /// Get UTC timestamp
    pub fn utc_timestamp() -> Timestamp {
        Self::current_timestamp()
    }

    /// Format timestamp as ISO 8601 string
    pub fn format_timestamp(timestamp: Timestamp) -> String {
        let system_time = Self::timestamp_to_system_time(timestamp);
        let datetime = chrono::DateTime::<chrono::Utc>::from(system_time);
        datetime.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
    }

    /// Parse ISO 8601 timestamp string
    pub fn parse_timestamp(timestamp_str: &str) -> Result<Timestamp> {
        let datetime = chrono::DateTime::parse_from_rfc3339(timestamp_str)
            .map_err(|e| Error::invalid_input(&format!("Invalid timestamp format: {}", e)))?;
        
        let system_time: SystemTime = datetime.into();
        let duration = system_time.duration_since(UNIX_EPOCH)
            .map_err(|e| Error::invalid_input(&format!("Invalid timestamp: {}", e)))?;
        
        Ok(duration.as_millis() as Timestamp)
    }
}

impl TemporalAnalysis {
    /// Analyze temporal patterns in a series of timestamps
    pub fn analyze_timestamps(timestamps: &[Timestamp]) -> Result<TemporalStats> {
        if timestamps.len() < 2 {
            return Err(Error::invalid_input("Need at least 2 timestamps for analysis"));
        }

        // Calculate intervals between consecutive timestamps
        let intervals: Vec<f64> = timestamps.windows(2)
            .map(|pair| (pair[1] as f64 - pair[0] as f64))
            .collect();

        let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
        
        let variance = intervals.iter()
            .map(|&interval| (interval - mean_interval).powi(2))
            .sum::<f64>() / intervals.len() as f64;
        
        let std_deviation = variance.sqrt();
        let min_interval = intervals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_interval = intervals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Calculate jitter as coefficient of variation
        let jitter = if mean_interval > 0.0 {
            std_deviation / mean_interval
        } else {
            0.0
        };

        // Calculate stability as inverse of jitter (normalized)
        let stability = 1.0 / (1.0 + jitter);

        Ok(TemporalStats {
            mean_interval,
            std_deviation,
            min_interval,
            max_interval,
            jitter,
            stability,
        })
    }

    /// Detect temporal outliers in timestamp series
    pub fn detect_outliers(timestamps: &[Timestamp], threshold_sigma: f64) -> Result<Vec<usize>> {
        if timestamps.len() < 3 {
            return Ok(Vec::new());
        }

        let stats = Self::analyze_timestamps(timestamps)?;
        let mut outliers = Vec::new();

        let intervals: Vec<f64> = timestamps.windows(2)
            .map(|pair| (pair[1] as f64 - pair[0] as f64))
            .collect();

        for (i, &interval) in intervals.iter().enumerate() {
            let z_score = (interval - stats.mean_interval) / stats.std_deviation;
            if z_score.abs() > threshold_sigma {
                outliers.push(i + 1); // +1 because outlier is the second timestamp in the pair
            }
        }

        Ok(outliers)
    }

    /// Calculate temporal entropy of timestamp series
    pub fn temporal_entropy(timestamps: &[Timestamp], bin_count: usize) -> Result<f64> {
        if timestamps.len() < 2 {
            return Err(Error::invalid_input("Need at least 2 timestamps"));
        }

        let intervals: Vec<f64> = timestamps.windows(2)
            .map(|pair| (pair[1] as f64 - pair[0] as f64))
            .collect();

        // Create histogram bins
        let min_interval = intervals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_interval = intervals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_interval == min_interval {
            return Ok(0.0); // No variation, zero entropy
        }

        let bin_width = (max_interval - min_interval) / bin_count as f64;
        let mut bins = vec![0; bin_count];

        // Populate bins
        for &interval in &intervals {
            let bin_index = ((interval - min_interval) / bin_width).floor() as usize;
            let bin_index = bin_index.min(bin_count - 1);
            bins[bin_index] += 1;
        }

        // Calculate entropy
        let total_count = intervals.len() as f64;
        let mut entropy = 0.0;

        for &count in &bins {
            if count > 0 {
                let probability = count as f64 / total_count;
                entropy -= probability * probability.log2();
            }
        }

        Ok(entropy)
    }

    /// Find periodic patterns in timestamps
    pub fn find_periodicity(timestamps: &[Timestamp]) -> Result<Option<f64>> {
        if timestamps.len() < 10 {
            return Ok(None);
        }

        let intervals: Vec<f64> = timestamps.windows(2)
            .map(|pair| (pair[1] as f64 - pair[0] as f64))
            .collect();

        // Simple autocorrelation-based period detection
        let max_lag = intervals.len() / 4;
        let mut best_correlation = 0.0;
        let mut best_period = None;

        for lag in 1..=max_lag {
            let correlation = Self::autocorrelation(&intervals, lag)?;
            if correlation > best_correlation && correlation > 0.5 {
                best_correlation = correlation;
                best_period = Some(lag as f64);
            }
        }

        Ok(best_period)
    }

    /// Calculate autocorrelation at given lag
    fn autocorrelation(data: &[f64], lag: usize) -> Result<f64> {
        if lag >= data.len() {
            return Ok(0.0);
        }

        let n = data.len() - lag;
        let mean = data.iter().sum::<f64>() / data.len() as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            let x_i = data[i] - mean;
            let x_i_lag = data[i + lag] - mean;
            numerator += x_i * x_i_lag;
        }

        for &value in data {
            let x_i = value - mean;
            denominator += x_i * x_i;
        }

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
}

impl TimeWindow {
    /// Check if timestamp falls within time window
    pub fn contains(window_start: Timestamp, window_end: Timestamp, timestamp: Timestamp) -> bool {
        timestamp >= window_start && timestamp <= window_end
    }

    /// Get window duration in milliseconds
    pub fn duration(window_start: Timestamp, window_end: Timestamp) -> u64 {
        if window_end >= window_start {
            window_end - window_start
        } else {
            0
        }
    }

    /// Create sliding time windows from timestamp series
    pub fn sliding_windows(timestamps: &[Timestamp], window_size_ms: u64, step_ms: u64) -> Vec<(Timestamp, Timestamp)> {
        let mut windows = Vec::new();
        
        if timestamps.is_empty() {
            return windows;
        }

        let start_time = timestamps[0];
        let end_time = timestamps[timestamps.len() - 1];
        
        let mut current_start = start_time;
        while current_start + window_size_ms <= end_time {
            let current_end = current_start + window_size_ms;
            windows.push((current_start, current_end));
            current_start += step_ms;
        }

        windows
    }

    /// Filter timestamps within time window
    pub fn filter_timestamps(timestamps: &[Timestamp], window_start: Timestamp, window_end: Timestamp) -> Vec<Timestamp> {
        timestamps.iter()
            .copied()
            .filter(|&ts| Self::contains(window_start, window_end, ts))
            .collect()
    }

    /// Calculate overlap between two time windows
    pub fn overlap(window1: (Timestamp, Timestamp), window2: (Timestamp, Timestamp)) -> u64 {
        let overlap_start = window1.0.max(window2.0);
        let overlap_end = window1.1.min(window2.1);
        
        if overlap_end > overlap_start {
            overlap_end - overlap_start
        } else {
            0
        }
    }
}

impl ClockDriftDetector {
    /// Create new clock drift detector
    pub fn new() -> Self {
        Self {
            reference_points: VecDeque::new(),
            max_references: 100,
            drift_rate: 0.0,
            last_calibration: Instant::now(),
        }
    }

    /// Add reference timestamp for drift detection
    pub fn add_reference(&mut self, reference_time: Timestamp, confidence: f64) {
        let reference = TimestampReference {
            local_time: Instant::now(),
            reference_time,
            confidence,
        };

        self.reference_points.push_back(reference);

        // Remove old references if we exceed maximum
        while self.reference_points.len() > self.max_references {
            self.reference_points.pop_front();
        }

        // Recalculate drift if we have enough references
        if self.reference_points.len() >= 3 {
            self.calculate_drift();
        }
    }

    /// Calculate current drift rate
    fn calculate_drift(&mut self) {
        if self.reference_points.len() < 2 {
            return;
        }

        let mut time_diffs = Vec::new();
        let mut ref_diffs = Vec::new();

        let first_point = &self.reference_points[0];
        for point in self.reference_points.iter().skip(1) {
            let local_diff = point.local_time.duration_since(first_point.local_time).as_secs_f64();
            let ref_diff = (point.reference_time as f64 - first_point.reference_time as f64) / 1000.0;
            
            time_diffs.push(local_diff);
            ref_diffs.push(ref_diff);
        }

        // Simple linear regression to find drift rate
        if let Ok(drift_rate) = Self::linear_regression(&time_diffs, &ref_diffs) {
            self.drift_rate = drift_rate - 1.0; // Drift rate relative to 1.0 (perfect sync)
            self.last_calibration = Instant::now();
        }
    }

    /// Simple linear regression
    fn linear_regression(x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(Error::invalid_input("Invalid regression data"));
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x_sq: f64 = x.iter().map(|a| a * a).sum();

        let denominator = n * sum_x_sq - sum_x * sum_x;
        if denominator == 0.0 {
            return Err(Error::invalid_input("Cannot calculate regression"));
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        Ok(slope)
    }

    /// Get current drift rate
    pub fn get_drift_rate(&self) -> f64 {
        self.drift_rate
    }

    /// Compensate timestamp for detected drift
    pub fn compensate_timestamp(&self, timestamp: Timestamp) -> Timestamp {
        if self.drift_rate == 0.0 {
            return timestamp;
        }

        let time_since_calibration = self.last_calibration.elapsed().as_secs_f64();
        let drift_adjustment = time_since_calibration * self.drift_rate * 1000.0; // Convert to ms
        
        (timestamp as f64 + drift_adjustment) as Timestamp
    }

    /// Check if calibration is needed
    pub fn needs_calibration(&self) -> bool {
        self.last_calibration.elapsed().as_secs() > 300 || // 5 minutes
        self.reference_points.len() < 3
    }
}

impl Default for PrecisionTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ClockDriftDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp (convenience function)
pub fn current_timestamp() -> Timestamp {
    TimeSync::current_timestamp()
}

/// Sleep for specified duration with high precision
pub fn precise_sleep(duration: Duration) {
    std::thread::sleep(duration);
}

/// Measure execution time of a function
pub fn measure_time<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_timer() {
        let mut timer = PrecisionTimer::new();
        assert!(!timer.is_running());

        timer.start();
        assert!(timer.is_running());

        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed >= Duration::from_millis(10));
        assert!(!timer.is_running());
    }

    #[test]
    fn test_time_sync() {
        let timestamp = TimeSync::current_timestamp();
        assert!(timestamp > 0);

        let valid = TimeSync::is_time_valid(timestamp, 1000);
        assert!(valid);

        let future_time = timestamp + 10000;
        let valid_future = TimeSync::is_time_valid(future_time, 1000);
        assert!(!valid_future);
    }

    #[test]
    fn test_temporal_analysis() {
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let stats = TemporalAnalysis::analyze_timestamps(&timestamps).unwrap();
        
        assert_eq!(stats.mean_interval, 1000.0);
        assert_eq!(stats.min_interval, 1000.0);
        assert_eq!(stats.max_interval, 1000.0);
        assert_eq!(stats.std_deviation, 0.0);
    }

    #[test]
    fn test_time_window() {
        let window_start = 1000;
        let window_end = 2000;
        
        assert!(TimeWindow::contains(window_start, window_end, 1500));
        assert!(!TimeWindow::contains(window_start, window_end, 500));
        assert!(!TimeWindow::contains(window_start, window_end, 2500));
        
        let duration = TimeWindow::duration(window_start, window_end);
        assert_eq!(duration, 1000);
    }

    #[test]
    fn test_clock_drift_detector() {
        let mut detector = ClockDriftDetector::new();
        assert_eq!(detector.get_drift_rate(), 0.0);
        
        detector.add_reference(TimeSync::current_timestamp(), 0.9);
        assert!(detector.needs_calibration());
    }

    #[test]
    fn test_measure_time() {
        let (result, duration) = measure_time(|| {
            std::thread::sleep(Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(10));
    }
}
