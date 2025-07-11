use crate::types::{DimensionMeasurement, DimensionType, MdtecError, MdtecResult};
use crate::utils::math::{VectorOps, Statistics};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Spatial coordinate system representation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpatialCoordinate {
    /// Latitude in decimal degrees
    pub latitude: f64,
    /// Longitude in decimal degrees
    pub longitude: f64,
    /// Altitude in meters above sea level
    pub altitude: f64,
    /// Accuracy in meters
    pub accuracy: f64,
}

/// Accelerometer data for motion detection
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AccelerometerData {
    /// X-axis acceleration in m/s²
    pub x: f64,
    /// Y-axis acceleration in m/s²
    pub y: f64,
    /// Z-axis acceleration in m/s²
    pub z: f64,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

/// Spatial measurement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialConfig {
    /// Minimum accuracy required for GPS measurements (meters)
    pub min_gps_accuracy: f64,
    /// Maximum age of GPS data before considered stale (seconds)
    pub max_gps_age: Duration,
    /// Minimum movement threshold for motion detection (m/s²)
    pub motion_threshold: f64,
    /// Sample window size for spatial analysis
    pub sample_window_size: usize,
    /// Enable accelerometer data collection
    pub enable_accelerometer: bool,
    /// Enable gyroscope data collection
    pub enable_gyroscope: bool,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            min_gps_accuracy: 10.0,
            max_gps_age: Duration::from_secs(30),
            motion_threshold: 0.5,
            sample_window_size: 100,
            enable_accelerometer: true,
            enable_gyroscope: true,
        }
    }
}

/// Spatial dimension analyzer
pub struct SpatialAnalyzer {
    config: SpatialConfig,
    gps_history: VecDeque<(SpatialCoordinate, Instant)>,
    accelerometer_history: VecDeque<AccelerometerData>,
    last_movement_time: Option<Instant>,
    baseline_coordinate: Option<SpatialCoordinate>,
}

impl SpatialAnalyzer {
    pub fn new(config: SpatialConfig) -> Self {
        Self {
            config,
            gps_history: VecDeque::new(),
            accelerometer_history: VecDeque::new(),
            last_movement_time: None,
            baseline_coordinate: None,
        }
    }

    /// Add GPS coordinate measurement
    pub fn add_gps_measurement(&mut self, coordinate: SpatialCoordinate) -> MdtecResult<()> {
        if coordinate.accuracy > self.config.min_gps_accuracy {
            return Err(MdtecError::InvalidInput(format!(
                "GPS accuracy {} exceeds threshold {}",
                coordinate.accuracy, self.config.min_gps_accuracy
            )));
        }

        let now = Instant::now();
        self.gps_history.push_back((coordinate, now));

        // Maintain window size
        while self.gps_history.len() > self.config.sample_window_size {
            self.gps_history.pop_front();
        }

        // Set baseline if not set
        if self.baseline_coordinate.is_none() {
            self.baseline_coordinate = Some(coordinate);
        }

        Ok(())
    }

    /// Add accelerometer measurement
    pub fn add_accelerometer_measurement(&mut self, data: AccelerometerData) -> MdtecResult<()> {
        if !self.config.enable_accelerometer {
            return Ok(());
        }

        let magnitude = (data.x * data.x + data.y * data.y + data.z * data.z).sqrt();
        
        if magnitude > self.config.motion_threshold {
            self.last_movement_time = Some(data.timestamp);
        }

        self.accelerometer_history.push_back(data);

        // Maintain window size
        while self.accelerometer_history.len() > self.config.sample_window_size {
            self.accelerometer_history.pop_front();
        }

        Ok(())
    }

    /// Calculate spatial entropy based on movement patterns
    pub fn calculate_spatial_entropy(&self) -> MdtecResult<f64> {
        if self.gps_history.is_empty() {
            return Err(MdtecError::InsufficientData("No GPS data available".to_string()));
        }

        let mut entropy = 0.0;

        // GPS position entropy
        if let Some(gps_entropy) = self.calculate_gps_entropy()? {
            entropy += gps_entropy * 0.6;
        }

        // Accelerometer entropy
        if let Some(accel_entropy) = self.calculate_accelerometer_entropy()? {
            entropy += accel_entropy * 0.4;
        }

        Ok(entropy.min(1.0))
    }

    /// Calculate GPS-based spatial entropy
    fn calculate_gps_entropy(&self) -> MdtecResult<Option<f64>> {
        if self.gps_history.len() < 2 {
            return Ok(None);
        }

        let coordinates: Vec<_> = self.gps_history.iter().map(|(coord, _)| coord).collect();
        let mut distances = Vec::new();

        // Calculate distances between consecutive points
        for i in 1..coordinates.len() {
            let prev = coordinates[i - 1];
            let curr = coordinates[i];
            
            let distance = self.haversine_distance(
                prev.latitude, prev.longitude,
                curr.latitude, curr.longitude
            );
            distances.push(distance);
        }

        if distances.is_empty() {
            return Ok(None);
        }

        // Calculate entropy based on distance variations
        let entropy = Statistics::shannon_entropy(&distances);
        Ok(Some(entropy))
    }

    /// Calculate accelerometer-based entropy
    fn calculate_accelerometer_entropy(&self) -> MdtecResult<Option<f64>> {
        if self.accelerometer_history.len() < 10 {
            return Ok(None);
        }

        let magnitudes: Vec<f64> = self.accelerometer_history
            .iter()
            .map(|data| (data.x * data.x + data.y * data.y + data.z * data.z).sqrt())
            .collect();

        let entropy = Statistics::shannon_entropy(&magnitudes);
        Ok(Some(entropy))
    }

    /// Calculate distance between two GPS coordinates using Haversine formula
    fn haversine_distance(&self, lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        const R: f64 = 6371000.0; // Earth's radius in meters
        
        let lat1_rad = lat1.to_radians();
        let lat2_rad = lat2.to_radians();
        let delta_lat = (lat2 - lat1).to_radians();
        let delta_lon = (lon2 - lon1).to_radians();

        let a = (delta_lat / 2.0).sin().powi(2) +
                lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        R * c
    }

    /// Get current spatial measurement
    pub fn get_measurement(&self) -> MdtecResult<DimensionMeasurement> {
        let entropy = self.calculate_spatial_entropy()?;
        let quality = self.calculate_measurement_quality();

        let mut metadata = std::collections::HashMap::new();
        
        if let Some((coord, timestamp)) = self.gps_history.back() {
            metadata.insert("latitude".to_string(), coord.latitude.to_string());
            metadata.insert("longitude".to_string(), coord.longitude.to_string());
            metadata.insert("altitude".to_string(), coord.altitude.to_string());
            metadata.insert("accuracy".to_string(), coord.accuracy.to_string());
            metadata.insert("gps_age".to_string(), timestamp.elapsed().as_secs().to_string());
        }

        if let Some(accel) = self.accelerometer_history.back() {
            let magnitude = (accel.x * accel.x + accel.y * accel.y + accel.z * accel.z).sqrt();
            metadata.insert("acceleration_magnitude".to_string(), magnitude.to_string());
        }

        if let Some(movement_time) = self.last_movement_time {
            metadata.insert("last_movement_age".to_string(), movement_time.elapsed().as_secs().to_string());
        }

        metadata.insert("gps_samples".to_string(), self.gps_history.len().to_string());
        metadata.insert("accelerometer_samples".to_string(), self.accelerometer_history.len().to_string());

        Ok(DimensionMeasurement {
            dimension_type: DimensionType::Spatial,
            value: entropy,
            quality,
            timestamp: std::time::SystemTime::now(),
            metadata,
        })
    }

    /// Calculate measurement quality based on data availability and accuracy
    fn calculate_measurement_quality(&self) -> f64 {
        let mut quality = 0.0;

        // GPS quality component
        if let Some((coord, timestamp)) = self.gps_history.back() {
            let age_penalty = (timestamp.elapsed().as_secs() as f64 / self.config.max_gps_age.as_secs() as f64).min(1.0);
            let accuracy_score = (self.config.min_gps_accuracy / coord.accuracy).min(1.0);
            quality += (1.0 - age_penalty) * accuracy_score * 0.6;
        }

        // Sample size quality
        let gps_coverage = (self.gps_history.len() as f64 / self.config.sample_window_size as f64).min(1.0);
        let accel_coverage = (self.accelerometer_history.len() as f64 / self.config.sample_window_size as f64).min(1.0);
        quality += (gps_coverage + accel_coverage) * 0.2;

        quality.min(1.0)
    }

    /// Reset spatial analyzer state
    pub fn reset(&mut self) {
        self.gps_history.clear();
        self.accelerometer_history.clear();
        self.last_movement_time = None;
        self.baseline_coordinate = None;
    }

    /// Get spatial statistics
    pub fn get_statistics(&self) -> SpatialStatistics {
        SpatialStatistics {
            total_gps_samples: self.gps_history.len(),
            total_accelerometer_samples: self.accelerometer_history.len(),
            current_entropy: self.calculate_spatial_entropy().unwrap_or(0.0),
            measurement_quality: self.calculate_measurement_quality(),
            has_recent_movement: self.last_movement_time
                .map(|t| t.elapsed() < Duration::from_secs(10))
                .unwrap_or(false),
            baseline_set: self.baseline_coordinate.is_some(),
        }
    }
}

/// Spatial measurement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialStatistics {
    pub total_gps_samples: usize,
    pub total_accelerometer_samples: usize,
    pub current_entropy: f64,
    pub measurement_quality: f64,
    pub has_recent_movement: bool,
    pub baseline_set: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_coordinate_creation() {
        let coord = SpatialCoordinate {
            latitude: 37.7749,
            longitude: -122.4194,
            altitude: 52.0,
            accuracy: 5.0,
        };

        assert_eq!(coord.latitude, 37.7749);
        assert_eq!(coord.longitude, -122.4194);
        assert_eq!(coord.altitude, 52.0);
        assert_eq!(coord.accuracy, 5.0);
    }

    #[test]
    fn test_spatial_analyzer_gps_measurement() {
        let mut analyzer = SpatialAnalyzer::new(SpatialConfig::default());

        let coord = SpatialCoordinate {
            latitude: 37.7749,
            longitude: -122.4194,
            altitude: 52.0,
            accuracy: 5.0,
        };

        assert!(analyzer.add_gps_measurement(coord).is_ok());
        assert_eq!(analyzer.gps_history.len(), 1);
    }

    #[test]
    fn test_haversine_distance() {
        let analyzer = SpatialAnalyzer::new(SpatialConfig::default());
        
        // Distance between San Francisco and Los Angeles (approximately)
        let distance = analyzer.haversine_distance(37.7749, -122.4194, 34.0522, -118.2437);
        
        // Should be approximately 559 km
        assert!((distance - 559000.0).abs() < 10000.0);
    }

    #[test]
    fn test_accelerometer_data() {
        let mut analyzer = SpatialAnalyzer::new(SpatialConfig::default());

        let data = AccelerometerData {
            x: 0.5,
            y: 0.3,
            z: 9.8,
            timestamp: Instant::now(),
        };

        assert!(analyzer.add_accelerometer_measurement(data).is_ok());
        assert_eq!(analyzer.accelerometer_history.len(), 1);
    }
} 