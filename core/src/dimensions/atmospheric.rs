use crate::types::{DimensionMeasurement, DimensionType, MdtecError, MdtecResult};
use crate::utils::math::Statistics;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

/// Atmospheric sensor reading
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AtmosphericReading {
    /// Atmospheric pressure in hPa (hectopascals)
    pub pressure: f64,
    /// Relative humidity as percentage (0-100)
    pub humidity: f64,
    /// Temperature in Celsius
    pub temperature: f64,
    /// Air quality index (0-500)
    pub air_quality_index: Option<f64>,
    /// Barometric altitude in meters
    pub barometric_altitude: Option<f64>,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

/// Atmospheric pressure trends
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PressureTrend {
    Rising,
    Falling,
    Stable,
    Rapid,
}

/// Atmospheric dimension configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtmosphericConfig {
    /// Sample window size for atmospheric analysis
    pub sample_window_size: usize,
    /// Minimum pressure change to detect trend (hPa)
    pub pressure_trend_threshold: f64,
    /// Temperature stability threshold (°C)
    pub temperature_stability_threshold: f64,
    /// Humidity change threshold (%)
    pub humidity_change_threshold: f64,
    /// Enable barometric altitude calculation
    pub enable_barometric_altitude: bool,
    /// Enable air quality monitoring
    pub enable_air_quality: bool,
    /// Pressure measurement precision (hPa)
    pub pressure_precision: f64,
}

impl Default for AtmosphericConfig {
    fn default() -> Self {
        Self {
            sample_window_size: 200,
            pressure_trend_threshold: 0.5,
            temperature_stability_threshold: 0.2,
            humidity_change_threshold: 2.0,
            enable_barometric_altitude: true,
            enable_air_quality: true,
            pressure_precision: 0.1,
        }
    }
}

/// Atmospheric dimension analyzer
pub struct AtmosphericAnalyzer {
    config: AtmosphericConfig,
    readings: VecDeque<AtmosphericReading>,
    pressure_trend: PressureTrend,
    baseline_pressure: Option<f64>,
    baseline_temperature: Option<f64>,
    baseline_humidity: Option<f64>,
}

impl AtmosphericAnalyzer {
    pub fn new(config: AtmosphericConfig) -> Self {
        Self {
            config,
            readings: VecDeque::new(),
            pressure_trend: PressureTrend::Stable,
            baseline_pressure: None,
            baseline_temperature: None,
            baseline_humidity: None,
        }
    }

    /// Add atmospheric reading
    pub fn add_reading(&mut self, reading: AtmosphericReading) -> MdtecResult<()> {
        // Validate reading
        if reading.pressure < 800.0 || reading.pressure > 1200.0 {
            return Err(MdtecError::InvalidInput(format!(
                "Pressure {} hPa is out of valid range (800-1200 hPa)",
                reading.pressure
            )));
        }

        if reading.humidity < 0.0 || reading.humidity > 100.0 {
            return Err(MdtecError::InvalidInput(format!(
                "Humidity {} is out of valid range (0-100%)",
                reading.humidity
            )));
        }

        if reading.temperature < -50.0 || reading.temperature > 60.0 {
            return Err(MdtecError::InvalidInput(format!(
                "Temperature {} °C is out of valid range (-50 to 60 °C)",
                reading.temperature
            )));
        }

        self.readings.push_back(reading);

        // Maintain window size
        while self.readings.len() > self.config.sample_window_size {
            self.readings.pop_front();
        }

        // Set baselines if not set
        if self.baseline_pressure.is_none() {
            self.baseline_pressure = Some(reading.pressure);
        }
        if self.baseline_temperature.is_none() {
            self.baseline_temperature = Some(reading.temperature);
        }
        if self.baseline_humidity.is_none() {
            self.baseline_humidity = Some(reading.humidity);
        }

        // Update pressure trend
        self.update_pressure_trend();

        Ok(())
    }

    /// Update pressure trend analysis
    fn update_pressure_trend(&mut self) {
        if self.readings.len() < 5 {
            return;
        }

        let recent_readings: Vec<_> = self.readings.iter().rev().take(5).collect();
        let pressures: Vec<f64> = recent_readings.iter().map(|r| r.pressure).collect();
        
        let first_pressure = pressures[0];
        let last_pressure = pressures[pressures.len() - 1];
        let pressure_change = last_pressure - first_pressure;

        self.pressure_trend = if pressure_change.abs() < self.config.pressure_trend_threshold {
            PressureTrend::Stable
        } else if pressure_change.abs() > self.config.pressure_trend_threshold * 3.0 {
            PressureTrend::Rapid
        } else if pressure_change > 0.0 {
            PressureTrend::Rising
        } else {
            PressureTrend::Falling
        };
    }

    /// Calculate atmospheric entropy
    pub fn calculate_atmospheric_entropy(&self) -> MdtecResult<f64> {
        if self.readings.len() < 10 {
            return Err(MdtecError::InsufficientData("Not enough atmospheric readings".to_string()));
        }

        let mut entropy = 0.0;

        // Pressure entropy (40% weight)
        let pressure_entropy = self.calculate_pressure_entropy()?;
        entropy += pressure_entropy * 0.4;

        // Temperature entropy (30% weight)
        let temperature_entropy = self.calculate_temperature_entropy()?;
        entropy += temperature_entropy * 0.3;

        // Humidity entropy (20% weight)
        let humidity_entropy = self.calculate_humidity_entropy()?;
        entropy += humidity_entropy * 0.2;

        // Air quality entropy (10% weight)
        if self.config.enable_air_quality {
            if let Some(aqi_entropy) = self.calculate_air_quality_entropy()? {
                entropy += aqi_entropy * 0.1;
            }
        }

        Ok(entropy.min(1.0))
    }

    /// Calculate pressure-based entropy
    fn calculate_pressure_entropy(&self) -> MdtecResult<f64> {
        let pressures: Vec<f64> = self.readings.iter().map(|r| r.pressure).collect();
        
        if pressures.is_empty() {
            return Ok(0.0);
        }

        // Calculate pressure variations
        let pressure_variations: Vec<f64> = pressures
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .collect();

        if pressure_variations.is_empty() {
            return Ok(0.0);
        }

        let entropy = Statistics::shannon_entropy(&pressure_variations);
        Ok(entropy)
    }

    /// Calculate temperature-based entropy
    fn calculate_temperature_entropy(&self) -> MdtecResult<f64> {
        let temperatures: Vec<f64> = self.readings.iter().map(|r| r.temperature).collect();
        
        if temperatures.is_empty() {
            return Ok(0.0);
        }

        // Calculate temperature variations
        let temp_variations: Vec<f64> = temperatures
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .collect();

        if temp_variations.is_empty() {
            return Ok(0.0);
        }

        let entropy = Statistics::shannon_entropy(&temp_variations);
        Ok(entropy)
    }

    /// Calculate humidity-based entropy
    fn calculate_humidity_entropy(&self) -> MdtecResult<f64> {
        let humidity_values: Vec<f64> = self.readings.iter().map(|r| r.humidity).collect();
        
        if humidity_values.is_empty() {
            return Ok(0.0);
        }

        // Calculate humidity variations
        let humidity_variations: Vec<f64> = humidity_values
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .collect();

        if humidity_variations.is_empty() {
            return Ok(0.0);
        }

        let entropy = Statistics::shannon_entropy(&humidity_variations);
        Ok(entropy)
    }

    /// Calculate air quality entropy
    fn calculate_air_quality_entropy(&self) -> MdtecResult<Option<f64>> {
        let aqi_values: Vec<f64> = self.readings
            .iter()
            .filter_map(|r| r.air_quality_index)
            .collect();

        if aqi_values.len() < 5 {
            return Ok(None);
        }

        let entropy = Statistics::shannon_entropy(&aqi_values);
        Ok(Some(entropy))
    }

    /// Calculate barometric altitude
    pub fn calculate_barometric_altitude(&self, sea_level_pressure: f64) -> Option<f64> {
        if !self.config.enable_barometric_altitude {
            return None;
        }

        if let Some(reading) = self.readings.back() {
            // International Standard Atmosphere formula
            let altitude = 44330.0 * (1.0 - (reading.pressure / sea_level_pressure).powf(0.1903));
            Some(altitude)
        } else {
            None
        }
    }

    /// Get atmospheric statistics
    pub fn get_atmospheric_statistics(&self) -> AtmosphericStatistics {
        let pressures: Vec<f64> = self.readings.iter().map(|r| r.pressure).collect();
        let temperatures: Vec<f64> = self.readings.iter().map(|r| r.temperature).collect();
        let humidity_values: Vec<f64> = self.readings.iter().map(|r| r.humidity).collect();

        AtmosphericStatistics {
            sample_count: self.readings.len(),
            pressure_mean: Statistics::mean(&pressures),
            pressure_std: Statistics::std_dev(&pressures).unwrap_or(0.0),
            temperature_mean: Statistics::mean(&temperatures),
            temperature_std: Statistics::std_dev(&temperatures).unwrap_or(0.0),
            humidity_mean: Statistics::mean(&humidity_values),
            humidity_std: Statistics::std_dev(&humidity_values).unwrap_or(0.0),
            pressure_trend: self.pressure_trend,
            current_entropy: self.calculate_atmospheric_entropy().unwrap_or(0.0),
        }
    }

    /// Get current atmospheric measurement
    pub fn get_measurement(&self) -> MdtecResult<DimensionMeasurement> {
        let entropy = self.calculate_atmospheric_entropy()?;
        let quality = self.calculate_measurement_quality();

        let mut metadata = std::collections::HashMap::new();
        
        if let Some(reading) = self.readings.back() {
            metadata.insert("pressure".to_string(), reading.pressure.to_string());
            metadata.insert("temperature".to_string(), reading.temperature.to_string());
            metadata.insert("humidity".to_string(), reading.humidity.to_string());
            
            if let Some(aqi) = reading.air_quality_index {
                metadata.insert("air_quality_index".to_string(), aqi.to_string());
            }
            
            if let Some(alt) = reading.barometric_altitude {
                metadata.insert("barometric_altitude".to_string(), alt.to_string());
            }
        }

        let stats = self.get_atmospheric_statistics();
        metadata.insert("pressure_trend".to_string(), format!("{:?}", stats.pressure_trend));
        metadata.insert("pressure_mean".to_string(), stats.pressure_mean.to_string());
        metadata.insert("temperature_mean".to_string(), stats.temperature_mean.to_string());
        metadata.insert("humidity_mean".to_string(), stats.humidity_mean.to_string());
        metadata.insert("sample_count".to_string(), stats.sample_count.to_string());

        Ok(DimensionMeasurement {
            dimension_type: DimensionType::Atmospheric,
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
        let sample_coverage = (self.readings.len() as f64 / self.config.sample_window_size as f64).min(1.0);
        quality += sample_coverage * 0.4;

        // Data freshness quality
        if let Some(reading) = self.readings.back() {
            let age = reading.timestamp.elapsed().as_secs();
            let freshness = (1.0 - (age as f64 / 300.0)).max(0.0); // 5 minutes max age
            quality += freshness * 0.3;
        }

        // Sensor precision quality
        quality += 0.2; // Assume good precision

        // Air quality availability bonus
        if self.config.enable_air_quality {
            let aqi_readings = self.readings.iter().filter(|r| r.air_quality_index.is_some()).count();
            let aqi_coverage = (aqi_readings as f64 / self.readings.len() as f64).min(1.0);
            quality += aqi_coverage * 0.1;
        }

        quality.min(1.0)
    }

    /// Reset atmospheric analyzer
    pub fn reset(&mut self) {
        self.readings.clear();
        self.pressure_trend = PressureTrend::Stable;
        self.baseline_pressure = None;
        self.baseline_temperature = None;
        self.baseline_humidity = None;
    }
}

/// Atmospheric measurement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtmosphericStatistics {
    pub sample_count: usize,
    pub pressure_mean: f64,
    pub pressure_std: f64,
    pub temperature_mean: f64,
    pub temperature_std: f64,
    pub humidity_mean: f64,
    pub humidity_std: f64,
    pub pressure_trend: PressureTrend,
    pub current_entropy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atmospheric_reading_creation() {
        let reading = AtmosphericReading {
            pressure: 1013.25,
            humidity: 65.0,
            temperature: 22.5,
            air_quality_index: Some(45.0),
            barometric_altitude: Some(100.0),
            timestamp: Instant::now(),
        };

        assert_eq!(reading.pressure, 1013.25);
        assert_eq!(reading.humidity, 65.0);
        assert_eq!(reading.temperature, 22.5);
        assert_eq!(reading.air_quality_index, Some(45.0));
    }

    #[test]
    fn test_atmospheric_analyzer_add_reading() {
        let mut analyzer = AtmosphericAnalyzer::new(AtmosphericConfig::default());

        let reading = AtmosphericReading {
            pressure: 1013.25,
            humidity: 65.0,
            temperature: 22.5,
            air_quality_index: None,
            barometric_altitude: None,
            timestamp: Instant::now(),
        };

        assert!(analyzer.add_reading(reading).is_ok());
        assert_eq!(analyzer.readings.len(), 1);
    }

    #[test]
    fn test_invalid_pressure() {
        let mut analyzer = AtmosphericAnalyzer::new(AtmosphericConfig::default());

        let reading = AtmosphericReading {
            pressure: 1300.0, // Invalid pressure
            humidity: 65.0,
            temperature: 22.5,
            air_quality_index: None,
            barometric_altitude: None,
            timestamp: Instant::now(),
        };

        assert!(analyzer.add_reading(reading).is_err());
    }

    #[test]
    fn test_barometric_altitude_calculation() {
        let mut analyzer = AtmosphericAnalyzer::new(AtmosphericConfig::default());

        let reading = AtmosphericReading {
            pressure: 1000.0,
            humidity: 65.0,
            temperature: 22.5,
            air_quality_index: None,
            barometric_altitude: None,
            timestamp: Instant::now(),
        };

        analyzer.add_reading(reading).unwrap();

        let altitude = analyzer.calculate_barometric_altitude(1013.25);
        assert!(altitude.is_some());
        assert!(altitude.unwrap() > 0.0);
    }

    #[test]
    fn test_pressure_trend_detection() {
        let mut analyzer = AtmosphericAnalyzer::new(AtmosphericConfig::default());

        // Add readings with rising pressure
        for i in 0..10 {
            let reading = AtmosphericReading {
                pressure: 1000.0 + i as f64,
                humidity: 65.0,
                temperature: 22.5,
                air_quality_index: None,
                barometric_altitude: None,
                timestamp: Instant::now(),
            };
            analyzer.add_reading(reading).unwrap();
        }

        // Should detect rising trend
        assert!(matches!(analyzer.pressure_trend, PressureTrend::Rising | PressureTrend::Rapid));
    }
} 