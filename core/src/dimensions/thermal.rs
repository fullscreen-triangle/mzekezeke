use crate::types::{DimensionMeasurement, DimensionType, MdtecError, MdtecResult};
use crate::utils::math::Statistics;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

/// Thermal measurement
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermalMeasurement {
    /// Temperature in Celsius
    pub temperature: f64,
    /// Heat flux in W/m²
    pub heat_flux: f64,
    /// Thermal resistance in K·m²/W
    pub thermal_resistance: Option<f64>,
    /// Humidity impact factor
    pub humidity_factor: Option<f64>,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

/// Thermal gradient measurement
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermalGradient {
    /// Temperature difference in K
    pub delta_temperature: f64,
    /// Distance over which gradient is measured (meters)
    pub distance: f64,
    /// Gradient direction (0-360 degrees)
    pub direction: f64,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

/// Thermal trend analysis
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ThermalTrend {
    /// Temperature increasing
    Heating,
    /// Temperature decreasing
    Cooling,
    /// Temperature stable
    Stable,
    /// Rapid temperature change
    Fluctuating,
}

/// Thermal dimension configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalConfig {
    /// Sample window size for thermal analysis
    pub sample_window_size: usize,
    /// Temperature change threshold for trend detection (°C)
    pub temperature_trend_threshold: f64,
    /// Thermal gradient threshold (K/m)
    pub gradient_threshold: f64,
    /// Enable heat flux monitoring
    pub enable_heat_flux: bool,
    /// Enable thermal gradient analysis
    pub enable_gradient_analysis: bool,
    /// Temperature measurement precision (°C)
    pub temperature_precision: f64,
    /// Thermal time constant (seconds)
    pub thermal_time_constant: f64,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            sample_window_size: 300,
            temperature_trend_threshold: 0.5,
            gradient_threshold: 1.0,
            enable_heat_flux: true,
            enable_gradient_analysis: true,
            temperature_precision: 0.1,
            thermal_time_constant: 60.0,
        }
    }
}

/// Thermal dimension analyzer
pub struct ThermalAnalyzer {
    config: ThermalConfig,
    thermal_readings: VecDeque<ThermalMeasurement>,
    gradient_readings: VecDeque<ThermalGradient>,
    thermal_trend: ThermalTrend,
    baseline_temperature: Option<f64>,
    peak_temperature: Option<f64>,
    min_temperature: Option<f64>,
    last_trend_change: Option<Instant>,
}

impl ThermalAnalyzer {
    pub fn new(config: ThermalConfig) -> Self {
        Self {
            config,
            thermal_readings: VecDeque::new(),
            gradient_readings: VecDeque::new(),
            thermal_trend: ThermalTrend::Stable,
            baseline_temperature: None,
            peak_temperature: None,
            min_temperature: None,
            last_trend_change: None,
        }
    }

    /// Add thermal measurement
    pub fn add_measurement(&mut self, measurement: ThermalMeasurement) -> MdtecResult<()> {
        // Validate measurement
        if measurement.temperature < -40.0 || measurement.temperature > 85.0 {
            return Err(MdtecError::InvalidInput(format!(
                "Temperature {} °C is out of valid range (-40 to 85 °C)",
                measurement.temperature
            )));
        }

        if measurement.heat_flux < 0.0 || measurement.heat_flux > 10000.0 {
            return Err(MdtecError::InvalidInput(format!(
                "Heat flux {} W/m² is out of valid range (0 to 10000 W/m²)",
                measurement.heat_flux
            )));
        }

        self.thermal_readings.push_back(measurement);

        // Maintain window size
        while self.thermal_readings.len() > self.config.sample_window_size {
            self.thermal_readings.pop_front();
        }

        // Update temperature statistics
        self.update_temperature_statistics(measurement.temperature);

        // Update thermal trend
        self.update_thermal_trend();

        Ok(())
    }

    /// Add thermal gradient measurement
    pub fn add_gradient(&mut self, gradient: ThermalGradient) -> MdtecResult<()> {
        if !self.config.enable_gradient_analysis {
            return Ok(());
        }

        if gradient.distance <= 0.0 {
            return Err(MdtecError::InvalidInput(
                "Gradient distance must be positive".to_string(),
            ));
        }

        self.gradient_readings.push_back(gradient);

        // Maintain window size
        while self.gradient_readings.len() > self.config.sample_window_size {
            self.gradient_readings.pop_front();
        }

        Ok(())
    }

    /// Update temperature statistics
    fn update_temperature_statistics(&mut self, temperature: f64) {
        // Set baseline if not set
        if self.baseline_temperature.is_none() {
            self.baseline_temperature = Some(temperature);
        }

        // Update peak temperature
        if self.peak_temperature.is_none() || temperature > self.peak_temperature.unwrap() {
            self.peak_temperature = Some(temperature);
        }

        // Update minimum temperature
        if self.min_temperature.is_none() || temperature < self.min_temperature.unwrap() {
            self.min_temperature = Some(temperature);
        }
    }

    /// Update thermal trend analysis
    fn update_thermal_trend(&mut self) {
        if self.thermal_readings.len() < 10 {
            return;
        }

        let recent_temps: Vec<f64> = self.thermal_readings
            .iter()
            .rev()
            .take(10)
            .map(|r| r.temperature)
            .collect();

        let first_temp = recent_temps[0];
        let last_temp = recent_temps[recent_temps.len() - 1];
        let temp_change = last_temp - first_temp;

        let new_trend = if temp_change.abs() < self.config.temperature_trend_threshold {
            ThermalTrend::Stable
        } else if temp_change.abs() > self.config.temperature_trend_threshold * 3.0 {
            ThermalTrend::Fluctuating
        } else if temp_change > 0.0 {
            ThermalTrend::Heating
        } else {
            ThermalTrend::Cooling
        };

        if new_trend != self.thermal_trend {
            self.thermal_trend = new_trend;
            self.last_trend_change = Some(Instant::now());
        }
    }

    /// Calculate thermal entropy
    pub fn calculate_thermal_entropy(&self) -> MdtecResult<f64> {
        if self.thermal_readings.len() < 10 {
            return Err(MdtecError::InsufficientData("Not enough thermal readings".to_string()));
        }

        let mut entropy = 0.0;

        // Temperature entropy (50% weight)
        let temp_entropy = self.calculate_temperature_entropy()?;
        entropy += temp_entropy * 0.5;

        // Heat flux entropy (30% weight)
        if self.config.enable_heat_flux {
            let heat_flux_entropy = self.calculate_heat_flux_entropy()?;
            entropy += heat_flux_entropy * 0.3;
        }

        // Gradient entropy (20% weight)
        if self.config.enable_gradient_analysis {
            if let Some(gradient_entropy) = self.calculate_gradient_entropy()? {
                entropy += gradient_entropy * 0.2;
            }
        }

        Ok(entropy.min(1.0))
    }

    /// Calculate temperature entropy
    fn calculate_temperature_entropy(&self) -> MdtecResult<f64> {
        let temperatures: Vec<f64> = self.thermal_readings.iter().map(|r| r.temperature).collect();
        
        if temperatures.is_empty() {
            return Ok(0.0);
        }

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

    /// Calculate heat flux entropy
    fn calculate_heat_flux_entropy(&self) -> MdtecResult<f64> {
        let heat_fluxes: Vec<f64> = self.thermal_readings.iter().map(|r| r.heat_flux).collect();
        
        if heat_fluxes.is_empty() {
            return Ok(0.0);
        }

        let flux_variations: Vec<f64> = heat_fluxes
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .collect();

        if flux_variations.is_empty() {
            return Ok(0.0);
        }

        let entropy = Statistics::shannon_entropy(&flux_variations);
        Ok(entropy)
    }

    /// Calculate gradient entropy
    fn calculate_gradient_entropy(&self) -> MdtecResult<Option<f64>> {
        if self.gradient_readings.is_empty() {
            return Ok(None);
        }

        let gradients: Vec<f64> = self.gradient_readings
            .iter()
            .map(|g| g.delta_temperature / g.distance)
            .collect();

        if gradients.is_empty() {
            return Ok(None);
        }

        let entropy = Statistics::shannon_entropy(&gradients);
        Ok(Some(entropy))
    }

    /// Calculate thermal time constant
    pub fn calculate_thermal_time_constant(&self) -> Option<f64> {
        if self.thermal_readings.len() < 20 {
            return None;
        }

        let temperatures: Vec<f64> = self.thermal_readings.iter().map(|r| r.temperature).collect();
        let n = temperatures.len();

        // Find step response (simplified)
        let mut max_change = 0.0;
        let mut change_index = 0;

        for i in 1..n {
            let change = (temperatures[i] - temperatures[i - 1]).abs();
            if change > max_change {
                max_change = change;
                change_index = i;
            }
        }

        if max_change < 0.5 {
            return None;
        }

        // Calculate time to reach 63% of final value
        let initial_temp = temperatures[change_index];
        let final_temp = temperatures[n - 1];
        let target_temp = initial_temp + 0.63 * (final_temp - initial_temp);

        for i in change_index..n {
            if (temperatures[i] - target_temp).abs() < 0.1 {
                return Some((i - change_index) as f64 * 0.1); // Assuming 0.1s sampling
            }
        }

        None
    }

    /// Calculate thermal diffusivity
    pub fn calculate_thermal_diffusivity(&self) -> Option<f64> {
        if self.thermal_readings.len() < 10 || self.gradient_readings.is_empty() {
            return None;
        }

        let recent_temps: Vec<f64> = self.thermal_readings
            .iter()
            .rev()
            .take(10)
            .map(|r| r.temperature)
            .collect();

        let temp_change_rate = if recent_temps.len() >= 2 {
            (recent_temps[0] - recent_temps[recent_temps.len() - 1]) / (recent_temps.len() as f64 * 0.1)
        } else {
            return None;
        };

        if let Some(gradient) = self.gradient_readings.back() {
            let spatial_gradient = gradient.delta_temperature / gradient.distance;
            if spatial_gradient.abs() > 0.001 {
                return Some(temp_change_rate / spatial_gradient);
            }
        }

        None
    }

    /// Get current thermal measurement
    pub fn get_measurement(&self) -> MdtecResult<DimensionMeasurement> {
        let entropy = self.calculate_thermal_entropy()?;
        let quality = self.calculate_measurement_quality();

        let mut metadata = std::collections::HashMap::new();
        
        if let Some(measurement) = self.thermal_readings.back() {
            metadata.insert("temperature".to_string(), measurement.temperature.to_string());
            metadata.insert("heat_flux".to_string(), measurement.heat_flux.to_string());
            
            if let Some(resistance) = measurement.thermal_resistance {
                metadata.insert("thermal_resistance".to_string(), resistance.to_string());
            }
            
            if let Some(humidity) = measurement.humidity_factor {
                metadata.insert("humidity_factor".to_string(), humidity.to_string());
            }
        }

        metadata.insert("thermal_trend".to_string(), format!("{:?}", self.thermal_trend));
        metadata.insert("sample_count".to_string(), self.thermal_readings.len().to_string());
        metadata.insert("gradient_count".to_string(), self.gradient_readings.len().to_string());

        if let Some(baseline) = self.baseline_temperature {
            metadata.insert("baseline_temperature".to_string(), baseline.to_string());
        }

        if let Some(peak) = self.peak_temperature {
            metadata.insert("peak_temperature".to_string(), peak.to_string());
        }

        if let Some(min) = self.min_temperature {
            metadata.insert("min_temperature".to_string(), min.to_string());
        }

        if let Some(time_constant) = self.calculate_thermal_time_constant() {
            metadata.insert("thermal_time_constant".to_string(), time_constant.to_string());
        }

        if let Some(diffusivity) = self.calculate_thermal_diffusivity() {
            metadata.insert("thermal_diffusivity".to_string(), diffusivity.to_string());
        }

        if let Some(last_change) = self.last_trend_change {
            metadata.insert("last_trend_change_age".to_string(), last_change.elapsed().as_secs().to_string());
        }

        Ok(DimensionMeasurement {
            dimension_type: DimensionType::Thermal,
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
        let sample_coverage = (self.thermal_readings.len() as f64 / self.config.sample_window_size as f64).min(1.0);
        quality += sample_coverage * 0.4;

        // Temperature range quality (better quality with more variation)
        if let (Some(peak), Some(min)) = (self.peak_temperature, self.min_temperature) {
            let temp_range = peak - min;
            let range_quality = (temp_range / 10.0).min(1.0); // Normalize to 10°C range
            quality += range_quality * 0.3;
        }

        // Gradient analysis quality
        if self.config.enable_gradient_analysis {
            let gradient_coverage = (self.gradient_readings.len() as f64 / (self.config.sample_window_size / 2) as f64).min(1.0);
            quality += gradient_coverage * 0.2;
        } else {
            quality += 0.2;
        }

        // Baseline establishment quality
        if self.baseline_temperature.is_some() {
            quality += 0.1;
        }

        quality.min(1.0)
    }

    /// Get thermal statistics
    pub fn get_statistics(&self) -> ThermalStatistics {
        let temperatures: Vec<f64> = self.thermal_readings.iter().map(|r| r.temperature).collect();
        let heat_fluxes: Vec<f64> = self.thermal_readings.iter().map(|r| r.heat_flux).collect();

        ThermalStatistics {
            sample_count: self.thermal_readings.len(),
            gradient_count: self.gradient_readings.len(),
            temperature_mean: Statistics::mean(&temperatures),
            temperature_std: Statistics::std_dev(&temperatures).unwrap_or(0.0),
            heat_flux_mean: Statistics::mean(&heat_fluxes),
            heat_flux_std: Statistics::std_dev(&heat_fluxes).unwrap_or(0.0),
            thermal_trend: self.thermal_trend,
            baseline_temperature: self.baseline_temperature,
            peak_temperature: self.peak_temperature,
            min_temperature: self.min_temperature,
            thermal_time_constant: self.calculate_thermal_time_constant(),
            thermal_diffusivity: self.calculate_thermal_diffusivity(),
            current_entropy: self.calculate_thermal_entropy().unwrap_or(0.0),
            measurement_quality: self.calculate_measurement_quality(),
        }
    }

    /// Reset thermal analyzer
    pub fn reset(&mut self) {
        self.thermal_readings.clear();
        self.gradient_readings.clear();
        self.thermal_trend = ThermalTrend::Stable;
        self.baseline_temperature = None;
        self.peak_temperature = None;
        self.min_temperature = None;
        self.last_trend_change = None;
    }
}

/// Thermal measurement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalStatistics {
    pub sample_count: usize,
    pub gradient_count: usize,
    pub temperature_mean: f64,
    pub temperature_std: f64,
    pub heat_flux_mean: f64,
    pub heat_flux_std: f64,
    pub thermal_trend: ThermalTrend,
    pub baseline_temperature: Option<f64>,
    pub peak_temperature: Option<f64>,
    pub min_temperature: Option<f64>,
    pub thermal_time_constant: Option<f64>,
    pub thermal_diffusivity: Option<f64>,
    pub current_entropy: f64,
    pub measurement_quality: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_measurement_creation() {
        let measurement = ThermalMeasurement {
            temperature: 25.0,
            heat_flux: 100.0,
            thermal_resistance: Some(0.1),
            humidity_factor: Some(0.6),
            timestamp: Instant::now(),
        };

        assert_eq!(measurement.temperature, 25.0);
        assert_eq!(measurement.heat_flux, 100.0);
        assert_eq!(measurement.thermal_resistance, Some(0.1));
        assert_eq!(measurement.humidity_factor, Some(0.6));
    }

    #[test]
    fn test_thermal_analyzer_add_measurement() {
        let mut analyzer = ThermalAnalyzer::new(ThermalConfig::default());

        let measurement = ThermalMeasurement {
            temperature: 25.0,
            heat_flux: 100.0,
            thermal_resistance: None,
            humidity_factor: None,
            timestamp: Instant::now(),
        };

        assert!(analyzer.add_measurement(measurement).is_ok());
        assert_eq!(analyzer.thermal_readings.len(), 1);
        assert_eq!(analyzer.baseline_temperature, Some(25.0));
    }

    #[test]
    fn test_thermal_gradient_analysis() {
        let mut analyzer = ThermalAnalyzer::new(ThermalConfig::default());

        let gradient = ThermalGradient {
            delta_temperature: 5.0,
            distance: 0.1,
            direction: 45.0,
            timestamp: Instant::now(),
        };

        assert!(analyzer.add_gradient(gradient).is_ok());
        assert_eq!(analyzer.gradient_readings.len(), 1);
    }

    #[test]
    fn test_thermal_trend_detection() {
        let mut analyzer = ThermalAnalyzer::new(ThermalConfig::default());

        // Add measurements with increasing temperature
        for i in 0..15 {
            let measurement = ThermalMeasurement {
                temperature: 20.0 + i as f64,
                heat_flux: 100.0,
                thermal_resistance: None,
                humidity_factor: None,
                timestamp: Instant::now(),
            };
            analyzer.add_measurement(measurement).unwrap();
        }

        // Should detect heating trend
        assert!(matches!(analyzer.thermal_trend, ThermalTrend::Heating | ThermalTrend::Fluctuating));
    }

    #[test]
    fn test_thermal_statistics() {
        let mut analyzer = ThermalAnalyzer::new(ThermalConfig::default());

        let measurements = vec![
            ThermalMeasurement {
                temperature: 20.0,
                heat_flux: 100.0,
                thermal_resistance: None,
                humidity_factor: None,
                timestamp: Instant::now(),
            },
            ThermalMeasurement {
                temperature: 25.0,
                heat_flux: 120.0,
                thermal_resistance: None,
                humidity_factor: None,
                timestamp: Instant::now(),
            },
            ThermalMeasurement {
                temperature: 30.0,
                heat_flux: 140.0,
                thermal_resistance: None,
                humidity_factor: None,
                timestamp: Instant::now(),
            },
        ];

        for measurement in measurements {
            analyzer.add_measurement(measurement).unwrap();
        }

        let stats = analyzer.get_statistics();
        assert_eq!(stats.sample_count, 3);
        assert_eq!(stats.temperature_mean, 25.0);
        assert_eq!(stats.heat_flux_mean, 120.0);
        assert_eq!(stats.baseline_temperature, Some(20.0));
        assert_eq!(stats.peak_temperature, Some(30.0));
        assert_eq!(stats.min_temperature, Some(20.0));
    }
} 