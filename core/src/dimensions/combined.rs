use crate::types::{DimensionMeasurement, DimensionType, MdtecError, MdtecResult};
use crate::utils::math::{Statistics, VectorOps};
use super::{
    spatial::{SpatialAnalyzer, SpatialConfig},
    temporal::{TemporalAnalyzer, TemporalConfig},
    atmospheric::{AtmosphericAnalyzer, AtmosphericConfig},
    electromagnetic::{ElectromagneticAnalyzer, ElectromagneticConfig},
    acoustic::{AcousticAnalyzer, AcousticConfig},
    thermal::{ThermalAnalyzer, ThermalConfig},
    network::{NetworkAnalyzer, NetworkConfig},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

/// Combined dimensional measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedMeasurement {
    /// Individual dimension measurements
    pub dimensions: HashMap<DimensionType, DimensionMeasurement>,
    /// Combined entropy score
    pub combined_entropy: f64,
    /// Dimensional correlation matrix
    pub correlation_matrix: Vec<Vec<f64>>,
    /// Environmental coherence score
    pub coherence_score: f64,
    /// Measurement timestamp
    pub timestamp: SystemTime,
}

/// Dimensional correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionalCorrelation {
    /// Correlation coefficient between dimensions
    pub correlation_coefficient: f64,
    /// Dimension pair
    pub dimension_pair: (DimensionType, DimensionType),
    /// Correlation strength
    pub strength: CorrelationStrength,
    /// Statistical significance
    pub significance: f64,
}

/// Correlation strength levels
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CorrelationStrength {
    /// Very weak correlation (< 0.2)
    VeryWeak,
    /// Weak correlation (0.2 - 0.4)
    Weak,
    /// Moderate correlation (0.4 - 0.6)
    Moderate,
    /// Strong correlation (0.6 - 0.8)
    Strong,
    /// Very strong correlation (> 0.8)
    VeryStrong,
}

/// Environmental coherence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalCoherence {
    /// Overall coherence score (0-1)
    pub overall_score: f64,
    /// Dimensional consistency scores
    pub dimensional_consistency: HashMap<DimensionType, f64>,
    /// Temporal stability score
    pub temporal_stability: f64,
    /// Spatial consistency score
    pub spatial_consistency: f64,
    /// Anomaly detection score
    pub anomaly_score: f64,
}

/// Combined dimension configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedConfig {
    /// Individual dimension configurations
    pub spatial_config: SpatialConfig,
    pub temporal_config: TemporalConfig,
    pub atmospheric_config: AtmosphericConfig,
    pub electromagnetic_config: ElectromagneticConfig,
    pub acoustic_config: AcousticConfig,
    pub thermal_config: ThermalConfig,
    pub network_config: NetworkConfig,
    
    /// Dimension weights for combined entropy calculation
    pub dimension_weights: HashMap<DimensionType, f64>,
    
    /// Enable cross-dimensional correlation analysis
    pub enable_correlation_analysis: bool,
    
    /// Enable environmental coherence analysis
    pub enable_coherence_analysis: bool,
    
    /// Correlation calculation window size
    pub correlation_window_size: usize,
    
    /// Coherence analysis threshold
    pub coherence_threshold: f64,
}

impl Default for CombinedConfig {
    fn default() -> Self {
        let mut dimension_weights = HashMap::new();
        dimension_weights.insert(DimensionType::Spatial, 0.15);
        dimension_weights.insert(DimensionType::Temporal, 0.15);
        dimension_weights.insert(DimensionType::Atmospheric, 0.15);
        dimension_weights.insert(DimensionType::Electromagnetic, 0.15);
        dimension_weights.insert(DimensionType::Acoustic, 0.15);
        dimension_weights.insert(DimensionType::Thermal, 0.15);
        dimension_weights.insert(DimensionType::Network, 0.10);

        Self {
            spatial_config: SpatialConfig::default(),
            temporal_config: TemporalConfig::default(),
            atmospheric_config: AtmosphericConfig::default(),
            electromagnetic_config: ElectromagneticConfig::default(),
            acoustic_config: AcousticConfig::default(),
            thermal_config: ThermalConfig::default(),
            network_config: NetworkConfig::default(),
            dimension_weights,
            enable_correlation_analysis: true,
            enable_coherence_analysis: true,
            correlation_window_size: 100,
            coherence_threshold: 0.7,
        }
    }
}

/// Combined dimensional analyzer
pub struct CombinedAnalyzer {
    config: CombinedConfig,
    
    // Individual dimension analyzers
    spatial_analyzer: SpatialAnalyzer,
    temporal_analyzer: TemporalAnalyzer,
    atmospheric_analyzer: AtmosphericAnalyzer,
    electromagnetic_analyzer: ElectromagneticAnalyzer,
    acoustic_analyzer: AcousticAnalyzer,
    thermal_analyzer: ThermalAnalyzer,
    network_analyzer: NetworkAnalyzer,
    
    // Combined analysis data
    measurement_history: Vec<CombinedMeasurement>,
    correlations: Vec<DimensionalCorrelation>,
    coherence_history: Vec<EnvironmentalCoherence>,
    
    // Analysis state
    last_analysis_time: Option<Instant>,
    baseline_entropy: Option<f64>,
}

impl CombinedAnalyzer {
    pub fn new(config: CombinedConfig) -> Self {
        Self {
            spatial_analyzer: SpatialAnalyzer::new(config.spatial_config.clone()),
            temporal_analyzer: TemporalAnalyzer::new(config.temporal_config.clone()),
            atmospheric_analyzer: AtmosphericAnalyzer::new(config.atmospheric_config.clone()),
            electromagnetic_analyzer: ElectromagneticAnalyzer::new(config.electromagnetic_config.clone()),
            acoustic_analyzer: AcousticAnalyzer::new(config.acoustic_config.clone()),
            thermal_analyzer: ThermalAnalyzer::new(config.thermal_config.clone()),
            network_analyzer: NetworkAnalyzer::new(config.network_config.clone()),
            config,
            measurement_history: Vec::new(),
            correlations: Vec::new(),
            coherence_history: Vec::new(),
            last_analysis_time: None,
            baseline_entropy: None,
        }
    }

    /// Get current combined measurement
    pub fn get_combined_measurement(&mut self) -> MdtecResult<CombinedMeasurement> {
        let mut dimensions = HashMap::new();
        
        // Collect measurements from all dimensions
        if let Ok(spatial_measurement) = self.spatial_analyzer.get_measurement() {
            dimensions.insert(DimensionType::Spatial, spatial_measurement);
        }
        
        if let Ok(temporal_measurement) = self.temporal_analyzer.get_measurement() {
            dimensions.insert(DimensionType::Temporal, temporal_measurement);
        }
        
        if let Ok(atmospheric_measurement) = self.atmospheric_analyzer.get_measurement() {
            dimensions.insert(DimensionType::Atmospheric, atmospheric_measurement);
        }
        
        if let Ok(electromagnetic_measurement) = self.electromagnetic_analyzer.get_measurement() {
            dimensions.insert(DimensionType::Electromagnetic, electromagnetic_measurement);
        }
        
        if let Ok(acoustic_measurement) = self.acoustic_analyzer.get_measurement() {
            dimensions.insert(DimensionType::Acoustic, acoustic_measurement);
        }
        
        if let Ok(thermal_measurement) = self.thermal_analyzer.get_measurement() {
            dimensions.insert(DimensionType::Thermal, thermal_measurement);
        }
        
        if let Ok(network_measurement) = self.network_analyzer.get_measurement() {
            dimensions.insert(DimensionType::Network, network_measurement);
        }

        // Calculate combined entropy
        let combined_entropy = self.calculate_combined_entropy(&dimensions)?;
        
        // Calculate correlation matrix
        let correlation_matrix = if self.config.enable_correlation_analysis {
            self.calculate_correlation_matrix(&dimensions)?
        } else {
            vec![]
        };
        
        // Calculate coherence score
        let coherence_score = if self.config.enable_coherence_analysis {
            self.calculate_coherence_score(&dimensions)?
        } else {
            0.0
        };

        let combined_measurement = CombinedMeasurement {
            dimensions,
            combined_entropy,
            correlation_matrix,
            coherence_score,
            timestamp: SystemTime::now(),
        };

        // Store in history
        self.measurement_history.push(combined_measurement.clone());
        
        // Maintain history size
        if self.measurement_history.len() > self.config.correlation_window_size {
            self.measurement_history.remove(0);
        }

        // Update baseline entropy
        if self.baseline_entropy.is_none() {
            self.baseline_entropy = Some(combined_entropy);
        }

        // Update analysis time
        self.last_analysis_time = Some(Instant::now());

        Ok(combined_measurement)
    }

    /// Calculate combined entropy from all dimensions
    fn calculate_combined_entropy(&self, dimensions: &HashMap<DimensionType, DimensionMeasurement>) -> MdtecResult<f64> {
        let mut weighted_entropy = 0.0;
        let mut total_weight = 0.0;

        for (dimension_type, measurement) in dimensions {
            if let Some(weight) = self.config.dimension_weights.get(dimension_type) {
                weighted_entropy += measurement.value * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            Ok(weighted_entropy / total_weight)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate correlation matrix between dimensions
    fn calculate_correlation_matrix(&self, dimensions: &HashMap<DimensionType, DimensionMeasurement>) -> MdtecResult<Vec<Vec<f64>>> {
        let dimension_types: Vec<_> = dimensions.keys().cloned().collect();
        let n = dimension_types.len();
        let mut correlation_matrix = vec![vec![0.0; n]; n];

        // Need historical data for correlation analysis
        if self.measurement_history.len() < 10 {
            return Ok(correlation_matrix);
        }

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    correlation_matrix[i][j] = 1.0;
                } else {
                    let correlation = self.calculate_dimension_correlation(
                        &dimension_types[i],
                        &dimension_types[j],
                    )?;
                    correlation_matrix[i][j] = correlation;
                }
            }
        }

        Ok(correlation_matrix)
    }

    /// Calculate correlation between two dimensions
    fn calculate_dimension_correlation(&self, dim1: &DimensionType, dim2: &DimensionType) -> MdtecResult<f64> {
        let mut values1 = Vec::new();
        let mut values2 = Vec::new();

        for measurement in &self.measurement_history {
            if let (Some(m1), Some(m2)) = (measurement.dimensions.get(dim1), measurement.dimensions.get(dim2)) {
                values1.push(m1.value);
                values2.push(m2.value);
            }
        }

        if values1.len() < 10 {
            return Ok(0.0);
        }

        let correlation = Statistics::correlation(&values1, &values2)?;
        Ok(correlation)
    }

    /// Calculate environmental coherence score
    fn calculate_coherence_score(&self, dimensions: &HashMap<DimensionType, DimensionMeasurement>) -> MdtecResult<f64> {
        let mut coherence_components = Vec::new();

        // Quality coherence (how consistent are the quality scores)
        let quality_scores: Vec<f64> = dimensions.values().map(|m| m.quality).collect();
        if !quality_scores.is_empty() {
            let quality_std = Statistics::std_dev(&quality_scores).unwrap_or(0.0);
            let quality_coherence = 1.0 - (quality_std / 2.0).min(1.0);
            coherence_components.push(quality_coherence);
        }

        // Temporal coherence (how close are the timestamps)
        let timestamps: Vec<SystemTime> = dimensions.values().map(|m| m.timestamp).collect();
        if timestamps.len() > 1 {
            let mut time_diffs = Vec::new();
            for i in 1..timestamps.len() {
                if let Ok(diff) = timestamps[i].duration_since(timestamps[0]) {
                    time_diffs.push(diff.as_secs_f64());
                }
            }
            if !time_diffs.is_empty() {
                let max_diff = time_diffs.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
                let temporal_coherence = 1.0 - (max_diff / 10.0).min(1.0); // 10 second window
                coherence_components.push(temporal_coherence);
            }
        }

        // Entropy coherence (how consistent are the entropy values)
        let entropy_values: Vec<f64> = dimensions.values().map(|m| m.value).collect();
        if !entropy_values.is_empty() {
            let entropy_std = Statistics::std_dev(&entropy_values).unwrap_or(0.0);
            let entropy_coherence = 1.0 - (entropy_std / 2.0).min(1.0);
            coherence_components.push(entropy_coherence);
        }

        // Calculate overall coherence
        let overall_coherence = if coherence_components.is_empty() {
            0.0
        } else {
            Statistics::mean(&coherence_components)
        };

        Ok(overall_coherence)
    }

    /// Analyze dimensional correlations
    pub fn analyze_correlations(&mut self) -> MdtecResult<Vec<DimensionalCorrelation>> {
        if self.measurement_history.len() < 20 {
            return Ok(Vec::new());
        }

        let mut correlations = Vec::new();
        let dimension_types = vec![
            DimensionType::Spatial,
            DimensionType::Temporal,
            DimensionType::Atmospheric,
            DimensionType::Electromagnetic,
            DimensionType::Acoustic,
            DimensionType::Thermal,
            DimensionType::Network,
        ];

        for i in 0..dimension_types.len() {
            for j in i + 1..dimension_types.len() {
                let dim1 = &dimension_types[i];
                let dim2 = &dimension_types[j];
                
                let correlation_coefficient = self.calculate_dimension_correlation(dim1, dim2)?;
                let strength = self.classify_correlation_strength(correlation_coefficient);
                
                // Simple significance test (placeholder)
                let significance = correlation_coefficient.abs();
                
                correlations.push(DimensionalCorrelation {
                    correlation_coefficient,
                    dimension_pair: (*dim1, *dim2),
                    strength,
                    significance,
                });
            }
        }

        // Sort by absolute correlation strength
        correlations.sort_by(|a, b| {
            b.correlation_coefficient.abs().partial_cmp(&a.correlation_coefficient.abs()).unwrap()
        });

        self.correlations = correlations.clone();
        Ok(correlations)
    }

    /// Classify correlation strength
    fn classify_correlation_strength(&self, correlation: f64) -> CorrelationStrength {
        let abs_corr = correlation.abs();
        
        if abs_corr < 0.2 {
            CorrelationStrength::VeryWeak
        } else if abs_corr < 0.4 {
            CorrelationStrength::Weak
        } else if abs_corr < 0.6 {
            CorrelationStrength::Moderate
        } else if abs_corr < 0.8 {
            CorrelationStrength::Strong
        } else {
            CorrelationStrength::VeryStrong
        }
    }

    /// Analyze environmental coherence
    pub fn analyze_environmental_coherence(&self) -> MdtecResult<EnvironmentalCoherence> {
        if self.measurement_history.is_empty() {
            return Err(MdtecError::InsufficientData("No measurement history available".to_string()));
        }

        let latest_measurement = &self.measurement_history[self.measurement_history.len() - 1];
        
        // Calculate dimensional consistency
        let mut dimensional_consistency = HashMap::new();
        for (dim_type, measurement) in &latest_measurement.dimensions {
            // Consistency based on quality and recent stability
            let consistency = measurement.quality * 0.7 + 0.3; // Base consistency
            dimensional_consistency.insert(*dim_type, consistency);
        }

        // Calculate temporal stability
        let temporal_stability = if self.measurement_history.len() >= 5 {
            let recent_entropies: Vec<f64> = self.measurement_history
                .iter()
                .rev()
                .take(5)
                .map(|m| m.combined_entropy)
                .collect();
            
            let entropy_std = Statistics::std_dev(&recent_entropies).unwrap_or(0.0);
            1.0 - (entropy_std / 2.0).min(1.0)
        } else {
            0.5
        };

        // Calculate spatial consistency (placeholder)
        let spatial_consistency = 0.8;

        // Calculate anomaly detection score
        let anomaly_score = if let Some(baseline) = self.baseline_entropy {
            let current_entropy = latest_measurement.combined_entropy;
            let deviation = (current_entropy - baseline).abs() / baseline.max(0.1);
            1.0 - deviation.min(1.0)
        } else {
            0.5
        };

        // Calculate overall coherence score
        let overall_score = (
            dimensional_consistency.values().sum::<f64>() / dimensional_consistency.len() as f64 * 0.4 +
            temporal_stability * 0.3 +
            spatial_consistency * 0.2 +
            anomaly_score * 0.1
        ).min(1.0);

        Ok(EnvironmentalCoherence {
            overall_score,
            dimensional_consistency,
            temporal_stability,
            spatial_consistency,
            anomaly_score,
        })
    }

    /// Get access to individual analyzers
    pub fn get_spatial_analyzer(&mut self) -> &mut SpatialAnalyzer {
        &mut self.spatial_analyzer
    }

    pub fn get_temporal_analyzer(&mut self) -> &mut TemporalAnalyzer {
        &mut self.temporal_analyzer
    }

    pub fn get_atmospheric_analyzer(&mut self) -> &mut AtmosphericAnalyzer {
        &mut self.atmospheric_analyzer
    }

    pub fn get_electromagnetic_analyzer(&mut self) -> &mut ElectromagneticAnalyzer {
        &mut self.electromagnetic_analyzer
    }

    pub fn get_acoustic_analyzer(&mut self) -> &mut AcousticAnalyzer {
        &mut self.acoustic_analyzer
    }

    pub fn get_thermal_analyzer(&mut self) -> &mut ThermalAnalyzer {
        &mut self.thermal_analyzer
    }

    pub fn get_network_analyzer(&mut self) -> &mut NetworkAnalyzer {
        &mut self.network_analyzer
    }

    /// Reset all analyzers
    pub fn reset(&mut self) {
        self.spatial_analyzer.reset();
        self.temporal_analyzer.reset();
        self.atmospheric_analyzer.reset();
        self.electromagnetic_analyzer.reset();
        self.acoustic_analyzer.reset();
        self.thermal_analyzer.reset();
        self.network_analyzer.reset();
        
        self.measurement_history.clear();
        self.correlations.clear();
        self.coherence_history.clear();
        self.last_analysis_time = None;
        self.baseline_entropy = None;
    }

    /// Get combined statistics
    pub fn get_combined_statistics(&self) -> CombinedStatistics {
        CombinedStatistics {
            measurement_count: self.measurement_history.len(),
            correlation_count: self.correlations.len(),
            coherence_history_length: self.coherence_history.len(),
            average_combined_entropy: if self.measurement_history.is_empty() {
                0.0
            } else {
                Statistics::mean(&self.measurement_history.iter().map(|m| m.combined_entropy).collect::<Vec<_>>())
            },
            baseline_entropy: self.baseline_entropy,
            last_analysis_age: self.last_analysis_time.map(|t| t.elapsed()),
        }
    }
}

/// Combined measurement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedStatistics {
    pub measurement_count: usize,
    pub correlation_count: usize,
    pub coherence_history_length: usize,
    pub average_combined_entropy: f64,
    pub baseline_entropy: Option<f64>,
    pub last_analysis_age: Option<Duration>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combined_analyzer_creation() {
        let analyzer = CombinedAnalyzer::new(CombinedConfig::default());
        assert!(analyzer.measurement_history.is_empty());
        assert!(analyzer.correlations.is_empty());
        assert!(analyzer.baseline_entropy.is_none());
    }

    #[test]
    fn test_correlation_strength_classification() {
        let analyzer = CombinedAnalyzer::new(CombinedConfig::default());
        
        assert!(matches!(analyzer.classify_correlation_strength(0.1), CorrelationStrength::VeryWeak));
        assert!(matches!(analyzer.classify_correlation_strength(0.3), CorrelationStrength::Weak));
        assert!(matches!(analyzer.classify_correlation_strength(0.5), CorrelationStrength::Moderate));
        assert!(matches!(analyzer.classify_correlation_strength(0.7), CorrelationStrength::Strong));
        assert!(matches!(analyzer.classify_correlation_strength(0.9), CorrelationStrength::VeryStrong));
    }

    #[test]
    fn test_dimension_weights() {
        let config = CombinedConfig::default();
        let total_weight: f64 = config.dimension_weights.values().sum();
        
        // Weights should sum to 1.0 (or close to it)
        assert!((total_weight - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_combined_measurement_creation() {
        let mut dimensions = HashMap::new();
        dimensions.insert(DimensionType::Spatial, DimensionMeasurement {
            dimension_type: DimensionType::Spatial,
            value: 0.5,
            quality: 0.8,
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
        });

        let measurement = CombinedMeasurement {
            dimensions,
            combined_entropy: 0.5,
            correlation_matrix: vec![vec![1.0]],
            coherence_score: 0.7,
            timestamp: SystemTime::now(),
        };

        assert_eq!(measurement.combined_entropy, 0.5);
        assert_eq!(measurement.coherence_score, 0.7);
        assert_eq!(measurement.dimensions.len(), 1);
    }
} 