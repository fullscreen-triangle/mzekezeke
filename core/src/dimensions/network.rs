use crate::types::{DimensionMeasurement, DimensionType, MdtecError, MdtecResult};
use crate::utils::math::Statistics;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

/// Network measurement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkMeasurement {
    /// Round-trip time in milliseconds
    pub rtt_ms: f64,
    /// Bandwidth in Mbps
    pub bandwidth_mbps: f64,
    /// Packet loss percentage (0-100)
    pub packet_loss: f64,
    /// Network jitter in milliseconds
    pub jitter_ms: f64,
    /// Connection quality score (0-1)
    pub quality_score: f64,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

/// Network connection information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConnection {
    /// Connection type (WiFi, Ethernet, Cellular)
    pub connection_type: String,
    /// Signal strength (dBm for wireless)
    pub signal_strength: Option<f64>,
    /// Connection speed in Mbps
    pub connection_speed: f64,
    /// IP address
    pub ip_address: String,
    /// MAC address
    pub mac_address: String,
    /// Network interface name
    pub interface_name: String,
}

/// Network topology information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopology {
    /// Number of network hops
    pub hop_count: usize,
    /// Gateway latency in milliseconds
    pub gateway_latency: f64,
    /// DNS resolution time in milliseconds
    pub dns_resolution_time: f64,
    /// Available network interfaces
    pub interfaces: Vec<String>,
    /// Active connections
    pub active_connections: usize,
}

/// Network anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAnomaly {
    /// Anomaly type
    pub anomaly_type: NetworkAnomalyType,
    /// Severity (0-1)
    pub severity: f64,
    /// Description
    pub description: String,
    /// Timestamp when detected
    pub timestamp: Instant,
}

/// Types of network anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkAnomalyType {
    /// High latency spike
    LatencySpike,
    /// Packet loss burst
    PacketLossBurst,
    /// Bandwidth degradation
    BandwidthDegradation,
    /// Connection instability
    ConnectionInstability,
    /// DNS resolution failure
    DnsFailure,
}

/// Network dimension configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Sample window size for network analysis
    pub sample_window_size: usize,
    /// Latency threshold for anomaly detection (ms)
    pub latency_threshold: f64,
    /// Packet loss threshold for anomaly detection (%)
    pub packet_loss_threshold: f64,
    /// Bandwidth threshold for anomaly detection (Mbps)
    pub bandwidth_threshold: f64,
    /// Enable topology analysis
    pub enable_topology_analysis: bool,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Network measurement interval (seconds)
    pub measurement_interval: f64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            sample_window_size: 200,
            latency_threshold: 100.0,
            packet_loss_threshold: 1.0,
            bandwidth_threshold: 1.0,
            enable_topology_analysis: true,
            enable_anomaly_detection: true,
            measurement_interval: 1.0,
        }
    }
}

/// Network dimension analyzer
pub struct NetworkAnalyzer {
    config: NetworkConfig,
    network_readings: VecDeque<NetworkMeasurement>,
    connections: Vec<NetworkConnection>,
    topology: Option<NetworkTopology>,
    detected_anomalies: Vec<NetworkAnomaly>,
    baseline_rtt: Option<f64>,
    baseline_bandwidth: Option<f64>,
    last_anomaly_time: Option<Instant>,
}

impl NetworkAnalyzer {
    pub fn new(config: NetworkConfig) -> Self {
        Self {
            config,
            network_readings: VecDeque::new(),
            connections: Vec::new(),
            topology: None,
            detected_anomalies: Vec::new(),
            baseline_rtt: None,
            baseline_bandwidth: None,
            last_anomaly_time: None,
        }
    }

    /// Add network measurement
    pub fn add_measurement(&mut self, measurement: NetworkMeasurement) -> MdtecResult<()> {
        // Validate measurement
        if measurement.rtt_ms < 0.0 || measurement.rtt_ms > 10000.0 {
            return Err(MdtecError::InvalidInput(format!(
                "RTT {} ms is out of valid range (0-10000 ms)",
                measurement.rtt_ms
            )));
        }

        if measurement.bandwidth_mbps < 0.0 || measurement.bandwidth_mbps > 10000.0 {
            return Err(MdtecError::InvalidInput(format!(
                "Bandwidth {} Mbps is out of valid range (0-10000 Mbps)",
                measurement.bandwidth_mbps
            )));
        }

        if measurement.packet_loss < 0.0 || measurement.packet_loss > 100.0 {
            return Err(MdtecError::InvalidInput(format!(
                "Packet loss {} is out of valid range (0-100%)",
                measurement.packet_loss
            )));
        }

        self.network_readings.push_back(measurement);

        // Maintain window size
        while self.network_readings.len() > self.config.sample_window_size {
            self.network_readings.pop_front();
        }

        // Update baseline measurements
        self.update_baseline_measurements();

        // Detect anomalies
        if self.config.enable_anomaly_detection {
            self.detect_anomalies(&measurement)?;
        }

        Ok(())
    }

    /// Update connection information
    pub fn update_connections(&mut self, connections: Vec<NetworkConnection>) {
        self.connections = connections;
    }

    /// Update network topology
    pub fn update_topology(&mut self, topology: NetworkTopology) {
        if self.config.enable_topology_analysis {
            self.topology = Some(topology);
        }
    }

    /// Update baseline measurements
    fn update_baseline_measurements(&mut self) {
        if self.network_readings.len() < 20 {
            return;
        }

        let recent_readings: Vec<_> = self.network_readings.iter().rev().take(50).collect();
        let rtt_values: Vec<f64> = recent_readings.iter().map(|r| r.rtt_ms).collect();
        let bandwidth_values: Vec<f64> = recent_readings.iter().map(|r| r.bandwidth_mbps).collect();

        // Use median as baseline to avoid outlier influence
        let mut sorted_rtt = rtt_values.clone();
        sorted_rtt.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.baseline_rtt = Some(sorted_rtt[sorted_rtt.len() / 2]);

        let mut sorted_bandwidth = bandwidth_values.clone();
        sorted_bandwidth.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.baseline_bandwidth = Some(sorted_bandwidth[sorted_bandwidth.len() / 2]);
    }

    /// Detect network anomalies
    fn detect_anomalies(&mut self, measurement: &NetworkMeasurement) -> MdtecResult<()> {
        // Latency anomaly detection
        if let Some(baseline_rtt) = self.baseline_rtt {
            if measurement.rtt_ms > baseline_rtt + self.config.latency_threshold {
                let anomaly = NetworkAnomaly {
                    anomaly_type: NetworkAnomalyType::LatencySpike,
                    severity: ((measurement.rtt_ms - baseline_rtt) / self.config.latency_threshold).min(1.0),
                    description: format!("RTT spike: {} ms (baseline: {} ms)", measurement.rtt_ms, baseline_rtt),
                    timestamp: measurement.timestamp,
                };
                self.detected_anomalies.push(anomaly);
                self.last_anomaly_time = Some(measurement.timestamp);
            }
        }

        // Packet loss anomaly detection
        if measurement.packet_loss > self.config.packet_loss_threshold {
            let anomaly = NetworkAnomaly {
                anomaly_type: NetworkAnomalyType::PacketLossBurst,
                severity: (measurement.packet_loss / 10.0).min(1.0),
                description: format!("Packet loss burst: {}%", measurement.packet_loss),
                timestamp: measurement.timestamp,
            };
            self.detected_anomalies.push(anomaly);
            self.last_anomaly_time = Some(measurement.timestamp);
        }

        // Bandwidth degradation detection
        if let Some(baseline_bandwidth) = self.baseline_bandwidth {
            if measurement.bandwidth_mbps < baseline_bandwidth - self.config.bandwidth_threshold {
                let anomaly = NetworkAnomaly {
                    anomaly_type: NetworkAnomalyType::BandwidthDegradation,
                    severity: ((baseline_bandwidth - measurement.bandwidth_mbps) / baseline_bandwidth).min(1.0),
                    description: format!("Bandwidth degradation: {} Mbps (baseline: {} Mbps)", 
                                       measurement.bandwidth_mbps, baseline_bandwidth),
                    timestamp: measurement.timestamp,
                };
                self.detected_anomalies.push(anomaly);
                self.last_anomaly_time = Some(measurement.timestamp);
            }
        }

        // Limit anomaly history
        if self.detected_anomalies.len() > 100 {
            self.detected_anomalies.drain(0..50);
        }

        Ok(())
    }

    /// Calculate network entropy
    pub fn calculate_network_entropy(&self) -> MdtecResult<f64> {
        if self.network_readings.len() < 10 {
            return Err(MdtecError::InsufficientData("Not enough network readings".to_string()));
        }

        let mut entropy = 0.0;

        // RTT entropy (35% weight)
        let rtt_entropy = self.calculate_rtt_entropy()?;
        entropy += rtt_entropy * 0.35;

        // Bandwidth entropy (30% weight)
        let bandwidth_entropy = self.calculate_bandwidth_entropy()?;
        entropy += bandwidth_entropy * 0.30;

        // Packet loss entropy (20% weight)
        let packet_loss_entropy = self.calculate_packet_loss_entropy()?;
        entropy += packet_loss_entropy * 0.20;

        // Jitter entropy (15% weight)
        let jitter_entropy = self.calculate_jitter_entropy()?;
        entropy += jitter_entropy * 0.15;

        Ok(entropy.min(1.0))
    }

    /// Calculate RTT entropy
    fn calculate_rtt_entropy(&self) -> MdtecResult<f64> {
        let rtt_values: Vec<f64> = self.network_readings.iter().map(|r| r.rtt_ms).collect();
        
        if rtt_values.is_empty() {
            return Ok(0.0);
        }

        let rtt_variations: Vec<f64> = rtt_values
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .collect();

        if rtt_variations.is_empty() {
            return Ok(0.0);
        }

        let entropy = Statistics::shannon_entropy(&rtt_variations);
        Ok(entropy)
    }

    /// Calculate bandwidth entropy
    fn calculate_bandwidth_entropy(&self) -> MdtecResult<f64> {
        let bandwidth_values: Vec<f64> = self.network_readings.iter().map(|r| r.bandwidth_mbps).collect();
        
        if bandwidth_values.is_empty() {
            return Ok(0.0);
        }

        let bandwidth_variations: Vec<f64> = bandwidth_values
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .collect();

        if bandwidth_variations.is_empty() {
            return Ok(0.0);
        }

        let entropy = Statistics::shannon_entropy(&bandwidth_variations);
        Ok(entropy)
    }

    /// Calculate packet loss entropy
    fn calculate_packet_loss_entropy(&self) -> MdtecResult<f64> {
        let packet_loss_values: Vec<f64> = self.network_readings.iter().map(|r| r.packet_loss).collect();
        
        if packet_loss_values.is_empty() {
            return Ok(0.0);
        }

        // Add small epsilon to avoid log(0)
        let adjusted_values: Vec<f64> = packet_loss_values.iter().map(|&v| v + 0.001).collect();
        let entropy = Statistics::shannon_entropy(&adjusted_values);
        Ok(entropy)
    }

    /// Calculate jitter entropy
    fn calculate_jitter_entropy(&self) -> MdtecResult<f64> {
        let jitter_values: Vec<f64> = self.network_readings.iter().map(|r| r.jitter_ms).collect();
        
        if jitter_values.is_empty() {
            return Ok(0.0);
        }

        let entropy = Statistics::shannon_entropy(&jitter_values);
        Ok(entropy)
    }

    /// Calculate network stability score
    pub fn calculate_stability_score(&self) -> f64 {
        if self.network_readings.len() < 10 {
            return 0.0;
        }

        let rtt_values: Vec<f64> = self.network_readings.iter().map(|r| r.rtt_ms).collect();
        let bandwidth_values: Vec<f64> = self.network_readings.iter().map(|r| r.bandwidth_mbps).collect();
        let packet_loss_values: Vec<f64> = self.network_readings.iter().map(|r| r.packet_loss).collect();

        let rtt_cv = Statistics::std_dev(&rtt_values).unwrap_or(0.0) / Statistics::mean(&rtt_values).max(1.0);
        let bandwidth_cv = Statistics::std_dev(&bandwidth_values).unwrap_or(0.0) / Statistics::mean(&bandwidth_values).max(1.0);
        let avg_packet_loss = Statistics::mean(&packet_loss_values);

        // Lower coefficient of variation and packet loss = higher stability
        let stability = 1.0 - (rtt_cv + bandwidth_cv + avg_packet_loss / 100.0).min(1.0);
        stability.max(0.0)
    }

    /// Analyze connection quality
    pub fn analyze_connection_quality(&self) -> HashMap<String, f64> {
        let mut quality_metrics = HashMap::new();

        if self.network_readings.is_empty() {
            return quality_metrics;
        }

        let rtt_values: Vec<f64> = self.network_readings.iter().map(|r| r.rtt_ms).collect();
        let bandwidth_values: Vec<f64> = self.network_readings.iter().map(|r| r.bandwidth_mbps).collect();
        let packet_loss_values: Vec<f64> = self.network_readings.iter().map(|r| r.packet_loss).collect();
        let jitter_values: Vec<f64> = self.network_readings.iter().map(|r| r.jitter_ms).collect();

        quality_metrics.insert("avg_rtt".to_string(), Statistics::mean(&rtt_values));
        quality_metrics.insert("avg_bandwidth".to_string(), Statistics::mean(&bandwidth_values));
        quality_metrics.insert("avg_packet_loss".to_string(), Statistics::mean(&packet_loss_values));
        quality_metrics.insert("avg_jitter".to_string(), Statistics::mean(&jitter_values));
        quality_metrics.insert("stability_score".to_string(), self.calculate_stability_score());

        quality_metrics
    }

    /// Get current network measurement
    pub fn get_measurement(&self) -> MdtecResult<DimensionMeasurement> {
        let entropy = self.calculate_network_entropy()?;
        let quality = self.calculate_measurement_quality();

        let mut metadata = std::collections::HashMap::new();
        
        if let Some(measurement) = self.network_readings.back() {
            metadata.insert("rtt_ms".to_string(), measurement.rtt_ms.to_string());
            metadata.insert("bandwidth_mbps".to_string(), measurement.bandwidth_mbps.to_string());
            metadata.insert("packet_loss".to_string(), measurement.packet_loss.to_string());
            metadata.insert("jitter_ms".to_string(), measurement.jitter_ms.to_string());
            metadata.insert("quality_score".to_string(), measurement.quality_score.to_string());
        }

        metadata.insert("sample_count".to_string(), self.network_readings.len().to_string());
        metadata.insert("connection_count".to_string(), self.connections.len().to_string());
        metadata.insert("detected_anomalies".to_string(), self.detected_anomalies.len().to_string());
        metadata.insert("stability_score".to_string(), self.calculate_stability_score().to_string());

        if let Some(baseline_rtt) = self.baseline_rtt {
            metadata.insert("baseline_rtt".to_string(), baseline_rtt.to_string());
        }

        if let Some(baseline_bandwidth) = self.baseline_bandwidth {
            metadata.insert("baseline_bandwidth".to_string(), baseline_bandwidth.to_string());
        }

        if let Some(topology) = &self.topology {
            metadata.insert("hop_count".to_string(), topology.hop_count.to_string());
            metadata.insert("gateway_latency".to_string(), topology.gateway_latency.to_string());
            metadata.insert("dns_resolution_time".to_string(), topology.dns_resolution_time.to_string());
            metadata.insert("active_connections".to_string(), topology.active_connections.to_string());
        }

        if let Some(last_anomaly) = self.last_anomaly_time {
            metadata.insert("last_anomaly_age".to_string(), last_anomaly.elapsed().as_secs().to_string());
        }

        Ok(DimensionMeasurement {
            dimension_type: DimensionType::Network,
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
        let sample_coverage = (self.network_readings.len() as f64 / self.config.sample_window_size as f64).min(1.0);
        quality += sample_coverage * 0.4;

        // Connection information quality
        let connection_quality = if self.connections.is_empty() { 0.0 } else { 1.0 };
        quality += connection_quality * 0.2;

        // Topology information quality
        let topology_quality = if self.topology.is_some() { 1.0 } else { 0.0 };
        quality += topology_quality * 0.2;

        // Baseline establishment quality
        let baseline_quality = if self.baseline_rtt.is_some() && self.baseline_bandwidth.is_some() { 1.0 } else { 0.0 };
        quality += baseline_quality * 0.2;

        quality.min(1.0)
    }

    /// Get network statistics
    pub fn get_statistics(&self) -> NetworkStatistics {
        let rtt_values: Vec<f64> = self.network_readings.iter().map(|r| r.rtt_ms).collect();
        let bandwidth_values: Vec<f64> = self.network_readings.iter().map(|r| r.bandwidth_mbps).collect();
        let packet_loss_values: Vec<f64> = self.network_readings.iter().map(|r| r.packet_loss).collect();
        let jitter_values: Vec<f64> = self.network_readings.iter().map(|r| r.jitter_ms).collect();

        NetworkStatistics {
            sample_count: self.network_readings.len(),
            connection_count: self.connections.len(),
            rtt_mean: Statistics::mean(&rtt_values),
            rtt_std: Statistics::std_dev(&rtt_values).unwrap_or(0.0),
            bandwidth_mean: Statistics::mean(&bandwidth_values),
            bandwidth_std: Statistics::std_dev(&bandwidth_values).unwrap_or(0.0),
            packet_loss_mean: Statistics::mean(&packet_loss_values),
            packet_loss_std: Statistics::std_dev(&packet_loss_values).unwrap_or(0.0),
            jitter_mean: Statistics::mean(&jitter_values),
            jitter_std: Statistics::std_dev(&jitter_values).unwrap_or(0.0),
            stability_score: self.calculate_stability_score(),
            detected_anomalies: self.detected_anomalies.len(),
            baseline_rtt: self.baseline_rtt,
            baseline_bandwidth: self.baseline_bandwidth,
            current_entropy: self.calculate_network_entropy().unwrap_or(0.0),
            measurement_quality: self.calculate_measurement_quality(),
        }
    }

    /// Reset network analyzer
    pub fn reset(&mut self) {
        self.network_readings.clear();
        self.connections.clear();
        self.topology = None;
        self.detected_anomalies.clear();
        self.baseline_rtt = None;
        self.baseline_bandwidth = None;
        self.last_anomaly_time = None;
    }
}

/// Network measurement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatistics {
    pub sample_count: usize,
    pub connection_count: usize,
    pub rtt_mean: f64,
    pub rtt_std: f64,
    pub bandwidth_mean: f64,
    pub bandwidth_std: f64,
    pub packet_loss_mean: f64,
    pub packet_loss_std: f64,
    pub jitter_mean: f64,
    pub jitter_std: f64,
    pub stability_score: f64,
    pub detected_anomalies: usize,
    pub baseline_rtt: Option<f64>,
    pub baseline_bandwidth: Option<f64>,
    pub current_entropy: f64,
    pub measurement_quality: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_measurement_creation() {
        let measurement = NetworkMeasurement {
            rtt_ms: 25.0,
            bandwidth_mbps: 100.0,
            packet_loss: 0.5,
            jitter_ms: 2.0,
            quality_score: 0.9,
            timestamp: Instant::now(),
        };

        assert_eq!(measurement.rtt_ms, 25.0);
        assert_eq!(measurement.bandwidth_mbps, 100.0);
        assert_eq!(measurement.packet_loss, 0.5);
        assert_eq!(measurement.jitter_ms, 2.0);
        assert_eq!(measurement.quality_score, 0.9);
    }

    #[test]
    fn test_network_analyzer_add_measurement() {
        let mut analyzer = NetworkAnalyzer::new(NetworkConfig::default());

        let measurement = NetworkMeasurement {
            rtt_ms: 25.0,
            bandwidth_mbps: 100.0,
            packet_loss: 0.5,
            jitter_ms: 2.0,
            quality_score: 0.9,
            timestamp: Instant::now(),
        };

        assert!(analyzer.add_measurement(measurement).is_ok());
        assert_eq!(analyzer.network_readings.len(), 1);
    }

    #[test]
    fn test_stability_score_calculation() {
        let mut analyzer = NetworkAnalyzer::new(NetworkConfig::default());

        // Add measurements with low variability (high stability)
        for _ in 0..20 {
            let measurement = NetworkMeasurement {
                rtt_ms: 25.0, // Constant RTT
                bandwidth_mbps: 100.0, // Constant bandwidth
                packet_loss: 0.0, // No packet loss
                jitter_ms: 1.0,
                quality_score: 0.95,
                timestamp: Instant::now(),
            };
            analyzer.add_measurement(measurement).unwrap();
        }

        let stability = analyzer.calculate_stability_score();
        assert!(stability > 0.8); // Should be high stability
    }

    #[test]
    fn test_anomaly_detection() {
        let mut analyzer = NetworkAnalyzer::new(NetworkConfig::default());

        // Add baseline measurements
        for _ in 0..25 {
            let measurement = NetworkMeasurement {
                rtt_ms: 25.0,
                bandwidth_mbps: 100.0,
                packet_loss: 0.0,
                jitter_ms: 1.0,
                quality_score: 0.95,
                timestamp: Instant::now(),
            };
            analyzer.add_measurement(measurement).unwrap();
        }

        // Add anomalous measurement
        let anomalous_measurement = NetworkMeasurement {
            rtt_ms: 200.0, // High RTT
            bandwidth_mbps: 10.0, // Low bandwidth
            packet_loss: 5.0, // High packet loss
            jitter_ms: 20.0,
            quality_score: 0.3,
            timestamp: Instant::now(),
        };

        analyzer.add_measurement(anomalous_measurement).unwrap();

        assert!(!analyzer.detected_anomalies.is_empty());
    }

    #[test]
    fn test_connection_quality_analysis() {
        let mut analyzer = NetworkAnalyzer::new(NetworkConfig::default());

        let measurements = vec![
            NetworkMeasurement {
                rtt_ms: 20.0,
                bandwidth_mbps: 100.0,
                packet_loss: 0.0,
                jitter_ms: 1.0,
                quality_score: 0.95,
                timestamp: Instant::now(),
            },
            NetworkMeasurement {
                rtt_ms: 25.0,
                bandwidth_mbps: 95.0,
                packet_loss: 0.1,
                jitter_ms: 1.5,
                quality_score: 0.9,
                timestamp: Instant::now(),
            },
        ];

        for measurement in measurements {
            analyzer.add_measurement(measurement).unwrap();
        }

        let quality_metrics = analyzer.analyze_connection_quality();
        assert!(quality_metrics.contains_key("avg_rtt"));
        assert!(quality_metrics.contains_key("avg_bandwidth"));
        assert!(quality_metrics.contains_key("stability_score"));
    }
} 