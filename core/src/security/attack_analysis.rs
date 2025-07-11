use crate::types::{MdtecError, MdtecResult};
use crate::utils::math::Statistics;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, Duration, Instant};

/// Attack pattern classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttackPattern {
    /// Brute force attack
    BruteForce,
    /// Dictionary attack
    Dictionary,
    /// Rainbow table attack
    RainbowTable,
    /// Timing attack
    TimingAttack,
    /// Power analysis attack
    PowerAnalysis,
    /// Electromagnetic analysis
    ElectromagneticAnalysis,
    /// Fault injection
    FaultInjection,
    /// Replay attack
    ReplayAttack,
    /// Man-in-the-middle
    ManInTheMiddle,
    /// Social engineering
    SocialEngineering,
    /// Phishing
    Phishing,
    /// Malware
    Malware,
    /// Ransomware
    Ransomware,
    /// Advanced Persistent Threat (APT)
    APT,
    /// Zero-day exploit
    ZeroDay,
    /// Privilege escalation
    PrivilegeEscalation,
    /// Data exfiltration
    DataExfiltration,
    /// Denial of Service
    DoS,
    /// Distributed Denial of Service
    DDoS,
    /// SQL injection
    SQLInjection,
    /// Cross-site scripting
    XSS,
    /// Buffer overflow
    BufferOverflow,
    /// Environmental manipulation
    EnvironmentalManipulation,
    /// Sensor spoofing
    SensorSpoofing,
    /// Temporal manipulation
    TemporalManipulation,
    /// Multi-dimensional attack
    MultiDimensional,
}

/// Attack phase classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttackPhase {
    /// Reconnaissance
    Reconnaissance,
    /// Weaponization
    Weaponization,
    /// Delivery
    Delivery,
    /// Exploitation
    Exploitation,
    /// Installation
    Installation,
    /// Command and Control
    CommandControl,
    /// Actions on Objectives
    ActionsOnObjectives,
}

/// Attack severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AttackSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Attack detection confidence
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DetectionConfidence {
    /// Very low confidence
    VeryLow,
    /// Low confidence
    Low,
    /// Medium confidence
    Medium,
    /// High confidence
    High,
    /// Very high confidence
    VeryHigh,
}

/// Attack indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackIndicator {
    /// Indicator type
    pub indicator_type: String,
    /// Indicator value
    pub value: String,
    /// Confidence in indicator
    pub confidence: f64,
    /// Timestamp when observed
    pub timestamp: SystemTime,
    /// Source of indicator
    pub source: String,
    /// Associated attack patterns
    pub associated_patterns: Vec<AttackPattern>,
}

/// Attack event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackEvent {
    /// Event ID
    pub event_id: String,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Attack pattern
    pub pattern: AttackPattern,
    /// Attack phase
    pub phase: AttackPhase,
    /// Severity
    pub severity: AttackSeverity,
    /// Detection confidence
    pub confidence: DetectionConfidence,
    /// Source IP/location
    pub source: Option<String>,
    /// Target component
    pub target: String,
    /// Attack indicators
    pub indicators: Vec<AttackIndicator>,
    /// Event description
    pub description: String,
    /// Raw event data
    pub raw_data: HashMap<String, String>,
}

/// Attack campaign
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackCampaign {
    /// Campaign ID
    pub campaign_id: String,
    /// Campaign name
    pub name: String,
    /// Associated events
    pub events: Vec<String>,
    /// Start time
    pub start_time: SystemTime,
    /// End time (if known)
    pub end_time: Option<SystemTime>,
    /// Attack patterns used
    pub patterns: Vec<AttackPattern>,
    /// Attributed actor
    pub attributed_actor: Option<String>,
    /// Campaign objectives
    pub objectives: Vec<String>,
    /// Tactics, Techniques, and Procedures (TTPs)
    pub ttps: Vec<String>,
    /// Impact assessment
    pub impact: AttackImpact,
}

/// Attack impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackImpact {
    /// Confidentiality impact
    pub confidentiality: AttackSeverity,
    /// Integrity impact
    pub integrity: AttackSeverity,
    /// Availability impact
    pub availability: AttackSeverity,
    /// Financial impact (estimated)
    pub financial_impact: Option<f64>,
    /// Affected systems count
    pub affected_systems: usize,
    /// Affected users count
    pub affected_users: usize,
    /// Recovery time (estimated)
    pub recovery_time: Option<Duration>,
}

/// Attack analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackAnalysisConfig {
    /// Event correlation window
    pub correlation_window: Duration,
    /// Minimum confidence threshold
    pub min_confidence_threshold: f64,
    /// Maximum events to retain
    pub max_events_retained: usize,
    /// Pattern detection sensitivity
    pub pattern_detection_sensitivity: f64,
    /// Enable real-time analysis
    pub enable_realtime_analysis: bool,
    /// Enable campaign detection
    pub enable_campaign_detection: bool,
    /// Forensic analysis depth
    pub forensic_analysis_depth: usize,
    /// Threat intelligence integration
    pub threat_intelligence_enabled: bool,
}

impl Default for AttackAnalysisConfig {
    fn default() -> Self {
        Self {
            correlation_window: Duration::from_secs(3600), // 1 hour
            min_confidence_threshold: 0.7,
            max_events_retained: 10000,
            pattern_detection_sensitivity: 0.8,
            enable_realtime_analysis: true,
            enable_campaign_detection: true,
            forensic_analysis_depth: 3,
            threat_intelligence_enabled: true,
        }
    }
}

/// Attack analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackAnalysisResult {
    /// Analysis timestamp
    pub timestamp: SystemTime,
    /// Detected patterns
    pub detected_patterns: Vec<AttackPattern>,
    /// Confidence scores
    pub confidence_scores: HashMap<AttackPattern, f64>,
    /// Attack timeline
    pub timeline: Vec<AttackEvent>,
    /// Risk assessment
    pub risk_assessment: f64,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
    /// Analysis metadata
    pub metadata: HashMap<String, String>,
}

/// Attack analyzer
pub struct AttackAnalyzer {
    config: AttackAnalysisConfig,
    events: VecDeque<AttackEvent>,
    campaigns: HashMap<String, AttackCampaign>,
    pattern_signatures: HashMap<AttackPattern, Vec<String>>,
    analysis_history: Vec<AttackAnalysisResult>,
    last_analysis_time: Option<Instant>,
}

impl AttackAnalyzer {
    pub fn new(config: AttackAnalysisConfig) -> Self {
        let mut analyzer = Self {
            config,
            events: VecDeque::new(),
            campaigns: HashMap::new(),
            pattern_signatures: HashMap::new(),
            analysis_history: Vec::new(),
            last_analysis_time: None,
        };
        
        analyzer.initialize_pattern_signatures();
        analyzer
    }

    /// Initialize pattern signatures for detection
    fn initialize_pattern_signatures(&mut self) {
        // Brute force signatures
        self.pattern_signatures.insert(AttackPattern::BruteForce, vec![
            "high_login_failure_rate".to_string(),
            "sequential_password_attempts".to_string(),
            "multiple_accounts_targeted".to_string(),
        ]);

        // Timing attack signatures
        self.pattern_signatures.insert(AttackPattern::TimingAttack, vec![
            "consistent_timing_patterns".to_string(),
            "microsecond_precision_timing".to_string(),
            "statistical_timing_analysis".to_string(),
        ]);

        // Environmental manipulation signatures
        self.pattern_signatures.insert(AttackPattern::EnvironmentalManipulation, vec![
            "unusual_sensor_readings".to_string(),
            "environmental_parameter_spikes".to_string(),
            "sensor_correlation_anomalies".to_string(),
        ]);

        // Power analysis signatures
        self.pattern_signatures.insert(AttackPattern::PowerAnalysis, vec![
            "power_consumption_monitoring".to_string(),
            "cryptographic_operation_correlation".to_string(),
            "side_channel_analysis_patterns".to_string(),
        ]);

        // Electromagnetic analysis signatures
        self.pattern_signatures.insert(AttackPattern::ElectromagneticAnalysis, vec![
            "em_emission_monitoring".to_string(),
            "rf_spectrum_analysis".to_string(),
            "electromagnetic_side_channel".to_string(),
        ]);

        // Advanced persistent threat signatures
        self.pattern_signatures.insert(AttackPattern::APT, vec![
            "long_term_persistence".to_string(),
            "lateral_movement".to_string(),
            "data_staging".to_string(),
            "covert_channels".to_string(),
        ]);

        // Multi-dimensional attack signatures
        self.pattern_signatures.insert(AttackPattern::MultiDimensional, vec![
            "coordinated_multi_vector".to_string(),
            "simultaneous_dimension_attacks".to_string(),
            "cross_dimensional_correlation".to_string(),
        ]);
    }

    /// Add attack event
    pub fn add_event(&mut self, event: AttackEvent) -> MdtecResult<()> {
        self.events.push_back(event);
        
        // Maintain event limit
        while self.events.len() > self.config.max_events_retained {
            self.events.pop_front();
        }
        
        // Real-time analysis if enabled
        if self.config.enable_realtime_analysis {
            self.analyze_realtime()?;
        }
        
        Ok(())
    }

    /// Analyze attack patterns
    pub fn analyze_patterns(&mut self) -> MdtecResult<AttackAnalysisResult> {
        let current_time = SystemTime::now();
        let analysis_start = Instant::now();
        
        // Collect events within correlation window
        let correlation_cutoff = current_time - self.config.correlation_window;
        let relevant_events: Vec<_> = self.events
            .iter()
            .filter(|e| e.timestamp >= correlation_cutoff)
            .collect();
        
        let mut detected_patterns = Vec::new();
        let mut confidence_scores = HashMap::new();
        
        // Analyze each attack pattern
        for pattern in &[
            AttackPattern::BruteForce,
            AttackPattern::TimingAttack,
            AttackPattern::EnvironmentalManipulation,
            AttackPattern::PowerAnalysis,
            AttackPattern::ElectromagneticAnalysis,
            AttackPattern::APT,
            AttackPattern::MultiDimensional,
        ] {
            let confidence = self.analyze_pattern(&relevant_events, *pattern)?;
            
            if confidence >= self.config.min_confidence_threshold {
                detected_patterns.push(*pattern);
                confidence_scores.insert(*pattern, confidence);
            }
        }
        
        // Calculate risk assessment
        let risk_assessment = self.calculate_risk_assessment(&detected_patterns, &confidence_scores);
        
        // Generate recommended actions
        let recommended_actions = self.generate_recommended_actions(&detected_patterns);
        
        // Create timeline
        let timeline = relevant_events.into_iter().cloned().collect();
        
        // Create analysis metadata
        let mut metadata = HashMap::new();
        metadata.insert("analysis_duration".to_string(), 
                        format!("{:.2}ms", analysis_start.elapsed().as_millis()));
        metadata.insert("events_analyzed".to_string(), 
                        self.events.len().to_string());
        metadata.insert("correlation_window".to_string(), 
                        format!("{:.0}s", self.config.correlation_window.as_secs()));
        
        let result = AttackAnalysisResult {
            timestamp: current_time,
            detected_patterns,
            confidence_scores,
            timeline,
            risk_assessment,
            recommended_actions,
            metadata,
        };
        
        // Store in analysis history
        self.analysis_history.push(result.clone());
        
        // Limit history size
        if self.analysis_history.len() > 100 {
            self.analysis_history.remove(0);
        }
        
        self.last_analysis_time = Some(analysis_start);
        
        Ok(result)
    }

    /// Analyze specific attack pattern
    fn analyze_pattern(&self, events: &[&AttackEvent], pattern: AttackPattern) -> MdtecResult<f64> {
        let pattern_events: Vec<_> = events
            .iter()
            .filter(|e| e.pattern == pattern)
            .collect();
        
        if pattern_events.is_empty() {
            return Ok(0.0);
        }
        
        let mut confidence = 0.0;
        
        // Base confidence from event count
        let event_count_factor = (pattern_events.len() as f64 / 10.0).min(1.0);
        confidence += event_count_factor * 0.3;
        
        // Confidence from detection confidence
        let avg_detection_confidence = pattern_events
            .iter()
            .map(|e| self.confidence_to_score(e.confidence))
            .sum::<f64>() / pattern_events.len() as f64;
        confidence += avg_detection_confidence * 0.4;
        
        // Temporal clustering analysis
        let temporal_clustering = self.analyze_temporal_clustering(&pattern_events)?;
        confidence += temporal_clustering * 0.3;
        
        // Check for signature matches
        if let Some(signatures) = self.pattern_signatures.get(&pattern) {
            let signature_matches = self.check_signature_matches(&pattern_events, signatures);
            confidence += signature_matches * 0.2;
        }
        
        // Apply sensitivity adjustment
        confidence *= self.config.pattern_detection_sensitivity;
        
        Ok(confidence.min(1.0))
    }

    /// Convert detection confidence to numeric score
    fn confidence_to_score(&self, confidence: DetectionConfidence) -> f64 {
        match confidence {
            DetectionConfidence::VeryLow => 0.2,
            DetectionConfidence::Low => 0.4,
            DetectionConfidence::Medium => 0.6,
            DetectionConfidence::High => 0.8,
            DetectionConfidence::VeryHigh => 1.0,
        }
    }

    /// Analyze temporal clustering of events
    fn analyze_temporal_clustering(&self, events: &[&AttackEvent]) -> MdtecResult<f64> {
        if events.len() < 2 {
            return Ok(0.0);
        }
        
        let mut timestamps = Vec::new();
        for event in events {
            if let Ok(duration) = event.timestamp.duration_since(SystemTime::UNIX_EPOCH) {
                timestamps.push(duration.as_secs_f64());
            }
        }
        
        if timestamps.len() < 2 {
            return Ok(0.0);
        }
        
        timestamps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate time differences
        let time_diffs: Vec<f64> = timestamps
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        
        // Calculate clustering coefficient
        let std_dev = Statistics::std_dev(&time_diffs).unwrap_or(0.0);
        let mean = Statistics::mean(&time_diffs);
        
        let clustering_score = if mean > 0.0 {
            1.0 - (std_dev / mean).min(1.0)
        } else {
            0.0
        };
        
        Ok(clustering_score)
    }

    /// Check signature matches
    fn check_signature_matches(&self, events: &[&AttackEvent], signatures: &[String]) -> f64 {
        let mut matches = 0;
        let mut total_checks = 0;
        
        for event in events {
            for signature in signatures {
                total_checks += 1;
                
                // Check if signature matches event indicators
                for indicator in &event.indicators {
                    if indicator.indicator_type.contains(signature) || 
                       indicator.value.contains(signature) {
                        matches += 1;
                        break;
                    }
                }
                
                // Check raw data
                for (key, value) in &event.raw_data {
                    if key.contains(signature) || value.contains(signature) {
                        matches += 1;
                        break;
                    }
                }
            }
        }
        
        if total_checks > 0 {
            matches as f64 / total_checks as f64
        } else {
            0.0
        }
    }

    /// Calculate overall risk assessment
    fn calculate_risk_assessment(&self, patterns: &[AttackPattern], confidence_scores: &HashMap<AttackPattern, f64>) -> f64 {
        if patterns.is_empty() {
            return 0.0;
        }
        
        let mut risk_score = 0.0;
        
        for pattern in patterns {
            let confidence = confidence_scores.get(pattern).unwrap_or(&0.0);
            let pattern_risk = self.get_pattern_risk_weight(*pattern);
            risk_score += confidence * pattern_risk;
        }
        
        // Normalize to 0-100 scale
        (risk_score * 100.0 / patterns.len() as f64).min(100.0)
    }

    /// Get risk weight for attack pattern
    fn get_pattern_risk_weight(&self, pattern: AttackPattern) -> f64 {
        match pattern {
            AttackPattern::MultiDimensional => 0.95,
            AttackPattern::APT => 0.90,
            AttackPattern::ZeroDay => 0.85,
            AttackPattern::PowerAnalysis => 0.80,
            AttackPattern::ElectromagneticAnalysis => 0.80,
            AttackPattern::TimingAttack => 0.75,
            AttackPattern::EnvironmentalManipulation => 0.70,
            AttackPattern::ManInTheMiddle => 0.65,
            AttackPattern::BruteForce => 0.60,
            AttackPattern::DDoS => 0.55,
            AttackPattern::Malware => 0.50,
            _ => 0.45,
        }
    }

    /// Generate recommended actions
    fn generate_recommended_actions(&self, patterns: &[AttackPattern]) -> Vec<String> {
        let mut actions = Vec::new();
        
        for pattern in patterns {
            match pattern {
                AttackPattern::BruteForce => {
                    actions.push("Implement account lockout policies".to_string());
                    actions.push("Enable multi-factor authentication".to_string());
                    actions.push("Monitor failed login attempts".to_string());
                },
                AttackPattern::TimingAttack => {
                    actions.push("Implement constant-time algorithms".to_string());
                    actions.push("Add timing noise to operations".to_string());
                    actions.push("Monitor timing patterns".to_string());
                },
                AttackPattern::EnvironmentalManipulation => {
                    actions.push("Validate sensor readings".to_string());
                    actions.push("Implement sensor correlation checks".to_string());
                    actions.push("Monitor environmental anomalies".to_string());
                },
                AttackPattern::PowerAnalysis => {
                    actions.push("Implement power analysis countermeasures".to_string());
                    actions.push("Use power line filtering".to_string());
                    actions.push("Monitor power consumption patterns".to_string());
                },
                AttackPattern::ElectromagneticAnalysis => {
                    actions.push("Implement electromagnetic shielding".to_string());
                    actions.push("Use spread spectrum techniques".to_string());
                    actions.push("Monitor EM emissions".to_string());
                },
                AttackPattern::APT => {
                    actions.push("Implement advanced threat detection".to_string());
                    actions.push("Monitor for lateral movement".to_string());
                    actions.push("Implement network segmentation".to_string());
                },
                AttackPattern::MultiDimensional => {
                    actions.push("Implement comprehensive monitoring".to_string());
                    actions.push("Use AI-based threat detection".to_string());
                    actions.push("Implement defense in depth".to_string());
                },
                _ => {
                    actions.push("Implement appropriate security controls".to_string());
                    actions.push("Monitor for attack indicators".to_string());
                },
            }
        }
        
        // Remove duplicates
        actions.sort();
        actions.dedup();
        
        actions
    }

    /// Real-time analysis
    fn analyze_realtime(&mut self) -> MdtecResult<()> {
        // Simplified real-time analysis
        if let Some(last_event) = self.events.back() {
            // Check for immediate threats
            if last_event.severity == AttackSeverity::Critical {
                // Trigger immediate response
                self.trigger_immediate_response(last_event)?;
            }
            
            // Check for pattern escalation
            self.check_pattern_escalation()?;
        }
        
        Ok(())
    }

    /// Trigger immediate response
    fn trigger_immediate_response(&self, event: &AttackEvent) -> MdtecResult<()> {
        // In a real implementation, this would trigger alerts, notifications, etc.
        println!("CRITICAL ALERT: {} detected on {}", 
                format!("{:?}", event.pattern), event.target);
        Ok(())
    }

    /// Check for pattern escalation
    fn check_pattern_escalation(&self) -> MdtecResult<()> {
        // Analyze recent events for escalation patterns
        let recent_events: Vec<_> = self.events.iter().rev().take(10).collect();
        
        let mut severity_trend = Vec::new();
        for event in recent_events {
            severity_trend.push(match event.severity {
                AttackSeverity::Low => 1,
                AttackSeverity::Medium => 2,
                AttackSeverity::High => 3,
                AttackSeverity::Critical => 4,
            });
        }
        
        // Check for increasing severity trend
        if severity_trend.len() >= 3 {
            let is_escalating = severity_trend.windows(2).all(|w| w[1] >= w[0]);
            if is_escalating {
                println!("WARNING: Attack pattern escalation detected");
            }
        }
        
        Ok(())
    }

    /// Detect attack campaigns
    pub fn detect_campaigns(&mut self) -> MdtecResult<Vec<AttackCampaign>> {
        if !self.config.enable_campaign_detection {
            return Ok(Vec::new());
        }
        
        let mut campaigns = Vec::new();
        
        // Group events by common characteristics
        let mut event_groups: HashMap<String, Vec<&AttackEvent>> = HashMap::new();
        
        for event in &self.events {
            let group_key = format!("{:?}_{}", event.pattern, 
                                   event.source.as_ref().unwrap_or(&"unknown".to_string()));
            event_groups.entry(group_key).or_insert_with(Vec::new).push(event);
        }
        
        // Analyze each group for campaign characteristics
        for (group_key, events) in event_groups {
            if events.len() >= 3 { // Minimum events for campaign
                let campaign = self.analyze_campaign_characteristics(&group_key, events)?;
                campaigns.push(campaign);
            }
        }
        
        Ok(campaigns)
    }

    /// Analyze campaign characteristics
    fn analyze_campaign_characteristics(&self, group_key: &str, events: Vec<&AttackEvent>) -> MdtecResult<AttackCampaign> {
        let campaign_id = format!("CAMPAIGN_{}", group_key);
        let name = format!("Campaign {}", group_key);
        
        let event_ids: Vec<String> = events.iter().map(|e| e.event_id.clone()).collect();
        
        let start_time = events.iter().map(|e| e.timestamp).min().unwrap_or(SystemTime::now());
        let end_time = events.iter().map(|e| e.timestamp).max();
        
        let mut patterns = Vec::new();
        let mut ttps = Vec::new();
        
        for event in &events {
            if !patterns.contains(&event.pattern) {
                patterns.push(event.pattern);
            }
            
            // Extract TTPs from event description
            if event.description.contains("lateral movement") {
                ttps.push("Lateral Movement".to_string());
            }
            if event.description.contains("privilege escalation") {
                ttps.push("Privilege Escalation".to_string());
            }
            if event.description.contains("data exfiltration") {
                ttps.push("Data Exfiltration".to_string());
            }
        }
        
        ttps.sort();
        ttps.dedup();
        
        let impact = AttackImpact {
            confidentiality: AttackSeverity::Medium,
            integrity: AttackSeverity::Medium,
            availability: AttackSeverity::Low,
            financial_impact: None,
            affected_systems: events.len(),
            affected_users: 0,
            recovery_time: None,
        };
        
        Ok(AttackCampaign {
            campaign_id,
            name,
            events: event_ids,
            start_time,
            end_time,
            patterns,
            attributed_actor: None,
            objectives: vec!["Unknown".to_string()],
            ttps,
            impact,
        })
    }

    /// Get attack statistics
    pub fn get_statistics(&self) -> AttackStatistics {
        let total_events = self.events.len();
        let total_campaigns = self.campaigns.len();
        let total_analyses = self.analysis_history.len();
        
        let pattern_distribution = self.calculate_pattern_distribution();
        let severity_distribution = self.calculate_severity_distribution();
        
        let avg_confidence = if !self.events.is_empty() {
            self.events.iter()
                .map(|e| self.confidence_to_score(e.confidence))
                .sum::<f64>() / self.events.len() as f64
        } else {
            0.0
        };
        
        AttackStatistics {
            total_events,
            total_campaigns,
            total_analyses,
            pattern_distribution,
            severity_distribution,
            avg_confidence,
            last_analysis: self.last_analysis_time,
        }
    }

    /// Calculate pattern distribution
    fn calculate_pattern_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        
        for event in &self.events {
            let pattern_name = format!("{:?}", event.pattern);
            *distribution.entry(pattern_name).or_insert(0) += 1;
        }
        
        distribution
    }

    /// Calculate severity distribution
    fn calculate_severity_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        
        for event in &self.events {
            let severity_name = format!("{:?}", event.severity);
            *distribution.entry(severity_name).or_insert(0) += 1;
        }
        
        distribution
    }

    /// Reset analyzer
    pub fn reset(&mut self) {
        self.events.clear();
        self.campaigns.clear();
        self.analysis_history.clear();
        self.last_analysis_time = None;
    }
}

/// Attack statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackStatistics {
    pub total_events: usize,
    pub total_campaigns: usize,
    pub total_analyses: usize,
    pub pattern_distribution: HashMap<String, usize>,
    pub severity_distribution: HashMap<String, usize>,
    pub avg_confidence: f64,
    pub last_analysis: Option<Instant>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attack_analyzer_creation() {
        let analyzer = AttackAnalyzer::new(AttackAnalysisConfig::default());
        assert_eq!(analyzer.events.len(), 0);
        assert!(!analyzer.pattern_signatures.is_empty());
    }

    #[test]
    fn test_attack_event_addition() {
        let mut analyzer = AttackAnalyzer::new(AttackAnalysisConfig::default());
        
        let event = AttackEvent {
            event_id: "TEST_001".to_string(),
            timestamp: SystemTime::now(),
            pattern: AttackPattern::BruteForce,
            phase: AttackPhase::Exploitation,
            severity: AttackSeverity::Medium,
            confidence: DetectionConfidence::High,
            source: Some("192.168.1.100".to_string()),
            target: "login_system".to_string(),
            indicators: vec![],
            description: "Brute force login attempt".to_string(),
            raw_data: HashMap::new(),
        };
        
        analyzer.add_event(event).unwrap();
        assert_eq!(analyzer.events.len(), 1);
    }

    #[test]
    fn test_pattern_analysis() {
        let mut analyzer = AttackAnalyzer::new(AttackAnalysisConfig::default());
        
        // Add multiple brute force events
        for i in 0..5 {
            let event = AttackEvent {
                event_id: format!("BRUTE_{}", i),
                timestamp: SystemTime::now(),
                pattern: AttackPattern::BruteForce,
                phase: AttackPhase::Exploitation,
                severity: AttackSeverity::Medium,
                confidence: DetectionConfidence::High,
                source: Some("192.168.1.100".to_string()),
                target: "login_system".to_string(),
                indicators: vec![],
                description: "Brute force login attempt".to_string(),
                raw_data: HashMap::new(),
            };
            analyzer.add_event(event).unwrap();
        }
        
        let analysis = analyzer.analyze_patterns().unwrap();
        assert!(analysis.detected_patterns.contains(&AttackPattern::BruteForce));
        assert!(analysis.risk_assessment > 0.0);
        assert!(!analysis.recommended_actions.is_empty());
    }

    #[test]
    fn test_campaign_detection() {
        let mut analyzer = AttackAnalyzer::new(AttackAnalysisConfig::default());
        
        // Add multiple related events
        for i in 0..4 {
            let event = AttackEvent {
                event_id: format!("APT_{}", i),
                timestamp: SystemTime::now(),
                pattern: AttackPattern::APT,
                phase: AttackPhase::Exploitation,
                severity: AttackSeverity::High,
                confidence: DetectionConfidence::High,
                source: Some("192.168.1.100".to_string()),
                target: "server".to_string(),
                indicators: vec![],
                description: "APT lateral movement".to_string(),
                raw_data: HashMap::new(),
            };
            analyzer.add_event(event).unwrap();
        }
        
        let campaigns = analyzer.detect_campaigns().unwrap();
        assert_eq!(campaigns.len(), 1);
        assert_eq!(campaigns[0].events.len(), 4);
    }

    #[test]
    fn test_risk_assessment() {
        let analyzer = AttackAnalyzer::new(AttackAnalysisConfig::default());
        
        let patterns = vec![AttackPattern::MultiDimensional, AttackPattern::APT];
        let mut confidence_scores = HashMap::new();
        confidence_scores.insert(AttackPattern::MultiDimensional, 0.9);
        confidence_scores.insert(AttackPattern::APT, 0.8);
        
        let risk = analyzer.calculate_risk_assessment(&patterns, &confidence_scores);
        assert!(risk > 80.0); // Should be high risk
    }

    #[test]
    fn test_temporal_clustering() {
        let analyzer = AttackAnalyzer::new(AttackAnalysisConfig::default());
        
        let now = SystemTime::now();
        let events = vec![
            AttackEvent {
                event_id: "1".to_string(),
                timestamp: now,
                pattern: AttackPattern::BruteForce,
                phase: AttackPhase::Exploitation,
                severity: AttackSeverity::Medium,
                confidence: DetectionConfidence::High,
                source: None,
                target: "test".to_string(),
                indicators: vec![],
                description: "test".to_string(),
                raw_data: HashMap::new(),
            },
            AttackEvent {
                event_id: "2".to_string(),
                timestamp: now + Duration::from_secs(1),
                pattern: AttackPattern::BruteForce,
                phase: AttackPhase::Exploitation,
                severity: AttackSeverity::Medium,
                confidence: DetectionConfidence::High,
                source: None,
                target: "test".to_string(),
                indicators: vec![],
                description: "test".to_string(),
                raw_data: HashMap::new(),
            },
        ];
        
        let event_refs: Vec<&AttackEvent> = events.iter().collect();
        let clustering = analyzer.analyze_temporal_clustering(&event_refs).unwrap();
        
        assert!(clustering >= 0.0 && clustering <= 1.0);
    }
} 