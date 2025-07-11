use crate::types::{MdtecError, MdtecResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, Duration};

/// Threat categories in the MDTEC system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreatCategory {
    /// Cryptographic attacks
    Cryptographic,
    /// Environmental manipulation
    Environmental,
    /// Side-channel attacks
    SideChannel,
    /// Network-based attacks
    Network,
    /// Physical attacks
    Physical,
    /// Temporal attacks
    Temporal,
    /// Data integrity attacks
    DataIntegrity,
    /// Availability attacks
    Availability,
    /// Privacy attacks
    Privacy,
    /// Multi-dimensional attacks
    MultiDimensional,
}

/// Threat severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreatSeverity {
    /// Low impact, unlikely
    Low,
    /// Moderate impact, possible
    Medium,
    /// High impact, likely
    High,
    /// Critical impact, imminent
    Critical,
}

/// Threat likelihood assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreatLikelihood {
    /// Very unlikely to occur
    VeryLow,
    /// Unlikely to occur
    Low,
    /// Moderate chance
    Medium,
    /// Likely to occur
    High,
    /// Very likely to occur
    VeryHigh,
}

/// Threat impact assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreatImpact {
    /// Minimal impact
    Minimal,
    /// Minor impact
    Minor,
    /// Moderate impact
    Moderate,
    /// Major impact
    Major,
    /// Catastrophic impact
    Catastrophic,
}

/// Threat vector classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreatVector {
    /// Remote network attack
    RemoteNetwork,
    /// Local network attack
    LocalNetwork,
    /// Physical access required
    PhysicalAccess,
    /// Insider threat
    Insider,
    /// Social engineering
    SocialEngineering,
    /// Supply chain attack
    SupplyChain,
    /// Environmental manipulation
    Environmental,
    /// Timing-based attack
    Timing,
    /// Power analysis
    PowerAnalysis,
    /// Electromagnetic analysis
    ElectromagneticAnalysis,
}

/// Threat actor classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreatActor {
    /// Nation-state actor
    NationState,
    /// Organized crime
    OrganizedCrime,
    /// Terrorist organization
    Terrorist,
    /// Hacktivist
    Hacktivist,
    /// Insider threat
    Insider,
    /// Script kiddie
    ScriptKiddie,
    /// Researcher
    Researcher,
    /// Competitor
    Competitor,
    /// Unknown actor
    Unknown,
}

/// Specific threat definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDefinition {
    /// Unique threat identifier
    pub id: String,
    /// Threat name
    pub name: String,
    /// Threat description
    pub description: String,
    /// Threat category
    pub category: ThreatCategory,
    /// Threat severity
    pub severity: ThreatSeverity,
    /// Threat likelihood
    pub likelihood: ThreatLikelihood,
    /// Threat impact
    pub impact: ThreatImpact,
    /// Attack vectors
    pub vectors: Vec<ThreatVector>,
    /// Potential threat actors
    pub actors: Vec<ThreatActor>,
    /// Affected MDTEC components
    pub affected_components: Vec<String>,
    /// Mitigation strategies
    pub mitigations: Vec<String>,
    /// Detection methods
    pub detection_methods: Vec<String>,
    /// CVSS score (if applicable)
    pub cvss_score: Option<f64>,
    /// Reference links
    pub references: Vec<String>,
}

/// Threat assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatAssessment {
    /// Threat definition
    pub threat: ThreatDefinition,
    /// Risk score (0-100)
    pub risk_score: f64,
    /// Exploitability score (0-10)
    pub exploitability_score: f64,
    /// Asset value affected (0-10)
    pub asset_value: f64,
    /// Current security posture (0-10)
    pub security_posture: f64,
    /// Residual risk after mitigations
    pub residual_risk: f64,
    /// Assessment timestamp
    pub assessment_time: SystemTime,
    /// Assessment confidence (0-1)
    pub confidence: f64,
}

/// Threat model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatModelConfig {
    /// Enable all threat categories
    pub enabled_categories: Vec<ThreatCategory>,
    /// Risk tolerance threshold
    pub risk_tolerance: f64,
    /// Assessment frequency
    pub assessment_frequency: Duration,
    /// Threat intelligence sources
    pub threat_intelligence_sources: Vec<String>,
    /// Asset valuation model
    pub asset_valuation: HashMap<String, f64>,
    /// Security control effectiveness
    pub control_effectiveness: HashMap<String, f64>,
}

impl Default for ThreatModelConfig {
    fn default() -> Self {
        let mut asset_valuation = HashMap::new();
        asset_valuation.insert("encryption_keys".to_string(), 10.0);
        asset_valuation.insert("environmental_data".to_string(), 8.0);
        asset_valuation.insert("temporal_data".to_string(), 8.0);
        asset_valuation.insert("user_data".to_string(), 9.0);
        asset_valuation.insert("system_integrity".to_string(), 10.0);

        let mut control_effectiveness = HashMap::new();
        control_effectiveness.insert("encryption".to_string(), 0.9);
        control_effectiveness.insert("access_control".to_string(), 0.8);
        control_effectiveness.insert("monitoring".to_string(), 0.7);
        control_effectiveness.insert("backup".to_string(), 0.6);
        control_effectiveness.insert("training".to_string(), 0.5);

        Self {
            enabled_categories: vec![
                ThreatCategory::Cryptographic,
                ThreatCategory::Environmental,
                ThreatCategory::SideChannel,
                ThreatCategory::Network,
                ThreatCategory::Physical,
                ThreatCategory::Temporal,
                ThreatCategory::DataIntegrity,
                ThreatCategory::Availability,
                ThreatCategory::Privacy,
                ThreatCategory::MultiDimensional,
            ],
            risk_tolerance: 30.0,
            assessment_frequency: Duration::from_secs(3600), // 1 hour
            threat_intelligence_sources: vec![
                "NIST".to_string(),
                "MITRE".to_string(),
                "CVE".to_string(),
                "OWASP".to_string(),
            ],
            asset_valuation,
            control_effectiveness,
        }
    }
}

/// Threat model analyzer
pub struct ThreatModelAnalyzer {
    config: ThreatModelConfig,
    threat_definitions: HashMap<String, ThreatDefinition>,
    threat_assessments: Vec<ThreatAssessment>,
    last_assessment_time: Option<SystemTime>,
}

impl ThreatModelAnalyzer {
    pub fn new(config: ThreatModelConfig) -> Self {
        let mut analyzer = Self {
            config,
            threat_definitions: HashMap::new(),
            threat_assessments: Vec::new(),
            last_assessment_time: None,
        };
        
        // Initialize with default threat definitions
        analyzer.initialize_default_threats();
        analyzer
    }

    /// Initialize default threat definitions for MDTEC
    fn initialize_default_threats(&mut self) {
        // Cryptographic threats
        self.add_threat_definition(ThreatDefinition {
            id: "CRYPTO_001".to_string(),
            name: "Quantum Computer Attack".to_string(),
            description: "Advanced quantum computers breaking traditional encryption".to_string(),
            category: ThreatCategory::Cryptographic,
            severity: ThreatSeverity::High,
            likelihood: ThreatLikelihood::Low,
            impact: ThreatImpact::Catastrophic,
            vectors: vec![ThreatVector::RemoteNetwork],
            actors: vec![ThreatActor::NationState, ThreatActor::Researcher],
            affected_components: vec!["encryption_keys".to_string(), "temporal_encryption".to_string()],
            mitigations: vec![
                "Implement quantum-resistant algorithms".to_string(),
                "Use post-quantum cryptography".to_string(),
                "Implement crypto-agility".to_string(),
            ],
            detection_methods: vec![
                "Monitor for unusual decryption patterns".to_string(),
                "Implement quantum key distribution".to_string(),
            ],
            cvss_score: Some(8.5),
            references: vec![
                "https://csrc.nist.gov/projects/post-quantum-cryptography".to_string(),
            ],
        });

        // Environmental threats
        self.add_threat_definition(ThreatDefinition {
            id: "ENV_001".to_string(),
            name: "Environmental Spoofing".to_string(),
            description: "Manipulation of environmental sensors to compromise entropy generation".to_string(),
            category: ThreatCategory::Environmental,
            severity: ThreatSeverity::Medium,
            likelihood: ThreatLikelihood::Medium,
            impact: ThreatImpact::Major,
            vectors: vec![ThreatVector::PhysicalAccess, ThreatVector::Environmental],
            actors: vec![ThreatActor::Insider, ThreatActor::Competitor],
            affected_components: vec!["environmental_sensors".to_string(), "entropy_generation".to_string()],
            mitigations: vec![
                "Implement sensor validation".to_string(),
                "Use multiple independent sensors".to_string(),
                "Implement anomaly detection".to_string(),
            ],
            detection_methods: vec![
                "Monitor sensor correlation".to_string(),
                "Implement statistical analysis".to_string(),
            ],
            cvss_score: Some(6.5),
            references: vec![],
        });

        // Side-channel threats
        self.add_threat_definition(ThreatDefinition {
            id: "SIDE_001".to_string(),
            name: "Timing Attack".to_string(),
            description: "Analysis of operation timing to extract cryptographic keys".to_string(),
            category: ThreatCategory::SideChannel,
            severity: ThreatSeverity::High,
            likelihood: ThreatLikelihood::High,
            impact: ThreatImpact::Major,
            vectors: vec![ThreatVector::Timing, ThreatVector::LocalNetwork],
            actors: vec![ThreatActor::Researcher, ThreatActor::Hacktivist],
            affected_components: vec!["cryptographic_operations".to_string()],
            mitigations: vec![
                "Implement constant-time algorithms".to_string(),
                "Add timing noise".to_string(),
                "Use blinding techniques".to_string(),
            ],
            detection_methods: vec![
                "Monitor timing patterns".to_string(),
                "Implement timing randomization".to_string(),
            ],
            cvss_score: Some(7.0),
            references: vec![],
        });

        // Network threats
        self.add_threat_definition(ThreatDefinition {
            id: "NET_001".to_string(),
            name: "Man-in-the-Middle Attack".to_string(),
            description: "Interception and manipulation of network communications".to_string(),
            category: ThreatCategory::Network,
            severity: ThreatSeverity::High,
            likelihood: ThreatLikelihood::Medium,
            impact: ThreatImpact::Major,
            vectors: vec![ThreatVector::RemoteNetwork, ThreatVector::LocalNetwork],
            actors: vec![ThreatActor::OrganizedCrime, ThreatActor::NationState],
            affected_components: vec!["network_communications".to_string()],
            mitigations: vec![
                "Implement mutual authentication".to_string(),
                "Use certificate pinning".to_string(),
                "Implement end-to-end encryption".to_string(),
            ],
            detection_methods: vec![
                "Monitor certificate changes".to_string(),
                "Implement connection fingerprinting".to_string(),
            ],
            cvss_score: Some(7.5),
            references: vec![],
        });

        // Physical threats
        self.add_threat_definition(ThreatDefinition {
            id: "PHYS_001".to_string(),
            name: "Hardware Tampering".to_string(),
            description: "Physical modification of hardware components".to_string(),
            category: ThreatCategory::Physical,
            severity: ThreatSeverity::High,
            likelihood: ThreatLikelihood::Low,
            impact: ThreatImpact::Catastrophic,
            vectors: vec![ThreatVector::PhysicalAccess, ThreatVector::SupplyChain],
            actors: vec![ThreatActor::NationState, ThreatActor::Insider],
            affected_components: vec!["hardware_components".to_string()],
            mitigations: vec![
                "Implement hardware security modules".to_string(),
                "Use tamper-evident seals".to_string(),
                "Implement secure boot".to_string(),
            ],
            detection_methods: vec![
                "Monitor hardware integrity".to_string(),
                "Implement attestation".to_string(),
            ],
            cvss_score: Some(8.0),
            references: vec![],
        });

        // Multi-dimensional threats
        self.add_threat_definition(ThreatDefinition {
            id: "MULTI_001".to_string(),
            name: "Coordinated Multi-Vector Attack".to_string(),
            description: "Simultaneous attack across multiple dimensions and vectors".to_string(),
            category: ThreatCategory::MultiDimensional,
            severity: ThreatSeverity::Critical,
            likelihood: ThreatLikelihood::Low,
            impact: ThreatImpact::Catastrophic,
            vectors: vec![
                ThreatVector::RemoteNetwork,
                ThreatVector::PhysicalAccess,
                ThreatVector::Environmental,
                ThreatVector::Timing,
            ],
            actors: vec![ThreatActor::NationState, ThreatActor::OrganizedCrime],
            affected_components: vec!["entire_system".to_string()],
            mitigations: vec![
                "Implement defense in depth".to_string(),
                "Use distributed architecture".to_string(),
                "Implement real-time monitoring".to_string(),
            ],
            detection_methods: vec![
                "Monitor correlation across dimensions".to_string(),
                "Implement AI-based threat detection".to_string(),
            ],
            cvss_score: Some(9.5),
            references: vec![],
        });
    }

    /// Add a new threat definition
    pub fn add_threat_definition(&mut self, threat: ThreatDefinition) {
        self.threat_definitions.insert(threat.id.clone(), threat);
    }

    /// Remove a threat definition
    pub fn remove_threat_definition(&mut self, threat_id: &str) -> Option<ThreatDefinition> {
        self.threat_definitions.remove(threat_id)
    }

    /// Get threat definition by ID
    pub fn get_threat_definition(&self, threat_id: &str) -> Option<&ThreatDefinition> {
        self.threat_definitions.get(threat_id)
    }

    /// Assess all threats
    pub fn assess_threats(&mut self) -> MdtecResult<Vec<ThreatAssessment>> {
        let mut assessments = Vec::new();
        
        for (_, threat) in &self.threat_definitions {
            if self.config.enabled_categories.contains(&threat.category) {
                let assessment = self.assess_individual_threat(threat)?;
                assessments.push(assessment);
            }
        }
        
        // Sort by risk score (highest first)
        assessments.sort_by(|a, b| b.risk_score.partial_cmp(&a.risk_score).unwrap());
        
        self.threat_assessments = assessments.clone();
        self.last_assessment_time = Some(SystemTime::now());
        
        Ok(assessments)
    }

    /// Assess an individual threat
    fn assess_individual_threat(&self, threat: &ThreatDefinition) -> MdtecResult<ThreatAssessment> {
        let likelihood_score = self.calculate_likelihood_score(threat);
        let impact_score = self.calculate_impact_score(threat);
        let exploitability_score = self.calculate_exploitability_score(threat);
        let asset_value = self.calculate_asset_value(threat);
        let security_posture = self.calculate_security_posture(threat);
        
        // Calculate risk score using a composite formula
        let risk_score = (likelihood_score * impact_score * exploitability_score * asset_value) / 
                        (security_posture * 10.0);
        
        let residual_risk = risk_score * (1.0 - self.calculate_mitigation_effectiveness(threat));
        
        let confidence = self.calculate_assessment_confidence(threat);
        
        Ok(ThreatAssessment {
            threat: threat.clone(),
            risk_score: risk_score.min(100.0),
            exploitability_score,
            asset_value,
            security_posture,
            residual_risk: residual_risk.min(100.0),
            assessment_time: SystemTime::now(),
            confidence,
        })
    }

    /// Calculate likelihood score (0-10)
    fn calculate_likelihood_score(&self, threat: &ThreatDefinition) -> f64 {
        match threat.likelihood {
            ThreatLikelihood::VeryLow => 1.0,
            ThreatLikelihood::Low => 3.0,
            ThreatLikelihood::Medium => 5.0,
            ThreatLikelihood::High => 7.0,
            ThreatLikelihood::VeryHigh => 9.0,
        }
    }

    /// Calculate impact score (0-10)
    fn calculate_impact_score(&self, threat: &ThreatDefinition) -> f64 {
        match threat.impact {
            ThreatImpact::Minimal => 1.0,
            ThreatImpact::Minor => 3.0,
            ThreatImpact::Moderate => 5.0,
            ThreatImpact::Major => 7.0,
            ThreatImpact::Catastrophic => 9.0,
        }
    }

    /// Calculate exploitability score (0-10)
    fn calculate_exploitability_score(&self, threat: &ThreatDefinition) -> f64 {
        let mut score = 5.0; // Base score
        
        // Adjust based on attack vectors
        for vector in &threat.vectors {
            match vector {
                ThreatVector::RemoteNetwork => score += 2.0,
                ThreatVector::LocalNetwork => score += 1.5,
                ThreatVector::PhysicalAccess => score -= 1.0,
                ThreatVector::Insider => score += 1.0,
                ThreatVector::SocialEngineering => score += 1.5,
                ThreatVector::SupplyChain => score -= 0.5,
                ThreatVector::Environmental => score += 0.5,
                ThreatVector::Timing => score += 1.0,
                ThreatVector::PowerAnalysis => score += 0.5,
                ThreatVector::ElectromagneticAnalysis => score += 0.5,
            }
        }
        
        // Adjust based on threat actors
        for actor in &threat.actors {
            match actor {
                ThreatActor::NationState => score += 1.0,
                ThreatActor::OrganizedCrime => score += 0.5,
                ThreatActor::Terrorist => score += 0.5,
                ThreatActor::Hacktivist => score += 0.5,
                ThreatActor::Insider => score += 1.0,
                ThreatActor::ScriptKiddie => score -= 1.0,
                ThreatActor::Researcher => score += 0.5,
                ThreatActor::Competitor => score += 0.5,
                ThreatActor::Unknown => score += 0.0,
            }
        }
        
        score.max(0.0).min(10.0)
    }

    /// Calculate asset value (0-10)
    fn calculate_asset_value(&self, threat: &ThreatDefinition) -> f64 {
        let mut total_value = 0.0;
        let mut count = 0;
        
        for component in &threat.affected_components {
            if let Some(value) = self.config.asset_valuation.get(component) {
                total_value += value;
                count += 1;
            }
        }
        
        if count > 0 {
            total_value / count as f64
        } else {
            5.0 // Default value
        }
    }

    /// Calculate security posture (0-10)
    fn calculate_security_posture(&self, threat: &ThreatDefinition) -> f64 {
        let mut posture_score = 0.0;
        let mut count = 0;
        
        for mitigation in &threat.mitigations {
            // Simplified mapping of mitigations to controls
            let control_key = if mitigation.contains("encryption") || mitigation.contains("crypto") {
                "encryption"
            } else if mitigation.contains("access") || mitigation.contains("authentication") {
                "access_control"
            } else if mitigation.contains("monitor") || mitigation.contains("detect") {
                "monitoring"
            } else if mitigation.contains("backup") || mitigation.contains("redundancy") {
                "backup"
            } else {
                "training"
            };
            
            if let Some(effectiveness) = self.config.control_effectiveness.get(control_key) {
                posture_score += effectiveness * 10.0;
                count += 1;
            }
        }
        
        if count > 0 {
            posture_score / count as f64
        } else {
            5.0 // Default posture
        }
    }

    /// Calculate mitigation effectiveness (0-1)
    fn calculate_mitigation_effectiveness(&self, threat: &ThreatDefinition) -> f64 {
        let mut effectiveness = 0.0;
        
        for mitigation in &threat.mitigations {
            // Simplified effectiveness calculation
            effectiveness += 0.1; // Each mitigation adds 10% effectiveness
        }
        
        effectiveness.min(0.9) // Maximum 90% effectiveness
    }

    /// Calculate assessment confidence (0-1)
    fn calculate_assessment_confidence(&self, threat: &ThreatDefinition) -> f64 {
        let mut confidence = 0.5; // Base confidence
        
        // Increase confidence based on available information
        if !threat.references.is_empty() {
            confidence += 0.1;
        }
        
        if threat.cvss_score.is_some() {
            confidence += 0.1;
        }
        
        if !threat.detection_methods.is_empty() {
            confidence += 0.1;
        }
        
        if !threat.mitigations.is_empty() {
            confidence += 0.1;
        }
        
        confidence.min(1.0)
    }

    /// Get high-risk threats
    pub fn get_high_risk_threats(&self) -> Vec<&ThreatAssessment> {
        self.threat_assessments
            .iter()
            .filter(|assessment| assessment.risk_score > self.config.risk_tolerance)
            .collect()
    }

    /// Get threat statistics
    pub fn get_threat_statistics(&self) -> ThreatStatistics {
        let total_threats = self.threat_definitions.len();
        let assessed_threats = self.threat_assessments.len();
        
        let high_risk_count = self.get_high_risk_threats().len();
        let medium_risk_count = self.threat_assessments
            .iter()
            .filter(|a| a.risk_score > 20.0 && a.risk_score <= self.config.risk_tolerance)
            .count();
        let low_risk_count = self.threat_assessments
            .iter()
            .filter(|a| a.risk_score <= 20.0)
            .count();
        
        let avg_risk_score = if !self.threat_assessments.is_empty() {
            self.threat_assessments.iter().map(|a| a.risk_score).sum::<f64>() / 
            self.threat_assessments.len() as f64
        } else {
            0.0
        };
        
        let category_distribution = self.calculate_category_distribution();
        
        ThreatStatistics {
            total_threats,
            assessed_threats,
            high_risk_count,
            medium_risk_count,
            low_risk_count,
            avg_risk_score,
            category_distribution,
            last_assessment: self.last_assessment_time,
        }
    }

    /// Calculate threat category distribution
    fn calculate_category_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        
        for threat in self.threat_definitions.values() {
            let category_name = format!("{:?}", threat.category);
            *distribution.entry(category_name).or_insert(0) += 1;
        }
        
        distribution
    }

    /// Reset threat model
    pub fn reset(&mut self) {
        self.threat_assessments.clear();
        self.last_assessment_time = None;
    }
}

/// Threat statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatStatistics {
    pub total_threats: usize,
    pub assessed_threats: usize,
    pub high_risk_count: usize,
    pub medium_risk_count: usize,
    pub low_risk_count: usize,
    pub avg_risk_score: f64,
    pub category_distribution: HashMap<String, usize>,
    pub last_assessment: Option<SystemTime>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threat_model_analyzer_creation() {
        let analyzer = ThreatModelAnalyzer::new(ThreatModelConfig::default());
        assert!(!analyzer.threat_definitions.is_empty());
        assert_eq!(analyzer.threat_assessments.len(), 0);
    }

    #[test]
    fn test_threat_definition_management() {
        let mut analyzer = ThreatModelAnalyzer::new(ThreatModelConfig::default());
        
        let threat = ThreatDefinition {
            id: "TEST_001".to_string(),
            name: "Test Threat".to_string(),
            description: "Test description".to_string(),
            category: ThreatCategory::Network,
            severity: ThreatSeverity::Medium,
            likelihood: ThreatLikelihood::Medium,
            impact: ThreatImpact::Moderate,
            vectors: vec![ThreatVector::RemoteNetwork],
            actors: vec![ThreatActor::Hacktivist],
            affected_components: vec!["test_component".to_string()],
            mitigations: vec!["test_mitigation".to_string()],
            detection_methods: vec!["test_detection".to_string()],
            cvss_score: Some(5.0),
            references: vec![],
        };
        
        analyzer.add_threat_definition(threat.clone());
        
        let retrieved = analyzer.get_threat_definition("TEST_001");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Test Threat");
        
        let removed = analyzer.remove_threat_definition("TEST_001");
        assert!(removed.is_some());
        assert!(analyzer.get_threat_definition("TEST_001").is_none());
    }

    #[test]
    fn test_threat_assessment() {
        let mut analyzer = ThreatModelAnalyzer::new(ThreatModelConfig::default());
        
        let assessments = analyzer.assess_threats().unwrap();
        assert!(!assessments.is_empty());
        
        // Check that assessments are sorted by risk score
        for i in 1..assessments.len() {
            assert!(assessments[i - 1].risk_score >= assessments[i].risk_score);
        }
    }

    #[test]
    fn test_risk_calculation() {
        let analyzer = ThreatModelAnalyzer::new(ThreatModelConfig::default());
        
        let threat = ThreatDefinition {
            id: "RISK_TEST".to_string(),
            name: "Risk Test".to_string(),
            description: "Test risk calculation".to_string(),
            category: ThreatCategory::Cryptographic,
            severity: ThreatSeverity::High,
            likelihood: ThreatLikelihood::High,
            impact: ThreatImpact::Major,
            vectors: vec![ThreatVector::RemoteNetwork],
            actors: vec![ThreatActor::NationState],
            affected_components: vec!["encryption_keys".to_string()],
            mitigations: vec!["Implement strong encryption".to_string()],
            detection_methods: vec!["Monitor key usage".to_string()],
            cvss_score: Some(8.0),
            references: vec![],
        };
        
        let assessment = analyzer.assess_individual_threat(&threat).unwrap();
        assert!(assessment.risk_score > 0.0);
        assert!(assessment.confidence > 0.0);
        assert!(assessment.confidence <= 1.0);
    }

    #[test]
    fn test_threat_statistics() {
        let mut analyzer = ThreatModelAnalyzer::new(ThreatModelConfig::default());
        analyzer.assess_threats().unwrap();
        
        let stats = analyzer.get_threat_statistics();
        assert!(stats.total_threats > 0);
        assert!(stats.assessed_threats > 0);
        assert!(stats.avg_risk_score >= 0.0);
        assert!(!stats.category_distribution.is_empty());
    }

    #[test]
    fn test_high_risk_threats() {
        let mut analyzer = ThreatModelAnalyzer::new(ThreatModelConfig::default());
        analyzer.assess_threats().unwrap();
        
        let high_risk = analyzer.get_high_risk_threats();
        
        for threat in high_risk {
            assert!(threat.risk_score > analyzer.config.risk_tolerance);
        }
    }
} 