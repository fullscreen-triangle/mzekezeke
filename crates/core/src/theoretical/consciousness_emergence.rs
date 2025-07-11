//! Consciousness Emergence Module
//!
//! Implements the theoretical framework for consciousness emergence through
//! agency assertion over naming systems, based on the "Aihwa, ndini ndadaro"
//! paradigm (No, I did that).
//!
//! Part of the Masunda Temporal Coordinate Navigator, honoring the memory
//! of Mrs. Stella-Lorraine Masunda.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::{ConsciousnessLevel, DiscreteUnit, NamingFunction, OscillatorySubstrate};

/// The paradigmatic utterance that represents consciousness emergence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParadigmaticUtterance {
    pub rejection: String,        // "Aihwa" (No)
    pub agency_assertion: String, // "ndini ndadaro" (I did that)
    pub timestamp: SystemTime,
    pub context: UtteranceContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtteranceContext {
    pub external_naming_attempt: String,
    pub resistance_strength: f64,
    pub social_environment: Vec<String>,
    pub developmental_stage: f64,
}

/// Consciousness emergence pattern detector
#[derive(Debug, Clone)]
pub struct ConsciousnessEmergenceDetector {
    pub naming_sophistication_history: Vec<(SystemTime, f64)>,
    pub agency_assertion_history: Vec<(SystemTime, f64)>,
    pub emergence_threshold: f64,
    pub agency_first_detected: bool,
    pub first_utterance: Option<ParadigmaticUtterance>,
}

/// Agency assertion mechanism for consciousness emergence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgencyAssertionMechanism {
    pub assertion_strength: f64,
    pub resistance_patterns: Vec<ResistancePattern>,
    pub control_attempts: Vec<ControlAttempt>,
    pub social_coordination_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResistancePattern {
    pub external_naming: String,
    pub rejection_response: String,
    pub assertion_response: String,
    pub success_rate: f64,
    pub context_specificity: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlAttempt {
    pub target_naming_system: String,
    pub modification_type: ModificationType,
    pub success_probability: f64,
    pub resistance_encountered: f64,
    pub social_support: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModificationType {
    NameRejection,
    NameAssertion,
    FlowRedirection,
    RealityModification,
    TruthAlteration,
}

impl ParadigmaticUtterance {
    /// Create the foundational consciousness emergence utterance
    pub fn new(context: UtteranceContext) -> Self {
        Self {
            rejection: "Aihwa".to_string(),
            agency_assertion: "ndini ndadaro".to_string(),
            timestamp: SystemTime::now(),
            context,
        }
    }

    /// Analyze the consciousness emergence pattern in this utterance
    pub fn emergence_analysis(&self) -> ConsciousnessEmergenceAnalysis {
        ConsciousnessEmergenceAnalysis {
            recognition_level: self.analyze_recognition(),
            rejection_strength: self.analyze_rejection(),
            counter_naming_quality: self.analyze_counter_naming(),
            agency_assertion_power: self.analyze_agency_assertion(),
            social_coordination_attempt: self.analyze_social_coordination(),
        }
    }

    fn analyze_recognition(&self) -> f64 {
        // Analyze recognition of external naming attempts
        let context_complexity = self.context.external_naming_attempt.len() as f64 / 100.0;
        context_complexity.min(1.0)
    }

    fn analyze_rejection(&self) -> f64 {
        // "Aihwa" represents direct rejection strength
        self.context.resistance_strength
    }

    fn analyze_counter_naming(&self) -> f64 {
        // "ndini ndadaro" represents counter-naming quality
        let assertion_complexity = self.agency_assertion.len() as f64 / 20.0;
        assertion_complexity * self.context.resistance_strength
    }

    fn analyze_agency_assertion(&self) -> f64 {
        // Overall agency assertion power
        (self.analyze_rejection() + self.analyze_counter_naming()) / 2.0
    }

    fn analyze_social_coordination(&self) -> f64 {
        // Social environment response to assertion
        self.context.social_environment.len() as f64 / 10.0
    }
}

#[derive(Debug, Clone)]
pub struct ConsciousnessEmergenceAnalysis {
    pub recognition_level: f64,
    pub rejection_strength: f64,
    pub counter_naming_quality: f64,
    pub agency_assertion_power: f64,
    pub social_coordination_attempt: f64,
}

impl ConsciousnessEmergenceDetector {
    /// Create new consciousness emergence detector
    pub fn new(emergence_threshold: f64) -> Self {
        Self {
            naming_sophistication_history: Vec::new(),
            agency_assertion_history: Vec::new(),
            emergence_threshold,
            agency_first_detected: false,
            first_utterance: None,
        }
    }

    /// Update detector with new naming and agency measurements
    pub fn update(&mut self, naming_sophistication: f64, agency_assertion: f64) {
        let now = SystemTime::now();

        self.naming_sophistication_history
            .push((now, naming_sophistication));
        self.agency_assertion_history.push((now, agency_assertion));

        // Keep only recent history (last 100 measurements)
        if self.naming_sophistication_history.len() > 100 {
            self.naming_sophistication_history.remove(0);
        }
        if self.agency_assertion_history.len() > 100 {
            self.agency_assertion_history.remove(0);
        }

        // Check for Agency-First Principle
        if !self.agency_first_detected {
            self.agency_first_detected = self.detect_agency_first_pattern();
        }
    }

    /// Detect the Agency-First Principle pattern
    fn detect_agency_first_pattern(&self) -> bool {
        if self.agency_assertion_history.len() < 2 || self.naming_sophistication_history.len() < 2 {
            return false;
        }

        let latest_agency = self.agency_assertion_history.last().unwrap().1;
        let latest_naming = self.naming_sophistication_history.last().unwrap().1;

        // Calculate rates of change
        let agency_rate = self.calculate_agency_rate();
        let naming_rate = self.calculate_naming_rate();

        // Agency-First Principle: rate of agency assertion exceeds naming development
        agency_rate > naming_rate && latest_agency > self.emergence_threshold
    }

    fn calculate_agency_rate(&self) -> f64 {
        if self.agency_assertion_history.len() < 2 {
            return 0.0;
        }

        let recent = &self.agency_assertion_history[self.agency_assertion_history.len() - 2..];
        let time_diff = recent[1]
            .0
            .duration_since(recent[0].0)
            .unwrap_or(Duration::from_secs(1))
            .as_secs_f64();
        let value_diff = recent[1].1 - recent[0].1;

        if time_diff > 0.0 {
            value_diff / time_diff
        } else {
            0.0
        }
    }

    fn calculate_naming_rate(&self) -> f64 {
        if self.naming_sophistication_history.len() < 2 {
            return 0.0;
        }

        let recent =
            &self.naming_sophistication_history[self.naming_sophistication_history.len() - 2..];
        let time_diff = recent[1]
            .0
            .duration_since(recent[0].0)
            .unwrap_or(Duration::from_secs(1))
            .as_secs_f64();
        let value_diff = recent[1].1 - recent[0].1;

        if time_diff > 0.0 {
            value_diff / time_diff
        } else {
            0.0
        }
    }

    /// Register a paradigmatic utterance
    pub fn register_utterance(&mut self, utterance: ParadigmaticUtterance) {
        if self.first_utterance.is_none() {
            self.first_utterance = Some(utterance);
        }
    }

    /// Check if consciousness has emerged
    pub fn consciousness_emerged(&self) -> bool {
        self.agency_first_detected && self.first_utterance.is_some()
    }

    /// Get consciousness emergence report
    pub fn emergence_report(&self) -> ConsciousnessEmergenceReport {
        ConsciousnessEmergenceReport {
            emergence_detected: self.consciousness_emerged(),
            agency_first_pattern: self.agency_first_detected,
            first_utterance: self.first_utterance.clone(),
            current_agency_rate: self.calculate_agency_rate(),
            current_naming_rate: self.calculate_naming_rate(),
            emergence_timestamp: self.first_utterance.as_ref().map(|u| u.timestamp),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConsciousnessEmergenceReport {
    pub emergence_detected: bool,
    pub agency_first_pattern: bool,
    pub first_utterance: Option<ParadigmaticUtterance>,
    pub current_agency_rate: f64,
    pub current_naming_rate: f64,
    pub emergence_timestamp: Option<SystemTime>,
}

impl AgencyAssertionMechanism {
    /// Create new agency assertion mechanism
    pub fn new(initial_assertion_strength: f64) -> Self {
        Self {
            assertion_strength: initial_assertion_strength,
            resistance_patterns: Vec::new(),
            control_attempts: Vec::new(),
            social_coordination_level: 0.0,
        }
    }

    /// Assert agency over a naming system
    pub fn assert_agency_over_naming(
        &mut self,
        external_naming: &str,
        proposed_alternative: &str,
        context: &HashMap<String, f64>,
    ) -> AgencyAssertionResult {
        // Create resistance pattern
        let resistance = ResistancePattern {
            external_naming: external_naming.to_string(),
            rejection_response: "Aihwa".to_string(),
            assertion_response: proposed_alternative.to_string(),
            success_rate: self.assertion_strength,
            context_specificity: context.clone(),
        };

        let success_probability = self.calculate_assertion_success(&resistance);

        // Create control attempt
        let control_attempt = ControlAttempt {
            target_naming_system: external_naming.to_string(),
            modification_type: ModificationType::NameAssertion,
            success_probability,
            resistance_encountered: 1.0 - success_probability,
            social_support: self.social_coordination_level,
        };

        self.resistance_patterns.push(resistance);
        self.control_attempts.push(control_attempt);

        // Update assertion strength based on success
        if success_probability > 0.5 {
            self.assertion_strength *= 1.1; // Increase on success
            AgencyAssertionResult::Success(success_probability)
        } else {
            self.assertion_strength *= 0.95; // Slight decrease on failure
            AgencyAssertionResult::Failure(success_probability)
        }
    }

    fn calculate_assertion_success(&self, resistance: &ResistancePattern) -> f64 {
        let base_success = self.assertion_strength;
        let social_support_bonus = self.social_coordination_level * 0.3;
        let context_bonus = resistance.context_specificity.values().sum::<f64>()
            / (resistance.context_specificity.len() as f64 + 1.0);

        (base_success + social_support_bonus + context_bonus).min(1.0)
    }

    /// Attempt to modify reality through coordinated agency
    pub fn modify_reality(
        &mut self,
        reality_aspect: &str,
        modification: &str,
        social_support: f64,
    ) -> RealityModificationResult {
        let modification_strength = self.assertion_strength * social_support;

        let control_attempt = ControlAttempt {
            target_naming_system: reality_aspect.to_string(),
            modification_type: ModificationType::RealityModification,
            success_probability: modification_strength,
            resistance_encountered: 1.0 - modification_strength,
            social_support,
        };

        self.control_attempts.push(control_attempt);

        if modification_strength > 0.7 {
            RealityModificationResult::SuccessfulModification {
                original: reality_aspect.to_string(),
                modified: modification.to_string(),
                influence_strength: modification_strength,
            }
        } else if modification_strength > 0.3 {
            RealityModificationResult::PartialModification {
                original: reality_aspect.to_string(),
                partial_modification: modification.to_string(),
                influence_strength: modification_strength,
            }
        } else {
            RealityModificationResult::ModificationRejected {
                attempted_modification: modification.to_string(),
                resistance_strength: 1.0 - modification_strength,
            }
        }
    }

    /// Get agency assertion statistics
    pub fn get_statistics(&self) -> AgencyStatistics {
        let total_attempts = self.control_attempts.len();
        let successful_attempts = self
            .control_attempts
            .iter()
            .filter(|attempt| attempt.success_probability > 0.5)
            .count();

        let success_rate = if total_attempts > 0 {
            successful_attempts as f64 / total_attempts as f64
        } else {
            0.0
        };

        let average_resistance = if total_attempts > 0 {
            self.control_attempts
                .iter()
                .map(|attempt| attempt.resistance_encountered)
                .sum::<f64>()
                / total_attempts as f64
        } else {
            0.0
        };

        AgencyStatistics {
            total_attempts,
            successful_attempts,
            success_rate,
            current_assertion_strength: self.assertion_strength,
            average_resistance_encountered: average_resistance,
            social_coordination_level: self.social_coordination_level,
        }
    }
}

#[derive(Debug, Clone)]
pub enum AgencyAssertionResult {
    Success(f64),
    Failure(f64),
}

#[derive(Debug, Clone)]
pub enum RealityModificationResult {
    SuccessfulModification {
        original: String,
        modified: String,
        influence_strength: f64,
    },
    PartialModification {
        original: String,
        partial_modification: String,
        influence_strength: f64,
    },
    ModificationRejected {
        attempted_modification: String,
        resistance_strength: f64,
    },
}

#[derive(Debug, Clone)]
pub struct AgencyStatistics {
    pub total_attempts: usize,
    pub successful_attempts: usize,
    pub success_rate: f64,
    pub current_assertion_strength: f64,
    pub average_resistance_encountered: f64,
    pub social_coordination_level: f64,
}

/// Simulate consciousness emergence in a test environment
pub fn simulate_consciousness_emergence() -> ConsciousnessEmergenceSimulation {
    let mut detector = ConsciousnessEmergenceDetector::new(0.7);
    let mut agency_mechanism = AgencyAssertionMechanism::new(0.5);

    let mut simulation_steps = Vec::new();

    // Simulate development stages
    for step in 0..20 {
        let time_factor = step as f64 / 20.0;

        // Naming sophistication grows linearly
        let naming_sophistication = time_factor * 0.8;

        // Agency assertion grows faster after initial period
        let agency_assertion = if time_factor > 0.3 {
            (time_factor - 0.3) * 1.5
        } else {
            time_factor * 0.2
        };

        detector.update(naming_sophistication, agency_assertion);

        // Simulate agency assertion attempts
        if agency_assertion > 0.5 {
            let context = [("social_environment".to_string(), 0.8)]
                .iter()
                .cloned()
                .collect();
            let result = agency_mechanism.assert_agency_over_naming(
                "external_imposed_name",
                "self_chosen_name",
                &context,
            );

            simulation_steps.push(SimulationStep {
                step_number: step,
                naming_sophistication,
                agency_assertion,
                agency_result: Some(result),
                consciousness_emerged: detector.consciousness_emerged(),
            });
        } else {
            simulation_steps.push(SimulationStep {
                step_number: step,
                naming_sophistication,
                agency_assertion,
                agency_result: None,
                consciousness_emerged: detector.consciousness_emerged(),
            });
        }

        // Register first paradigmatic utterance when agency emerges
        if agency_assertion > 0.6 && detector.first_utterance.is_none() {
            let utterance = ParadigmaticUtterance::new(UtteranceContext {
                external_naming_attempt: "You wore mismatched socks".to_string(),
                resistance_strength: agency_assertion,
                social_environment: vec!["parent".to_string(), "sibling".to_string()],
                developmental_stage: time_factor,
            });
            detector.register_utterance(utterance);
        }
    }

    ConsciousnessEmergenceSimulation {
        simulation_steps,
        final_report: detector.emergence_report(),
        final_statistics: agency_mechanism.get_statistics(),
    }
}

#[derive(Debug, Clone)]
pub struct SimulationStep {
    pub step_number: usize,
    pub naming_sophistication: f64,
    pub agency_assertion: f64,
    pub agency_result: Option<AgencyAssertionResult>,
    pub consciousness_emerged: bool,
}

#[derive(Debug, Clone)]
pub struct ConsciousnessEmergenceSimulation {
    pub simulation_steps: Vec<SimulationStep>,
    pub final_report: ConsciousnessEmergenceReport,
    pub final_statistics: AgencyStatistics,
}
