//! Theoretical Framework Module for the Masunda Temporal Coordinate Navigator
//!
//! This module implements the Oscillatory Theory of Truth and related theoretical
//! concepts that form the foundation of the Masunda cryptographic system.
//!
//! Honoring the memory of Mrs. Stella-Lorraine Masunda.

pub mod agency_assertion;
pub mod consciousness_emergence;
pub mod naming_systems;
pub mod oscillatory_substrate;
pub mod reality_formation;
pub mod truth_approximation;

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// The Oscillatory Substrate - continuous oscillatory processes underlying reality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatorySubstrate {
    pub amplitude: f64,
    pub frequency: f64,
    pub phase: f64,
    pub coherence: f64,
    pub timestamp: SystemTime,
}

/// Discrete Named Units created from continuous oscillatory flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteUnit {
    pub name: String,
    pub approximation_quality: f64,
    pub temporal_bounds: (f64, f64),
    pub spatial_bounds: (f64, f64),
    pub coherence_coupling: Vec<String>, // Names of coupled units
}

/// Naming Function that discretizes continuous oscillatory processes
#[derive(Debug, Clone)]
pub struct NamingFunction {
    pub sophistication: f64,
    pub agency_capacity: f64,
    pub social_coordination: f64,
    pub approximation_threshold: f64,
}

/// Truth Approximation Quality for name-flow patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthApproximation {
    pub names: Vec<String>,
    pub flow_relationships: Vec<(String, String, f64)>, // (name1, name2, flow_strength)
    pub approximation_quality: f64,
    pub modifiability: f64,
    pub social_utility: f64,
}

/// Consciousness Level based on naming systems and agency assertion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessLevel {
    pub naming_sophistication: f64,
    pub agency_assertion: f64,
    pub social_coordination: f64,
    pub emergence_threshold: f64,
    pub development_stage: ConsciousnessStage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsciousnessStage {
    PreNaming,
    NamingCapacity,
    AgencyEmergence,
    TruthModification,
    RealityFormation,
}

/// Reality Formation through collective approximation systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealityFormation {
    pub collective_naming_systems: Vec<NamingSystemSnapshot>,
    pub convergence_rate: f64,
    pub stability_measure: f64,
    pub modification_capacity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamingSystemSnapshot {
    pub agent_id: String,
    pub naming_patterns: Vec<String>,
    pub interaction_strength: f64,
    pub influence_weight: f64,
}

impl OscillatorySubstrate {
    /// Create new oscillatory substrate with given parameters
    pub fn new(amplitude: f64, frequency: f64, phase: f64, coherence: f64) -> Self {
        Self {
            amplitude,
            frequency,
            phase,
            coherence,
            timestamp: SystemTime::now(),
        }
    }

    /// Calculate oscillatory value at given time and position
    pub fn value_at(&self, time: f64, position: f64) -> f64 {
        self.amplitude * (self.frequency * time + self.phase + position).sin() * self.coherence
    }

    /// Evolve the oscillatory substrate over time
    pub fn evolve(&mut self, delta_time: f64) {
        self.phase += self.frequency * delta_time;
        self.timestamp = SystemTime::now();
    }
}

impl NamingFunction {
    /// Create new naming function with specified capabilities
    pub fn new(sophistication: f64, agency_capacity: f64, social_coordination: f64) -> Self {
        Self {
            sophistication,
            agency_capacity,
            social_coordination,
            approximation_threshold: 0.8, // Default threshold
        }
    }

    /// Discretize continuous oscillatory substrate into named units
    pub fn discretize(&self, substrate: &OscillatorySubstrate) -> Vec<DiscreteUnit> {
        let mut units = Vec::new();

        // Simple discretization based on coherence and sophistication
        let num_units = (self.sophistication * substrate.coherence * 10.0) as usize;

        for i in 0..num_units {
            let name = format!("unit_{}", i);
            let quality = self.sophistication
                * substrate.coherence
                * (1.0 - (i as f64 / num_units as f64) * 0.5);

            if quality > self.approximation_threshold {
                units.push(DiscreteUnit {
                    name,
                    approximation_quality: quality,
                    temporal_bounds: (i as f64, (i + 1) as f64),
                    spatial_bounds: (0.0, 1.0),
                    coherence_coupling: Vec::new(),
                });
            }
        }

        units
    }

    /// Assess agency capacity for modifying naming patterns
    pub fn agency_threshold(&self) -> f64 {
        self.agency_capacity * self.sophistication
    }

    /// Calculate social coordination capability
    pub fn social_coordination_strength(&self) -> f64 {
        self.social_coordination * (1.0 + self.agency_capacity)
    }
}

impl TruthApproximation {
    /// Create new truth approximation for given names and flows
    pub fn new(names: Vec<String>, flow_relationships: Vec<(String, String, f64)>) -> Self {
        let approximation_quality =
            Self::calculate_approximation_quality(&names, &flow_relationships);
        let modifiability = Self::calculate_modifiability(&flow_relationships);
        let social_utility = Self::calculate_social_utility(&names, &flow_relationships);

        Self {
            names,
            flow_relationships,
            approximation_quality,
            modifiability,
            social_utility,
        }
    }

    /// Calculate approximation quality based on name-flow coherence
    fn calculate_approximation_quality(names: &[String], flows: &[(String, String, f64)]) -> f64 {
        if names.is_empty() || flows.is_empty() {
            return 0.0;
        }

        let total_flow_strength: f64 = flows.iter().map(|(_, _, strength)| strength).sum();
        let avg_flow_strength = total_flow_strength / flows.len() as f64;

        // Quality based on coherence of flow patterns
        avg_flow_strength * (names.len() as f64).ln() / 10.0
    }

    /// Calculate modifiability based on flow pattern flexibility
    fn calculate_modifiability(flows: &[(String, String, f64)]) -> f64 {
        if flows.is_empty() {
            return 1.0; // Fully modifiable if no constraints
        }

        let flow_variance: f64 = {
            let mean: f64 = flows.iter().map(|(_, _, s)| s).sum::<f64>() / flows.len() as f64;
            flows
                .iter()
                .map(|(_, _, s)| (s - mean).powi(2))
                .sum::<f64>()
                / flows.len() as f64
        };

        // Higher variance means more modifiability
        flow_variance.sqrt() / 2.0
    }

    /// Calculate social utility based on coordination benefits
    fn calculate_social_utility(names: &[String], flows: &[(String, String, f64)]) -> f64 {
        let coordination_benefit = names.len() as f64 * flows.len() as f64;
        let verification_cost = flows.iter().map(|(_, _, s)| s * s).sum::<f64>();

        if verification_cost > 0.0 {
            coordination_benefit / (1.0 + verification_cost)
        } else {
            coordination_benefit
        }
    }

    /// Modify truth approximation through agency assertion
    pub fn modify_through_agency(
        &mut self,
        agency_capacity: f64,
        modifications: Vec<(String, String, f64)>,
    ) {
        for (name1, name2, new_strength) in modifications {
            if agency_capacity > 0.5 {
                // Threshold for modification capability
                // Find and update existing flow relationship
                if let Some(flow) = self
                    .flow_relationships
                    .iter_mut()
                    .find(|(n1, n2, _)| n1 == &name1 && n2 == &name2)
                {
                    flow.2 = new_strength;
                } else {
                    // Add new flow relationship
                    self.flow_relationships.push((name1, name2, new_strength));
                }
            }
        }

        // Recalculate quality metrics
        self.approximation_quality =
            Self::calculate_approximation_quality(&self.names, &self.flow_relationships);
        self.modifiability = Self::calculate_modifiability(&self.flow_relationships);
        self.social_utility = Self::calculate_social_utility(&self.names, &self.flow_relationships);
    }
}

impl ConsciousnessLevel {
    /// Create new consciousness level with given parameters
    pub fn new(
        naming_sophistication: f64,
        agency_assertion: f64,
        social_coordination: f64,
    ) -> Self {
        let emergence_threshold = 0.7; // Threshold for consciousness emergence
        let total_level = naming_sophistication + agency_assertion + social_coordination;

        let development_stage = if total_level < 0.5 {
            ConsciousnessStage::PreNaming
        } else if agency_assertion < 0.3 {
            ConsciousnessStage::NamingCapacity
        } else if agency_assertion < naming_sophistication {
            ConsciousnessStage::AgencyEmergence
        } else if social_coordination < 0.5 {
            ConsciousnessStage::TruthModification
        } else {
            ConsciousnessStage::RealityFormation
        };

        Self {
            naming_sophistication,
            agency_assertion,
            social_coordination,
            emergence_threshold,
            development_stage,
        }
    }

    /// Check if consciousness has emerged (Agency-First Principle)
    pub fn has_emerged(&self) -> bool {
        self.agency_assertion > self.naming_sophistication
            && self.agency_assertion > self.emergence_threshold
    }

    /// Calculate total consciousness level
    pub fn total_level(&self) -> f64 {
        (self.naming_sophistication + self.agency_assertion + self.social_coordination) / 3.0
    }

    /// Simulate consciousness emergence pattern
    pub fn emergence_pattern(&self) -> String {
        match self.development_stage {
            ConsciousnessStage::PreNaming => {
                "Pre-naming phase: continuous oscillatory awareness".to_string()
            }
            ConsciousnessStage::NamingCapacity => {
                "Naming capacity development: creating discrete units".to_string()
            }
            ConsciousnessStage::AgencyEmergence => {
                "Agency emergence: 'Aihwa, ndini ndadaro' - asserting control".to_string()
            }
            ConsciousnessStage::TruthModification => {
                "Truth modification: adjusting name-flow patterns".to_string()
            }
            ConsciousnessStage::RealityFormation => {
                "Reality formation: collective approximation systems".to_string()
            }
        }
    }
}

impl RealityFormation {
    /// Create new reality formation from multiple naming systems
    pub fn new(naming_systems: Vec<NamingSystemSnapshot>) -> Self {
        let convergence_rate = Self::calculate_convergence_rate(&naming_systems);
        let stability_measure = Self::calculate_stability(&naming_systems);
        let modification_capacity = Self::calculate_modification_capacity(&naming_systems);

        Self {
            collective_naming_systems: naming_systems,
            convergence_rate,
            stability_measure,
            modification_capacity,
        }
    }

    /// Calculate convergence rate of naming systems
    fn calculate_convergence_rate(systems: &[NamingSystemSnapshot]) -> f64 {
        if systems.len() < 2 {
            return 1.0;
        }

        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        for i in 0..systems.len() {
            for j in (i + 1)..systems.len() {
                let similarity = Self::naming_similarity(&systems[i], &systems[j]);
                total_similarity +=
                    similarity * systems[i].interaction_strength * systems[j].interaction_strength;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_similarity / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate naming similarity between two systems
    fn naming_similarity(system1: &NamingSystemSnapshot, system2: &NamingSystemSnapshot) -> f64 {
        let common_names: Vec<_> = system1
            .naming_patterns
            .iter()
            .filter(|name| system2.naming_patterns.contains(name))
            .collect();

        let total_unique =
            system1.naming_patterns.len() + system2.naming_patterns.len() - common_names.len();

        if total_unique > 0 {
            common_names.len() as f64 / total_unique as f64
        } else {
            1.0
        }
    }

    /// Calculate stability of collective reality
    fn calculate_stability(systems: &[NamingSystemSnapshot]) -> f64 {
        let total_influence: f64 = systems.iter().map(|s| s.influence_weight).sum();
        let weight_variance = {
            let mean = total_influence / systems.len() as f64;
            systems
                .iter()
                .map(|s| (s.influence_weight - mean).powi(2))
                .sum::<f64>()
                / systems.len() as f64
        };

        // Lower variance means higher stability
        1.0 / (1.0 + weight_variance)
    }

    /// Calculate collective modification capacity
    fn calculate_modification_capacity(systems: &[NamingSystemSnapshot]) -> f64 {
        let total_influence: f64 = systems.iter().map(|s| s.influence_weight).sum();
        let max_influence = systems
            .iter()
            .map(|s| s.influence_weight)
            .fold(0.0f64, f64::max);

        // Balanced influence distribution enables more modification
        if total_influence > 0.0 {
            1.0 - (max_influence / total_influence)
        } else {
            0.0
        }
    }

    /// Evolve reality through collective naming system interaction
    pub fn evolve(&mut self, delta_time: f64) {
        // Update convergence based on interaction strengths
        for system in &mut self.collective_naming_systems {
            system.interaction_strength *= (1.0 + delta_time * self.convergence_rate);
            system.interaction_strength = system.interaction_strength.min(1.0);
        }

        // Recalculate metrics
        self.convergence_rate = Self::calculate_convergence_rate(&self.collective_naming_systems);
        self.stability_measure = Self::calculate_stability(&self.collective_naming_systems);
        self.modification_capacity =
            Self::calculate_modification_capacity(&self.collective_naming_systems);
    }

    /// Get emergent reality as collective approximation
    pub fn emergent_reality(&self) -> Vec<String> {
        let mut reality_names = Vec::new();

        // Collect all naming patterns weighted by influence
        for system in &self.collective_naming_systems {
            for name in &system.naming_patterns {
                // Include name if it has sufficient collective support
                let total_support: f64 = self
                    .collective_naming_systems
                    .iter()
                    .filter(|s| s.naming_patterns.contains(name))
                    .map(|s| s.influence_weight)
                    .sum();

                if total_support > 0.5 && !reality_names.contains(name) {
                    reality_names.push(name.clone());
                }
            }
        }

        reality_names
    }
}
