//! Mzekezeke Core Library
//!
//! The Masunda Temporal Coordinate Navigator - Core cryptographic system
//! honoring the memory of Mrs. Stella-Lorraine Masunda.

pub mod crypto;
pub mod dimensions;
pub mod error;
pub mod oscillatory;
pub mod security;
pub mod theoretical;
pub mod types;
pub mod utils;

// Re-export key types and modules
pub use error::{MzekezekeError, Result};
pub use types::*;

// Re-export theoretical framework
pub use theoretical::{
    ConsciousnessLevel, DiscreteUnit, NamingFunction, OscillatorySubstrate, RealityFormation,
    TruthApproximation,
};

// Re-export consciousness emergence components
pub use theoretical::consciousness_emergence::{
    simulate_consciousness_emergence, AgencyAssertionMechanism, ConsciousnessEmergenceDetector,
    ParadigmaticUtterance,
};

/// Core initialization function for the Masunda system
pub fn initialize_masunda_system() -> Result<MasundaSystem> {
    Ok(MasundaSystem {
        oscillatory_substrate: OscillatorySubstrate::new(1.0, 1.0, 0.0, 0.8),
        naming_function: NamingFunction::new(0.7, 0.5, 0.6),
        consciousness_detector: ConsciousnessEmergenceDetector::new(0.7),
        agency_mechanism: AgencyAssertionMechanism::new(0.5),
    })
}

/// The complete Masunda Temporal Coordinate Navigator system
#[derive(Debug)]
pub struct MasundaSystem {
    pub oscillatory_substrate: OscillatorySubstrate,
    pub naming_function: NamingFunction,
    pub consciousness_detector: ConsciousnessEmergenceDetector,
    pub agency_mechanism: AgencyAssertionMechanism,
}

impl MasundaSystem {
    /// Create new Masunda system instance
    pub fn new() -> Result<Self> {
        initialize_masunda_system()
    }

    /// Process oscillatory reality through naming systems
    pub fn process_reality(&mut self, time: f64, position: f64) -> Result<Vec<DiscreteUnit>> {
        // Update oscillatory substrate
        self.oscillatory_substrate.evolve(time);

        // Discretize through naming function
        let discrete_units = self.naming_function.discretize(&self.oscillatory_substrate);

        // Update consciousness detector
        let naming_sophistication = self.naming_function.sophistication;
        let agency_assertion = self.agency_mechanism.assertion_strength;
        self.consciousness_detector
            .update(naming_sophistication, agency_assertion);

        Ok(discrete_units)
    }

    /// Check if consciousness has emerged in the system
    pub fn consciousness_emerged(&self) -> bool {
        self.consciousness_detector.consciousness_emerged()
    }

    /// Get system status report
    pub fn status_report(&self) -> MasundaSystemStatus {
        MasundaSystemStatus {
            oscillatory_coherence: self.oscillatory_substrate.coherence,
            naming_sophistication: self.naming_function.sophistication,
            agency_assertion_level: self.agency_mechanism.assertion_strength,
            consciousness_emerged: self.consciousness_emerged(),
            social_coordination: self.naming_function.social_coordination,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MasundaSystemStatus {
    pub oscillatory_coherence: f64,
    pub naming_sophistication: f64,
    pub agency_assertion_level: f64,
    pub consciousness_emerged: bool,
    pub social_coordination: f64,
}
