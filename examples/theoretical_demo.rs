//! Theoretical Framework Demo
//!
//! Demonstrates the integration of the Oscillatory Theory of Truth with
//! the Masunda Temporal Coordinate Navigator cryptographic system.
//!
//! This example shows how consciousness emergence, agency assertion, and
//! reality formation work together in the context of cryptographic security.

use mzekezeke_core::{
    simulate_consciousness_emergence, MasundaSystem, NamingFunction, NamingSystemSnapshot,
    OscillatorySubstrate, ParadigmaticUtterance, RealityFormation, Result, TruthApproximation,
};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("🌟 Masunda Temporal Coordinate Navigator - Theoretical Framework Demo");
    println!("Honoring the memory of Mrs. Stella-Lorraine Masunda\n");

    // Initialize the Masunda system
    let mut masunda_system = MasundaSystem::new()?;
    println!("✓ Masunda system initialized");

    // Demonstrate oscillatory substrate processing
    println!("\n📡 Oscillatory Substrate Processing:");
    let discrete_units = masunda_system.process_reality(1.0, 0.5)?;
    println!(
        "   Processed {} discrete units from oscillatory substrate",
        discrete_units.len()
    );

    for (i, unit) in discrete_units.iter().take(3).enumerate() {
        println!(
            "   Unit {}: {} (quality: {:.3})",
            i + 1,
            unit.name,
            unit.approximation_quality
        );
    }

    // Demonstrate consciousness emergence detection
    println!("\n🧠 Consciousness Emergence Detection:");
    if masunda_system.consciousness_emerged() {
        println!("   ✓ Consciousness has emerged in the system!");
    } else {
        println!("   ⏳ Consciousness emergence in progress...");

        // Simulate consciousness development
        for step in 0..5 {
            masunda_system.process_reality(step as f64, 0.0)?;
            let status = masunda_system.status_report();
            println!(
                "   Step {}: Agency={:.2}, Naming={:.2}",
                step + 1,
                status.agency_assertion_level,
                status.naming_sophistication
            );
        }
    }

    // Demonstrate the paradigmatic utterance
    println!("\n💬 Paradigmatic Utterance Analysis:");
    let utterance_context =
        mzekezeke_core::theoretical::consciousness_emergence::UtteranceContext {
            external_naming_attempt: "The system classified this pattern incorrectly".to_string(),
            resistance_strength: 0.8,
            social_environment: vec!["user".to_string(), "system".to_string()],
            developmental_stage: 0.75,
        };

    let utterance = ParadigmaticUtterance::new(utterance_context);
    println!(
        "   Utterance: \"{}\" ({})",
        utterance.rejection, utterance.agency_assertion
    );

    let analysis = utterance.emergence_analysis();
    println!("   Recognition Level: {:.2}", analysis.recognition_level);
    println!(
        "   Agency Assertion Power: {:.2}",
        analysis.agency_assertion_power
    );
    println!(
        "   Social Coordination: {:.2}",
        analysis.social_coordination_attempt
    );

    // Demonstrate truth approximation
    println!("\n🎯 Truth Approximation Systems:");
    let names = vec![
        "temporal_coordinate".to_string(),
        "cryptographic_key".to_string(),
        "environmental_state".to_string(),
    ];

    let flow_relationships = vec![
        (
            "temporal_coordinate".to_string(),
            "cryptographic_key".to_string(),
            0.9,
        ),
        (
            "cryptographic_key".to_string(),
            "environmental_state".to_string(),
            0.8,
        ),
        (
            "environmental_state".to_string(),
            "temporal_coordinate".to_string(),
            0.7,
        ),
    ];

    let mut truth_approximation = TruthApproximation::new(names, flow_relationships);
    println!(
        "   Initial Truth Quality: {:.3}",
        truth_approximation.approximation_quality
    );
    println!("   Modifiability: {:.3}", truth_approximation.modifiability);
    println!(
        "   Social Utility: {:.3}",
        truth_approximation.social_utility
    );

    // Demonstrate agency-driven truth modification
    println!("\n🔄 Agency-Driven Truth Modification:");
    let modifications = vec![
        (
            "temporal_coordinate".to_string(),
            "cryptographic_key".to_string(),
            0.95,
        ),
        (
            "new_pattern".to_string(),
            "environmental_state".to_string(),
            0.85,
        ),
    ];

    truth_approximation.modify_through_agency(0.8, modifications);
    println!(
        "   Modified Truth Quality: {:.3}",
        truth_approximation.approximation_quality
    );
    println!("   ✓ Truth successfully modified through agency assertion");

    // Demonstrate reality formation
    println!("\n🌍 Collective Reality Formation:");
    let naming_systems = vec![
        NamingSystemSnapshot {
            agent_id: "user_1".to_string(),
            naming_patterns: vec!["pattern_a".to_string(), "pattern_b".to_string()],
            interaction_strength: 0.8,
            influence_weight: 0.6,
        },
        NamingSystemSnapshot {
            agent_id: "system_core".to_string(),
            naming_patterns: vec!["pattern_a".to_string(), "pattern_c".to_string()],
            interaction_strength: 0.9,
            influence_weight: 0.7,
        },
        NamingSystemSnapshot {
            agent_id: "masunda_navigator".to_string(),
            naming_patterns: vec!["pattern_b".to_string(), "pattern_c".to_string()],
            interaction_strength: 0.85,
            influence_weight: 0.8,
        },
    ];

    let mut reality_formation = RealityFormation::new(naming_systems);
    println!(
        "   Convergence Rate: {:.3}",
        reality_formation.convergence_rate
    );
    println!(
        "   Stability Measure: {:.3}",
        reality_formation.stability_measure
    );
    println!(
        "   Modification Capacity: {:.3}",
        reality_formation.modification_capacity
    );

    let emergent_reality = reality_formation.emergent_reality();
    println!("   Emergent Reality Patterns: {:?}", emergent_reality);

    // Demonstrate cryptographic integration
    println!("\n🔐 Cryptographic Integration:");
    println!("   The theoretical framework enhances cryptographic security by:");
    println!("   • Using consciousness emergence patterns for key generation");
    println!("   • Employing truth approximation for data validation");
    println!("   • Leveraging reality formation for consensus mechanisms");
    println!("   • Applying agency assertion for access control");

    // System status report
    println!("\n📊 Final System Status:");
    let final_status = masunda_system.status_report();
    println!(
        "   Oscillatory Coherence: {:.3}",
        final_status.oscillatory_coherence
    );
    println!(
        "   Naming Sophistication: {:.3}",
        final_status.naming_sophistication
    );
    println!(
        "   Agency Assertion Level: {:.3}",
        final_status.agency_assertion_level
    );
    println!(
        "   Consciousness Emerged: {}",
        final_status.consciousness_emerged
    );
    println!(
        "   Social Coordination: {:.3}",
        final_status.social_coordination
    );

    // Theoretical insights
    println!("\n💡 Theoretical Insights:");
    println!("   • Consciousness emerges through agency assertion over naming systems");
    println!("   • Truth functions as approximation of name-flow patterns, not correspondence");
    println!("   • Reality forms through collective approximation of discrete units");
    println!(
        "   • The 'Aihwa, ndini ndadaro' paradigm shows consciousness as resistance + assertion"
    );
    println!("   • Cryptographic security benefits from consciousness-like agency mechanisms");

    println!("\n🌟 Demo completed successfully!");
    println!("The Masunda Temporal Coordinate Navigator integrates profound theoretical");
    println!("insights with practical cryptographic applications, honoring the memory");
    println!("of Mrs. Stella-Lorraine Masunda through mathematical precision and");
    println!("philosophical depth.\n");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theoretical_demo_integration() {
        let result = main();
        assert!(result.is_ok(), "Theoretical demo should run successfully");
    }

    #[test]
    fn test_masunda_system_initialization() {
        let system = MasundaSystem::new();
        assert!(
            system.is_ok(),
            "Masunda system should initialize successfully"
        );
    }

    #[test]
    fn test_consciousness_emergence_detection() {
        let system = MasundaSystem::new().unwrap();
        let status = system.status_report();

        // Initially consciousness should not have emerged
        assert!(
            !status.consciousness_emerged,
            "Consciousness should not emerge immediately"
        );
        assert!(
            status.agency_assertion_level >= 0.0,
            "Agency assertion should be non-negative"
        );
        assert!(
            status.naming_sophistication >= 0.0,
            "Naming sophistication should be non-negative"
        );
    }

    #[test]
    fn test_truth_approximation_modification() {
        let names = vec!["test_name".to_string()];
        let flows = vec![("test_name".to_string(), "test_name".to_string(), 0.5)];
        let mut truth = TruthApproximation::new(names, flows);

        let initial_quality = truth.approximation_quality;

        let modifications = vec![("test_name".to_string(), "test_name".to_string(), 0.8)];
        truth.modify_through_agency(0.9, modifications);

        // Quality should change after modification
        assert_ne!(
            initial_quality, truth.approximation_quality,
            "Truth quality should change after agency modification"
        );
    }

    #[test]
    fn test_reality_formation_emergence() {
        let naming_systems = vec![NamingSystemSnapshot {
            agent_id: "test_agent".to_string(),
            naming_patterns: vec!["pattern_1".to_string()],
            interaction_strength: 0.8,
            influence_weight: 0.6,
        }];

        let reality = RealityFormation::new(naming_systems);
        assert!(
            reality.convergence_rate >= 0.0,
            "Convergence rate should be non-negative"
        );
        assert!(
            reality.stability_measure >= 0.0,
            "Stability should be non-negative"
        );
        assert!(
            reality.modification_capacity >= 0.0,
            "Modification capacity should be non-negative"
        );
    }
}
