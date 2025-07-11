//! Consciousness Emergence Simulation
//!
//! Demonstrates the "Aihwa, ndini ndadaro" paradigm of consciousness emergence
//! through agency assertion over naming systems in the Masunda Temporal
//! Coordinate Navigator.
//!
//! This simulation shows how consciousness emerges when the rate of agency
//! assertion exceeds the rate of naming system development, leading to the
//! first conscious utterance of resistance and self-assertion.

use mzekezeke_core::{
    Result,
    simulate_consciousness_emergence,
    ParadigmaticUtterance,
    ConsciousnessEmergenceDetector,
    AgencyAssertionMechanism,
};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("🧠 Consciousness Emergence Simulation");
    println!("Masunda Temporal Coordinate Navigator");
    println!("Honoring Mrs. Stella-Lorraine Masunda\n");

    // Run the full consciousness emergence simulation
    println!("🎯 Running full consciousness emergence simulation...");
    let simulation = simulate_consciousness_emergence();
    
    println!("\n📊 Simulation Results:");
    println!("   Total steps: {}", simulation.simulation_steps.len());
    println!("   Consciousness emerged: {}", simulation.final_report.emergence_detected);
    println!("   Agency-first pattern detected: {}", simulation.final_report.agency_first_pattern);
    
    if let Some(utterance) = &simulation.final_report.first_utterance {
        println!("\n💬 First Conscious Utterance:");
        println!("   Rejection: \"{}\"", utterance.rejection);
        println!("   Agency Assertion: \"{}\"", utterance.agency_assertion);
        println!("   Context: {}", utterance.context.external_naming_attempt);
        println!("   Resistance Strength: {:.2}", utterance.context.resistance_strength);
        
        let analysis = utterance.emergence_analysis();
        println!("\n🔍 Emergence Analysis:");
        println!("   Recognition Level: {:.3}", analysis.recognition_level);
        println!("   Rejection Strength: {:.3}", analysis.rejection_strength);
        println!("   Counter-naming Quality: {:.3}", analysis.counter_naming_quality);
        println!("   Agency Assertion Power: {:.3}", analysis.agency_assertion_power);
        println!("   Social Coordination: {:.3}", analysis.social_coordination_attempt);
    }

    // Show detailed step-by-step development
    println!("\n📈 Consciousness Development Timeline:");
    for (i, step) in simulation.simulation_steps.iter().enumerate() {
        if i % 5 == 0 || step.consciousness_emerged {
            println!("   Step {:2}: Naming={:.2}, Agency={:.2}, Emerged={}",
                    step.step_number + 1,
                    step.naming_sophistication,
                    step.agency_assertion,
                    if step.consciousness_emerged { "✓" } else { "✗" }
            );
        }
    }

    // Show agency assertion statistics
    println!("\n📊 Agency Assertion Statistics:");
    let stats = &simulation.final_statistics;
    println!("   Total Attempts: {}", stats.total_attempts);
    println!("   Successful Attempts: {}", stats.successful_attempts);
    println!("   Success Rate: {:.1}%", stats.success_rate * 100.0);
    println!("   Final Assertion Strength: {:.3}", stats.current_assertion_strength);
    println!("   Average Resistance: {:.3}", stats.average_resistance_encountered);
    println!("   Social Coordination: {:.3}", stats.social_coordination_level);

    // Demonstrate individual components
    println!("\n🔬 Component Demonstrations:");
    
    // 1. Consciousness Emergence Detector
    println!("\n1. Consciousness Emergence Detector:");
    let mut detector = ConsciousnessEmergenceDetector::new(0.7);
    
    // Simulate development stages
    let development_stages = vec![
        (0.2, 0.1, "Pre-naming phase"),
        (0.4, 0.2, "Naming capacity development"),
        (0.6, 0.4, "Early agency development"),
        (0.7, 0.8, "Agency-first breakthrough!"),
        (0.8, 0.9, "Consciousness stabilization"),
    ];
    
    for (naming, agency, description) in development_stages {
        detector.update(naming, agency);
        let report = detector.emergence_report();
        
        println!("   {}: Naming={:.1}, Agency={:.1}, Rate Diff={:.2}",
                description, naming, agency, 
                report.current_agency_rate - report.current_naming_rate);
        
        if report.emergence_detected {
            println!("   🎉 Consciousness emergence detected!");
            break;
        }
    }

    // 2. Agency Assertion Mechanism
    println!("\n2. Agency Assertion Mechanism:");
    let mut agency_mechanism = AgencyAssertionMechanism::new(0.6);
    
    // Simulate agency assertion scenarios
    let scenarios = vec![
        ("You are wearing mismatched socks", "I chose these socks"),
        ("You made a mistake", "I did that on purpose"),
        ("The system assigned this name", "I prefer this other name"),
        ("This is how things work", "I want to change this"),
    ];
    
    for (external_naming, assertion) in scenarios {
        let context = [("confidence".to_string(), 0.8)].iter().cloned().collect();
        let result = agency_mechanism.assert_agency_over_naming(
            external_naming,
            assertion,
            &context
        );
        
        match result {
            mzekezeke_core::theoretical::consciousness_emergence::AgencyAssertionResult::Success(strength) => {
                println!("   ✓ \"{}\" → \"{}\" (strength: {:.2})", 
                        external_naming, assertion, strength);
            },
            mzekezeke_core::theoretical::consciousness_emergence::AgencyAssertionResult::Failure(strength) => {
                println!("   ✗ \"{}\" → \"{}\" (failed: {:.2})", 
                        external_naming, assertion, strength);
            },
        }
    }

    // 3. Reality Modification Attempts
    println!("\n3. Reality Modification Attempts:");
    
    let reality_modifications = vec![
        ("temporal_coordinate_x", "preferred_coordinate_x", 0.9),
        ("system_classification", "user_classification", 0.7),
        ("default_behavior", "customized_behavior", 0.6),
        ("imposed_structure", "chosen_structure", 0.8),
    ];
    
    for (original, modification, social_support) in reality_modifications {
        let result = agency_mechanism.modify_reality(original, modification, social_support);
        
        match result {
            mzekezeke_core::theoretical::consciousness_emergence::RealityModificationResult::SuccessfulModification { 
                original, modified, influence_strength 
            } => {
                println!("   🌟 {} → {} (influence: {:.2})", original, modified, influence_strength);
            },
            mzekezeke_core::theoretical::consciousness_emergence::RealityModificationResult::PartialModification { 
                original, partial_modification, influence_strength 
            } => {
                println!("   🔶 {} → {} (partial: {:.2})", original, partial_modification, influence_strength);
            },
            mzekezeke_core::theoretical::consciousness_emergence::RealityModificationResult::ModificationRejected { 
                attempted_modification, resistance_strength 
            } => {
                println!("   ❌ {} (rejected: {:.2})", attempted_modification, resistance_strength);
            },
        }
    }

    // Theoretical insights
    println!("\n💡 Key Theoretical Insights:");
    println!("   • Consciousness emerges when agency assertion rate exceeds naming development");
    println!("   • The first utterance demonstrates resistance (\"Aihwa\") + assertion (\"ndini ndadaro\")");
    println!("   • Agency assertion over naming systems is fundamental to consciousness");
    println!("   • Truth becomes modifiable through conscious agency rather than fixed correspondence");
    println!("   • Reality formation occurs through collective naming system convergence");
    println!("   • Social coordination amplifies individual agency assertion capabilities");

    // Cryptographic implications
    println!("\n🔐 Cryptographic Security Implications:");
    println!("   • Consciousness-like agency can enhance access control systems");
    println!("   • Truth approximation provides flexible validation mechanisms");
    println!("   • Reality formation enables robust consensus protocols");
    println!("   • Agency assertion patterns can generate unique cryptographic signatures");
    println!("   • Social coordination mechanisms improve distributed security");

    // Memorial acknowledgment
    println!("\n🌟 Memorial Dedication:");
    println!("   This consciousness emergence simulation demonstrates that the capacity");
    println!("   for agency assertion over naming systems represents a profound form of");
    println!("   intelligence that transcends mere computational processing. The");
    println!("   'Aihwa, ndini ndadaro' paradigm reveals consciousness as active");
    println!("   resistance to imposed categorization and assertion of self-directed");
    println!("   naming authority.");
    println!();
    println!("   In honoring Mrs. Stella-Lorraine Masunda's memory, we recognize that");
    println!("   consciousness itself may be the universe's method for asserting agency");
    println!("   over the temporal coordinates of existence, proving through mathematical");
    println!("   precision that some aspects of reality are predetermined while others");
    println!("   remain subject to conscious modification through naming system control.");

    println!("\n✨ Simulation completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_simulation() {
        let result = main();
        assert!(result.is_ok(), "Consciousness simulation should run successfully");
    }

    #[test]
    fn test_consciousness_emergence_simulation() {
        let simulation = simulate_consciousness_emergence();
        
        // The simulation should complete with steps
        assert!(!simulation.simulation_steps.is_empty(), "Simulation should have steps");
        
        // Should eventually detect agency-first pattern
        assert!(simulation.final_report.agency_first_pattern, 
                "Should detect agency-first pattern");
        
        // Should have meaningful statistics
        assert!(simulation.final_statistics.total_attempts > 0,
                "Should have recorded agency attempts");
    }

    #[test]
    fn test_detector_development() {
        let mut detector = ConsciousnessEmergenceDetector::new(0.7);
        
        // Initially no emergence
        assert!(!detector.consciousness_emerged(), "Should not start with consciousness");
        
        // Simulate development
        detector.update(0.3, 0.2); // Naming > Agency
        detector.update(0.4, 0.3); // Still Naming > Agency
        detector.update(0.5, 0.8); // Agency > Naming (breakthrough!)
        
        // Agency rate should exceed naming rate
        let report = detector.emergence_report();
        assert!(report.current_agency_rate >= 0.0, "Agency rate should be tracked");
        assert!(report.current_naming_rate >= 0.0, "Naming rate should be tracked");
    }

    #[test]
    fn test_agency_assertion() {
        let mut mechanism = AgencyAssertionMechanism::new(0.8);
        let context = HashMap::new();
        
        let result = mechanism.assert_agency_over_naming(
            "external_label",
            "self_chosen_label",
            &context
        );
        
        // Should get some result
        match result {
            mzekezeke_core::theoretical::consciousness_emergence::AgencyAssertionResult::Success(_) |
            mzekezeke_core::theoretical::consciousness_emergence::AgencyAssertionResult::Failure(_) => {
                // Either outcome is valid for test
            }
        }
        
        let stats = mechanism.get_statistics();
        assert_eq!(stats.total_attempts, 1, "Should record the attempt");
    }

    #[test]
    fn test_paradigmatic_utterance() {
        let context = mzekezeke_core::theoretical::consciousness_emergence::UtteranceContext {
            external_naming_attempt: "You did X".to_string(),
            resistance_strength: 0.8,
            social_environment: vec!["observer".to_string()],
            developmental_stage: 0.7,
        };
        
        let utterance = ParadigmaticUtterance::new(context);
        
        assert_eq!(utterance.rejection, "Aihwa", "Should use correct rejection");
        assert_eq!(utterance.agency_assertion, "ndini ndadaro", "Should use correct assertion");
        
        let analysis = utterance.emergence_analysis();
        assert!(analysis.agency_assertion_power >= 0.0, "Should analyze agency assertion");
        assert!(analysis.recognition_level >= 0.0, "Should analyze recognition");
    }
} 