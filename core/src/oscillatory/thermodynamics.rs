use crate::types::{MdtecError, MdtecResult};
use crate::utils::math::{Statistics, MathConstants};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Thermodynamic state representation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermodynamicState {
    /// Temperature in Kelvin
    pub temperature: f64,
    /// Pressure in Pascal
    pub pressure: f64,
    /// Volume in m³
    pub volume: f64,
    /// Internal energy in Joules
    pub internal_energy: f64,
    /// Entropy in J/K
    pub entropy: f64,
    /// Enthalpy in Joules
    pub enthalpy: f64,
}

impl ThermodynamicState {
    pub fn new(temperature: f64, pressure: f64, volume: f64) -> Self {
        // Calculate derived properties (simplified ideal gas)
        let internal_energy = 1.5 * MathConstants::R * temperature; // For monatomic gas
        let entropy = MathConstants::R * (temperature.ln() + 1.5 * volume.ln()) + 100.0; // Reference entropy
        let enthalpy = internal_energy + pressure * volume;

        Self {
            temperature,
            pressure,
            volume,
            internal_energy,
            entropy,
            enthalpy,
        }
    }

    /// Calculate Gibbs free energy
    pub fn gibbs_free_energy(&self) -> f64 {
        self.enthalpy - self.temperature * self.entropy
    }

    /// Calculate Helmholtz free energy
    pub fn helmholtz_free_energy(&self) -> f64 {
        self.internal_energy - self.temperature * self.entropy
    }

    /// Calculate heat capacity at constant volume (simplified)
    pub fn heat_capacity_cv(&self) -> f64 {
        1.5 * MathConstants::R // For monatomic ideal gas
    }

    /// Calculate heat capacity at constant pressure (simplified)
    pub fn heat_capacity_cp(&self) -> f64 {
        2.5 * MathConstants::R // For monatomic ideal gas
    }

    /// Calculate thermal efficiency for Carnot cycle
    pub fn carnot_efficiency(&self, cold_temperature: f64) -> f64 {
        if self.temperature > cold_temperature && cold_temperature > 0.0 {
            1.0 - cold_temperature / self.temperature
        } else {
            0.0
        }
    }
}

/// Heat transfer mechanisms
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HeatTransferType {
    /// Conduction
    Conduction,
    /// Convection
    Convection,
    /// Radiation
    Radiation,
    /// Phase change
    PhaseChange,
}

/// Heat transfer analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatTransfer {
    /// Heat transfer type
    pub transfer_type: HeatTransferType,
    /// Heat transfer rate (W)
    pub heat_rate: f64,
    /// Temperature difference (K)
    pub temperature_difference: f64,
    /// Heat transfer coefficient
    pub heat_transfer_coefficient: f64,
    /// Contact area (m²)
    pub area: f64,
    /// Thermal resistance (K/W)
    pub thermal_resistance: f64,
}

impl HeatTransfer {
    pub fn new_conduction(thermal_conductivity: f64, area: f64, thickness: f64, temp_diff: f64) -> Self {
        let thermal_resistance = thickness / (thermal_conductivity * area);
        let heat_rate = temp_diff / thermal_resistance;
        
        Self {
            transfer_type: HeatTransferType::Conduction,
            heat_rate,
            temperature_difference: temp_diff,
            heat_transfer_coefficient: thermal_conductivity / thickness,
            area,
            thermal_resistance,
        }
    }

    pub fn new_convection(convection_coefficient: f64, area: f64, temp_diff: f64) -> Self {
        let thermal_resistance = 1.0 / (convection_coefficient * area);
        let heat_rate = temp_diff / thermal_resistance;
        
        Self {
            transfer_type: HeatTransferType::Convection,
            heat_rate,
            temperature_difference: temp_diff,
            heat_transfer_coefficient: convection_coefficient,
            area,
            thermal_resistance,
        }
    }

    pub fn new_radiation(emissivity: f64, area: f64, hot_temp: f64, cold_temp: f64) -> Self {
        let stefan_boltzmann = 5.67e-8; // W/(m²·K⁴)
        let heat_rate = emissivity * stefan_boltzmann * area * 
                       (hot_temp.powi(4) - cold_temp.powi(4));
        let temp_diff = hot_temp - cold_temp;
        let thermal_resistance = if heat_rate > 0.0 { temp_diff / heat_rate } else { f64::INFINITY };
        
        Self {
            transfer_type: HeatTransferType::Radiation,
            heat_rate,
            temperature_difference: temp_diff,
            heat_transfer_coefficient: emissivity * stefan_boltzmann * 
                                     (hot_temp.powi(3) + hot_temp.powi(2) * cold_temp + 
                                      hot_temp * cold_temp.powi(2) + cold_temp.powi(3)),
            area,
            thermal_resistance,
        }
    }
}

/// Thermodynamic process types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ThermodynamicProcess {
    /// Constant temperature
    Isothermal,
    /// Constant pressure
    Isobaric,
    /// Constant volume
    Isochoric,
    /// No heat transfer
    Adiabatic,
    /// Constant entropy
    Isentropic,
    /// Constant enthalpy
    Isenthalpic,
}

/// Thermodynamic cycle analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicCycle {
    /// Cycle name
    pub name: String,
    /// States in the cycle
    pub states: Vec<ThermodynamicState>,
    /// Processes between states
    pub processes: Vec<ThermodynamicProcess>,
    /// Work done by the cycle (J)
    pub work_done: f64,
    /// Heat input (J)
    pub heat_input: f64,
    /// Heat output (J)
    pub heat_output: f64,
    /// Cycle efficiency
    pub efficiency: f64,
}

impl ThermodynamicCycle {
    pub fn new(name: String) -> Self {
        Self {
            name,
            states: Vec::new(),
            processes: Vec::new(),
            work_done: 0.0,
            heat_input: 0.0,
            heat_output: 0.0,
            efficiency: 0.0,
        }
    }

    pub fn add_state(&mut self, state: ThermodynamicState) {
        self.states.push(state);
    }

    pub fn add_process(&mut self, process: ThermodynamicProcess) {
        self.processes.push(process);
    }

    pub fn calculate_cycle_properties(&mut self) -> MdtecResult<()> {
        if self.states.len() < 2 {
            return Err(MdtecError::InvalidInput("Cycle must have at least 2 states".to_string()));
        }

        let mut total_work = 0.0;
        let mut total_heat_input = 0.0;
        let mut total_heat_output = 0.0;

        for i in 0..self.states.len() {
            let current_state = &self.states[i];
            let next_state = &self.states[(i + 1) % self.states.len()];
            
            if i < self.processes.len() {
                let process = self.processes[i];
                let (work, heat) = self.calculate_process_work_heat(current_state, next_state, process);
                
                total_work += work;
                if heat > 0.0 {
                    total_heat_input += heat;
                } else {
                    total_heat_output += heat.abs();
                }
            }
        }

        self.work_done = total_work;
        self.heat_input = total_heat_input;
        self.heat_output = total_heat_output;
        self.efficiency = if total_heat_input > 0.0 { total_work / total_heat_input } else { 0.0 };

        Ok(())
    }

    fn calculate_process_work_heat(&self, initial: &ThermodynamicState, final: &ThermodynamicState, process: ThermodynamicProcess) -> (f64, f64) {
        match process {
            ThermodynamicProcess::Isothermal => {
                // Work = nRT ln(Vf/Vi)
                let work = initial.pressure * initial.volume * (final.volume / initial.volume).ln();
                let heat = work; // For isothermal process, ΔU = 0, so Q = W
                (work, heat)
            },
            ThermodynamicProcess::Isobaric => {
                // Work = P(Vf - Vi)
                let work = initial.pressure * (final.volume - initial.volume);
                let heat = final.internal_energy - initial.internal_energy + work;
                (work, heat)
            },
            ThermodynamicProcess::Isochoric => {
                // Work = 0 (no volume change)
                let work = 0.0;
                let heat = final.internal_energy - initial.internal_energy;
                (work, heat)
            },
            ThermodynamicProcess::Adiabatic => {
                // Heat = 0 (no heat transfer)
                let work = initial.internal_energy - final.internal_energy;
                let heat = 0.0;
                (work, heat)
            },
            _ => (0.0, 0.0), // Simplified for other processes
        }
    }
}

/// Thermodynamic equilibrium analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquilibriumAnalysis {
    /// System states
    pub states: Vec<ThermodynamicState>,
    /// Equilibrium criteria
    pub equilibrium_criteria: Vec<EquilibriumCriterion>,
    /// Equilibrium achieved
    pub is_equilibrium: bool,
    /// Time to equilibrium (seconds)
    pub equilibrium_time: Option<f64>,
}

/// Equilibrium criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquilibriumCriterion {
    /// Criterion type
    pub criterion_type: EquilibriumType,
    /// Threshold value
    pub threshold: f64,
    /// Current deviation
    pub deviation: f64,
    /// Criterion satisfied
    pub satisfied: bool,
}

/// Types of equilibrium
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EquilibriumType {
    /// Thermal equilibrium (equal temperatures)
    Thermal,
    /// Mechanical equilibrium (equal pressures)
    Mechanical,
    /// Chemical equilibrium (equal chemical potentials)
    Chemical,
    /// Diffusive equilibrium (equal particle concentrations)
    Diffusive,
}

/// Thermodynamics analyzer
pub struct ThermodynamicsAnalyzer {
    /// System states
    states: Vec<ThermodynamicState>,
    /// Heat transfer processes
    heat_transfers: Vec<HeatTransfer>,
    /// Thermodynamic cycles
    cycles: Vec<ThermodynamicCycle>,
    /// Equilibrium analysis
    equilibrium_analysis: Option<EquilibriumAnalysis>,
}

impl ThermodynamicsAnalyzer {
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            heat_transfers: Vec::new(),
            cycles: Vec::new(),
            equilibrium_analysis: None,
        }
    }

    /// Add thermodynamic state
    pub fn add_state(&mut self, state: ThermodynamicState) -> usize {
        self.states.push(state);
        self.states.len() - 1
    }

    /// Add heat transfer process
    pub fn add_heat_transfer(&mut self, heat_transfer: HeatTransfer) {
        self.heat_transfers.push(heat_transfer);
    }

    /// Add thermodynamic cycle
    pub fn add_cycle(&mut self, cycle: ThermodynamicCycle) {
        self.cycles.push(cycle);
    }

    /// Analyze thermal equilibrium
    pub fn analyze_thermal_equilibrium(&mut self, tolerance: f64) -> MdtecResult<()> {
        if self.states.len() < 2 {
            return Err(MdtecError::InvalidInput("Need at least 2 states for equilibrium analysis".to_string()));
        }

        let mut criteria = Vec::new();
        
        // Check thermal equilibrium
        let temperatures: Vec<f64> = self.states.iter().map(|s| s.temperature).collect();
        let temp_std = Statistics::std_dev(&temperatures).unwrap_or(0.0);
        let temp_mean = Statistics::mean(&temperatures);
        let temp_deviation = temp_std / temp_mean;
        
        criteria.push(EquilibriumCriterion {
            criterion_type: EquilibriumType::Thermal,
            threshold: tolerance,
            deviation: temp_deviation,
            satisfied: temp_deviation < tolerance,
        });

        // Check mechanical equilibrium
        let pressures: Vec<f64> = self.states.iter().map(|s| s.pressure).collect();
        let pressure_std = Statistics::std_dev(&pressures).unwrap_or(0.0);
        let pressure_mean = Statistics::mean(&pressures);
        let pressure_deviation = pressure_std / pressure_mean.max(1e-10);
        
        criteria.push(EquilibriumCriterion {
            criterion_type: EquilibriumType::Mechanical,
            threshold: tolerance,
            deviation: pressure_deviation,
            satisfied: pressure_deviation < tolerance,
        });

        let is_equilibrium = criteria.iter().all(|c| c.satisfied);

        self.equilibrium_analysis = Some(EquilibriumAnalysis {
            states: self.states.clone(),
            equilibrium_criteria: criteria,
            is_equilibrium,
            equilibrium_time: None, // Would need time series data
        });

        Ok(())
    }

    /// Calculate system entropy
    pub fn calculate_system_entropy(&self) -> f64 {
        if self.states.is_empty() {
            return 0.0;
        }

        let total_entropy: f64 = self.states.iter().map(|s| s.entropy).sum();
        total_entropy / self.states.len() as f64
    }

    /// Calculate entropy production rate
    pub fn calculate_entropy_production_rate(&self) -> MdtecResult<f64> {
        if self.heat_transfers.is_empty() {
            return Ok(0.0);
        }

        let mut entropy_production = 0.0;
        
        for heat_transfer in &self.heat_transfers {
            if heat_transfer.temperature_difference > 0.0 {
                // Entropy production = Heat flow / Temperature
                let avg_temperature = 300.0; // Simplified assumption
                entropy_production += heat_transfer.heat_rate / avg_temperature;
            }
        }

        Ok(entropy_production)
    }

    /// Calculate exergy (available work)
    pub fn calculate_exergy(&self, reference_state: &ThermodynamicState) -> Vec<f64> {
        self.states.iter().map(|state| {
            let exergy = (state.internal_energy - reference_state.internal_energy) +
                        reference_state.pressure * (state.volume - reference_state.volume) -
                        reference_state.temperature * (state.entropy - reference_state.entropy);
            exergy
        }).collect()
    }

    /// Analyze heat engine performance
    pub fn analyze_heat_engine(&self, hot_reservoir_temp: f64, cold_reservoir_temp: f64) -> MdtecResult<EnginePerformance> {
        if hot_reservoir_temp <= cold_reservoir_temp {
            return Err(MdtecError::InvalidInput("Hot reservoir must be hotter than cold reservoir".to_string()));
        }

        let carnot_efficiency = 1.0 - cold_reservoir_temp / hot_reservoir_temp;
        let max_work = carnot_efficiency;

        // Calculate actual efficiency from cycles
        let actual_efficiency = if !self.cycles.is_empty() {
            let efficiencies: Vec<f64> = self.cycles.iter().map(|c| c.efficiency).collect();
            Statistics::mean(&efficiencies)
        } else {
            0.0
        };

        let performance_ratio = if carnot_efficiency > 0.0 { actual_efficiency / carnot_efficiency } else { 0.0 };

        Ok(EnginePerformance {
            carnot_efficiency,
            actual_efficiency,
            performance_ratio,
            max_theoretical_work: max_work,
            entropy_production_rate: self.calculate_entropy_production_rate()?,
        })
    }

    /// Get thermodynamic statistics
    pub fn get_statistics(&self) -> ThermodynamicStatistics {
        let temperatures: Vec<f64> = self.states.iter().map(|s| s.temperature).collect();
        let pressures: Vec<f64> = self.states.iter().map(|s| s.pressure).collect();
        let volumes: Vec<f64> = self.states.iter().map(|s| s.volume).collect();
        let entropies: Vec<f64> = self.states.iter().map(|s| s.entropy).collect();

        ThermodynamicStatistics {
            state_count: self.states.len(),
            heat_transfer_count: self.heat_transfers.len(),
            cycle_count: self.cycles.len(),
            avg_temperature: Statistics::mean(&temperatures),
            avg_pressure: Statistics::mean(&pressures),
            avg_volume: Statistics::mean(&volumes),
            system_entropy: Statistics::mean(&entropies),
            is_equilibrium: self.equilibrium_analysis.as_ref().map(|e| e.is_equilibrium).unwrap_or(false),
            entropy_production_rate: self.calculate_entropy_production_rate().unwrap_or(0.0),
        }
    }

    /// Reset analyzer
    pub fn reset(&mut self) {
        self.states.clear();
        self.heat_transfers.clear();
        self.cycles.clear();
        self.equilibrium_analysis = None;
    }
}

/// Heat engine performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnginePerformance {
    pub carnot_efficiency: f64,
    pub actual_efficiency: f64,
    pub performance_ratio: f64,
    pub max_theoretical_work: f64,
    pub entropy_production_rate: f64,
}

/// Thermodynamic statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicStatistics {
    pub state_count: usize,
    pub heat_transfer_count: usize,
    pub cycle_count: usize,
    pub avg_temperature: f64,
    pub avg_pressure: f64,
    pub avg_volume: f64,
    pub system_entropy: f64,
    pub is_equilibrium: bool,
    pub entropy_production_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermodynamic_state_creation() {
        let state = ThermodynamicState::new(300.0, 101325.0, 1.0);
        
        assert_eq!(state.temperature, 300.0);
        assert_eq!(state.pressure, 101325.0);
        assert_eq!(state.volume, 1.0);
        assert!(state.internal_energy > 0.0);
        assert!(state.entropy > 0.0);
    }

    #[test]
    fn test_carnot_efficiency() {
        let state = ThermodynamicState::new(400.0, 101325.0, 1.0);
        let efficiency = state.carnot_efficiency(300.0);
        
        assert_eq!(efficiency, 1.0 - 300.0 / 400.0);
        assert_eq!(efficiency, 0.25);
    }

    #[test]
    fn test_heat_transfer_conduction() {
        let heat_transfer = HeatTransfer::new_conduction(50.0, 1.0, 0.1, 10.0);
        
        assert_eq!(heat_transfer.transfer_type, HeatTransferType::Conduction);
        assert_eq!(heat_transfer.temperature_difference, 10.0);
        assert!(heat_transfer.heat_rate > 0.0);
    }

    #[test]
    fn test_thermodynamic_cycle() {
        let mut cycle = ThermodynamicCycle::new("Test Cycle".to_string());
        
        let state1 = ThermodynamicState::new(300.0, 101325.0, 1.0);
        let state2 = ThermodynamicState::new(400.0, 101325.0, 1.33);
        
        cycle.add_state(state1);
        cycle.add_state(state2);
        cycle.add_process(ThermodynamicProcess::Isobaric);
        
        assert!(cycle.calculate_cycle_properties().is_ok());
        assert!(cycle.work_done != 0.0);
    }

    #[test]
    fn test_thermodynamics_analyzer() {
        let mut analyzer = ThermodynamicsAnalyzer::new();
        
        let state1 = ThermodynamicState::new(300.0, 101325.0, 1.0);
        let state2 = ThermodynamicState::new(301.0, 101325.0, 1.0);
        
        analyzer.add_state(state1);
        analyzer.add_state(state2);
        
        assert!(analyzer.analyze_thermal_equilibrium(0.01).is_ok());
        
        let stats = analyzer.get_statistics();
        assert_eq!(stats.state_count, 2);
        assert!(stats.avg_temperature > 300.0);
    }

    #[test]
    fn test_entropy_calculation() {
        let mut analyzer = ThermodynamicsAnalyzer::new();
        
        let state1 = ThermodynamicState::new(300.0, 101325.0, 1.0);
        let state2 = ThermodynamicState::new(400.0, 101325.0, 1.0);
        
        analyzer.add_state(state1);
        analyzer.add_state(state2);
        
        let entropy = analyzer.calculate_system_entropy();
        assert!(entropy > 0.0);
    }
} 