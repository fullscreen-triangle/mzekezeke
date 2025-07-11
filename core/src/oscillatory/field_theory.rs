use crate::types::{MdtecError, MdtecResult};
use crate::utils::math::{Statistics, VectorOps, MathConstants};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Three-dimensional vector representation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vector3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag > 0.0 {
            Self::new(self.x / mag, self.y / mag, self.z / mag)
        } else {
            Self::zero()
        }
    }

    pub fn dot(&self, other: &Vector3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vector3D) -> Vector3D {
        Vector3D::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn scale(&self, scalar: f64) -> Vector3D {
        Vector3D::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }

    pub fn add(&self, other: &Vector3D) -> Vector3D {
        Vector3D::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    pub fn subtract(&self, other: &Vector3D) -> Vector3D {
        Vector3D::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

/// Electromagnetic field representation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ElectromagneticField {
    /// Electric field vector (V/m)
    pub electric_field: Vector3D,
    /// Magnetic field vector (Tesla)
    pub magnetic_field: Vector3D,
    /// Field frequency (Hz)
    pub frequency: f64,
    /// Field phase (radians)
    pub phase: f64,
}

impl ElectromagneticField {
    pub fn new(electric_field: Vector3D, magnetic_field: Vector3D, frequency: f64, phase: f64) -> Self {
        Self {
            electric_field,
            magnetic_field,
            frequency,
            phase,
        }
    }

    /// Calculate field energy density (J/m³)
    pub fn energy_density(&self) -> f64 {
        let epsilon_0 = MathConstants::EPSILON_0;
        let mu_0 = MathConstants::MU_0;
        
        let electric_energy = 0.5 * epsilon_0 * self.electric_field.magnitude().powi(2);
        let magnetic_energy = 0.5 * self.magnetic_field.magnitude().powi(2) / mu_0;
        
        electric_energy + magnetic_energy
    }

    /// Calculate Poynting vector (W/m²)
    pub fn poynting_vector(&self) -> Vector3D {
        let mu_0 = MathConstants::MU_0;
        let cross_product = self.electric_field.cross(&self.magnetic_field);
        cross_product.scale(1.0 / mu_0)
    }

    /// Calculate field impedance (Ohms)
    pub fn impedance(&self) -> f64 {
        let e_mag = self.electric_field.magnitude();
        let h_mag = self.magnetic_field.magnitude();
        
        if h_mag > 0.0 {
            e_mag / h_mag
        } else {
            0.0
        }
    }

    /// Check if field satisfies Maxwell's equations (simplified)
    pub fn is_physically_consistent(&self) -> bool {
        // Check if E and B are perpendicular (for plane waves)
        let dot_product = self.electric_field.dot(&self.magnetic_field);
        let tolerance = 1e-10;
        
        dot_product.abs() < tolerance
    }
}

/// Wave propagation parameters
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WaveParameters {
    /// Wave vector (rad/m)
    pub wave_vector: Vector3D,
    /// Angular frequency (rad/s)
    pub angular_frequency: f64,
    /// Wavelength (m)
    pub wavelength: f64,
    /// Wave speed (m/s)
    pub wave_speed: f64,
}

impl WaveParameters {
    pub fn new(frequency: f64, wave_vector: Vector3D) -> Self {
        let angular_frequency = 2.0 * PI * frequency;
        let k_magnitude = wave_vector.magnitude();
        let wavelength = if k_magnitude > 0.0 { 2.0 * PI / k_magnitude } else { 0.0 };
        let wave_speed = if k_magnitude > 0.0 { angular_frequency / k_magnitude } else { 0.0 };
        
        Self {
            wave_vector,
            angular_frequency,
            wavelength,
            wave_speed,
        }
    }

    /// Calculate phase velocity
    pub fn phase_velocity(&self) -> f64 {
        self.wave_speed
    }

    /// Calculate group velocity (simplified)
    pub fn group_velocity(&self) -> f64 {
        // For electromagnetic waves in vacuum, group velocity = phase velocity
        self.wave_speed
    }
}

/// Field interaction types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FieldInteractionType {
    /// Superposition of fields
    Superposition,
    /// Interference pattern
    Interference,
    /// Resonance
    Resonance,
    /// Scattering
    Scattering,
    /// Diffraction
    Diffraction,
}

/// Field interaction analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldInteraction {
    /// Type of interaction
    pub interaction_type: FieldInteractionType,
    /// Interaction strength
    pub strength: f64,
    /// Resulting field
    pub resulting_field: ElectromagneticField,
    /// Interaction efficiency
    pub efficiency: f64,
}

/// Field theory analyzer
pub struct FieldTheoryAnalyzer {
    /// Current electromagnetic fields
    fields: Vec<ElectromagneticField>,
    /// Field interaction history
    interactions: Vec<FieldInteraction>,
    /// Wave parameters
    wave_parameters: HashMap<usize, WaveParameters>,
}

impl FieldTheoryAnalyzer {
    pub fn new() -> Self {
        Self {
            fields: Vec::new(),
            interactions: Vec::new(),
            wave_parameters: HashMap::new(),
        }
    }

    /// Add electromagnetic field
    pub fn add_field(&mut self, field: ElectromagneticField) -> usize {
        self.fields.push(field);
        let field_id = self.fields.len() - 1;
        
        // Calculate wave parameters
        let k_magnitude = 2.0 * PI * field.frequency / MathConstants::SPEED_OF_LIGHT;
        let wave_vector = Vector3D::new(k_magnitude, 0.0, 0.0); // Simplified assumption
        let wave_params = WaveParameters::new(field.frequency, wave_vector);
        self.wave_parameters.insert(field_id, wave_params);
        
        field_id
    }

    /// Calculate field superposition
    pub fn calculate_superposition(&self, field_indices: &[usize]) -> MdtecResult<ElectromagneticField> {
        if field_indices.is_empty() {
            return Err(MdtecError::InvalidInput("No fields provided for superposition".to_string()));
        }

        let mut total_electric = Vector3D::zero();
        let mut total_magnetic = Vector3D::zero();
        let mut weighted_frequency = 0.0;
        let mut weighted_phase = 0.0;
        let mut total_weight = 0.0;

        for &index in field_indices {
            if index >= self.fields.len() {
                return Err(MdtecError::InvalidInput(format!("Field index {} out of bounds", index)));
            }

            let field = &self.fields[index];
            let weight = field.electric_field.magnitude();
            
            total_electric = total_electric.add(&field.electric_field);
            total_magnetic = total_magnetic.add(&field.magnetic_field);
            weighted_frequency += field.frequency * weight;
            weighted_phase += field.phase * weight;
            total_weight += weight;
        }

        let result_frequency = if total_weight > 0.0 { weighted_frequency / total_weight } else { 0.0 };
        let result_phase = if total_weight > 0.0 { weighted_phase / total_weight } else { 0.0 };

        Ok(ElectromagneticField::new(
            total_electric,
            total_magnetic,
            result_frequency,
            result_phase,
        ))
    }

    /// Analyze field interference
    pub fn analyze_interference(&self, field1_id: usize, field2_id: usize) -> MdtecResult<FieldInteraction> {
        if field1_id >= self.fields.len() || field2_id >= self.fields.len() {
            return Err(MdtecError::InvalidInput("Invalid field indices".to_string()));
        }

        let field1 = &self.fields[field1_id];
        let field2 = &self.fields[field2_id];

        // Calculate phase difference
        let phase_diff = (field1.phase - field2.phase).abs();
        let normalized_phase_diff = phase_diff % (2.0 * PI);

        // Determine interference type and strength
        let (strength, interaction_type) = if normalized_phase_diff < PI / 4.0 || normalized_phase_diff > 7.0 * PI / 4.0 {
            // Constructive interference
            (1.0 - normalized_phase_diff / (PI / 4.0), FieldInteractionType::Interference)
        } else if normalized_phase_diff > 3.0 * PI / 4.0 && normalized_phase_diff < 5.0 * PI / 4.0 {
            // Destructive interference
            (1.0 - (PI - normalized_phase_diff).abs() / (PI / 4.0), FieldInteractionType::Interference)
        } else {
            // Partial interference
            (0.5, FieldInteractionType::Interference)
        };

        // Calculate resulting field
        let resulting_field = self.calculate_superposition(&[field1_id, field2_id])?;

        // Calculate efficiency
        let original_energy = field1.energy_density() + field2.energy_density();
        let resulting_energy = resulting_field.energy_density();
        let efficiency = if original_energy > 0.0 { resulting_energy / original_energy } else { 0.0 };

        Ok(FieldInteraction {
            interaction_type,
            strength,
            resulting_field,
            efficiency,
        })
    }

    /// Analyze field resonance
    pub fn analyze_resonance(&self, field_id: usize, resonant_frequency: f64) -> MdtecResult<FieldInteraction> {
        if field_id >= self.fields.len() {
            return Err(MdtecError::InvalidInput("Invalid field index".to_string()));
        }

        let field = &self.fields[field_id];
        let frequency_diff = (field.frequency - resonant_frequency).abs();
        let resonance_bandwidth = resonant_frequency * 0.01; // 1% bandwidth

        let strength = if frequency_diff < resonance_bandwidth {
            1.0 - frequency_diff / resonance_bandwidth
        } else {
            0.0
        };

        // Resonance amplifies the field
        let amplification_factor = 1.0 + strength * 10.0;
        let amplified_electric = field.electric_field.scale(amplification_factor);
        let amplified_magnetic = field.magnetic_field.scale(amplification_factor);

        let resulting_field = ElectromagneticField::new(
            amplified_electric,
            amplified_magnetic,
            field.frequency,
            field.phase,
        );

        Ok(FieldInteraction {
            interaction_type: FieldInteractionType::Resonance,
            strength,
            resulting_field,
            efficiency: amplification_factor,
        })
    }

    /// Calculate field gradient
    pub fn calculate_field_gradient(&self, field_id: usize, direction: Vector3D) -> MdtecResult<Vector3D> {
        if field_id >= self.fields.len() {
            return Err(MdtecError::InvalidInput("Invalid field index".to_string()));
        }

        let field = &self.fields[field_id];
        let normalized_direction = direction.normalize();

        // Simplified gradient calculation
        let gradient_magnitude = field.electric_field.magnitude() * 0.1; // Placeholder
        let gradient = normalized_direction.scale(gradient_magnitude);

        Ok(gradient)
    }

    /// Calculate field divergence
    pub fn calculate_divergence(&self, field_id: usize) -> MdtecResult<f64> {
        if field_id >= self.fields.len() {
            return Err(MdtecError::InvalidInput("Invalid field index".to_string()));
        }

        let field = &self.fields[field_id];
        
        // For electromagnetic fields in vacuum, div(E) = 0 (Gauss's law)
        // This is a simplified calculation
        let divergence = 0.0; // Idealized case
        
        Ok(divergence)
    }

    /// Calculate field curl
    pub fn calculate_curl(&self, field_id: usize) -> MdtecResult<Vector3D> {
        if field_id >= self.fields.len() {
            return Err(MdtecError::InvalidInput("Invalid field index".to_string()));
        }

        let field = &self.fields[field_id];
        let wave_params = self.wave_parameters.get(&field_id).unwrap();

        // Simplified curl calculation using Maxwell's equations
        // curl(E) = -∂B/∂t
        let curl_magnitude = wave_params.angular_frequency * field.magnetic_field.magnitude();
        let curl_direction = field.magnetic_field.normalize();
        let curl = curl_direction.scale(-curl_magnitude);

        Ok(curl)
    }

    /// Analyze wave propagation
    pub fn analyze_wave_propagation(&self, field_id: usize, distance: f64) -> MdtecResult<ElectromagneticField> {
        if field_id >= self.fields.len() {
            return Err(MdtecError::InvalidInput("Invalid field index".to_string()));
        }

        let field = &self.fields[field_id];
        let wave_params = self.wave_parameters.get(&field_id).unwrap();

        // Calculate phase change due to propagation
        let phase_change = wave_params.wave_vector.magnitude() * distance;
        let new_phase = field.phase + phase_change;

        // Calculate amplitude attenuation (simplified)
        let attenuation_factor = (-distance / 1000.0).exp(); // 1km characteristic distance
        let attenuated_electric = field.electric_field.scale(attenuation_factor);
        let attenuated_magnetic = field.magnetic_field.scale(attenuation_factor);

        Ok(ElectromagneticField::new(
            attenuated_electric,
            attenuated_magnetic,
            field.frequency,
            new_phase,
        ))
    }

    /// Calculate field entropy
    pub fn calculate_field_entropy(&self) -> MdtecResult<f64> {
        if self.fields.is_empty() {
            return Ok(0.0);
        }

        let energies: Vec<f64> = self.fields.iter().map(|f| f.energy_density()).collect();
        let entropy = Statistics::shannon_entropy(&energies);
        Ok(entropy)
    }

    /// Get field statistics
    pub fn get_field_statistics(&self) -> FieldStatistics {
        let electric_magnitudes: Vec<f64> = self.fields.iter().map(|f| f.electric_field.magnitude()).collect();
        let magnetic_magnitudes: Vec<f64> = self.fields.iter().map(|f| f.magnetic_field.magnitude()).collect();
        let frequencies: Vec<f64> = self.fields.iter().map(|f| f.frequency).collect();
        let energy_densities: Vec<f64> = self.fields.iter().map(|f| f.energy_density()).collect();

        FieldStatistics {
            field_count: self.fields.len(),
            interaction_count: self.interactions.len(),
            avg_electric_magnitude: Statistics::mean(&electric_magnitudes),
            avg_magnetic_magnitude: Statistics::mean(&magnetic_magnitudes),
            avg_frequency: Statistics::mean(&frequencies),
            avg_energy_density: Statistics::mean(&energy_densities),
            field_entropy: self.calculate_field_entropy().unwrap_or(0.0),
        }
    }

    /// Reset analyzer
    pub fn reset(&mut self) {
        self.fields.clear();
        self.interactions.clear();
        self.wave_parameters.clear();
    }
}

/// Field theory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldStatistics {
    pub field_count: usize,
    pub interaction_count: usize,
    pub avg_electric_magnitude: f64,
    pub avg_magnetic_magnitude: f64,
    pub avg_frequency: f64,
    pub avg_energy_density: f64,
    pub field_entropy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector3d_operations() {
        let v1 = Vector3D::new(1.0, 2.0, 3.0);
        let v2 = Vector3D::new(4.0, 5.0, 6.0);

        assert_eq!(v1.magnitude(), (1.0 + 4.0 + 9.0_f64).sqrt());
        assert_eq!(v1.dot(&v2), 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
        
        let cross = v1.cross(&v2);
        assert_eq!(cross.x, 2.0 * 6.0 - 3.0 * 5.0);
        assert_eq!(cross.y, 3.0 * 4.0 - 1.0 * 6.0);
        assert_eq!(cross.z, 1.0 * 5.0 - 2.0 * 4.0);
    }

    #[test]
    fn test_electromagnetic_field_energy() {
        let electric = Vector3D::new(1.0, 0.0, 0.0);
        let magnetic = Vector3D::new(0.0, 1.0, 0.0);
        let field = ElectromagneticField::new(electric, magnetic, 1e9, 0.0);

        let energy = field.energy_density();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_field_theory_analyzer() {
        let mut analyzer = FieldTheoryAnalyzer::new();

        let electric = Vector3D::new(1.0, 0.0, 0.0);
        let magnetic = Vector3D::new(0.0, 1.0, 0.0);
        let field = ElectromagneticField::new(electric, magnetic, 1e9, 0.0);

        let field_id = analyzer.add_field(field);
        assert_eq!(field_id, 0);
        assert_eq!(analyzer.fields.len(), 1);
    }

    #[test]
    fn test_field_superposition() {
        let mut analyzer = FieldTheoryAnalyzer::new();

        let field1 = ElectromagneticField::new(
            Vector3D::new(1.0, 0.0, 0.0),
            Vector3D::new(0.0, 1.0, 0.0),
            1e9,
            0.0,
        );

        let field2 = ElectromagneticField::new(
            Vector3D::new(0.0, 1.0, 0.0),
            Vector3D::new(0.0, 0.0, 1.0),
            1e9,
            0.0,
        );

        let id1 = analyzer.add_field(field1);
        let id2 = analyzer.add_field(field2);

        let superposition = analyzer.calculate_superposition(&[id1, id2]).unwrap();
        assert_eq!(superposition.electric_field.x, 1.0);
        assert_eq!(superposition.electric_field.y, 1.0);
        assert_eq!(superposition.electric_field.z, 0.0);
    }

    #[test]
    fn test_wave_propagation() {
        let mut analyzer = FieldTheoryAnalyzer::new();

        let field = ElectromagneticField::new(
            Vector3D::new(1.0, 0.0, 0.0),
            Vector3D::new(0.0, 1.0, 0.0),
            1e9,
            0.0,
        );

        let field_id = analyzer.add_field(field);
        let propagated = analyzer.analyze_wave_propagation(field_id, 100.0).unwrap();

        // Field should be attenuated
        assert!(propagated.electric_field.magnitude() < field.electric_field.magnitude());
        
        // Phase should change
        assert!(propagated.phase != field.phase);
    }
} 