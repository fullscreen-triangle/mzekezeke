//! Mathematical utilities for MDTEC
//!
//! This module provides mathematical functions for vector operations,
//! statistical analysis, and coordinate transformations used throughout
//! the MDTEC system.

use crate::error::{Error, Result};
use std::f64::consts::{E, PI};

/// Vector operations for environmental data
pub struct VectorOps;

/// Statistical analysis utilities
pub struct Statistics;

/// Coordinate transformation utilities
pub struct CoordinateTransform;

/// Constants for mathematical operations
pub struct MathConstants;

impl VectorOps {
    /// Calculate dot product of two vectors
    pub fn dot_product(a: &[f64], b: &[f64]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(Error::invalid_input("Vector dimensions must match"));
        }

        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
    }

    /// Calculate cross product of two 3D vectors
    pub fn cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    /// Calculate magnitude (length) of a vector
    pub fn magnitude(vector: &[f64]) -> f64 {
        vector.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Normalize a vector to unit length
    pub fn normalize(vector: &[f64]) -> Result<Vec<f64>> {
        let mag = Self::magnitude(vector);
        if mag == 0.0 {
            return Err(Error::invalid_input("Cannot normalize zero vector"));
        }

        Ok(vector.iter().map(|x| x / mag).collect())
    }

    /// Calculate angle between two vectors in radians
    pub fn angle_between(a: &[f64], b: &[f64]) -> Result<f64> {
        let dot = Self::dot_product(a, b)?;
        let mag_a = Self::magnitude(a);
        let mag_b = Self::magnitude(b);

        if mag_a == 0.0 || mag_b == 0.0 {
            return Err(Error::invalid_input("Cannot calculate angle with zero vector"));
        }

        let cos_theta = dot / (mag_a * mag_b);
        Ok(cos_theta.clamp(-1.0, 1.0).acos())
    }

    /// Calculate distance between two points
    pub fn euclidean_distance(a: &[f64], b: &[f64]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(Error::invalid_input("Point dimensions must match"));
        }

        let sum_squares: f64 = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();

        Ok(sum_squares.sqrt())
    }

    /// Perform vector addition
    pub fn add(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        if a.len() != b.len() {
            return Err(Error::invalid_input("Vector dimensions must match"));
        }

        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }

    /// Perform vector subtraction
    pub fn subtract(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        if a.len() != b.len() {
            return Err(Error::invalid_input("Vector dimensions must match"));
        }

        Ok(a.iter().zip(b.iter()).map(|(x, y)| x - y).collect())
    }

    /// Scale vector by scalar
    pub fn scale(vector: &[f64], scalar: f64) -> Vec<f64> {
        vector.iter().map(|x| x * scalar).collect()
    }

    /// Calculate weighted average of vectors
    pub fn weighted_average(vectors: &[Vec<f64>], weights: &[f64]) -> Result<Vec<f64>> {
        if vectors.is_empty() {
            return Err(Error::invalid_input("No vectors provided"));
        }

        if vectors.len() != weights.len() {
            return Err(Error::invalid_input("Number of vectors and weights must match"));
        }

        let dim = vectors[0].len();
        if !vectors.iter().all(|v| v.len() == dim) {
            return Err(Error::invalid_input("All vectors must have same dimension"));
        }

        let weight_sum: f64 = weights.iter().sum();
        if weight_sum == 0.0 {
            return Err(Error::invalid_input("Sum of weights cannot be zero"));
        }

        let mut result = vec![0.0; dim];
        for (vector, &weight) in vectors.iter().zip(weights.iter()) {
            for (i, &val) in vector.iter().enumerate() {
                result[i] += val * weight;
            }
        }

        for val in &mut result {
            *val /= weight_sum;
        }

        Ok(result)
    }
}

impl Statistics {
    /// Calculate mean of a dataset
    pub fn mean(data: &[f64]) -> Result<f64> {
        if data.is_empty() {
            return Err(Error::invalid_input("Cannot calculate mean of empty dataset"));
        }

        Ok(data.iter().sum::<f64>() / data.len() as f64)
    }

    /// Calculate variance of a dataset
    pub fn variance(data: &[f64]) -> Result<f64> {
        if data.len() < 2 {
            return Err(Error::invalid_input("Need at least 2 data points for variance"));
        }

        let mean = Self::mean(data)?;
        let sum_squares: f64 = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum();

        Ok(sum_squares / (data.len() - 1) as f64)
    }

    /// Calculate standard deviation of a dataset
    pub fn standard_deviation(data: &[f64]) -> Result<f64> {
        Ok(Self::variance(data)?.sqrt())
    }

    /// Calculate correlation coefficient between two datasets
    pub fn correlation(x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(Error::invalid_input("Datasets must have same length and at least 2 points"));
        }

        let mean_x = Self::mean(x)?;
        let mean_y = Self::mean(y)?;

        let numerator: f64 = x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator == 0.0 {
            return Ok(0.0);
        }

        Ok(numerator / denominator)
    }

    /// Calculate Shannon entropy of a probability distribution
    pub fn shannon_entropy(probabilities: &[f64]) -> Result<f64> {
        if probabilities.is_empty() {
            return Err(Error::invalid_input("Cannot calculate entropy of empty distribution"));
        }

        let sum: f64 = probabilities.iter().sum();
        if (sum - 1.0).abs() > 1e-10 {
            return Err(Error::invalid_input("Probabilities must sum to 1.0"));
        }

        let entropy = probabilities.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum();

        Ok(entropy)
    }

    /// Calculate percentile of a dataset
    pub fn percentile(data: &[f64], p: f64) -> Result<f64> {
        if data.is_empty() {
            return Err(Error::invalid_input("Cannot calculate percentile of empty dataset"));
        }

        if !(0.0..=100.0).contains(&p) {
            return Err(Error::invalid_input("Percentile must be between 0 and 100"));
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (p / 100.0) * (sorted_data.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            Ok(sorted_data[lower])
        } else {
            let weight = index - lower as f64;
            Ok(sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight)
        }
    }

    /// Calculate moving average with specified window size
    pub fn moving_average(data: &[f64], window_size: usize) -> Result<Vec<f64>> {
        if window_size == 0 || window_size > data.len() {
            return Err(Error::invalid_input("Invalid window size"));
        }

        let mut result = Vec::new();
        for i in 0..=(data.len() - window_size) {
            let window_sum: f64 = data[i..i + window_size].iter().sum();
            result.push(window_sum / window_size as f64);
        }

        Ok(result)
    }

    /// Calculate exponential moving average
    pub fn exponential_moving_average(data: &[f64], alpha: f64) -> Result<Vec<f64>> {
        if data.is_empty() {
            return Err(Error::invalid_input("Cannot calculate EMA of empty dataset"));
        }

        if !(0.0..=1.0).contains(&alpha) {
            return Err(Error::invalid_input("Alpha must be between 0 and 1"));
        }

        let mut result = Vec::with_capacity(data.len());
        result.push(data[0]);

        for &value in &data[1..] {
            let ema = alpha * value + (1.0 - alpha) * result.last().unwrap();
            result.push(ema);
        }

        Ok(result)
    }
}

impl CoordinateTransform {
    /// Convert Cartesian coordinates to spherical
    pub fn cartesian_to_spherical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r = (x * x + y * y + z * z).sqrt();
        let theta = (z / r).acos(); // polar angle
        let phi = y.atan2(x); // azimuthal angle

        (r, theta, phi)
    }

    /// Convert spherical coordinates to Cartesian
    pub fn spherical_to_cartesian(r: f64, theta: f64, phi: f64) -> (f64, f64, f64) {
        let x = r * theta.sin() * phi.cos();
        let y = r * theta.sin() * phi.sin();
        let z = r * theta.cos();

        (x, y, z)
    }

    /// Convert GPS coordinates to Cartesian (simplified Earth model)
    pub fn gps_to_cartesian(lat: f64, lon: f64, alt: f64) -> (f64, f64, f64) {
        let earth_radius = 6371000.0; // meters
        let lat_rad = lat.to_radians();
        let lon_rad = lon.to_radians();
        let r = earth_radius + alt;

        let x = r * lat_rad.cos() * lon_rad.cos();
        let y = r * lat_rad.cos() * lon_rad.sin();
        let z = r * lat_rad.sin();

        (x, y, z)
    }

    /// Calculate distance between two GPS coordinates (Haversine formula)
    pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        let earth_radius = 6371000.0; // meters

        let dlat = (lat2 - lat1).to_radians();
        let dlon = (lon2 - lon1).to_radians();
        let lat1_rad = lat1.to_radians();
        let lat2_rad = lat2.to_radians();

        let a = (dlat / 2.0).sin().powi(2) +
                lat1_rad.cos() * lat2_rad.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        earth_radius * c
    }

    /// Apply rotation matrix to 3D point
    pub fn rotate_3d(point: &[f64; 3], rotation_matrix: &[[f64; 3]; 3]) -> [f64; 3] {
        [
            rotation_matrix[0][0] * point[0] + rotation_matrix[0][1] * point[1] + rotation_matrix[0][2] * point[2],
            rotation_matrix[1][0] * point[0] + rotation_matrix[1][1] * point[1] + rotation_matrix[1][2] * point[2],
            rotation_matrix[2][0] * point[0] + rotation_matrix[2][1] * point[1] + rotation_matrix[2][2] * point[2],
        ]
    }

    /// Create rotation matrix around X-axis
    pub fn rotation_matrix_x(angle: f64) -> [[f64; 3]; 3] {
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        [
            [1.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a],
            [0.0, sin_a, cos_a],
        ]
    }

    /// Create rotation matrix around Y-axis
    pub fn rotation_matrix_y(angle: f64) -> [[f64; 3]; 3] {
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ]
    }

    /// Create rotation matrix around Z-axis
    pub fn rotation_matrix_z(angle: f64) -> [[f64; 3]; 3] {
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    /// Transform coordinates using homogeneous transformation matrix
    pub fn homogeneous_transform(point: &[f64; 3], transform: &[[f64; 4]; 4]) -> [f64; 3] {
        let homogeneous_point = [point[0], point[1], point[2], 1.0];
        let mut result = [0.0; 4];

        for i in 0..4 {
            for j in 0..4 {
                result[i] += transform[i][j] * homogeneous_point[j];
            }
        }

        // Convert back to 3D by dividing by w component
        if result[3] != 0.0 {
            [result[0] / result[3], result[1] / result[3], result[2] / result[3]]
        } else {
            [result[0], result[1], result[2]]
        }
    }
}

impl MathConstants {
    /// Euler's number
    pub const E: f64 = E;

    /// Pi
    pub const PI: f64 = PI;

    /// Golden ratio (φ) - Masunda memorial constant
    pub const GOLDEN_RATIO: f64 = 1.618033988749894;

    /// Square root of 2
    pub const SQRT_2: f64 = 1.414213562373095;

    /// Square root of 3
    pub const SQRT_3: f64 = 1.7320508075688772;

    /// Natural log of 2
    pub const LN_2: f64 = 0.6931471805599453;

    /// Speed of light in m/s
    pub const SPEED_OF_LIGHT: f64 = 299792458.0;

    /// Planck constant
    pub const PLANCK_CONSTANT: f64 = 6.62607015e-34;

    /// Boltzmann constant
    pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;
}

/// Fast Fourier Transform utilities
pub struct FFT;

impl FFT {
    /// Compute the magnitude spectrum of a real signal
    pub fn magnitude_spectrum(signal: &[f64]) -> Result<Vec<f64>> {
        if signal.is_empty() {
            return Err(Error::invalid_input("Cannot compute FFT of empty signal"));
        }

        // Simplified FFT for magnitude computation
        // In practice, this would use a proper FFT library
        let n = signal.len();
        let mut magnitudes = Vec::with_capacity(n / 2 + 1);

        for k in 0..=(n / 2) {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for j in 0..n {
                let angle = -2.0 * PI * (k as f64) * (j as f64) / (n as f64);
                real_sum += signal[j] * angle.cos();
                imag_sum += signal[j] * angle.sin();
            }

            let magnitude = (real_sum * real_sum + imag_sum * imag_sum).sqrt();
            magnitudes.push(magnitude);
        }

        Ok(magnitudes)
    }

    /// Find dominant frequency in a signal
    pub fn dominant_frequency(signal: &[f64], sample_rate: f64) -> Result<f64> {
        let magnitudes = Self::magnitude_spectrum(signal)?;
        
        let max_index = magnitudes.iter()
            .enumerate()
            .skip(1) // Skip DC component
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(max_index as f64 * sample_rate / signal.len() as f64)
    }
}

/// Interpolation utilities
pub struct Interpolation;

impl Interpolation {
    /// Linear interpolation between two points
    pub fn linear(x: f64, x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
        if x1 == x0 {
            return y0;
        }
        
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }

    /// Bilinear interpolation
    pub fn bilinear(x: f64, y: f64, 
                   x0: f64, x1: f64, y0: f64, y1: f64,
                   f00: f64, f01: f64, f10: f64, f11: f64) -> f64 {
        let f_x0 = Self::linear(y, y0, f00, y1, f01);
        let f_x1 = Self::linear(y, y0, f10, y1, f11);
        Self::linear(x, x0, f_x0, x1, f_x1)
    }

    /// Cubic spline interpolation (simplified)
    pub fn cubic_spline(x: f64, x_points: &[f64], y_points: &[f64]) -> Result<f64> {
        if x_points.len() != y_points.len() || x_points.len() < 2 {
            return Err(Error::invalid_input("Invalid interpolation points"));
        }

        // Find the interval containing x
        let mut i = 0;
        while i < x_points.len() - 1 && x > x_points[i + 1] {
            i += 1;
        }

        if i == x_points.len() - 1 {
            return Ok(y_points[i]);
        }

        // Simple cubic interpolation (in practice, would use proper spline coefficients)
        let t = (x - x_points[i]) / (x_points[i + 1] - x_points[i]);
        let t2 = t * t;
        let t3 = t2 * t;

        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        Ok(h00 * y_points[i] + h01 * y_points[i + 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_operations() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let dot = VectorOps::dot_product(&a, &b).unwrap();
        assert_eq!(dot, 32.0);

        let magnitude = VectorOps::magnitude(&a);
        assert!((magnitude - 3.7416573867739413).abs() < 1e-10);

        let sum = VectorOps::add(&a, &b).unwrap();
        assert_eq!(sum, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mean = Statistics::mean(&data).unwrap();
        assert_eq!(mean, 3.0);

        let variance = Statistics::variance(&data).unwrap();
        assert_eq!(variance, 2.5);

        let std_dev = Statistics::standard_deviation(&data).unwrap();
        assert!((std_dev - 1.5811388300841898).abs() < 1e-10);
    }

    #[test]
    fn test_coordinate_transform() {
        let (x, y, z) = CoordinateTransform::spherical_to_cartesian(1.0, PI / 2.0, 0.0);
        assert!((x - 1.0).abs() < 1e-10);
        assert!(y.abs() < 1e-10);
        assert!(z.abs() < 1e-10);

        let distance = CoordinateTransform::haversine_distance(0.0, 0.0, 0.0, 1.0);
        assert!(distance > 110000.0 && distance < 112000.0); // Approximately 111 km
    }

    #[test]
    fn test_math_constants() {
        assert_eq!(MathConstants::PI, PI);
        assert_eq!(MathConstants::E, E);
        assert_eq!(MathConstants::GOLDEN_RATIO, 1.618033988749894);
    }

    #[test]
    fn test_fft() {
        let signal = vec![1.0, 0.0, -1.0, 0.0]; // Simple square wave
        let magnitudes = FFT::magnitude_spectrum(&signal).unwrap();
        assert_eq!(magnitudes.len(), 3); // DC + 2 frequency bins
    }

    #[test]
    fn test_interpolation() {
        let result = Interpolation::linear(1.5, 1.0, 10.0, 2.0, 20.0);
        assert_eq!(result, 15.0);

        let x_points = vec![0.0, 1.0, 2.0];
        let y_points = vec![0.0, 1.0, 4.0];
        let result = Interpolation::cubic_spline(0.5, &x_points, &y_points).unwrap();
        assert!(result >= 0.0 && result <= 1.0);
    }
}
