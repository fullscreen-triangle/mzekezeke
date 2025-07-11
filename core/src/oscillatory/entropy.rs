use crate::types::{MdtecError, MdtecResult};
use crate::utils::math::Statistics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::{E, LN_2};

/// Entropy calculation methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EntropyMethod {
    /// Shannon entropy (information theory)
    Shannon,
    /// Rényi entropy (generalized entropy)
    Renyi,
    /// Tsallis entropy (non-extensive entropy)
    Tsallis,
    /// Kolmogorov-Sinai entropy (dynamical systems)
    KolmogorovSinai,
    /// Approximate entropy (regularity measure)
    Approximate,
    /// Sample entropy (improved approximate entropy)
    Sample,
    /// Permutation entropy (ordinal patterns)
    Permutation,
    /// Spectral entropy (frequency domain)
    Spectral,
    /// Multiscale entropy (across scales)
    Multiscale,
}

/// Entropy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyConfig {
    /// Primary entropy method
    pub method: EntropyMethod,
    /// Rényi entropy parameter (q)
    pub renyi_q: f64,
    /// Tsallis entropy parameter (q)
    pub tsallis_q: f64,
    /// Approximate entropy parameters
    pub approximate_m: usize,
    pub approximate_r: f64,
    /// Sample entropy parameters
    pub sample_m: usize,
    pub sample_r: f64,
    /// Permutation entropy parameters
    pub permutation_m: usize,
    pub permutation_tau: usize,
    /// Multiscale entropy parameters
    pub multiscale_scales: Vec<usize>,
    pub multiscale_method: EntropyMethod,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        Self {
            method: EntropyMethod::Shannon,
            renyi_q: 2.0,
            tsallis_q: 2.0,
            approximate_m: 2,
            approximate_r: 0.2,
            sample_m: 2,
            sample_r: 0.2,
            permutation_m: 3,
            permutation_tau: 1,
            multiscale_scales: vec![1, 2, 4, 8, 16],
            multiscale_method: EntropyMethod::Sample,
        }
    }
}

/// Entropy calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyResult {
    /// Entropy value
    pub value: f64,
    /// Calculation method used
    pub method: EntropyMethod,
    /// Confidence in the result
    pub confidence: f64,
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
    /// Normalized entropy (0-1)
    pub normalized: f64,
}

/// Entropy analyzer
pub struct EntropyAnalyzer {
    config: EntropyConfig,
    data_history: Vec<Vec<f64>>,
    entropy_history: Vec<EntropyResult>,
}

impl EntropyAnalyzer {
    pub fn new(config: EntropyConfig) -> Self {
        Self {
            config,
            data_history: Vec::new(),
            entropy_history: Vec::new(),
        }
    }

    /// Calculate entropy using configured method
    pub fn calculate_entropy(&mut self, data: &[f64]) -> MdtecResult<EntropyResult> {
        if data.is_empty() {
            return Err(MdtecError::InvalidInput("Data array is empty".to_string()));
        }

        let result = match self.config.method {
            EntropyMethod::Shannon => self.calculate_shannon_entropy(data)?,
            EntropyMethod::Renyi => self.calculate_renyi_entropy(data, self.config.renyi_q)?,
            EntropyMethod::Tsallis => self.calculate_tsallis_entropy(data, self.config.tsallis_q)?,
            EntropyMethod::KolmogorovSinai => self.calculate_kolmogorov_sinai_entropy(data)?,
            EntropyMethod::Approximate => self.calculate_approximate_entropy(data)?,
            EntropyMethod::Sample => self.calculate_sample_entropy(data)?,
            EntropyMethod::Permutation => self.calculate_permutation_entropy(data)?,
            EntropyMethod::Spectral => self.calculate_spectral_entropy(data)?,
            EntropyMethod::Multiscale => self.calculate_multiscale_entropy(data)?,
        };

        // Store result in history
        self.entropy_history.push(result.clone());
        
        // Maintain history size
        if self.entropy_history.len() > 1000 {
            self.entropy_history.remove(0);
        }

        Ok(result)
    }

    /// Calculate Shannon entropy
    fn calculate_shannon_entropy(&self, data: &[f64]) -> MdtecResult<EntropyResult> {
        let entropy = Statistics::shannon_entropy(data);
        let max_entropy = (data.len() as f64).log2();
        let normalized = if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 };

        Ok(EntropyResult {
            value: entropy,
            method: EntropyMethod::Shannon,
            confidence: 1.0,
            parameters: HashMap::new(),
            normalized,
        })
    }

    /// Calculate Rényi entropy
    fn calculate_renyi_entropy(&self, data: &[f64], q: f64) -> MdtecResult<EntropyResult> {
        if q == 1.0 {
            return self.calculate_shannon_entropy(data);
        }

        let histogram = self.create_histogram(data, 256);
        let mut sum = 0.0;
        let total = data.len() as f64;

        for count in histogram.values() {
            if *count > 0 {
                let p = *count as f64 / total;
                sum += p.powf(q);
            }
        }

        let entropy = if q != 1.0 && sum > 0.0 {
            (1.0 / (1.0 - q)) * sum.ln()
        } else {
            0.0
        };

        let mut parameters = HashMap::new();
        parameters.insert("q".to_string(), q);

        Ok(EntropyResult {
            value: entropy,
            method: EntropyMethod::Renyi,
            confidence: 0.9,
            parameters,
            normalized: entropy / 10.0, // Simplified normalization
        })
    }

    /// Calculate Tsallis entropy
    fn calculate_tsallis_entropy(&self, data: &[f64], q: f64) -> MdtecResult<EntropyResult> {
        if q == 1.0 {
            return self.calculate_shannon_entropy(data);
        }

        let histogram = self.create_histogram(data, 256);
        let mut sum = 0.0;
        let total = data.len() as f64;

        for count in histogram.values() {
            if *count > 0 {
                let p = *count as f64 / total;
                sum += p.powf(q);
            }
        }

        let entropy = if q != 1.0 {
            (1.0 - sum) / (q - 1.0)
        } else {
            0.0
        };

        let mut parameters = HashMap::new();
        parameters.insert("q".to_string(), q);

        Ok(EntropyResult {
            value: entropy,
            method: EntropyMethod::Tsallis,
            confidence: 0.9,
            parameters,
            normalized: entropy.abs() / 2.0, // Simplified normalization
        })
    }

    /// Calculate Kolmogorov-Sinai entropy (simplified)
    fn calculate_kolmogorov_sinai_entropy(&self, data: &[f64]) -> MdtecResult<EntropyResult> {
        if data.len() < 10 {
            return Err(MdtecError::InvalidInput("Insufficient data for KS entropy".to_string()));
        }

        // Simplified calculation using correlation sum
        let mut entropy = 0.0;
        let n = data.len();
        let embed_dim = 3;
        let tolerance = Statistics::std_dev(data).unwrap_or(0.0) * 0.1;

        for m in 1..=embed_dim {
            let correlation_sum = self.calculate_correlation_sum(data, m, tolerance);
            if correlation_sum > 0.0 {
                entropy += correlation_sum.ln();
            }
        }

        entropy /= embed_dim as f64;

        let mut parameters = HashMap::new();
        parameters.insert("embed_dim".to_string(), embed_dim as f64);
        parameters.insert("tolerance".to_string(), tolerance);

        Ok(EntropyResult {
            value: entropy.abs(),
            method: EntropyMethod::KolmogorovSinai,
            confidence: 0.7,
            parameters,
            normalized: (entropy.abs() / 10.0).min(1.0),
        })
    }

    /// Calculate approximate entropy
    fn calculate_approximate_entropy(&self, data: &[f64]) -> MdtecResult<EntropyResult> {
        let m = self.config.approximate_m;
        let r = self.config.approximate_r * Statistics::std_dev(data).unwrap_or(0.0);
        
        if data.len() < m + 1 {
            return Err(MdtecError::InvalidInput("Insufficient data for approximate entropy".to_string()));
        }

        let phi_m = self.calculate_phi(data, m, r);
        let phi_m_plus_1 = self.calculate_phi(data, m + 1, r);
        
        let entropy = phi_m - phi_m_plus_1;

        let mut parameters = HashMap::new();
        parameters.insert("m".to_string(), m as f64);
        parameters.insert("r".to_string(), r);

        Ok(EntropyResult {
            value: entropy,
            method: EntropyMethod::Approximate,
            confidence: 0.8,
            parameters,
            normalized: entropy / 2.0,
        })
    }

    /// Calculate sample entropy
    fn calculate_sample_entropy(&self, data: &[f64]) -> MdtecResult<EntropyResult> {
        let m = self.config.sample_m;
        let r = self.config.sample_r * Statistics::std_dev(data).unwrap_or(0.0);
        
        if data.len() < m + 1 {
            return Err(MdtecError::InvalidInput("Insufficient data for sample entropy".to_string()));
        }

        let (a, b) = self.calculate_sample_entropy_counts(data, m, r);
        
        let entropy = if b > 0.0 {
            -(a / b).ln()
        } else {
            0.0
        };

        let mut parameters = HashMap::new();
        parameters.insert("m".to_string(), m as f64);
        parameters.insert("r".to_string(), r);
        parameters.insert("a".to_string(), a);
        parameters.insert("b".to_string(), b);

        Ok(EntropyResult {
            value: entropy,
            method: EntropyMethod::Sample,
            confidence: 0.9,
            parameters,
            normalized: entropy / 3.0,
        })
    }

    /// Calculate permutation entropy
    fn calculate_permutation_entropy(&self, data: &[f64]) -> MdtecResult<EntropyResult> {
        let m = self.config.permutation_m;
        let tau = self.config.permutation_tau;
        
        if data.len() < m * tau {
            return Err(MdtecError::InvalidInput("Insufficient data for permutation entropy".to_string()));
        }

        let ordinal_patterns = self.extract_ordinal_patterns(data, m, tau);
        let total_patterns = ordinal_patterns.values().sum::<usize>() as f64;
        
        let mut entropy = 0.0;
        for count in ordinal_patterns.values() {
            if *count > 0 {
                let p = *count as f64 / total_patterns;
                entropy -= p * p.ln();
            }
        }

        let max_entropy = (Self::factorial(m) as f64).ln();
        let normalized = if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 };

        let mut parameters = HashMap::new();
        parameters.insert("m".to_string(), m as f64);
        parameters.insert("tau".to_string(), tau as f64);

        Ok(EntropyResult {
            value: entropy,
            method: EntropyMethod::Permutation,
            confidence: 0.9,
            parameters,
            normalized,
        })
    }

    /// Calculate spectral entropy
    fn calculate_spectral_entropy(&self, data: &[f64]) -> MdtecResult<EntropyResult> {
        // Simplified spectral entropy using power spectral density
        let power_spectrum = self.calculate_power_spectrum(data);
        let total_power: f64 = power_spectrum.iter().sum();
        
        if total_power == 0.0 {
            return Ok(EntropyResult {
                value: 0.0,
                method: EntropyMethod::Spectral,
                confidence: 0.5,
                parameters: HashMap::new(),
                normalized: 0.0,
            });
        }

        let mut entropy = 0.0;
        for power in power_spectrum {
            if power > 0.0 {
                let p = power / total_power;
                entropy -= p * p.ln();
            }
        }

        let max_entropy = (power_spectrum.len() as f64).ln();
        let normalized = if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 };

        Ok(EntropyResult {
            value: entropy,
            method: EntropyMethod::Spectral,
            confidence: 0.8,
            parameters: HashMap::new(),
            normalized,
        })
    }

    /// Calculate multiscale entropy
    fn calculate_multiscale_entropy(&self, data: &[f64]) -> MdtecResult<EntropyResult> {
        let mut scale_entropies = Vec::new();
        let mut total_entropy = 0.0;

        for &scale in &self.config.multiscale_scales {
            let coarse_grained = self.coarse_grain(data, scale);
            if coarse_grained.len() > 10 {
                let scale_entropy = match self.config.multiscale_method {
                    EntropyMethod::Sample => self.calculate_sample_entropy(&coarse_grained)?,
                    EntropyMethod::Shannon => self.calculate_shannon_entropy(&coarse_grained)?,
                    _ => self.calculate_shannon_entropy(&coarse_grained)?,
                };
                scale_entropies.push(scale_entropy.value);
                total_entropy += scale_entropy.value;
            }
        }

        let avg_entropy = if !scale_entropies.is_empty() {
            total_entropy / scale_entropies.len() as f64
        } else {
            0.0
        };

        let mut parameters = HashMap::new();
        parameters.insert("scales".to_string(), self.config.multiscale_scales.len() as f64);
        
        for (i, entropy) in scale_entropies.iter().enumerate() {
            parameters.insert(format!("scale_{}", i), *entropy);
        }

        Ok(EntropyResult {
            value: avg_entropy,
            method: EntropyMethod::Multiscale,
            confidence: 0.9,
            parameters,
            normalized: avg_entropy / 3.0,
        })
    }

    /// Create histogram from data
    fn create_histogram(&self, data: &[f64], bins: usize) -> HashMap<usize, usize> {
        let mut histogram = HashMap::new();
        
        if data.is_empty() {
            return histogram;
        }

        let min_val = data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_val = data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let range = max_val - min_val;
        
        if range == 0.0 {
            histogram.insert(0, data.len());
            return histogram;
        }

        for value in data {
            let bin = ((value - min_val) / range * bins as f64) as usize;
            let bin = bin.min(bins - 1);
            *histogram.entry(bin).or_insert(0) += 1;
        }

        histogram
    }

    /// Calculate correlation sum for KS entropy
    fn calculate_correlation_sum(&self, data: &[f64], m: usize, tolerance: f64) -> f64 {
        let n = data.len();
        let mut count = 0;
        let mut total = 0;

        for i in 0..n - m {
            for j in i + 1..n - m {
                let mut max_diff = 0.0;
                for k in 0..m {
                    let diff = (data[i + k] - data[j + k]).abs();
                    max_diff = max_diff.max(diff);
                }
                
                total += 1;
                if max_diff < tolerance {
                    count += 1;
                }
            }
        }

        if total > 0 {
            count as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Calculate phi function for approximate entropy
    fn calculate_phi(&self, data: &[f64], m: usize, r: f64) -> f64 {
        let n = data.len();
        let mut phi = 0.0;

        for i in 0..n - m {
            let mut count = 0;
            
            for j in 0..n - m {
                let mut max_diff = 0.0;
                for k in 0..m {
                    if i + k < n && j + k < n {
                        let diff = (data[i + k] - data[j + k]).abs();
                        max_diff = max_diff.max(diff);
                    }
                }
                
                if max_diff <= r {
                    count += 1;
                }
            }
            
            if count > 0 {
                phi += (count as f64 / (n - m) as f64).ln();
            }
        }

        phi / (n - m) as f64
    }

    /// Calculate sample entropy counts
    fn calculate_sample_entropy_counts(&self, data: &[f64], m: usize, r: f64) -> (f64, f64) {
        let n = data.len();
        let mut a = 0.0;
        let mut b = 0.0;

        for i in 0..n - m {
            let mut count_m = 0;
            let mut count_m_plus_1 = 0;

            for j in 0..n - m {
                if i != j {
                    let mut max_diff_m = 0.0;
                    let mut max_diff_m_plus_1 = 0.0;

                    for k in 0..m {
                        if i + k < n && j + k < n {
                            let diff = (data[i + k] - data[j + k]).abs();
                            max_diff_m = max_diff_m.max(diff);
                        }
                    }

                    if max_diff_m <= r {
                        count_m += 1;
                        
                        if i + m < n && j + m < n {
                            let diff = (data[i + m] - data[j + m]).abs();
                            max_diff_m_plus_1 = max_diff_m.max(diff);
                            
                            if max_diff_m_plus_1 <= r {
                                count_m_plus_1 += 1;
                            }
                        }
                    }
                }
            }

            a += count_m_plus_1 as f64;
            b += count_m as f64;
        }

        (a, b)
    }

    /// Extract ordinal patterns for permutation entropy
    fn extract_ordinal_patterns(&self, data: &[f64], m: usize, tau: usize) -> HashMap<Vec<usize>, usize> {
        let mut patterns = HashMap::new();
        
        for i in 0..data.len() - (m - 1) * tau {
            let mut pattern = Vec::new();
            let mut values = Vec::new();
            
            for j in 0..m {
                if i + j * tau < data.len() {
                    values.push((data[i + j * tau], j));
                }
            }
            
            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            
            for (_, original_index) in values {
                pattern.push(original_index);
            }
            
            *patterns.entry(pattern).or_insert(0) += 1;
        }
        
        patterns
    }

    /// Calculate power spectrum (simplified)
    fn calculate_power_spectrum(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut power_spectrum = vec![0.0; n / 2];
        
        // Simplified power spectrum calculation
        for i in 0..n / 2 {
            let freq = i as f64 / n as f64;
            let mut power = 0.0;
            
            for j in 0..n {
                let phase = 2.0 * std::f64::consts::PI * freq * j as f64;
                power += data[j] * phase.cos();
            }
            
            power_spectrum[i] = power * power;
        }
        
        power_spectrum
    }

    /// Coarse grain data for multiscale entropy
    fn coarse_grain(&self, data: &[f64], scale: usize) -> Vec<f64> {
        let mut coarse_grained = Vec::new();
        
        for i in (0..data.len()).step_by(scale) {
            let end = (i + scale).min(data.len());
            let mean = data[i..end].iter().sum::<f64>() / (end - i) as f64;
            coarse_grained.push(mean);
        }
        
        coarse_grained
    }

    /// Calculate factorial
    fn factorial(n: usize) -> usize {
        match n {
            0 | 1 => 1,
            _ => n * Self::factorial(n - 1),
        }
    }

    /// Get entropy statistics
    pub fn get_statistics(&self) -> EntropyStatistics {
        let entropy_values: Vec<f64> = self.entropy_history.iter().map(|e| e.value).collect();
        let confidence_values: Vec<f64> = self.entropy_history.iter().map(|e| e.confidence).collect();
        let normalized_values: Vec<f64> = self.entropy_history.iter().map(|e| e.normalized).collect();

        EntropyStatistics {
            calculation_count: self.entropy_history.len(),
            avg_entropy: Statistics::mean(&entropy_values),
            entropy_std: Statistics::std_dev(&entropy_values).unwrap_or(0.0),
            avg_confidence: Statistics::mean(&confidence_values),
            avg_normalized: Statistics::mean(&normalized_values),
            method_distribution: self.get_method_distribution(),
        }
    }

    /// Get distribution of entropy methods used
    fn get_method_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        
        for result in &self.entropy_history {
            let method_name = format!("{:?}", result.method);
            *distribution.entry(method_name).or_insert(0) += 1;
        }
        
        distribution
    }

    /// Reset analyzer
    pub fn reset(&mut self) {
        self.data_history.clear();
        self.entropy_history.clear();
    }
}

/// Entropy statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyStatistics {
    pub calculation_count: usize,
    pub avg_entropy: f64,
    pub entropy_std: f64,
    pub avg_confidence: f64,
    pub avg_normalized: f64,
    pub method_distribution: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_analyzer_creation() {
        let analyzer = EntropyAnalyzer::new(EntropyConfig::default());
        assert_eq!(analyzer.entropy_history.len(), 0);
    }

    #[test]
    fn test_shannon_entropy_calculation() {
        let mut analyzer = EntropyAnalyzer::new(EntropyConfig::default());
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let result = analyzer.calculate_entropy(&data).unwrap();
        assert_eq!(result.method, EntropyMethod::Shannon);
        assert!(result.value > 0.0);
        assert!(result.normalized >= 0.0 && result.normalized <= 1.0);
    }

    #[test]
    fn test_renyi_entropy_calculation() {
        let mut config = EntropyConfig::default();
        config.method = EntropyMethod::Renyi;
        config.renyi_q = 2.0;
        
        let mut analyzer = EntropyAnalyzer::new(config);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let result = analyzer.calculate_entropy(&data).unwrap();
        assert_eq!(result.method, EntropyMethod::Renyi);
        assert!(result.parameters.contains_key("q"));
    }

    #[test]
    fn test_sample_entropy_calculation() {
        let mut config = EntropyConfig::default();
        config.method = EntropyMethod::Sample;
        
        let mut analyzer = EntropyAnalyzer::new(config);
        let data = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        
        let result = analyzer.calculate_entropy(&data).unwrap();
        assert_eq!(result.method, EntropyMethod::Sample);
        assert!(result.value >= 0.0);
    }

    #[test]
    fn test_permutation_entropy_calculation() {
        let mut config = EntropyConfig::default();
        config.method = EntropyMethod::Permutation;
        
        let mut analyzer = EntropyAnalyzer::new(config);
        let data = vec![1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        let result = analyzer.calculate_entropy(&data).unwrap();
        assert_eq!(result.method, EntropyMethod::Permutation);
        assert!(result.value >= 0.0);
    }

    #[test]
    fn test_multiscale_entropy_calculation() {
        let mut config = EntropyConfig::default();
        config.method = EntropyMethod::Multiscale;
        
        let mut analyzer = EntropyAnalyzer::new(config);
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        
        let result = analyzer.calculate_entropy(&data).unwrap();
        assert_eq!(result.method, EntropyMethod::Multiscale);
        assert!(result.parameters.contains_key("scales"));
    }

    #[test]
    fn test_entropy_history() {
        let mut analyzer = EntropyAnalyzer::new(EntropyConfig::default());
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        
        analyzer.calculate_entropy(&data1).unwrap();
        analyzer.calculate_entropy(&data2).unwrap();
        
        assert_eq!(analyzer.entropy_history.len(), 2);
        
        let stats = analyzer.get_statistics();
        assert_eq!(stats.calculation_count, 2);
        assert!(stats.avg_entropy > 0.0);
    }

    #[test]
    fn test_coarse_graining() {
        let analyzer = EntropyAnalyzer::new(EntropyConfig::default());
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let coarse_grained = analyzer.coarse_grain(&data, 2);
        
        assert_eq!(coarse_grained.len(), 3);
        assert_eq!(coarse_grained[0], 1.5); // (1+2)/2
        assert_eq!(coarse_grained[1], 3.5); // (3+4)/2
        assert_eq!(coarse_grained[2], 5.5); // (5+6)/2
    }
} 