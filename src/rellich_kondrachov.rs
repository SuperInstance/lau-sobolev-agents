//! Rellich-Kondrachov theorem: compact embedding Wᵏᵖ ⊂ Lᵍ on bounded domains.
//!
//! If Ω is a bounded Lipschitz domain, k ≥ 1, 1 ≤ p < ∞, and
//! 1 ≤ q < p* = np/(n - kp) (when kp < n), then Wᵏᵖ(Ω) ⊂⊂ Lᵠ(Ω)
//! is a compact embedding.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::SobolevSpace;

/// Rellich-Kondrachov compact embedding checker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RellichKondrachov {
    pub space: SobolevSpace,
    /// Whether the domain is bounded
    pub bounded_domain: bool,
}

/// Result of checking Rellich-Kondrachov compact embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactEmbeddingResult {
    pub is_compact: bool,
    /// Optimal target exponent p*
    pub sobolev_conjugate: Option<f64>,
    /// Reasoning
    pub reason: String,
}

impl RellichKondrachov {
    pub fn new(space: SobolevSpace, bounded_domain: bool) -> Self {
        Self { space, bounded_domain }
    }

    /// Compute the Sobolev conjugate exponent p* = np/(n - kp).
    pub fn sobolev_conjugate(&self) -> Option<f64> {
        let n = self.space.n_dims as f64;
        let k = self.space.k as f64;
        let p = self.space.p;

        let kp = k * p;
        if kp < n {
            Some(n * p / (n - kp))
        } else {
            None // kp ≥ n: embedding into L^∞ (Holder continuous)
        }
    }

    /// Check if Wᵏᵖ(Ω) ⊂⊂ Lᵠ(Ω) is compact for a given target q.
    pub fn check_compact_embedding(&self, target_q: f64) -> CompactEmbeddingResult {
        if !self.bounded_domain {
            return CompactEmbeddingResult {
                is_compact: false,
                sobolev_conjugate: self.sobolev_conjugate(),
                reason: "Domain is not bounded".to_string(),
            };
        }

        let n = self.space.n_dims as f64;
        let k = self.space.k as f64;
        let p = self.space.p;
        let kp = k * p;

        if kp < n {
            let p_star = n * p / (n - kp);
            if target_q < p_star {
                CompactEmbeddingResult {
                    is_compact: true,
                    sobolev_conjugate: Some(p_star),
                    reason: format!(
                        "kp = {} < n = {}: compact for q = {} < p* = {:.2}",
                        kp, n, target_q, p_star
                    ),
                }
            } else {
                CompactEmbeddingResult {
                    is_compact: false,
                    sobolev_conjugate: Some(p_star),
                    reason: format!(
                        "q = {} ≥ p* = {:.2}: not compact at critical exponent",
                        target_q, p_star
                    ),
                }
            }
        } else if kp == n {
            // Compact for all q < ∞
            CompactEmbeddingResult {
                is_compact: target_q.is_finite(),
                sobolev_conjugate: None,
                reason: format!("kp = n: compact for all finite q, asked q = {}", target_q),
            }
        } else {
            // kp > n: compact into C⁰ (Holder)
            CompactEmbeddingResult {
                is_compact: true,
                sobolev_conjugate: None,
                reason: format!(
                    "kp = {} > n = {}: compact into C⁰",
                    kp, n
                ),
            }
        }
    }

    /// Check compactness into a Sobolev space Wˡᵠ.
    pub fn check_compact_into_sobolev(&self, target_k: usize, target_p: f64) -> CompactEmbeddingResult {
        let n = self.space.n_dims as f64;
        let k = self.space.k as f64;
        let p = self.space.p;
        let tk = target_k as f64;
        let tp = target_p;

        if !self.bounded_domain {
            return CompactEmbeddingResult {
                is_compact: false,
                sobolev_conjugate: self.sobolev_conjugate(),
                reason: "Domain is not bounded".to_string(),
            };
        }

        // Wᵏᵖ ⊂⊂ Wˡᵠ when k > l and k - n/p > l - n/q (with q finite)
        let diff_source = k - n / p;
        let diff_target = tk - n / tp;

        if diff_source > diff_target && k > target_k as f64 {
            CompactEmbeddingResult {
                is_compact: true,
                sobolev_conjugate: self.sobolev_conjugate(),
                reason: format!(
                    "k - n/p = {:.2} > l - n/q = {:.2}: compact",
                    diff_source, diff_target
                ),
            }
        } else {
            CompactEmbeddingResult {
                is_compact: false,
                sobolev_conjugate: self.sobolev_conjugate(),
                reason: format!(
                    "k - n/p = {:.2} ≤ l - n/q = {:.2}: not compact",
                    diff_source, diff_target
                ),
            }
        }
    }

    /// Estimate the compactness modulus for a bounded sequence.
    /// This measures how "tight" the embedding is.
    pub fn compactness_modulus(
        &self,
        functions: &[DVector<f64>],
        _spacings: &[f64],
    ) -> f64 {
        if functions.is_empty() {
            return 0.0;
        }

        let n = functions[0].len();
        let mut mean = DVector::zeros(n);

        for f in functions {
            mean += f;
        }
        mean /= functions.len() as f64;

        // Compute variance (measure of spread in L²)
        let mut variance = 0.0;
        for f in functions {
            let diff = f - &mean;
            variance += diff.norm_squared() / n as f64;
        }
        variance /= functions.len() as f64;

        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobolev_conjugate_1d() {
        let rk = RellichKondrachov::new(SobolevSpace::h1(1), true);
        // kp = 1*2 = 2 > n = 1, so no finite conjugate
        assert!(rk.sobolev_conjugate().is_none());
    }

    #[test]
    fn test_sobolev_conjugate_3d() {
        let rk = RellichKondrachov::new(SobolevSpace::h1(3), true);
        // kp = 1*2 = 2 < n = 3, p* = 3*2/(3-2) = 6
        let ps = rk.sobolev_conjugate().unwrap();
        assert!((ps - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_compact_embedding_bounded() {
        let rk = RellichKondrachov::new(SobolevSpace::h1(3), true);
        let result = rk.check_compact_embedding(4.0);
        assert!(result.is_compact);
    }

    #[test]
    fn test_compact_embedding_unbounded() {
        let rk = RellichKondrachov::new(SobolevSpace::h1(3), false);
        let result = rk.check_compact_embedding(4.0);
        assert!(!result.is_compact);
    }

    #[test]
    fn test_compact_at_critical_exponent() {
        let rk = RellichKondrachov::new(SobolevSpace::h1(3), true);
        // p* = 6, asking q = 6 → not compact at critical exponent
        let result = rk.check_compact_embedding(6.0);
        assert!(!result.is_compact);
    }

    #[test]
    fn test_compact_kp_equals_n() {
        // W¹²(ℝ²): kp = 2 = n = 2
        let rk = RellichKondrachov::new(SobolevSpace::h1(2), true);
        let result = rk.check_compact_embedding(100.0);
        assert!(result.is_compact);
    }

    #[test]
    fn test_compact_kp_greater_than_n() {
        // W¹²(ℝ¹): kp = 2 > n = 1
        let rk = RellichKondrachov::new(SobolevSpace::h1(1), true);
        let result = rk.check_compact_embedding(2.0);
        assert!(result.is_compact);
    }

    #[test]
    fn test_compact_into_sobolev() {
        let rk = RellichKondrachov::new(SobolevSpace::h2(2), true);
        // W²² ⊂⊂ W¹²: diff_source = 2 - 1 = 1, diff_target = 1 - 1 = 0, 1 > 0 ✓
        let result = rk.check_compact_into_sobolev(1, 2.0);
        assert!(result.is_compact);
    }

    #[test]
    fn test_compact_into_same_space() {
        let rk = RellichKondrachov::new(SobolevSpace::h1(2), true);
        let result = rk.check_compact_into_sobolev(1, 2.0);
        assert!(!result.is_compact); // Same space, not a proper embedding
    }

    #[test]
    fn test_compactness_modulus() {
        let fns: Vec<DVector<f64>> = (0..5)
            .map(|i| DVector::from_vec(vec![(i as f64 + 1.0) * 0.1; 10]))
            .collect();
        let rk = RellichKondrachov::new(SobolevSpace::h1(1), true);
        let modulus = rk.compactness_modulus(&fns, &[0.1]);
        assert!(modulus > 0.0);
    }

    #[test]
    fn test_compactness_modulus_identical() {
        let fns: Vec<DVector<f64>> = (0..5)
            .map(|_| DVector::from_vec(vec![1.0; 10]))
            .collect();
        let rk = RellichKondrachov::new(SobolevSpace::h1(1), true);
        let modulus = rk.compactness_modulus(&fns, &[0.1]);
        assert!((modulus).abs() < 1e-10);
    }

    #[test]
    fn test_compactness_modulus_empty() {
        let rk = RellichKondrachov::new(SobolevSpace::h1(1), true);
        let modulus = rk.compactness_modulus(&[], &[0.1]);
        assert!((modulus - 0.0).abs() < 1e-10);
    }
}
