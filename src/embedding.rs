//! Sobolev embedding theorem: Wᵏᵖ ⊂ Cᵐ when k > n/p + m.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::SobolevSpace;

/// Result of a Sobolev embedding check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    /// Whether the embedding holds
    pub holds: bool,
    /// Source space
    pub source: SobolevSpace,
    /// Target smoothness m (continuous derivatives)
    pub target_m: usize,
    /// The critical exponent: k - n/p
    pub critical_exponent: f64,
    /// Embedding type
    pub embedding_type: EmbeddingType,
}

/// Types of Sobolev embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EmbeddingType {
    /// Wᵏᵖ ⊂ Cᵐ: functions are m-times continuously differentiable
    Continuous,
    /// Wᵏᵖ ⊂ Lᵍ: embedding into a higher integrability space
    Integrability,
    /// Compact embedding (Rellich-Kondrachov)
    Compact,
}

/// Sobolev embedding calculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SobolevEmbedding {
    pub source: SobolevSpace,
}

impl SobolevEmbedding {
    pub fn new(source: SobolevSpace) -> Self {
        Self { source }
    }

    /// Check if Wᵏᵖ ⊂ Cᵐ (Morrey/ Sobolev embedding).
    ///
    /// The embedding holds when k - n/p > m (strict inequality),
    /// or k - n/p ≥ m with appropriate modifications.
    pub fn check_continuous_embedding(&self, target_m: usize) -> EmbeddingResult {
        let critical = self.source.k as f64 - self.source.n_dims as f64 / self.source.p;
        let holds = critical > target_m as f64;

        EmbeddingResult {
            holds,
            source: self.source.clone(),
            target_m,
            critical_exponent: critical,
            embedding_type: EmbeddingType::Continuous,
        }
    }

    /// Compute the optimal Sobolev conjugate exponent q.
    /// For W¹ᵖ(ℝⁿ) with p < n: 1/p* = 1/p - 1/n
    pub fn conjugate_exponent(&self) -> Option<f64> {
        let n = self.source.n_dims as f64;
        let p = self.source.p;

        if p < n {
            let q = 1.0 / (1.0 / p - 1.0 / n);
            Some(q)
        } else {
            None
        }
    }

    /// Check integrability embedding Wᵏᵖ ⊂ Lᵍ.
    pub fn check_integrability_embedding(&self, target_q: f64) -> EmbeddingResult {
        let p = self.source.p;
        let k = self.source.k as f64;
        let n = self.source.n_dims as f64;

        // Optimal exponent: 1/q = 1/p - k/n (when k < n/p)
        let optimal_q = if p * k as f64 > n as f64 * p {
            f64::INFINITY
        } else {
            1.0 / (1.0 / p - k / n)
        };

        let holds = target_q <= optimal_q + 1e-10 || optimal_q.is_infinite();

        EmbeddingResult {
            holds,
            source: self.source.clone(),
            target_m: 0,
            critical_exponent: k - n / p,
            embedding_type: EmbeddingType::Integrability,
        }
    }

    /// Maximum continuity order m such that Wᵏᵖ ⊂ Cᵐ.
    pub fn max_continuity_order(&self) -> usize {
        let critical = self.source.k as f64 - self.source.n_dims as f64 / self.source.p;
        if critical > 0.0 {
            critical.floor() as usize
        } else {
            0
        }
    }

    /// Check embedding for a specific function by computing norms.
    pub fn verify_embedding(
        &self,
        values: &DVector<f64>,
        spacings: &[f64],
        target_m: usize,
    ) -> bool {
        let result = self.check_continuous_embedding(target_m);
        if !result.holds {
            return false;
        }
        // If the embedding theorem says it holds, verify the function is in Wᵏᵖ
        let membership = self.source.membership(values, spacings);
        membership.is_member
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_h1_1d_into_c0() {
        // W¹²(ℝ¹): k=1, n=1, p=2 → critical = 1 - 1/2 = 0.5 > 0 ✓
        let emb = SobolevEmbedding::new(SobolevSpace::h1(1));
        let result = emb.check_continuous_embedding(0);
        assert!(result.holds);
        assert_eq!(emb.max_continuity_order(), 0);
    }

    #[test]
    fn test_embedding_h1_1d_not_into_c1() {
        // W¹²(ℝ¹): critical = 0.5, NOT > 1
        let emb = SobolevEmbedding::new(SobolevSpace::h1(1));
        let result = emb.check_continuous_embedding(1);
        assert!(!result.holds);
    }

    #[test]
    fn test_embedding_h2_1d_into_c0() {
        // W²²(ℝ¹): critical = 2 - 0.5 = 1.5 > 0 ✓
        let emb = SobolevEmbedding::new(SobolevSpace::h2(1));
        let result = emb.check_continuous_embedding(0);
        assert!(result.holds);
    }

    #[test]
    fn test_embedding_h2_1d_into_c1() {
        // W²²(ℝ¹): critical = 1.5 > 1 ✓
        let emb = SobolevEmbedding::new(SobolevSpace::h2(1));
        let result = emb.check_continuous_embedding(1);
        assert!(result.holds);
    }

    #[test]
    fn test_embedding_h1_3d_into_c0() {
        // W¹²(ℝ³): critical = 1 - 3/2 = -0.5 < 0 ✗
        let emb = SobolevEmbedding::new(SobolevSpace::h1(3));
        let result = emb.check_continuous_embedding(0);
        assert!(!result.holds);
    }

    #[test]
    fn test_embedding_h2_3d_into_c0() {
        // W²²(ℝ³): critical = 2 - 3/2 = 0.5 > 0 ✓
        let emb = SobolevEmbedding::new(SobolevSpace::h2(3));
        let result = emb.check_continuous_embedding(0);
        assert!(result.holds);
    }

    #[test]
    fn test_conjugate_exponent_1d() {
        // W¹²(ℝ¹): p=2 < n=1? No. p > n.
        let emb = SobolevEmbedding::new(SobolevSpace::h1(1));
        assert!(emb.conjugate_exponent().is_none()); // p=2 > n=1
    }

    #[test]
    fn test_conjugate_exponent_3d() {
        // W¹²(ℝ³): p=2 < n=3 ✓ → q = 1/(1/2 - 1/3) = 6
        let emb = SobolevEmbedding::new(SobolevSpace::h1(3));
        let q = emb.conjugate_exponent().unwrap();
        assert!((q - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_max_continuity_h2_1d() {
        let emb = SobolevEmbedding::new(SobolevSpace::h2(1));
        assert_eq!(emb.max_continuity_order(), 1);
    }

    #[test]
    fn test_max_continuity_h1_3d() {
        // critical = 1 - 3/2 = -0.5 < 0
        let emb = SobolevEmbedding::new(SobolevSpace::h1(3));
        assert_eq!(emb.max_continuity_order(), 0);
    }

    #[test]
    fn test_verify_embedding_smooth_function() {
        let emb = SobolevEmbedding::new(SobolevSpace::h1(1));
        let n = 50;
        let h = 0.1;
        let u: DVector<f64> = DVector::from_vec((0..n).map(|i| (i as f64 * h).sin()).collect::<Vec<_>>());
        assert!(emb.verify_embedding(&u, &[h], 0));
    }

    #[test]
    fn test_integrability_embedding() {
        let emb = SobolevEmbedding::new(SobolevSpace::h1(3));
        let result = emb.check_integrability_embedding(6.0);
        assert!(result.holds);
    }

    #[test]
    fn test_embedding_high_order() {
        // W⁵²(ℝ²): critical = 5 - 2/2 = 4 > 3 ✓
        let w = SobolevSpace::new(5, 2.0, 2);
        let emb = SobolevEmbedding::new(w);
        let result = emb.check_continuous_embedding(3);
        assert!(result.holds);
        assert_eq!(emb.max_continuity_order(), 4);
    }
}
