//! Sobolev space Wᵏᵖ definitions and operations.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::{SobolevNorm, WeakDerivative};

/// Sobolev space parameters Wᵏᵖ(Ω).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SobolevSpace {
    /// Differentiability order k
    pub k: usize,
    /// Integrability exponent p (1 ≤ p < ∞)
    pub p: f64,
    /// Domain dimension n
    pub n_dims: usize,
}

impl SobolevSpace {
    /// Create a new Sobolev space Wᵏᵖ(Ω) where Ω ⊂ ℝⁿ.
    pub fn new(k: usize, p: f64, n_dims: usize) -> Self {
        assert!(p >= 1.0, "p must be ≥ 1");
        Self { k, p, n_dims }
    }

    /// The standard L² Sobolev space Hᵏ = Wᵏ².
    pub fn hilbert(k: usize, n_dims: usize) -> Self {
        Self::new(k, 2.0, n_dims)
    }

    /// H¹ = W¹² — the most common Sobolev space.
    pub fn h1(n_dims: usize) -> Self {
        Self::hilbert(1, n_dims)
    }

    /// H² = W²² — used for fourth-order PDEs.
    pub fn h2(n_dims: usize) -> Self {
        Self::hilbert(2, n_dims)
    }

    /// Check if a function (given as grid values) belongs to this Sobolev space.
    /// Returns the Sobolev norm if it's finite (i.e., function is in Wᵏᵖ).
    pub fn membership(&self, values: &DVector<f64>, spacings: &[f64]) -> MembershipResult {
        let norm_calculator = SobolevNorm::new(self.k, self.p);
        let norm = norm_calculator.compute(values, spacings);

        MembershipResult {
            is_member: norm.is_finite(),
            sobolev_norm: norm,
            space: self.clone(),
        }
    }

    /// Compute all weak derivatives up to order k.
    pub fn all_derivatives(&self, values: &DVector<f64>, spacings: &[f64]) -> Vec<WeakDerivative> {
        let mut derivs = Vec::new();

        if self.n_dims == 1 {
            for total_order in 1..=self.k {
                let mut orders = vec![0; 1];
                orders[0] = total_order;
                let wd = WeakDerivative::compute_higher_order(values, &orders, spacings);
                derivs.push(wd);
            }
        } else {
            // For multi-dim, compute partial derivatives for each dimension up to order k
            for dim in 0..self.n_dims {
                for order in 1..=self.k {
                    let mut orders = vec![0; self.n_dims];
                    orders[dim] = order;
                    let wd = WeakDerivative::compute_higher_order(values, &orders, spacings);
                    derivs.push(wd);
                }
            }
        }
        derivs
    }

    /// Compare smoothness of two functions in this space.
    pub fn compare_smoothness(
        &self,
        f1: &DVector<f64>,
        f2: &DVector<f64>,
        spacings: &[f64],
    ) -> SmoothnessComparison {
        let norm_calc = SobolevNorm::new(self.k, self.p);
        let norm1 = norm_calc.compute(f1, spacings);
        let norm2 = norm_calc.compute(f2, spacings);

        SmoothnessComparison {
            smoother: if norm1 <= norm2 { 0 } else { 1 },
            norm_1: norm1,
            norm_2: norm2,
            ratio: if norm2 > 1e-15 { norm1 / norm2 } else { f64::INFINITY },
        }
    }
}

/// Result of checking Sobolev space membership.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembershipResult {
    pub is_member: bool,
    pub sobolev_norm: f64,
    pub space: SobolevSpace,
}

/// Comparison of smoothness between two functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothnessComparison {
    /// Index of the smoother function (0 or 1)
    pub smoother: usize,
    pub norm_1: f64,
    pub norm_2: f64,
    /// Ratio norm_1 / norm_2
    pub ratio: f64,
}

/// Direct sum of Sobolev spaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SobolevSpaceSum {
    pub spaces: Vec<SobolevSpace>,
}

impl SobolevSpaceSum {
    pub fn new(spaces: Vec<SobolevSpace>) -> Self {
        Self { spaces }
    }

    /// Compute the sum norm (max of individual norms).
    pub fn sum_norm(&self, values: &DVector<f64>, spacings: &[f64]) -> f64 {
        self.spaces.iter()
            .map(|s| {
                let calc = SobolevNorm::new(s.k, s.p);
                calc.compute(values, spacings)
            })
            .fold(0.0_f64, f64::max)
    }
}

/// Intersection of Sobolev spaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SobolevSpaceIntersection {
    pub spaces: Vec<SobolevSpace>,
}

impl SobolevSpaceIntersection {
    pub fn new(spaces: Vec<SobolevSpace>) -> Self {
        Self { spaces }
    }

    /// A function is in the intersection iff it's in every space.
    pub fn is_in_intersection(&self, values: &DVector<f64>, spacings: &[f64]) -> bool {
        self.spaces.iter().all(|s| {
            let calc = SobolevNorm::new(s.k, s.p);
            calc.compute(values, spacings).is_finite()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobolev_space_creation() {
        let w = SobolevSpace::new(1, 2.0, 1);
        assert_eq!(w.k, 1);
        assert_eq!(w.p, 2.0);
        assert_eq!(w.n_dims, 1);
    }

    #[test]
    fn test_hilbert_space() {
        let h = SobolevSpace::hilbert(2, 3);
        assert_eq!(h.k, 2);
        assert_eq!(h.p, 2.0);
        assert_eq!(h.n_dims, 3);
    }

    #[test]
    fn test_h1_space() {
        let h = SobolevSpace::h1(1);
        assert_eq!(h.k, 1);
        assert_eq!(h.p, 2.0);
    }

    #[test]
    fn test_h2_space() {
        let h = SobolevSpace::h2(2);
        assert_eq!(h.k, 2);
        assert_eq!(h.p, 2.0);
    }

    #[test]
    fn test_membership_smooth() {
        let w = SobolevSpace::h1(1);
        let n = 50;
        let h = 0.1;
        let u: DVector<f64> = DVector::from_vec((0..n).map(|i| (i as f64 * h).sin()).collect::<Vec<_>>());
        let result = w.membership(&u, &[h]);
        assert!(result.is_member);
        assert!(result.sobolev_norm.is_finite());
    }

    #[test]
    fn test_membership_constant() {
        let w = SobolevSpace::h1(1);
        let u = DVector::from_vec(vec![1.0; 20]);
        let result = w.membership(&u, &[0.1]);
        assert!(result.is_member);
    }

    #[test]
    fn test_compare_smoothness() {
        let w = SobolevSpace::h1(1);
        let n = 50;
        let h = 0.1;
        // Smooth function
        let smooth: DVector<f64> = DVector::from_vec((0..n).map(|i| (i as f64 * h).sin()).collect::<Vec<_>>());
        // Less smooth (noise)
        let rough: DVector<f64> = DVector::from_vec((0..n).map(|i| {
            let x = i as f64 * h;
            x.sin() + 0.5 * (x * 20.0).sin()
        }).collect::<Vec<_>>());

        let cmp = w.compare_smoothness(&smooth, &rough, &[h]);
        assert_eq!(cmp.smoother, 0); // smooth has lower norm
        assert!(cmp.norm_1 < cmp.norm_2);
    }

    #[test]
    fn test_all_derivatives_1d() {
        let w = SobolevSpace::new(2, 2.0, 1);
        let u: DVector<f64> = DVector::from_vec((0..20).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());
        let derivs = w.all_derivatives(&u, &[0.1]);
        assert_eq!(derivs.len(), 2); // 1st and 2nd order
        assert_eq!(derivs[0].total_order(), 1);
        assert_eq!(derivs[1].total_order(), 2);
    }

    #[test]
    fn test_all_derivatives_2d() {
        let w = SobolevSpace::new(1, 2.0, 2);
        let u = DVector::from_vec(vec![1.0; 10]);
        let derivs = w.all_derivatives(&u, &[0.1, 0.1]);
        assert_eq!(derivs.len(), 2); // one per dimension
    }

    #[test]
    fn test_space_sum() {
        let s = SobolevSpaceSum::new(vec![
            SobolevSpace::h1(1),
            SobolevSpace::h2(1),
        ]);
        let u = DVector::from_vec(vec![1.0; 20]);
        let norm = s.sum_norm(&u, &[0.1]);
        assert!(norm.is_finite());
    }

    #[test]
    fn test_space_intersection() {
        let inter = SobolevSpaceIntersection::new(vec![
            SobolevSpace::h1(1),
            SobolevSpace::h2(1),
        ]);
        let u: DVector<f64> = DVector::from_vec((0..20).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());
        assert!(inter.is_in_intersection(&u, &[0.1]));
    }

    #[test]
    #[should_panic]
    fn test_invalid_p() {
        SobolevSpace::new(1, 0.5, 1);
    }
}
