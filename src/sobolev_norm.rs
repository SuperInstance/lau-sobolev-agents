//! Sobolev norm computation: ‖u‖_{Wᵏᵖ} = (Σ|α|≤k ‖Dα u‖_p^p)^{1/p}

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::WeakDerivative;

/// Calculator for Sobolev norms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SobolevNorm {
    /// Differentiability order k
    pub k: usize,
    /// Integrability exponent p
    pub p: f64,
}

impl SobolevNorm {
    pub fn new(k: usize, p: f64) -> Self {
        assert!(p >= 1.0, "p must be ≥ 1");
        Self { k, p }
    }

    /// Compute the Lᵖ norm of a vector.
    /// ‖u‖_p = (Σ|uᵢ|ᵖ)^{1/p}
    pub fn lp_norm(u: &DVector<f64>, p: f64) -> f64 {
        if p.is_infinite() {
            u.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
        } else {
            let sum: f64 = u.iter().map(|v| v.abs().powf(p)).sum();
            let n = u.len() as f64;
            // Normalize by number of elements for grid functions
            (sum / n).powf(1.0 / p)
        }
    }

    /// Compute the L² norm.
    pub fn l2_norm(u: &DVector<f64>) -> f64 {
        Self::lp_norm(u, 2.0)
    }

    /// Compute the Sobolev norm ‖u‖_{Wᵏᵖ}.
    pub fn compute(&self, values: &DVector<f64>, spacings: &[f64]) -> f64 {
        let p = self.p;
        let _h = spacings.first().copied().unwrap_or(1.0);

        // Sum ‖Dα u‖_p^p for all |α| ≤ k
        let mut sum = 0.0;

        // |α| = 0: just the Lᵖ norm of u itself
        let u_norm = Self::lp_norm(values, p);
        sum += u_norm.powf(p);

        // Higher-order derivatives
        for order in 1..=self.k {
            let mut orders = vec![0; spacings.len().max(1)];
            orders[0] = order;
            let wd = WeakDerivative::compute_higher_order(values, &orders, spacings);
            let d_norm = Self::lp_norm(&wd.values, p);
            sum += d_norm.powf(p);

            // For multi-dim, also compute mixed derivatives
            if spacings.len() > 1 {
                for dim in 1..spacings.len() {
                    let mut mixed_orders = vec![0; spacings.len()];
                    mixed_orders[dim] = order;
                    let wd_mixed = WeakDerivative::compute_higher_order(values, &mixed_orders, spacings);
                    let d_mixed_norm = Self::lp_norm(&wd_mixed.values, p);
                    sum += d_mixed_norm.powf(p);
                }
            }
        }

        sum.powf(1.0 / p)
    }

    /// Compute the Sobolev semi-norm (only derivatives, not the function itself).
    /// |u|_{Wᵏᵖ} = (Σ|α|=k ‖Dα u‖_p^p)^{1/p}
    pub fn semi_norm(&self, values: &DVector<f64>, spacings: &[f64]) -> f64 {
        let p = self.p;
        let mut sum = 0.0;

        let mut orders = vec![0; spacings.len().max(1)];
        orders[0] = self.k;
        let wd = WeakDerivative::compute_higher_order(values, &orders, spacings);
        let d_norm = Self::lp_norm(&wd.values, p);
        sum += d_norm.powf(p);

        if spacings.len() > 1 {
            for dim in 1..spacings.len() {
                let mut mixed_orders = vec![0; spacings.len()];
                mixed_orders[dim] = self.k;
                let wd_mixed = WeakDerivative::compute_higher_order(values, &mixed_orders, spacings);
                let d_mixed_norm = Self::lp_norm(&wd_mixed.values, p);
                sum += d_mixed_norm.powf(p);
            }
        }

        sum.powf(1.0 / p)
    }

    /// Compute the H¹ norm (W¹²): ‖u‖_{H¹}² = ‖u‖_{L²}² + ‖∇u‖_{L²}²
    pub fn h1_norm(values: &DVector<f64>, spacings: &[f64]) -> f64 {
        let calc = Self::new(1, 2.0);
        calc.compute(values, spacings)
    }

    /// Compute the H¹ semi-norm (just the gradient part).
    pub fn h1_semi_norm(values: &DVector<f64>, spacings: &[f64]) -> f64 {
        let calc = Self::new(1, 2.0);
        calc.semi_norm(values, spacings)
    }

    /// Check if function is in Wᵏᵖ by verifying the norm is finite.
    pub fn is_in_sobolev_space(&self, values: &DVector<f64>, spacings: &[f64]) -> bool {
        self.compute(values, spacings).is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lp_norm_l1() {
        let v = DVector::from_vec(vec![1.0, -2.0, 3.0]);
        let norm = SobolevNorm::lp_norm(&v, 1.0);
        assert_relative_eq!(norm, (1.0 + 2.0 + 3.0) / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lp_norm_l2() {
        let v = DVector::from_vec(vec![3.0, 4.0]);
        let norm = SobolevNorm::lp_norm(&v, 2.0);
        assert_relative_eq!(norm, (25.0_f64 / 2.0).sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_lp_norm_linf() {
        let v = DVector::from_vec(vec![1.0, -5.0, 3.0]);
        let norm = SobolevNorm::lp_norm(&v, f64::INFINITY);
        assert_relative_eq!(norm, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_l2_norm() {
        let v = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let norm = SobolevNorm::l2_norm(&v);
        assert_relative_eq!(norm, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_sobolev_norm_constant() {
        let u = DVector::from_vec(vec![5.0; 20]);
        let calc = SobolevNorm::new(1, 2.0);
        let norm = calc.compute(&u, &[0.1]);
        // Only the function value contributes; derivatives are zero
        assert!(norm > 0.0);
        assert!(norm.is_finite());
    }

    #[test]
    fn test_sobolev_norm_linear() {
        let n = 50;
        let h = 0.1;
        let u: DVector<f64> = DVector::from_vec((0..n).map(|i| i as f64 * h).collect::<Vec<_>>());
        let calc = SobolevNorm::new(1, 2.0);
        let norm = calc.compute(&u, &[h]);
        assert!(norm.is_finite());
        assert!(norm > 0.0);
    }

    #[test]
    fn test_sobolev_norm_higher_order() {
        let n = 30;
        let h = 0.1;
        let u: DVector<f64> = DVector::from_vec((0..n).map(|i| { let x = i as f64 * h; x * x * x }).collect::<Vec<_>>());
        let calc = SobolevNorm::new(2, 2.0);
        let norm = calc.compute(&u, &[h]);
        assert!(norm.is_finite());
    }

    #[test]
    fn test_semi_norm() {
        let u = DVector::from_vec(vec![5.0; 20]);
        let calc = SobolevNorm::new(1, 2.0);
        let semi = calc.semi_norm(&u, &[0.1]);
        assert_relative_eq!(semi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_h1_norm() {
        let n = 30;
        let h = 0.1;
        let u: DVector<f64> = DVector::from_vec((0..n).map(|i| (i as f64 * h).sin()).collect::<Vec<_>>());
        let norm = SobolevNorm::h1_norm(&u, &[h]);
        assert!(norm.is_finite());
        assert!(norm > 0.0);
    }

    #[test]
    fn test_h1_semi_norm_nonconstant() {
        let n = 30;
        let h = 0.1;
        let u: DVector<f64> = DVector::from_vec((0..n).map(|i| (i as f64 * h).sin()).collect::<Vec<_>>());
        let semi = SobolevNorm::h1_semi_norm(&u, &[h]);
        assert!(semi > 0.0);
    }

    #[test]
    fn test_is_in_sobolev_space() {
        let u = DVector::from_vec(vec![1.0; 20]);
        let calc = SobolevNorm::new(1, 2.0);
        assert!(calc.is_in_sobolev_space(&u, &[0.1]));
    }

    #[test]
    fn test_norm_increases_with_roughness() {
        let n = 100;
        let h = 0.05;
        let smooth: DVector<f64> = DVector::from_vec((0..n).map(|i| (i as f64 * h).sin()).collect::<Vec<_>>());
        let rough: DVector<f64> = DVector::from_vec((0..n).map(|i| {
            let x = i as f64 * h;
            x.sin() + 0.3 * (x * 50.0).sin()
        }).collect::<Vec<_>>());

        let calc = SobolevNorm::new(1, 2.0);
        let norm_smooth = calc.compute(&smooth, &[h]);
        let norm_rough = calc.compute(&rough, &[h]);
        assert!(norm_smooth < norm_rough);
    }

    #[test]
    fn test_norm_increases_with_order() {
        let n = 50;
        let h = 0.1;
        let u: DVector<f64> = DVector::from_vec((0..n).map(|i| { let x = i as f64 * h; x * x }).collect::<Vec<_>>());

        let calc1 = SobolevNorm::new(1, 2.0);
        let calc2 = SobolevNorm::new(2, 2.0);
        let norm1 = calc1.compute(&u, &[h]);
        let norm2 = calc2.compute(&u, &[h]);
        // W² norm ≥ W¹ norm
        assert!(norm2 >= norm1);
    }
}
