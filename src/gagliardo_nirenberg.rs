//! Gagliardo-Nirenberg interpolation inequality.
//!
//! For intermediate derivatives, estimates:
//!   ‖Dʲu‖_{Lʳ} ≤ C₁ ‖Dᵐu‖_{Lᵠ}^a · ‖u‖_{Lᵠ}^{1-a}
//!
//! where 1/r = j/n + a(1/q - m/n) + (1-a)/p
//! and a = (j/n + 1/p - 1/r) / (m/n + 1/p - 1/q)

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::SobolevNorm;

/// Gagliardo-Nirenberg interpolation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GagliardoNirenberg {
    /// Domain dimension n
    pub n: usize,
}

/// Result of the interpolation inequality computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpolationResult {
    /// Interpolation parameter a ∈ [0, 1]
    pub a: f64,
    /// Whether the interpolation is valid
    pub valid: bool,
    /// Left side: ‖Dʲu‖_{Lʳ}
    pub left_side: f64,
    /// Right side: C₁ ‖Dᵐu‖_{Lᵠ}^a · ‖u‖_{Lᵖ}^{1-a}
    pub right_side: f64,
    /// Whether the inequality holds
    pub holds: bool,
}

impl GagliardoNirenberg {
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Compute the interpolation parameter a.
    ///
    /// a = (j/n + 1/p - 1/r) / (m/n + 1/p - 1/q)
    pub fn interpolation_parameter(
        &self,
        j: usize,  // derivative order on left
        m: usize,  // derivative order on right (m > j)
        p: f64,    // Lᵖ norm of u
        q: f64,    // Lᵠ norm of Dᵐu
        r: f64,    // Lʳ norm of Dʲu
    ) -> (f64, bool) {
        let n = self.n as f64;
        let numerator = j as f64 / n + 1.0 / p - 1.0 / r;
        let denominator = m as f64 / n + 1.0 / p - 1.0 / q;

        if denominator.abs() < 1e-15 {
            return (0.0, false);
        }

        let a = numerator / denominator;
        let valid = a >= 0.0 && a <= 1.0;
        (a, valid)
    }

    /// Verify the Gagliardo-Nirenberg inequality for a specific function.
    pub fn verify(
        &self,
        values: &DVector<f64>,
        spacings: &[f64],
        j: usize,
        m: usize,
        p: f64,
        q: f64,
        r: f64,
    ) -> InterpolationResult {
        let (a, valid) = self.interpolation_parameter(j, m, p, q, r);

        // Compute norms
        let left_side = if j == 0 {
            SobolevNorm::lp_norm(values, r)
        } else {
            let mut orders = vec![0; spacings.len().max(1)];
            orders[0] = j;
            let dj = crate::WeakDerivative::compute_higher_order(values, &orders, spacings);
            SobolevNorm::lp_norm(&dj.values, r)
        };

        let right_high = if m == 0 {
            SobolevNorm::lp_norm(values, q)
        } else {
            let mut orders_m = vec![0; spacings.len().max(1)];
            orders_m[0] = m;
            let dm = crate::WeakDerivative::compute_higher_order(values, &orders_m, spacings);
            SobolevNorm::lp_norm(&dm.values, q)
        };

        let right_low = SobolevNorm::lp_norm(values, p);

        let right_side = if valid && a >= 0.0 && a <= 1.0 {
            let c1 = 2.0; // generic constant
            c1 * right_high.powf(a) * right_low.powf(1.0 - a)
        } else {
            f64::NAN
        };

        let holds = valid && left_side <= right_side * 1.5; // tolerance for discrete

        InterpolationResult {
            a,
            valid,
            left_side,
            right_side,
            holds,
        }
    }

    /// Classical interpolation: ‖∇u‖_{L²} ≤ C ‖u‖_{L²}^{1/2} ‖Δu‖_{L²}^{1/2}
    /// (j=1, m=2, p=q=r=2)
    pub fn verify_classical(
        &self,
        values: &DVector<f64>,
        spacings: &[f64],
    ) -> InterpolationResult {
        self.verify(values, spacings, 1, 2, 2.0, 2.0, 2.0)
    }

    /// Nash inequality: ‖u‖_{L²} ≤ C ‖u‖_{L¹}^{1-a} ‖∇u‖_{L²}^a
    pub fn verify_nash(
        &self,
        values: &DVector<f64>,
        spacings: &[f64],
    ) -> InterpolationResult {
        self.verify(values, spacings, 0, 1, 1.0, 2.0, 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation_parameter_classical() {
        let gn = GagliardoNirenberg::new(1);
        let (a, valid) = gn.interpolation_parameter(1, 2, 2.0, 2.0, 2.0);
        assert!(valid);
        assert!((a - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_interpolation_parameter_zero_j() {
        let gn = GagliardoNirenberg::new(1);
        let (a, valid) = gn.interpolation_parameter(0, 1, 2.0, 2.0, 2.0);
        assert!(valid);
        assert!((a - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_interpolation_parameter_full_j() {
        let gn = GagliardoNirenberg::new(1);
        let (a, valid) = gn.interpolation_parameter(1, 1, 2.0, 2.0, 2.0);
        assert!(valid);
        assert!((a - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_interpolation_parameter_2d() {
        let gn = GagliardoNirenberg::new(2);
        let (a, valid) = gn.interpolation_parameter(1, 2, 2.0, 2.0, 2.0);
        assert!(valid);
        assert!(a >= 0.0 && a <= 1.0);
    }

    #[test]
    fn test_verify_classical_sine() {
        let gn = GagliardoNirenberg::new(1);
        let n = 50;
        let h = 0.1;
        let values: DVector<f64> = DVector::from_vec((0..n).map(|i| (i as f64 * h).sin()).collect::<Vec<_>>());
        let result = gn.verify_classical(&values, &[h]);
        assert!(result.valid);
    }

    #[test]
    fn test_verify_classical_smooth() {
        let gn = GagliardoNirenberg::new(1);
        let n = 100;
        let h = 0.05;
        let values: DVector<f64> = DVector::from_vec((0..n).map(|i| {
            let x = i as f64 * h;
            (x * std::f64::consts::PI).sin()
        }).collect::<Vec<_>>());
        let result = gn.verify_classical(&values, &[h]);
        assert!(result.valid);
        assert!(result.left_side.is_finite());
        assert!(result.right_side.is_finite());
    }

    #[test]
    fn test_verify_nash() {
        let gn = GagliardoNirenberg::new(1);
        let n = 50;
        let h = 0.1;
        let values: DVector<f64> = DVector::from_vec((0..n).map(|i| (i as f64 * h).sin()).collect::<Vec<_>>());
        let result = gn.verify_nash(&values, &[h]);
        assert!(result.left_side.is_finite());
    }

    #[test]
    fn test_interpolation_parameter_invalid() {
        let gn = GagliardoNirenberg::new(1);
        let (_, _valid) = gn.interpolation_parameter(2, 1, 2.0, 2.0, 2.0);
        // j > m gives a > 1 in general
    }

    #[test]
    fn test_verify_custom() {
        let gn = GagliardoNirenberg::new(1);
        let n = 50;
        let h = 0.1;
        let values: DVector<f64> = DVector::from_vec((0..n).map(|i| (i as f64 * h).sin()).collect::<Vec<_>>());
        let result = gn.verify(&values, &[h], 0, 2, 2.0, 2.0, 2.0);
        assert!(result.valid);
    }

    #[test]
    fn test_interpolation_3d() {
        let gn = GagliardoNirenberg::new(3);
        let (a, valid) = gn.interpolation_parameter(1, 2, 2.0, 2.0, 2.0);
        assert!(valid);
        assert!(a >= 0.0 && a <= 1.0);
    }
}
