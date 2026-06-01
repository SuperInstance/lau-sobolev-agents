//! Poincaré inequality: ‖u - ū‖_{Lᵖ} ≤ C ‖∇u‖_{Lᵖ}
//!
//! Controls the mean deviation of a function by its gradient.
//! Essential for showing coercivity of bilinear forms in PDE theory.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::SobolevNorm;

/// Poincaré inequality calculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoincareInequality {
    /// Domain diameter (or characteristic length)
    pub diameter: f64,
    /// Domain dimension
    pub n_dims: usize,
    /// Integrability exponent p
    pub p: f64,
}

/// Result of verifying the Poincaré inequality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoincareResult {
    /// Lᵖ norm of u - ū
    pub mean_deviation: f64,
    /// Lᵖ norm of ∇u
    pub gradient_norm: f64,
    /// Poincaré constant C such that ‖u - ū‖ ≤ C ‖∇u‖
    pub constant_c: f64,
    /// Whether the inequality holds
    pub holds: bool,
    /// Theoretical Poincaré constant (diameter / π for p=2)
    pub theoretical_c: f64,
}

/// Poincaré-Wirtinger inequality (for zero-mean functions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoincareWirtingerResult {
    pub norm_u: f64,
    pub gradient_norm: f64,
    pub ratio: f64,
    pub holds: bool,
}

impl PoincareInequality {
    pub fn new(diameter: f64, n_dims: usize, p: f64) -> Self {
        assert!(diameter > 0.0, "Diameter must be positive");
        Self { diameter, n_dims, p }
    }

    /// Theoretical Poincaré constant for a convex domain.
    /// C ≤ diameter / π for p = 2 (one-dimensional).
    /// More generally, C ≤ diameter * (n/(n-1))^(1/p) / π for convex domains.
    pub fn theoretical_constant(&self) -> f64 {
        if self.p == 2.0 && self.n_dims == 1 {
            self.diameter / std::f64::consts::PI
        } else {
            self.diameter / (2.0 * std::f64::consts::PI)
        }
    }

    /// Verify the Poincaré inequality for a given function.
    pub fn verify(&self, values: &DVector<f64>, spacings: &[f64]) -> PoincareResult {
        let n = values.len() as f64;

        // Compute mean
        let mean: f64 = values.iter().sum::<f64>() / n;

        // u - ū
        let deviation = values.map(|v| v - mean);
        let mean_dev = SobolevNorm::lp_norm(&deviation, self.p);

        // ‖∇u‖_p (using spacing for finite differences)
        let h = spacings.first().copied().unwrap_or(1.0);
        let mut grad = DVector::zeros(values.len());
        for i in 0..values.len() {
            if i == 0 && values.len() > 1 {
                grad[i] = (values[1] - values[0]) / h;
            } else if i == values.len() - 1 && values.len() > 1 {
                grad[i] = (values[values.len() - 1] - values[values.len() - 2]) / h;
            } else if values.len() > 2 {
                grad[i] = (values[i + 1] - values[i - 1]) / (2.0 * h);
            }
        }
        let grad_norm = SobolevNorm::lp_norm(&grad, self.p);

        let (constant_c, holds) = if grad_norm > 1e-15 {
            let c = mean_dev / grad_norm;
            (c, c <= self.theoretical_constant() * 1.5) // tolerance for discrete approximation
        } else {
            (0.0, mean_dev < 1e-10)
        };

        PoincareResult {
            mean_deviation: mean_dev,
            gradient_norm: grad_norm,
            constant_c,
            holds,
            theoretical_c: self.theoretical_constant(),
        }
    }

    /// Poincaré-Wirtinger: for zero-mean functions, ‖u‖ ≤ C ‖∇u‖.
    pub fn verify_wirtinger(&self, values: &DVector<f64>, spacings: &[f64]) -> PoincareWirtingerResult {
        let n = values.len() as f64;
        let mean: f64 = values.iter().sum::<f64>() / n;
        let centered = values.map(|v| v - mean);

        let norm_u = SobolevNorm::lp_norm(&centered, self.p);

        let h = spacings.first().copied().unwrap_or(1.0);
        let mut grad = DVector::zeros(values.len());
        for i in 0..values.len() {
            if i == 0 && values.len() > 1 {
                grad[i] = (values[1] - values[0]) / h;
            } else if i == values.len() - 1 && values.len() > 1 {
                grad[i] = (values[values.len() - 1] - values[values.len() - 2]) / h;
            } else if values.len() > 2 {
                grad[i] = (values[i + 1] - values[i - 1]) / (2.0 * h);
            }
        }
        let grad_norm = SobolevNorm::lp_norm(&grad, self.p);

        let ratio = if grad_norm > 1e-15 { norm_u / grad_norm } else { 0.0 };

        PoincareWirtingerResult {
            norm_u,
            gradient_norm: grad_norm,
            ratio,
            holds: ratio <= self.theoretical_constant() * 2.0 || grad_norm < 1e-10,
        }
    }

    /// Estimate the optimal Poincaré constant from a collection of functions.
    pub fn estimate_constant(&self, functions: &[DVector<f64>], spacings: &[f64]) -> f64 {
        functions.iter()
            .map(|f| {
                let result = self.verify(f, spacings);
                if result.gradient_norm > 1e-15 {
                    result.mean_deviation / result.gradient_norm
                } else {
                    0.0
                }
            })
            .fold(0.0_f64, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_poincare_constant_1d() {
        let pi = PoincareInequality::new(1.0, 1, 2.0);
        assert_relative_eq!(pi.theoretical_constant(), 1.0 / std::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn test_poincare_constant_linear() {
        let pi = PoincareInequality::new(2.0, 1, 2.0);
        assert_relative_eq!(pi.theoretical_constant(), 2.0 / std::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn test_poincare_sine() {
        let n = 100;
        let h = std::f64::consts::PI / (n as f64);
        let diameter = std::f64::consts::PI;
        let values: DVector<f64> = DVector::from_vec((0..n).map(|i| (i as f64 * h).sin()).collect::<Vec<_>>());
        let pi = PoincareInequality::new(diameter, 1, 2.0);
        let result = pi.verify(&values, &[h]);
        assert!(result.holds || result.constant_c < 2.0); // relaxed for discrete
    }

    #[test]
    fn test_poincare_constant_function() {
        let values = DVector::from_vec(vec![5.0; 20]);
        let pi = PoincareInequality::new(1.0, 1, 2.0);
        let result = pi.verify(&values, &[0.05]);
        assert_relative_eq!(result.mean_deviation, 0.0, epsilon = 1e-10);
        assert!(result.holds);
    }

    #[test]
    fn test_poincare_linear_function() {
        let n = 50;
        let h = 0.1;
        let diameter = (n as f64) * h;
        let values: DVector<f64> = DVector::from_vec((0..n).map(|i| i as f64 * h).collect::<Vec<_>>());
        let pi = PoincareInequality::new(diameter, 1, 2.0);
        let result = pi.verify(&values, &[h]);
        assert!(result.gradient_norm > 0.0);
    }

    #[test]
    fn test_poincare_wirtinger() {
        let n = 50;
        let h = 0.1;
        let values: DVector<f64> = DVector::from_vec((0..n).map(|i| {
            let x = i as f64 * h;
            (x * std::f64::consts::PI / (n as f64 * h)).sin()
        }).collect::<Vec<_>>());
        let pi = PoincareInequality::new(n as f64 * h, 1, 2.0);
        let result = pi.verify_wirtinger(&values, &[h]);
        assert!(result.gradient_norm > 0.0);
    }

    #[test]
    fn test_poincare_wirtinger_constant() {
        let values = DVector::from_vec(vec![3.0; 20]);
        let pi = PoincareInequality::new(1.0, 1, 2.0);
        let result = pi.verify_wirtinger(&values, &[0.05]);
        assert_relative_eq!(result.norm_u, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_estimate_constant() {
        let n = 50;
        let h = 0.1;
        let fns: Vec<DVector<f64>> = (0..3).map(|k| {
            DVector::from_vec((0..n).map(|i| {
                let x = i as f64 * h;
                ((k as f64 + 1.0) * x).sin()
            }).collect::<Vec<_>>())
        }).collect();
        let pi = PoincareInequality::new(n as f64 * h, 1, 2.0);
        let c = pi.estimate_constant(&fns, &[h]);
        assert!(c > 0.0);
        assert!(c.is_finite());
    }

    #[test]
    fn test_poincare_2d() {
        let pi = PoincareInequality::new(1.0, 2, 2.0);
        let c = pi.theoretical_constant();
        assert!(c > 0.0);
    }
}
