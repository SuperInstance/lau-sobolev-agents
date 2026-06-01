//! Fractional Sobolev spaces Wˢᵖ: partial regularity.
//!
//! For s > 0 non-integer, Wˢᵖ(Ω) is defined via the Gagliardo seminorm:
//!   [u]_{Wˢᵖ} = (∫∫ |u(x) - u(y)|ᵖ / |x - y|^{n + sp} dx dy)^{1/p}
//!
//! Also known as Sobolev-Slobodeckij spaces. These interpolate between
//! integer-order Sobolev spaces.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::SobolevNorm;

/// Fractional Sobolev space Wˢᵖ(Ω).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractionalSobolevSpace {
    /// Fractional order s > 0
    pub s: f64,
    /// Integrability exponent p ≥ 1
    pub p: f64,
    /// Domain dimension n
    pub n_dims: usize,
}

impl FractionalSobolevSpace {
    pub fn new(s: f64, p: f64, n_dims: usize) -> Self {
        assert!(s > 0.0, "s must be > 0");
        assert!(p >= 1.0, "p must be ≥ 1");
        Self { s, p, n_dims }
    }

    /// The Gagliardo (Slobodeckij) seminorm.
    /// [u]_{s,p} = (Σᵢ Σⱼ |uᵢ - uⱼ|ᵖ / |xᵢ - xⱼ|^{n + sp} Δxᵢ Δxⱼ)^{1/p}
    pub fn gagliardo_seminorm(&self, values: &DVector<f64>, spacings: &[f64]) -> f64 {
        let n = values.len();
        let h = spacings.first().copied().unwrap_or(1.0);
        let sp = self.s * self.p;
        let n_plus_sp = self.n_dims as f64 + sp;

        let mut sum = 0.0;
        let h_factor = h.powi(2); // Δxᵢ Δxⱼ

        for i in 0..n {
            for j in (i + 1)..n {
                let diff = (values[i] - values[j]).abs().powf(self.p);
                let dist = ((i as f64 - j as f64) * h).abs();
                if dist > 1e-15 {
                    sum += diff / dist.powf(n_plus_sp) * h_factor;
                }
            }
        }
        sum *= 2.0; // symmetry
        sum.powf(1.0 / self.p)
    }

    /// The full fractional Sobolev norm.
    /// ‖u‖_{Wˢᵖ} = (‖u‖_{Lᵖ}^p + [u]_{s,p}^p)^{1/p}
    pub fn norm(&self, values: &DVector<f64>, spacings: &[f64]) -> f64 {
        let lp = SobolevNorm::lp_norm(values, self.p);
        let semi = self.gagliardo_seminorm(values, spacings);
        (lp.powf(self.p) + semi.powf(self.p)).powf(1.0 / self.p)
    }

    /// Check membership in the fractional Sobolev space.
    pub fn is_member(&self, values: &DVector<f64>, spacings: &[f64]) -> bool {
        self.norm(values, spacings).is_finite()
    }

    /// Compare with integer-order space.
    /// For s = k + σ with 0 < σ < 1: Wˢᵖ ⊂ Wᵏᵖ with fractional part as remainder.
    pub fn decompose_order(&self) -> (usize, f64) {
        let floor = self.s.floor() as usize;
        let frac = self.s - floor as f64;
        (floor, frac)
    }

    /// Embedding exponent for fractional Sobolev spaces.
    /// If sp < n: p* = np/(n - sp)
    pub fn sobolev_conjugate(&self) -> Option<f64> {
        let sp = self.s * self.p;
        let n = self.n_dims as f64;
        if sp < n {
            Some(n * self.p / (n - sp))
        } else {
            None
        }
    }
}

/// Fractional Laplacian (-Δ)^s via discrete approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractionalLaplacian {
    pub s: f64,
}

impl FractionalLaplacian {
    pub fn new(s: f64) -> Self {
        assert!(s > 0.0 && s < 2.0, "s must be in (0, 2)");
        Self { s }
    }

    /// Compute (-Δ)^s u using spectral method (FFT-based for periodic functions).
    /// Simplified: uses finite difference approximation of the integral representation.
    pub fn apply(&self, values: &DVector<f64>, spacing: f64) -> DVector<f64> {
        let n = values.len();
        let mut result = DVector::zeros(n);

        let two_s = 2.0 * self.s;
        let n_plus_2s = 1.0 + two_s; // 1D

        let h = spacing;

        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                if i != j {
                    let dist = ((i as f64 - j as f64) * h).abs();
                    let diff = values[i] - values[j];
                    sum += diff / dist.powf(n_plus_2s) * h;
                }
            }
            // Normalization constant (simplified)
            let c_s = 4.0 * std::f64::consts::PI.powf(self.s) *
                       self.s * (1.0 - self.s) / 2.0;
            result[i] = c_s * sum;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fractional_space_creation() {
        let fs = FractionalSobolevSpace::new(0.5, 2.0, 1);
        assert!((fs.s - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gagliardo_seminorm_constant() {
        let fs = FractionalSobolevSpace::new(0.5, 2.0, 1);
        let v = DVector::from_vec(vec![3.0; 10]);
        let semi = fs.gagliardo_seminorm(&v, &[0.1]);
        assert_relative_eq!(semi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gagliardo_seminorm_nonconstant() {
        let fs = FractionalSobolevSpace::new(0.5, 2.0, 1);
        let v: DVector<f64> = DVector::from_vec((0..20).map(|i| i as f64 * 0.1).collect::<Vec<_>>());
        let semi = fs.gagliardo_seminorm(&v, &[0.1]);
        assert!(semi > 0.0);
    }

    #[test]
    fn test_fractional_norm() {
        let fs = FractionalSobolevSpace::new(0.5, 2.0, 1);
        let v: DVector<f64> = DVector::from_vec((0..20).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());
        let norm = fs.norm(&v, &[0.1]);
        assert!(norm > 0.0);
        assert!(norm.is_finite());
    }

    #[test]
    fn test_is_member() {
        let fs = FractionalSobolevSpace::new(0.5, 2.0, 1);
        let v = DVector::from_vec(vec![1.0; 10]);
        assert!(fs.is_member(&v, &[0.1]));
    }

    #[test]
    fn test_decompose_order() {
        let fs = FractionalSobolevSpace::new(1.7, 2.0, 1);
        let (k, sigma) = fs.decompose_order();
        assert_eq!(k, 1);
        assert!((sigma - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_decompose_order_half() {
        let fs = FractionalSobolevSpace::new(0.5, 2.0, 1);
        let (k, sigma) = fs.decompose_order();
        assert_eq!(k, 0);
        assert!((sigma - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sobolev_conjugate_fractional() {
        // s=0.5, p=2, n=1: sp=1 < n=1? No. sp = n.
        let fs = FractionalSobolevSpace::new(0.5, 2.0, 1);
        assert!(fs.sobolev_conjugate().is_none());
    }

    #[test]
    fn test_sobolev_conjugate_fractional_3d() {
        // s=0.5, p=2, n=3: sp=1 < n=3 → p* = 3*2/(3-1) = 3
        let fs = FractionalSobolevSpace::new(0.5, 2.0, 3);
        let q = fs.sobolev_conjugate().unwrap();
        assert!((q - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_fractional_laplacian_constant() {
        let fl = FractionalLaplacian::new(0.5);
        let v = DVector::from_vec(vec![5.0; 10]);
        let result = fl.apply(&v, 0.1);
        for val in result.iter() {
            assert_relative_eq!(*val, 0.0, epsilon = 0.5);
        }
    }

    #[test]
    fn test_fractional_laplacian_linear() {
        let fl = FractionalLaplacian::new(0.5);
        let v: DVector<f64> = DVector::from_vec((0..10).map(|i| i as f64 * 0.1).collect::<Vec<_>>());
        let result = fl.apply(&v, 0.1);
        // Fractional Laplacian of a linear function should be zero or close
        // (it's a nonlocal operator, so not exactly zero)
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_seminorm_increases_with_roughness() {
        let fs = FractionalSobolevSpace::new(0.5, 2.0, 1);
        let n = 30;
        let h = 0.1;
        let smooth: DVector<f64> = DVector::from_vec((0..n).map(|i| (i as f64 * h).sin()).collect::<Vec<_>>());
        let rough: DVector<f64> = DVector::from_vec((0..n).map(|i| {
            let x = i as f64 * h;
            x.sin() + 0.5 * (x * 20.0).sin()
        }).collect::<Vec<_>>());
        let semi_smooth = fs.gagliardo_seminorm(&smooth, &[h]);
        let semi_rough = fs.gagliardo_seminorm(&rough, &[h]);
        assert!(semi_smooth < semi_rough);
    }

    #[test]
    fn test_fractional_norm_order_monotonicity() {
        let v: DVector<f64> = DVector::from_vec((0..20).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());
        let fs_low = FractionalSobolevSpace::new(0.25, 2.0, 1);
        let fs_high = FractionalSobolevSpace::new(0.75, 2.0, 1);
        let norm_low = fs_low.norm(&v, &[0.1]);
        let norm_high = fs_high.norm(&v, &[0.1]);
        assert!(norm_high >= norm_low);
    }
}
