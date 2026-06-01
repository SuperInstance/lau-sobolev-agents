//! Trace theorem: boundary values of Sobolev functions.
//!
//! For u ∈ Wᵏᵖ(Ω), the trace γu = u|∂Ω is well-defined in Wᵏ⁻¹/ᵖ,ᵖ(∂Ω)
//! when k > 1/p. The trace operator is bounded and surjective (with a
//! continuous right inverse).
//!
//! For agents: what the agent "looks like" from outside — its observable behavior
//! at the boundary of its domain.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// Trace theorem result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceResult {
    /// Values of the trace (boundary values)
    pub trace_values: DVector<f64>,
    /// Trace norm
    pub trace_norm: f64,
    /// Interior norm (Sobolev norm of original function)
    pub interior_norm: f64,
    /// Trace constant (bound on trace operator)
    pub trace_constant: f64,
    /// Whether the trace is well-defined
    pub is_well_defined: bool,
}

/// Trace theorem calculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceTheorem {
    /// Sobolev order k
    pub k: usize,
    /// Integrability exponent p
    pub p: f64,
    /// Domain dimension
    pub n_dims: usize,
}

impl TraceTheorem {
    pub fn new(k: usize, p: f64, n_dims: usize) -> Self {
        Self { k, p, n_dims }
    }

    /// Check if the trace is well-defined: k > 1/p.
    pub fn trace_is_well_defined(&self) -> bool {
        (self.k as f64) > 1.0 / self.p
    }

    /// Compute the trace regularity: k - 1/p.
    pub fn trace_regularity(&self) -> f64 {
        self.k as f64 - 1.0 / self.p
    }

    /// Extract boundary values (trace) of a 1D function.
    /// In 1D, the boundary is just the two endpoints.
    pub fn trace_1d(&self, values: &DVector<f64>) -> TraceResult {
        let n = values.len();
        let trace_values = if n >= 2 {
            DVector::from_vec(vec![values[0], values[n - 1]])
        } else {
            values.clone()
        };

        let trace_norm = trace_values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let interior_norm = SobolevNormShim::l2_norm(values);

        let constant = if interior_norm > 1e-15 {
            trace_norm / interior_norm
        } else {
            0.0
        };

        TraceResult {
            trace_values,
            trace_norm,
            interior_norm,
            trace_constant: constant,
            is_well_defined: self.trace_is_well_defined(),
        }
    }

    /// Extract boundary values for a 2D grid.
    /// Returns values on the boundary of the grid.
    pub fn trace_2d(&self, grid: &nalgebra::DMatrix<f64>) -> TraceResult {
        let (rows, cols) = grid.shape();
        let mut boundary = Vec::new();

        // Top and bottom rows
        for j in 0..cols {
            boundary.push(grid[(0, j)]);
            boundary.push(grid[(rows - 1, j)]);
        }
        // Left and right columns (excluding corners)
        for i in 1..rows - 1 {
            boundary.push(grid[(i, 0)]);
            boundary.push(grid[(i, cols - 1)]);
        }

        let trace_values = DVector::from_vec(boundary);
        let trace_norm = trace_values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

        let total = rows * cols;
        let mut sum = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                sum += grid[(i, j)] * grid[(i, j)];
            }
        }
        let interior_norm = (sum / total as f64).sqrt();

        let constant = if interior_norm > 1e-15 {
            trace_norm / interior_norm
        } else {
            0.0
        };

        TraceResult {
            trace_values,
            trace_norm,
            interior_norm,
            trace_constant: constant,
            is_well_defined: self.trace_is_well_defined(),
        }
    }

    /// Compute the theoretical trace constant estimate.
    /// For W¹²: ‖γu‖ ≤ C ‖u‖_{H¹} with C depending on domain geometry.
    pub fn theoretical_trace_constant(&self, domain_size: f64) -> f64 {
        if self.trace_is_well_defined() {
            // Rough estimate: C ~ domain_size^{1/2} / boundary_measure^{1/2}
            domain_size.sqrt() * 2.0
        } else {
            f64::INFINITY
        }
    }

    /// Extension: given boundary data, construct an extension to the interior.
    /// Uses simple linear interpolation for 1D.
    pub fn extend_from_trace_1d(&self, boundary: &[f64], n_interior: usize) -> DVector<f64> {
        let n = n_interior.max(2);
        let mut result = DVector::zeros(n);
        let left = boundary.first().copied().unwrap_or(0.0);
        let right = boundary.get(1).copied().unwrap_or(left);

        for i in 0..n {
            let t = i as f64 / (n - 1).max(1) as f64;
            result[i] = left * (1.0 - t) + right * t;
        }
        result
    }
}

/// Minimal L² norm helper to avoid circular dependency issues.
struct SobolevNormShim;

impl SobolevNormShim {
    fn l2_norm(u: &DVector<f64>) -> f64 {
        let n = u.len() as f64;
        let sum: f64 = u.iter().map(|v| v * v).sum();
        (sum / n).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_well_defined_h1() {
        let tt = TraceTheorem::new(1, 2.0, 1);
        assert!(tt.trace_is_well_defined()); // 1 > 1/2
    }

    #[test]
    fn test_trace_not_defined_low_order() {
        let tt = TraceTheorem::new(0, 2.0, 1);
        assert!(!tt.trace_is_well_defined()); // 0 < 1/2
    }

    #[test]
    fn test_trace_regularity() {
        let tt = TraceTheorem::new(1, 2.0, 1);
        assert!((tt.trace_regularity() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_trace_1d() {
        let tt = TraceTheorem::new(1, 2.0, 1);
        let v = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = tt.trace_1d(&v);
        assert_eq!(result.trace_values.len(), 2);
        assert!((result.trace_values[0] - 1.0).abs() < 1e-10);
        assert!((result.trace_values[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_trace_1d_constant() {
        let tt = TraceTheorem::new(1, 2.0, 1);
        let v = DVector::from_vec(vec![3.0; 10]);
        let result = tt.trace_1d(&v);
        assert!((result.trace_values[0] - 3.0).abs() < 1e-10);
        assert!((result.trace_values[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_trace_2d() {
        let tt = TraceTheorem::new(1, 2.0, 2);
        let m = nalgebra::DMatrix::from_row_slice(3, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let result = tt.trace_2d(&m);
        assert!(result.trace_values.len() > 0);
        assert!(result.trace_norm > 0.0);
    }

    #[test]
    fn test_trace_bounded() {
        let tt = TraceTheorem::new(1, 2.0, 1);
        let n = 50;
        let v: DVector<f64> = DVector::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());
        let result = tt.trace_1d(&v);
        assert!(result.trace_constant.is_finite());
    }

    #[test]
    fn test_extend_from_trace() {
        let tt = TraceTheorem::new(1, 2.0, 1);
        let extended = tt.extend_from_trace_1d(&[0.0, 1.0], 10);
        assert_eq!(extended.len(), 10);
        assert!((extended[0] - 0.0).abs() < 1e-10);
        assert!((extended[9] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_extend_matches_boundary() {
        let tt = TraceTheorem::new(1, 2.0, 1);
        let extended = tt.extend_from_trace_1d(&[2.0, 8.0], 5);
        assert!((extended[0] - 2.0).abs() < 1e-10);
        assert!((extended[4] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_theoretical_trace_constant() {
        let tt = TraceTheorem::new(1, 2.0, 1);
        let c = tt.theoretical_trace_constant(1.0);
        assert!(c.is_finite());
    }

    #[test]
    fn test_theoretical_trace_undefined() {
        let tt = TraceTheorem::new(0, 2.0, 1);
        let c = tt.theoretical_trace_constant(1.0);
        assert!(c.is_infinite());
    }
}
