//! Weak (distributional) derivatives for non-smooth agents.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// A weak derivative of a function in the distributional sense.
///
/// A function v is the weak derivative of u if for all test functions φ:
///   ∫ v·φ dx = -∫ u·φ' dx
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakDerivative {
    /// Order of the derivative (multi-index representation per dimension)
    pub orders: Vec<usize>,
    /// Values of the weak derivative on a discrete grid
    pub values: DVector<f64>,
    /// Domain spacing (h for each dimension)
    pub spacing: Vec<f64>,
}

impl WeakDerivative {
    /// Create a new weak derivative.
    pub fn new(orders: Vec<usize>, values: DVector<f64>, spacing: Vec<f64>) -> Self {
        Self { orders, values, spacing }
    }

    /// Compute the weak derivative of order 1 in dimension `dim` via
    /// finite differences with averaging (mimics distributional derivative).
    pub fn compute_weak(u: &DVector<f64>, dim: usize, spacing: f64, n_dims: usize) -> Self {
        let n = u.len();
        let mut dv = DVector::zeros(n);

        if n < 2 {
            return WeakDerivative::new(vec![0; n_dims], dv, vec![spacing; n_dims]);
        }

        let mut orders = vec![0; n_dims];
        orders[dim] = 1;

        // Central differences interior, forward/backward at boundaries
        for i in 0..n {
            if i == 0 {
                dv[i] = (u[1] - u[0]) / spacing;
            } else if i == n - 1 {
                dv[i] = (u[n - 1] - u[n - 2]) / spacing;
            } else {
                dv[i] = (u[i + 1] - u[i - 1]) / (2.0 * spacing);
            }
        }

        WeakDerivative::new(orders, dv, vec![spacing; n_dims])
    }

    /// Compute higher-order weak derivative by repeated differentiation.
    pub fn compute_higher_order(u: &DVector<f64>, orders: &[usize], spacings: &[f64]) -> Self {
        let n_dims = orders.len();
        let mut current = u.clone();
        let current_spacings = spacings.to_vec();

        for (dim, &order) in orders.iter().enumerate() {
            let h = if dim < spacings.len() { spacings[dim] } else { spacings[0] };
            for _ in 0..order {
                let wd = Self::compute_weak(&current, 0, h, n_dims);
                current = wd.values;
            }
        }

        WeakDerivative::new(orders.to_vec(), current, current_spacings)
    }

    /// Check if two functions have the same weak derivative (up to tolerance).
    pub fn approx_eq(&self, other: &WeakDerivative, tol: f64) -> bool {
        if self.orders != other.orders || self.values.len() != other.values.len() {
            return false;
        }
        self.values.iter().zip(other.values.iter())
            .all(|(&a, &b)| (a - b).abs() < tol)
    }

    /// Total order of the weak derivative.
    pub fn total_order(&self) -> usize {
        self.orders.iter().sum()
    }
}

/// Compute gradient (vector of first-order weak derivatives) in 1D.
pub fn gradient_1d(u: &DVector<f64>, spacing: f64) -> DVector<f64> {
    let n = u.len();
    if n < 2 {
        return DVector::zeros(n);
    }
    let mut grad = DVector::zeros(n);
    for i in 0..n {
        if i == 0 {
            grad[i] = (u[1] - u[0]) / spacing;
        } else if i == n - 1 {
            grad[i] = (u[n - 1] - u[n - 2]) / spacing;
        } else {
            grad[i] = (u[i + 1] - u[i - 1]) / (2.0 * spacing);
        }
    }
    grad
}

/// Compute Laplacian (sum of second-order weak derivatives) in 1D.
pub fn laplacian_1d(u: &DVector<f64>, spacing: f64) -> DVector<f64> {
    let n = u.len();
    if n < 3 {
        return DVector::zeros(n);
    }
    let mut lap = DVector::zeros(n);
    let h2 = spacing * spacing;
    for i in 1..n - 1 {
        lap[i] = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / h2;
    }
    lap[0] = lap[1.min(n - 1)];
    lap[n - 1] = lap[(n - 2).max(0)];
    lap
}

/// Multi-dimensional gradient (Jacobian) via finite differences.
pub fn gradient_nd(values: &DMatrix<f64>, spacings: &[f64]) -> Vec<DMatrix<f64>> {
    let (nrows, ncols) = values.shape();
    let n_dims = spacings.len().min(2);
    let mut grads = Vec::new();

    // Derivative along rows (dim 0)
    if n_dims > 0 {
        let mut d0 = DMatrix::zeros(nrows, ncols);
        for j in 0..ncols {
            for i in 0..nrows {
                if i == 0 && nrows > 1 {
                    d0[(i, j)] = (values[(1, j)] - values[(0, j)]) / spacings[0];
                } else if i == nrows - 1 && nrows > 1 {
                    d0[(i, j)] = (values[(nrows - 1, j)] - values[(nrows - 2, j)]) / spacings[0];
                } else if nrows > 2 {
                    d0[(i, j)] = (values[(i + 1, j)] - values[(i - 1, j)]) / (2.0 * spacings[0]);
                }
            }
        }
        grads.push(d0);
    }

    // Derivative along columns (dim 1)
    if n_dims > 1 {
        let mut d1 = DMatrix::zeros(nrows, ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                if j == 0 && ncols > 1 {
                    d1[(i, j)] = (values[(i, 1)] - values[(i, 0)]) / spacings[1];
                } else if j == ncols - 1 && ncols > 1 {
                    d1[(i, j)] = (values[(i, ncols - 1)] - values[(i, ncols - 2)]) / spacings[1];
                } else if ncols > 2 {
                    d1[(i, j)] = (values[(i, j + 1)] - values[(i, j - 1)]) / (2.0 * spacings[1]);
                }
            }
        }
        grads.push(d1);
    }

    grads
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_weak_derivative_constant() {
        let u = DVector::from_vec(vec![5.0; 10]);
        let wd = WeakDerivative::compute_weak(&u, 0, 0.1, 1);
        for v in wd.values.iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_weak_derivative_linear() {
        let n = 20;
        let h = 0.1;
        let u: DVector<f64> = DVector::from_vec((0..n).map(|i| 3.0 * (i as f64) * h).collect::<Vec<_>>());
        let wd = WeakDerivative::compute_weak(&u, 0, h, 1);
        for v in wd.values.iter() {
            assert_relative_eq!(*v, 3.0, epsilon = 0.5);
        }
    }

    #[test]
    fn test_weak_derivative_quadratic() {
        let n = 50;
        let h = 0.05;
        let u: DVector<f64> = DVector::from_vec((0..n).map(|i| { let x = i as f64 * h; x * x }).collect::<Vec<_>>());
        let wd = WeakDerivative::compute_weak(&u, 0, h, 1);
        // d/dx(x^2) = 2x
        for i in 5..n - 5 {
            let x = i as f64 * h;
            assert_relative_eq!(wd.values[i], 2.0 * x, epsilon = 0.1);
        }
    }

    #[test]
    fn test_weak_derivative_sine() {
        let n = 100;
        let h = 2.0 * std::f64::consts::PI / (n as f64);
        let u: DVector<f64> = DVector::from_vec((0..n).map(|i| (i as f64 * h).sin()).collect::<Vec<_>>());
        let wd = WeakDerivative::compute_weak(&u, 0, h, 1);
        for i in 5..n - 5 {
            let x = i as f64 * h;
            assert_relative_eq!(wd.values[i], x.cos(), epsilon = 0.05);
        }
    }

    #[test]
    fn test_weak_derivative_orders() {
        let orders = vec![2, 0, 1];
        let wd = WeakDerivative::new(orders.clone(), DVector::zeros(5), vec![0.1; 3]);
        assert_eq!(wd.orders, orders);
        assert_eq!(wd.total_order(), 3);
    }

    #[test]
    fn test_approx_eq_weak_derivatives() {
        let v1 = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = DVector::from_vec(vec![1.001, 2.001, 3.001]);
        let wd1 = WeakDerivative::new(vec![1], v1, vec![0.1]);
        let wd2 = WeakDerivative::new(vec![1], v2, vec![0.1]);
        assert!(wd1.approx_eq(&wd2, 0.01));
        assert!(!wd1.approx_eq(&wd2, 0.0001));
    }

    #[test]
    fn test_higher_order_second_derivative() {
        let n = 50;
        let h = 0.1;
        let u: DVector<f64> = DVector::from_vec((0..n).map(|i| { let x = i as f64 * h; x * x }).collect::<Vec<_>>());
        let wd = WeakDerivative::compute_higher_order(&u, &[2], &[h]);
        // d²/dx²(x²) = 2
        for i in 5..n - 5 {
            assert_relative_eq!(wd.values[i], 2.0, epsilon = 0.5);
        }
    }

    #[test]
    fn test_gradient_1d_constant() {
        let u = DVector::from_vec(vec![3.0; 10]);
        let g = gradient_1d(&u, 0.1);
        assert!(g.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_gradient_1d_linear() {
        let u: DVector<f64> = DVector::from_vec((0..10).map(|i| 2.0 * i as f64 * 0.1).collect::<Vec<_>>());
        let g = gradient_1d(&u, 0.1);
        for v in g.iter() {
            assert_relative_eq!(*v, 2.0, epsilon = 0.3);
        }
    }

    #[test]
    fn test_laplacian_1d_quadratic() {
        let n = 30;
        let h = 0.1;
        let u: DVector<f64> = DVector::from_vec((0..n).map(|i| { let x = i as f64 * h; x * x }).collect::<Vec<_>>());
        let lap = laplacian_1d(&u, h);
        for i in 2..n - 2 {
            assert_relative_eq!(lap[i], 2.0, epsilon = 0.01);
        }
    }

    #[test]
    fn test_gradient_nd_2d() {
        let m = DMatrix::from_row_slice(3, 3, &[
            0.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
            2.0, 2.0, 2.0,
        ]);
        let grads = gradient_nd(&m, &[1.0, 1.0]);
        assert_eq!(grads.len(), 2);
        // Row derivative should detect constant vertical gradient
        assert_relative_eq!(grads[0][(1, 1)], 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_weak_derivative_single_point() {
        let u = DVector::from_vec(vec![1.0]);
        let wd = WeakDerivative::compute_weak(&u, 0, 0.1, 1);
        assert_eq!(wd.values.len(), 1);
        assert_relative_eq!(wd.values[0], 0.0);
    }

    #[test]
    fn test_weak_derivative_two_points() {
        let u = DVector::from_vec(vec![0.0, 1.0]);
        let wd = WeakDerivative::compute_weak(&u, 0, 0.5, 1);
        assert_relative_eq!(wd.values[0], 2.0, epsilon = 0.01);
    }
}
