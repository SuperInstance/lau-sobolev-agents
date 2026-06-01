//! Agent policy smoothness classification.
//!
//! Apply Sobolev space theory to classify agent behavior:
//! smooth policies → stable, generalizable, robust
//! rough policies → unpredictable, fragile, overfitting-prone

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::SobolevNorm;

/// Smoothness class of an agent policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmoothnessClass {
    /// C⁰ or rougher — discontinuous, highly erratic
    Rough,
    /// W¹ᵖ with moderate regularity — mostly smooth with occasional jumps
    ModeratelySmooth,
    /// W²ᵖ or higher — smooth, well-behaved, differentiable policy
    Smooth,
    /// Wᵏᵖ with large k — very smooth, essentially polynomial
    VerySmooth,
}

impl std::fmt::Display for SmoothnessClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SmoothnessClass::Rough => write!(f, "Rough (erratic/fragile)"),
            SmoothnessClass::ModeratelySmooth => write!(f, "ModeratelySmooth (acceptable)"),
            SmoothnessClass::Smooth => write!(f, "Smooth (stable/generalizable)"),
            SmoothnessClass::VerySmooth => write!(f, "VerySmooth (highly regular)"),
        }
    }
}

/// Classification result for an agent policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyClassification {
    /// The smoothness class
    pub class: SmoothnessClass,
    /// W¹² norm
    pub h1_norm: f64,
    /// W²² norm (if computed)
    pub h2_norm: Option<f64>,
    /// H¹ semi-norm (measures gradient energy)
    pub h1_semi_norm: f64,
    /// L² norm of the policy
    pub l2_norm: f64,
    /// Estimated robustness score (0–1)
    pub robustness_score: f64,
    /// Estimated generalization score (0–1)
    pub generalization_score: f64,
    /// Gradient energy (total variation of policy)
    pub gradient_energy: f64,
}

/// Agent policy analyzer using Sobolev space theory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPolicyAnalyzer {
    /// Input dimension for the policy
    pub input_dim: usize,
    /// Grid spacing for discrete approximation
    pub spacing: Vec<f64>,
}

impl AgentPolicyAnalyzer {
    pub fn new(input_dim: usize, spacing: Vec<f64>) -> Self {
        Self { input_dim, spacing }
    }

    /// 1D analyzer with uniform spacing.
    pub fn new_1d(spacing: f64) -> Self {
        Self::new(1, vec![spacing])
    }

    /// Classify a policy's smoothness.
    pub fn classify(&self, policy: &DVector<f64>) -> PolicyClassification {
        let h1_norm = SobolevNorm::h1_norm(policy, &self.spacing);
        let h1_semi = SobolevNorm::h1_semi_norm(policy, &self.spacing);
        let l2_norm = SobolevNorm::l2_norm(policy);

        let _h1_calc = SobolevNorm::new(1, 2.0);
        let h2_calc = SobolevNorm::new(2, 2.0);
        let h2_norm = h2_calc.compute(policy, &self.spacing);

        // Gradient energy
        let grad = crate::gradient_1d(policy, self.spacing[0]);
        let gradient_energy = SobolevNorm::l2_norm(&grad);

        // Ratio of gradient energy to function energy (regularity measure)
        let regularity = if l2_norm > 1e-15 {
            h1_semi / l2_norm
        } else {
            0.0
        };

        // Classify based on regularity ratios
        let class = if regularity < 0.3 {
            SmoothnessClass::VerySmooth
        } else if regularity < 1.0 {
            SmoothnessClass::Smooth
        } else if regularity < 3.0 {
            SmoothnessClass::ModeratelySmooth
        } else {
            SmoothnessClass::Rough
        };

        // Robustness: inversely related to gradient magnitude
        let robustness_score = 1.0 / (1.0 + regularity).min(10.0);

        // Generalization: higher smoothness = better generalization
        let generalization_score = {
            let h2_ratio = if l2_norm > 1e-15 { h2_norm / l2_norm } else { 0.0 };
            1.0 / (1.0 + h2_ratio).min(10.0)
        };

        PolicyClassification {
            class,
            h1_norm,
            h2_norm: Some(h2_norm),
            h1_semi_norm: h1_semi,
            l2_norm,
            robustness_score,
            generalization_score,
            gradient_energy,
        }
    }

    /// Compare two policies' smoothness.
    pub fn compare(&self, policy_a: &DVector<f64>, policy_b: &DVector<f64>) -> PolicyComparison {
        let cls_a = self.classify(policy_a);
        let cls_b = self.classify(policy_b);

        let smoother = if cls_a.h1_norm <= cls_b.h1_norm { 0 } else { 1 };
        let norm_ratio = if cls_b.h1_norm > 1e-15 {
                cls_a.h1_norm / cls_b.h1_norm
            } else {
                f64::INFINITY
            };

        PolicyComparison {
            policy_a: cls_a,
            policy_b: cls_b,
            smoother,
            norm_ratio,
        }
    }

    /// Predict robustness: policies with lower Sobolev norms are more robust.
    pub fn predict_robustness(&self, policy: &DVector<f64>) -> RobustnessPrediction {
        let cls = self.classify(policy);

        // Estimate Lipschitz constant from gradient
        let grad = crate::gradient_1d(policy, self.spacing[0]);
        let lip_estimate = grad.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));

        RobustnessPrediction {
            robustness_score: cls.robustness_score,
            lipschitz_estimate: lip_estimate,
            smoothness_class: cls.class,
            is_stable: lip_estimate < 5.0,
            perturbation_tolerance: 1.0 / (1.0 + lip_estimate).min(10.0),
        }
    }

    /// Predict generalization from Sobolev regularity.
    pub fn predict_generalization(&self, policy: &DVector<f64>) -> GeneralizationPrediction {
        let cls = self.classify(policy);
        let h2_calc = SobolevNorm::new(2, 2.0);
        let h2_semi = h2_calc.semi_norm(policy, &self.spacing);

        GeneralizationPrediction {
            generalization_score: cls.generalization_score,
            h2_semi_norm: h2_semi,
            smoothness_class: cls.class,
            overfitting_risk: if cls.generalization_score > 0.7 {
                OverfittingRisk::Low
            } else if cls.generalization_score > 0.3 {
                OverfittingRisk::Medium
            } else {
                OverfittingRisk::High
            },
        }
    }

    /// Batch classify multiple policies.
    pub fn batch_classify(&self, policies: &[DVector<f64>]) -> Vec<PolicyClassification> {
        policies.iter().map(|p| self.classify(p)).collect()
    }

    /// Find the smoothest policy in a batch.
    pub fn find_smoothest(&self, policies: &[DVector<f64>]) -> (usize, PolicyClassification) {
        let classifications = self.batch_classify(policies);
        let (idx, cls) = classifications.iter().enumerate()
            .min_by(|(_, a), (_, b)| a.h1_norm.partial_cmp(&b.h1_norm).unwrap())
            .unwrap();
        (idx, cls.clone())
    }

    /// Compute Sobolev regularization loss for training.
    /// Encourages smooth policies by penalizing high Sobolev norms.
    pub fn regularization_loss(&self, policy: &DVector<f64>, order: usize, weight: f64) -> f64 {
        let calc = SobolevNorm::new(order, 2.0);
        weight * calc.semi_norm(policy, &self.spacing).powi(2)
    }
}

/// Overfitting risk level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverfittingRisk {
    Low,
    Medium,
    High,
}

/// Comparison of two policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyComparison {
    pub policy_a: PolicyClassification,
    pub policy_b: PolicyClassification,
    /// Index of the smoother policy (0 or 1)
    pub smoother: usize,
    /// Ratio of norms (a/b)
    pub norm_ratio: f64,
}

/// Robustness prediction for a policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessPrediction {
    pub robustness_score: f64,
    pub lipschitz_estimate: f64,
    pub smoothness_class: SmoothnessClass,
    pub is_stable: bool,
    pub perturbation_tolerance: f64,
}

/// Generalization prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralizationPrediction {
    pub generalization_score: f64,
    pub h2_semi_norm: f64,
    pub smoothness_class: SmoothnessClass,
    pub overfitting_risk: OverfittingRisk,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_analyzer() -> AgentPolicyAnalyzer {
        AgentPolicyAnalyzer::new_1d(0.1)
    }

    #[test]
    fn test_classify_constant() {
        let a = make_analyzer();
        let p = DVector::from_vec(vec![1.0; 50]);
        let cls = a.classify(&p);
        assert_eq!(cls.class, SmoothnessClass::VerySmooth);
        assert!(cls.robustness_score > 0.5);
    }

    #[test]
    fn test_classify_smooth() {
        let a = make_analyzer();
        let p: DVector<f64> = DVector::from_vec((0..50).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());
        let cls = a.classify(&p);
        assert!(cls.h1_norm > 0.0);
        assert!(cls.robustness_score > 0.0);
    }

    #[test]
    fn test_classify_rough() {
        let a = make_analyzer();
        let p: DVector<f64> = DVector::from_vec((0..50).map(|i| {
            let x = i as f64 * 0.1;
            x.sin() + 5.0 * (x * 100.0).sin()
        }).collect::<Vec<_>>());
        let cls = a.classify(&p);
        assert_eq!(cls.class, SmoothnessClass::Rough);
    }

    #[test]
    fn test_compare_policies() {
        let a = make_analyzer();
        let smooth: DVector<f64> = DVector::from_vec((0..50).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());
        let rough: DVector<f64> = DVector::from_vec((0..50).map(|i| {
            let x = i as f64 * 0.1;
            x.sin() + 2.0 * (x * 30.0).sin()
        }).collect::<Vec<_>>());
        let cmp = a.compare(&smooth, &rough);
        assert_eq!(cmp.smoother, 0);
        assert!(cmp.norm_ratio < 1.0);
    }

    #[test]
    fn test_predict_robustness() {
        let a = make_analyzer();
        let p: DVector<f64> = DVector::from_vec((0..50).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());
        let pred = a.predict_robustness(&p);
        assert!(pred.robustness_score > 0.0);
        assert!(pred.lipschitz_estimate > 0.0);
        assert!(pred.lipschitz_estimate.is_finite());
    }

    #[test]
    fn test_predict_generalization() {
        let a = make_analyzer();
        let p: DVector<f64> = DVector::from_vec((0..50).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());
        let pred = a.predict_generalization(&p);
        assert!(pred.generalization_score > 0.0);
    }

    #[test]
    fn test_overfitting_risk_smooth() {
        let a = make_analyzer();
        let p = DVector::from_vec(vec![1.0; 50]);
        let pred = a.predict_generalization(&p);
        assert_eq!(pred.overfitting_risk, OverfittingRisk::Medium);
    }

    #[test]
    fn test_overfitting_risk_rough() {
        let a = make_analyzer();
        let p: DVector<f64> = DVector::from_vec((0..50).map(|i| {
            let x = i as f64 * 0.1;
            10.0 * (x * 100.0).sin()
        }).collect::<Vec<_>>());
        let pred = a.predict_generalization(&p);
        assert_eq!(pred.overfitting_risk, OverfittingRisk::High);
    }

    #[test]
    fn test_batch_classify() {
        let a = make_analyzer();
        let policies: Vec<DVector<f64>> = (0..3).map(|k| {
            DVector::from_vec((0..50).map(|i| ((i as f64 * 0.1) * (k as f64 + 1.0)).sin()).collect::<Vec<_>>())
        }).collect();
        let results = a.batch_classify(&policies);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_find_smoothest() {
        let a = make_analyzer();
        let policies = vec![
            DVector::from_vec(vec![1.0; 50]), // smoothest
            DVector::from_vec((0..50).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>()),
            DVector::from_vec((0..50).map(|i| 5.0 * (i as f64 * 0.1 * 20.0).sin()).collect::<Vec<_>>()),
        ];
        let (idx, cls) = a.find_smoothest(&policies);
        assert!(idx == 0 || idx == 1);
        assert!(cls.h1_norm > 0.0);
    }

    #[test]
    fn test_regularization_loss() {
        let a = make_analyzer();
        let p: DVector<f64> = DVector::from_vec((0..50).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());
        let loss = a.regularization_loss(&p, 1, 0.01);
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_regularization_loss_constant() {
        let a = make_analyzer();
        let p = DVector::from_vec(vec![5.0; 50]);
        let loss = a.regularization_loss(&p, 1, 0.01);
        assert!((loss).abs() < 1e-10);
    }

    #[test]
    fn test_smoothness_class_display() {
        assert!(format!("{}", SmoothnessClass::Smooth).contains("Smooth"));
        assert!(format!("{}", SmoothnessClass::Rough).contains("Rough"));
    }

    #[test]
    fn test_classify_moderately_smooth() {
        let a = make_analyzer();
        let p: DVector<f64> = DVector::from_vec((0..50).map(|i| {
            let x = i as f64 * 0.1;
            x.sin() + 0.5 * (x * 5.0).sin()
        }).collect::<Vec<_>>());
        let cls = a.classify(&p);
        // Should be at least ModeratelySmooth
        assert!(cls.class == SmoothnessClass::Smooth || cls.class == SmoothnessClass::ModeratelySmooth);
    }

    #[test]
    fn test_stability() {
        let a = make_analyzer();
        let smooth: DVector<f64> = DVector::from_vec((0..50).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());
        let pred = a.predict_robustness(&smooth);
        assert!(pred.is_stable);
    }
}
