# lau-sobolev-agents

**Sobolev spaces for agents — smoothness classes of agent behavior.**

[![Tests](https://img.shields.io/badge/tests-121-passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)]()

---

## What This Does

Not all agent policies are created equal. A smooth policy — one whose derivatives are well-controlled — generalizes well, resists perturbations, and behaves predictably. A rough policy — full of sharp transitions and high-frequency oscillations — is fragile, overfits easily, and may be unstable.

This crate provides the full machinery of **Sobolev space theory** — weak derivatives, Sobolev norms, embedding theorems, compactness (Rellich-Kondrachov), Poincaré inequalities, trace theorems, Gagliardo-Nirenberg interpolation, fractional Sobolev spaces, and the fractional Laplacian — all applied to classifying and analyzing agent policy smoothness.

You can:
- **Compute Sobolev norms** W^{k,p} to measure how smooth a policy is
- **Classify policies** as Rough / ModeratelySmooth / Smooth / VerySmooth
- **Predict robustness and generalization** from regularity
- **Verify embedding theorems**: W^{k,p} ⊂ C^m when k > n/p + m
- **Check compactness** via Rellich-Kondrachov
- **Verify Poincaré inequalities** on bounded domains
- **Extract traces** (boundary values) of Sobolev functions
- **Interpolate norms** via Gagliardo-Nirenberg
- **Work with fractional regularity** W^{s,p} for non-integer s
- **Apply the fractional Laplacian** (−Δ)^s

---

## Key Idea

The central question: **how smooth is this agent's behavior?**

| Classical View | Sobolev View |
|---|---|
| Is the policy differentiable? | What's the W^{k,p} norm? |
| Is it continuous? | Does the Sobolev embedding hold? |
| How much does it oscillate? | What's the H¹ semi-norm? |
| Will it generalize? | Higher smoothness ⟹ better generalization |
| Is it robust to noise? | Lower gradient energy ⟹ more robust |

A **Sobolev space** W^{k,p}(Ω) consists of functions whose derivatives up to order k exist (in the weak/distributional sense) and are p-integrable. The norm:

$$\|u\|_{W^{k,p}} = \left(\sum_{|\alpha| \leq k} \|D^\alpha u\|_{L^p}^p\right)^{1/p}$$

measures both the size of the function and the size of its derivatives — a combined measure of magnitude and smoothness.

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-sobolev-agents = { git = "https://github.com/SuperInstance/lau-sobolev-agents" }
```

**Dependencies:** `serde` (with `derive`), `nalgebra` (with `serde-serialize`), `num-traits`.

---

## Quick Start

### Classify a Policy's Smoothness

```rust
use lau_sobolev_agents::{AgentPolicyAnalyzer, SmoothnessClass};

let analyzer = AgentPolicyAnalyzer::new_1d(0.1);

// A smooth sinusoidal policy
let smooth: DVector<f64> = DVector::from_vec(
    (0..50).map(|i| (i as f64 * 0.1).sin()).collect()
);
let cls = analyzer.classify(&smooth);
println!("Class: {} (H¹ norm: {:.4})", cls.class, cls.h1_norm);
// → Smooth (stable/generalizable)

// A noisy, erratic policy
let rough: DVector<f64> = DVector::from_vec(
    (0..50).map(|i| {
        let x = i as f64 * 0.1;
        x.sin() + 5.0 * (x * 100.0).sin()
    }).collect()
);
let cls = analyzer.classify(&rough);
println!("Robustness: {:.2}, Overfitting risk: {:?}", 
    cls.robustness_score, /* ... */);
// → Rough (erratic/fragile)
```

### Predict Robustness and Generalization

```rust
let pred = analyzer.predict_robustness(&policy);
println!("Lipschitz estimate: {:.2}", pred.lipschitz_estimate);
println!("Perturbation tolerance: {:.2}", pred.perturbation_tolerance);
println!("Stable: {}", pred.is_stable);

let gen = analyzer.predict_generalization(&policy);
println!("Generalization score: {:.2}", gen.generalization_score);
println!("Overfitting risk: {:?}", gen.overfitting_risk);
```

### Sobolev Embedding Theorem

```rust
use lau_sobolev_agents::{SobolevSpace, SobolevEmbedding};

// H¹(ℝ) = W^{1,2}(ℝ¹)
let h1 = SobolevSpace::h1(1);
let emb = SobolevEmbedding::new(h1);

// W^{1,2}(ℝ) ⊂ C⁰? Yes — critical exponent = 1 - 1/2 = 0.5 > 0
let result = emb.check_continuous_embedding(0);
assert!(result.holds);

// W^{1,2}(ℝ) ⊂ C¹? No — 0.5 < 1
let result = emb.check_continuous_embedding(1);
assert!(!result.holds);

// Maximum continuity: floor(0.5) = 0 → C⁰ but not C¹
assert_eq!(emb.max_continuity_order(), 0);
```

### Fractional Sobolev Spaces

```rust
use lau_sobolev_agents::{FractionalSobolevSpace, FractionalLaplacian};

// W^{1/2, 2}(ℝ¹) — the fractional Sobolev space of order 1/2
let w_half = FractionalSobolevSpace::new(0.5, 2.0, 1);
let seminorm = w_half.gagliardo_seminorm(&values, &[0.1]);
let full_norm = w_half.norm(&values, &[0.1]);

// Fractional Laplacian (-Δ)^{1/2}
let fl = FractionalLaplacian::new(0.5);
let result = fl.apply(&values, 0.1);
```

---

## API Reference

### Core Types

| Type | Module | Description |
|---|---|---|
| `WeakDerivative` | `weak_derivative` | Distributional derivative via finite differences |
| `SobolevNorm` | `sobolev_norm` | W^{k,p} and L^p norm computation |
| `SobolevSpace` | `sobolev_space` | W^{k,p} space with membership testing |
| `SobolevEmbedding` | `embedding` | Sobolev embedding theorem checks |
| `RellichKondrachov` | `rellich_kondrachov` | Compact embedding verification |
| `PoincareInequality` | `poincare` | Poincaré and Poincaré-Wirtinger inequalities |
| `TraceTheorem` | `trace` | Boundary value extraction and extension |
| `GagliardoNirenberg` | `gagliardo_nirenberg` | Interpolation inequality verification |
| `FractionalSobolevSpace` | `fractional` | W^{s,p} for non-integer s (Gagliardo seminorm) |
| `FractionalLaplacian` | `fractional` | (−Δ)^s via discrete approximation |
| `AgentPolicyAnalyzer` | `agent_policy` | High-level smoothness classification |

### `WeakDerivative` — Distributional Differentiation

```rust
// Compute first-order weak derivative
let wd = WeakDerivative::compute_weak(&u, dim, spacing, n_dims);

// Higher-order: d²/dx²(x²) = 2
let wd2 = WeakDerivative::compute_higher_order(&u, &[2], &[h]);

// Gradient and Laplacian helpers
let grad = gradient_1d(&u, h);
let lap = laplacian_1d(&u, h);

// Multi-dimensional
let grads = gradient_nd(&matrix_2d, &[h1, h2]);
```

### `SobolevNorm` — Norm Computation

```rust
let calc = SobolevNorm::new(k, p);

// L^p norm
calc.lp_norm(&u, p);

// Sobolev norm ‖u‖_{W^{k,p}}
calc.compute(&u, &spacings);

// Semi-norm (derivatives only, no L^p part)
calc.semi_norm(&u, &spacings);

// H¹ = W^{1,2} convenience
SobolevNorm::h1_norm(&u, &spacings);
SobolevNorm::h1_semi_norm(&u, &spacings);
SobolevNorm::l2_norm(&u);
```

### `SobolevSpace` — Space Construction

```rust
SobolevSpace::new(k, p, n_dims)   // General W^{k,p}(ℝⁿ)
SobolevSpace::hilbert(k, n)       // H^k = W^{k,2}(ℝⁿ)
SobolevSpace::h1(n)               // H¹(ℝⁿ)
SobolevSpace::h2(n)               // H²(ℝⁿ)

let result = space.membership(&u, &spacings);
// result.is_member, result.sobolev_norm

let cmp = space.compare_smoothness(&u1, &u2, &spacings);
// cmp.smoother (0 or 1), cmp.norm_1, cmp.norm_2
```

### `AgentPolicyAnalyzer` — High-Level API

```rust
let analyzer = AgentPolicyAnalyzer::new_1d(0.1);

// Classify smoothness
analyzer.classify(&policy)
// → PolicyClassification { class, h1_norm, h2_norm, robustness_score, ... }

// Predict robustness
analyzer.predict_robustness(&policy)
// → RobustnessPrediction { lipschitz_estimate, is_stable, perturbation_tolerance }

// Predict generalization
analyzer.predict_generalization(&policy)
// → GeneralizationPrediction { overfitting_risk: Low/Medium/High }

// Compare two policies
analyzer.compare(&policy_a, &policy_b)
// → PolicyComparison { smoother: 0|1, norm_ratio }

// Batch operations
analyzer.batch_classify(&[p1, p2, p3]);
analyzer.find_smoothest(&[p1, p2, p3]);

// Regularization loss for training
analyzer.regularization_loss(&policy, order: 1, weight: 0.01);
```

### `PoincareInequality` — Mean Deviation Bounds

```rust
let pi = PoincareInequality::new(diameter, n_dims, p);
pi.theoretical_constant()           // C = diameter / π (for p=2, 1D)
pi.verify(&values, &spacings)       // PoincareResult { holds, constant_c }
pi.verify_wirtinger(&values, &sp)   // For zero-mean functions
pi.estimate_constant(&fns, &sp)     // From a collection of functions
```

### `TraceTheorem` — Boundary Values

```rust
let tt = TraceTheorem::new(k, p, n_dims);
tt.trace_is_well_defined()          // k > 1/p
tt.trace_regularity()               // k - 1/p
tt.trace_1d(&values)                // Extract endpoints
tt.trace_2d(&matrix)                // Extract boundary of 2D grid
tt.extend_from_trace_1d(&[a, b], n) // Linear interpolation to interior
```

---

## How It Works

### Layer 1: Weak Derivatives (`weak_derivative`)

Classical derivatives require functions to be smooth. **Weak derivatives** relax this: v is the weak derivative of u if for all test functions φ:

$$\int v \cdot \varphi \, dx = -\int u \cdot \varphi' \, dx$$

This is computed via finite differences (central for interior, one-sided at boundaries). Higher-order derivatives are obtained by repeated application. The module also provides `gradient_1d`, `laplacian_1d`, and `gradient_nd` (multi-dimensional) helpers.

### Layer 2: Sobolev Norms (`sobolev_norm`)

The Sobolev norm combines the L^p norm of a function with the L^p norms of its derivatives:

$$\|u\|_{W^{k,p}} = \left(\sum_{|\alpha| \leq k} \|D^\alpha u\|_{L^p}^p\right)^{1/p}$$

The **semi-norm** uses only the highest-order derivatives:

$$|u|_{W^{k,p}} = \left(\sum_{|\alpha| = k} \|D^\alpha u\|_{L^p}^p\right)^{1/p}$$

Convenience methods for H¹ = W^{1,2} and L² norms are provided.

### Layer 3: Sobolev Spaces (`sobolev_space`)

A `SobolevSpace` bundles the parameters (k, p, n_dims) with membership testing. A function is "in" W^{k,p} if its Sobolev norm is finite. The module supports:
- **Membership testing**: is the function's W^{k,p} norm finite?
- **Smoothness comparison**: which of two functions has lower norm?
- **All derivatives up to order k**: enumerate all multi-index derivatives
- **Sum and intersection** of Sobolev spaces

### Layer 4: Sobolev Embedding (`embedding`)

The Sobolev embedding theorem answers: when is a Sobolev function continuous? The key criterion:

$$W^{k,p}(\mathbb{R}^n) \subset C^m \quad \text{when } k - \frac{n}{p} > m$$

The **critical exponent** k − n/p determines everything. In 1D with p = 2:
- W^{1,2}(ℝ): critical = 0.5 → embeds into C⁰ but not C¹
- W^{2,2}(ℝ): critical = 1.5 → embeds into C¹ but not C²
- W^{1,2}(ℝ³): critical = −0.5 → no continuous embedding!

The **Sobolev conjugate** p* = np/(n − kp) gives the integrability embedding W^{1,p} ⊂ L^{p*}.

### Layer 5: Compact Embeddings (`rellich_kondrachov`)

The Rellich-Kondrachov theorem: on bounded domains, the Sobolev embedding is **compact** — bounded sequences in W^{k,p} have convergent subsequences in L^q (for q < p*). This is the analytical foundation for:
- Existence of minimizers in variational problems
- Spectral theory of differential operators
- Finite element convergence

The crate checks compactness for embeddings into L^q and into W^{l,q}, and computes the compactness modulus (variance) of a bounded sequence.

### Layer 6: Poincaré Inequality (`poincare`)

The Poincaré inequality controls the mean deviation by the gradient:

$$\|u - \bar{u}\|_{L^p} \leq C \|\nabla u\|_{L^p}$$

For a convex domain with diameter d, C ≤ d/π (in 1D with p = 2). The Poincaré-Wirtinger variant states that for zero-mean functions, ‖u‖_{L^p} ≤ C‖∇u‖_{L^p}. The crate verifies these numerically and estimates the optimal constant from a collection of functions.

### Layer 7: Trace Theorem (`trace`)

For u ∈ W^{k,p}(Ω), the **trace** γu = u|_{∂Ω} is well-defined when k > 1/p. The trace operator is bounded: ‖γu‖ ≤ C‖u‖_{W^{k,p}}. In 1D, the trace is just the endpoints. In 2D, it's the boundary of the grid. The crate also supports extension from boundary data to the interior via linear interpolation.

### Layer 8: Gagliardo-Nirenberg Interpolation (`gagliardo_nirenberg`)

The Gagliardo-Nirenberg inequality estimates intermediate derivatives:

$$\|D^j u\|_{L^r} \leq C_1 \|D^m u\|_{L^q}^a \cdot \|u\|_{L^p}^{1-a}$$

where a = (j/n + 1/p − 1/r) / (m/n + 1/p − 1/q). Special cases:
- **Classical**: ‖∇u‖_{L²} ≤ C‖u‖_{L²}^{1/2}‖Δu‖_{L²}^{1/2} (a = 1/2)
- **Nash**: ‖u‖_{L²} ≤ C‖u‖_{L¹}^{1−a}‖∇u‖_{L²}^{a}

### Layer 9: Fractional Sobolev Spaces (`fractional`)

For non-integer s > 0, W^{s,p} is defined via the **Gagliardo (Slobodeckij) seminorm**:

$$[u]_{s,p} = \left(\iint \frac{|u(x) - u(y)|^p}{|x - y|^{n + sp}} \, dx \, dy\right)^{1/p}$$

These spaces interpolate between integer-order Sobolev spaces. The crate also implements the **fractional Laplacian** (−Δ)^s via a discrete integral approximation.

### Layer 10: Agent Policy Analysis (`agent_policy`)

The top-level API applies all this theory to agent policies:

- **Smoothness classification**: Rough / ModeratelySmooth / Smooth / VerySmooth based on regularity ratios
- **Robustness prediction**: estimated Lipschitz constant, perturbation tolerance, stability
- **Generalization prediction**: overfitting risk (Low/Medium/High) from H² semi-norm
- **Batch operations**: classify multiple policies, find the smoothest
- **Regularization loss**: Sobolev penalty for training smooth policies

---

## The Math

### Weak Derivatives

The weak derivative extends differentiation to non-smooth functions. v = Du in the weak sense if:

$$\int_\Omega v \varphi \, dx = -\int_\Omega u \varphi' \, dx \quad \forall \varphi \in C_c^\infty(\Omega)$$

Key property: the weak derivative is **unique** (up to measure zero) and agrees with the classical derivative when both exist.

### Sobolev Embedding Theorem

For k − n/p > m, the embedding W^{k,p}(ℝⁿ) ↪ C^m(ℝⁿ) is continuous. The **critical exponent** is:

$$p^* = \frac{np}{n - kp} \quad (kp < n)$$

giving W^{1,p}(ℝⁿ) ⊂ L^{p*}(ℝⁿ).

### Rellich-Kondrachov

On bounded Lipschitz domains: W^{k,p}(Ω) ⊂⊂ L^q(Ω) is compact for q < p*. This means every bounded sequence has a convergent subsequence.

### Poincaré Inequality

On bounded convex domains with diameter d:

$$\|u - \bar{u}\|_{L^p} \leq \frac{d}{\pi} \|\nabla u\|_{L^p}$$

### Trace Theorem

The trace γ: W^{k,p}(Ω) → W^{k−1/p,p}(∂Ω) is bounded when k > 1/p. The trace regularity is k − 1/p.

### Gagliardo-Nirenberg

The interpolation parameter a = (j/n + 1/p − 1/r) / (m/n + 1/p − 1/q) satisfies a ∈ [0, 1] and:

$$\|D^j u\|_{L^r} \leq C \|D^m u\|_{L^q}^a \|u\|_{L^p}^{1-a}$$

### Fractional Sobolev Spaces

W^{s,p}(ℝⁿ) for non-integer s = k + σ (k ∈ ℕ, 0 < σ < 1):

$$\|u\|_{W^{s,p}} = \|u\|_{W^{k,p}} + \sum_{|\alpha|=k} [D^\alpha u]_{\sigma, p}$$

The Gagliardo seminorm [u]_{σ,p} = (∫∫ |u(x)−u(y)|^p / |x−y|^{n+σp} dx dy)^{1/p}.

---

## Test Suite

121 tests across 11 modules:

| Module | Tests | Coverage |
|---|---|---|
| `weak_derivative` | 13 | Constant, linear, quadratic, sine derivatives; orders; gradient; Laplacian; 2D |
| `sobolev_norm` | 13 | L^p norms, H¹/H² norms, semi-norms, smoothness comparison |
| `sobolev_space` | 12 | H¹/H²/Hilbert construction, membership, comparison, sum/intersection |
| `embedding` | 13 | Continuous/integrability embeddings, conjugate exponents, verification |
| `rellich_kondrachov` | 12 | Compact embeddings, conjugate, bounded/unbounded, modulus |
| `poincare` | 9 | Poincaré constant, sine/constant/linear verification, Wirtinger, 2D |
| `trace` | 11 | Well-definedness, 1D/2D traces, extension, boundedness |
| `gagliardo_nirenberg` | 10 | Interpolation parameters, classical/Nash verification, 2D/3D |
| `fractional` | 13 | Gagliardo seminorm, norm, membership, decomposition, fractional Laplacian |
| `agent_policy` | 15 | Classification, comparison, robustness, generalization, batch, regularization |

Run all tests:

```bash
cargo test
```

---

## License

MIT
