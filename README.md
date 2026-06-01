# lau-sobolev-agents

Sobolev spaces for agents — smoothness classes of agent behavior.

A Sobolev space Wᵏᵖ measures how smooth a function is by controlling its derivatives up to order k. For agents: smooth policies = stable, differentiable, well-behaved; rough policies = unpredictable, fragile.

## Features

- **Sobolev space Wᵏᵖ**: functions with k derivatives in Lᵖ
- **Sobolev embedding**: Wᵏᵖ ⊂ Cᵐ when k > n/p + m
- **Rellich-Kondrachov**: compact embedding Wᵏᵖ ⊂⊂ Lᵍ on bounded domains
- **Poincaré inequality**: ‖u - ū‖ ≤ C‖∇u‖
- **Sobolev norm**: ‖u‖_{Wᵏᵖ} = (Σ|α|≤k ‖Dα u‖_p^p)^{1/p}
- **Weak derivatives**: distributional derivatives for non-smooth agents
- **Trace theorem**: boundary values of Sobolev functions
- **Gagliardo-Nirenberg interpolation**: between Sobolev spaces
- **Fractional Sobolev spaces Wˢᵖ**: partial regularity
- **Agent policy classification**: smoothness → robustness & generalization

## Usage

```rust
use lau_sobolev_agents::*;
use nalgebra::DVector;

// Create an analyzer for 1D policies
let analyzer = AgentPolicyAnalyzer::new_1d(0.1);

// Classify a policy
let policy: DVector<f64> = DVector::from_vec((0..100).map(|i| {
    (i as f64 * 0.1).sin()
}).collect());
let classification = analyzer.classify(&policy);
println!("Smoothness: {}", classification.class);
println!("Robustness: {:.3}", classification.robustness_score);
```

## License

MIT
