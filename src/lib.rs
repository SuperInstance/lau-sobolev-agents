//! # lau-sobolev-agents
//!
//! Sobolev spaces for agents — smoothness classes of agent behavior.
//!
//! A Sobolev space Wᵏᵖ measures how smooth a function is by controlling
//! its derivatives up to order k. For agents: smooth policies = stable,
//! differentiable, well-behaved; rough policies = unpredictable, fragile.

mod weak_derivative;
mod sobolev_space;
mod sobolev_norm;
mod embedding;
mod rellich_kondrachov;
mod poincare;
mod trace;
mod gagliardo_nirenberg;
mod fractional;
mod agent_policy;

pub use weak_derivative::*;
pub use sobolev_space::*;
pub use sobolev_norm::*;
pub use embedding::*;
pub use rellich_kondrachov::*;
pub use poincare::*;
pub use trace::*;
pub use gagliardo_nirenberg::*;
pub use fractional::*;
pub use agent_policy::*;
