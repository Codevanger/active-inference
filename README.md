# Active Inference

TypeScript implementation of the Active Inference framework based on Karl Friston's Free Energy Principle.

## What is Active Inference?

Active Inference is a theory of how biological agents perceive and act in the world. Agents maintain beliefs about hidden states and select actions to minimize **Expected Free Energy** — a quantity that balances:

- **Risk**: avoiding unpreferred outcomes
- **Ambiguity**: seeking informative observations

This library provides building blocks for creating agents that perceive, learn, plan, and act using these principles.

## Installation

```bash
npm install active-inference
```

## Quick Start

```typescript
import {
    createAgent,
    DiscreteBelief,
    DiscreteTransition,
    DiscreteObservation,
} from 'active-inference';

const agent = createAgent({
    belief: new DiscreteBelief({ left: 0.5, right: 0.5 }),
    transitionModel: new DiscreteTransition({
        go_left: {
            left: { left: 1.0, right: 0.0 },
            right: { left: 0.8, right: 0.2 },
        },
        go_right: {
            left: { left: 0.2, right: 0.8 },
            right: { left: 0.0, right: 1.0 },
        },
    }),
    observationModel: new DiscreteObservation({
        see_reward: { left: 0.9, right: 0.1 },
        see_nothing: { left: 0.1, right: 0.9 },
    }),
    preferences: { see_reward: 0, see_nothing: -5 },
});

const action = agent.step('see_reward');
```

## API

### createAgent(config)

| Parameter | Description |
|-----------|-------------|
| `belief` | Initial belief over hidden states |
| `transitionModel` | P(s'\|s, a) — how actions change states |
| `observationModel` | P(o\|s) — how states generate observations |
| `preferences` | Log probabilities of preferred observations |
| `planningHorizon` | Steps to look ahead (default: 1) |
| `precision` | Action selection temperature (default: 1) |
| `habits` | Prior over actions / E matrix (default: uniform) |
| `seed` | Random seed for reproducibility |

### Agent

| Method | Description |
|--------|-------------|
| `step(obs)` | Observe and act |
| `observe(obs)` | Update beliefs from observation |
| `act()` | Select action minimizing EFE |
| `state` | Most likely hidden state |
| `uncertainty` | Belief entropy (confidence) |
| `freeEnergy` | Variational Free Energy |
| `exportBelief()` | Get full belief distribution |

## Learning

The library supports **Dirichlet-categorical learning** — agents that update their generative models from experience. Instead of fixed probability matrices, learnable models maintain pseudo-count concentrations that are refined over time.

- `DirichletObservation` and `DirichletTransition` are drop-in replacements for their Discrete counterparts. Learning happens automatically on every `step()` call.
- `DirichletPreferences` provides learnable preferred observations — call `.learn()` manually and pass `.preferences` to the agent config.

Low concentrations encode weak priors (learns fast). High concentrations encode strong priors (resists change).

```typescript
import {
    createAgent,
    DiscreteBelief,
    DirichletTransition,
    DirichletObservation,
} from 'active-inference';

const agent = createAgent({
    belief: new DiscreteBelief({ safe: 0.5, danger: 0.5 }),
    transitionModel: new DirichletTransition({
        flee: {
            safe:   { safe: 1, danger: 1 },
            danger: { safe: 1, danger: 1 },
        },
        stay: {
            safe:   { safe: 1, danger: 1 },
            danger: { safe: 1, danger: 1 },
        },
    }),
    observationModel: new DirichletObservation({
        see_safe:   { safe: 1, danger: 1 },
        see_danger: { safe: 1, danger: 1 },
    }),
    preferences: { see_safe: 0, see_danger: -5 },
    seed: 42,
});

// Models update automatically on each step
const action = agent.step('see_safe');
```

## Examples

### Cart-Pole Balancing

Interactive browser demo — an Active Inference agent balances an inverted pendulum using a 49-state generative model with 3-step planning horizon.

```bash
npm run build:examples
open examples/cart-pole/index.html
```

## Contributing

```bash
git clone https://github.com/codevanger/active-inference
cd active-inference
npm install
npm test
```

PRs welcome

## References

- [Friston, K. (2010). The free-energy principle: a unified brain theory?](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf)
- [Active Inference: A Process Theory](https://direct.mit.edu/neco/article/29/1/1/8207/Active-Inference-A-Process-Theory)

## License

MIT
