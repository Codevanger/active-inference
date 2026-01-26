/**
 * Probability distribution over states.
 * Maps each state to its probability value (0 to 1).
 *
 * @typeParam S - Union type of possible state names
 *
 * @example
 * ```typescript
 * const dist: Distribution<'sunny' | 'rainy'> = {
 *   sunny: 0.7,
 *   rainy: 0.3
 * };
 * ```
 */
export type Distribution<S extends string = string> = Record<S, number>;

/**
 * Agent's preferences over observations expressed as log probabilities.
 * Higher values indicate more preferred observations.
 * Typically 0 for neutral, negative for undesired outcomes.
 *
 * In Active Inference, preferences define the "goal" of the agent -
 * what observations it wants to experience.
 *
 * @typeParam O - Union type of possible observation names
 *
 * @example
 * ```typescript
 * const prefs: Preferences<'reward' | 'punishment'> = {
 *   reward: 0,      // neutral/desired
 *   punishment: -5  // strongly undesired
 * };
 * ```
 */
export type Preferences<O extends string = string> = Record<O, number>;

/**
 * Abstract base class representing an agent's beliefs about the world state.
 *
 * In Active Inference, beliefs (denoted Q(s) or sometimes D) represent
 * the agent's probability distribution over hidden states of the world.
 * The agent cannot directly observe the true state - it must infer it
 * from observations.
 *
 * Beliefs are updated via Bayesian inference when new observations arrive,
 * and used to predict future states when planning actions.
 *
 * @typeParam S - Union type of possible state names
 *
 * @example
 * ```typescript
 * class MyBelief extends Belief<'hot' | 'cold'> {
 *   // ... implement abstract methods
 * }
 * ```
 */
export abstract class Belief<S extends string = string> {
    /**
     * List of all possible states in this belief's state space.
     * @returns Array of state names
     */
    abstract get states(): S[];

    /**
     * Get the probability of a specific state.
     * @param state - The state to query
     * @returns Probability value between 0 and 1
     */
    abstract probability(state: S): number;

    /**
     * Perform Bayesian update given observation likelihood.
     * Computes posterior: P(state|obs) ∝ P(obs|state) × P(state)
     *
     * @param likelihood - P(observation|state) for each state
     * @returns New belief representing the posterior distribution
     */
    abstract update(likelihood: Distribution<S>): Belief<S>;

    /**
     * Create a deep copy of this belief.
     * @returns Independent copy with same probability distribution
     */
    abstract copy(): Belief<S>;

    /**
     * Find the most likely state (Maximum A Posteriori estimate).
     *
     * @returns The state with highest probability
     *
     * @example
     * ```typescript
     * const belief = new DiscreteBelief({ sunny: 0.8, rainy: 0.2 });
     * belief.argmax(); // 'sunny'
     * ```
     */
    argmax(): S {
        let maxState = this.states[0];
        let maxProb = 0;
        for (const state of this.states) {
            const prob = this.probability(state);
            if (prob > maxProb) {
                maxProb = prob;
                maxState = state;
            }
        }
        return maxState;
    }

    /**
     * Compute Kullback-Leibler divergence from another belief.
     * KL(P||Q) = Σ P(s) × log(P(s)/Q(s))
     *
     * Measures how different this distribution (P) is from another (Q).
     * Used in Active Inference to quantify epistemic value - how much
     * information an action would provide.
     *
     * @param other - The reference distribution Q
     * @returns KL divergence (non-negative, 0 if identical)
     *
     * @example
     * ```typescript
     * const prior = new DiscreteBelief({ a: 0.5, b: 0.5 });
     * const posterior = new DiscreteBelief({ a: 0.9, b: 0.1 });
     * posterior.kl(prior); // Information gained from update
     * ```
     */
    kl(other: Belief<S>): number {
        let result = 0;
        for (const state of this.states) {
            const p = this.probability(state);
            const q = other.probability(state) || 1e-10;
            if (p > 0) {
                result += p * Math.log(p / q);
            }
        }
        return result;
    }

    /**
     * Compute Shannon entropy of the belief distribution.
     * H(P) = -Σ P(s) × log(P(s))
     *
     * Measures uncertainty in the belief. High entropy means
     * the agent is uncertain about the true state.
     *
     * In Active Inference, entropy relates to expected surprise
     * and is minimized through perception and action.
     *
     * @returns Entropy in nats (natural log units), non-negative
     *
     * @example
     * ```typescript
     * const uncertain = new DiscreteBelief({ a: 0.5, b: 0.5 });
     * const certain = new DiscreteBelief({ a: 0.99, b: 0.01 });
     * uncertain.entropy(); // ~0.693 (high uncertainty)
     * certain.entropy();   // ~0.056 (low uncertainty)
     * ```
     */
    entropy(): number {
        let result = 0;
        for (const state of this.states) {
            const p = this.probability(state);
            if (p > 0) {
                result -= p * Math.log(p);
            }
        }
        return result;
    }
}
