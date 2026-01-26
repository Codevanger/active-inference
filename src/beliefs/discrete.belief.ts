import { Belief, Distribution } from '../models/belief.model';

/**
 * Discrete probability distribution over a finite set of states.
 *
 * This is the standard implementation for discrete state spaces,
 * representing beliefs as a categorical distribution stored as
 * a simple key-value object.
 *
 * The distribution should sum to 1.0, though this is not strictly
 * enforced (it will be normalized during Bayesian updates).
 *
 * @typeParam S - Union type of possible state names
 *
 * @example
 * ```typescript
 * // Create a belief about weather
 * const belief = new DiscreteBelief({
 *   sunny: 0.7,
 *   rainy: 0.3
 * });
 *
 * console.log(belief.argmax());           // 'sunny'
 * console.log(belief.probability('sunny')); // 0.7
 * console.log(belief.entropy());           // ~0.61 nats
 *
 * // Update with observation
 * const likelihood = { sunny: 0.1, rainy: 0.9 }; // saw clouds
 * const posterior = belief.update(likelihood);
 * console.log(posterior.argmax()); // 'rainy'
 * ```
 */
export class DiscreteBelief<S extends string = string> extends Belief<S> {
    /**
     * Create a discrete belief from a probability distribution.
     *
     * @param distribution - Object mapping state names to probabilities.
     *                       Values should be non-negative and ideally sum to 1.
     *
     * @example
     * ```typescript
     * // Uniform prior over 3 states
     * const uniform = new DiscreteBelief({
     *   a: 1/3,
     *   b: 1/3,
     *   c: 1/3
     * });
     *
     * // Certain belief
     * const certain = new DiscreteBelief({
     *   known: 1.0,
     *   unknown: 0.0
     * });
     * ```
     */
    constructor(public distribution: Distribution<S>) {
        super();
    }

    /**
     * Get all states in this belief's state space.
     * Order is determined by object key order (insertion order in ES6+).
     *
     * @returns Array of state names
     */
    get states(): S[] {
        return Object.keys(this.distribution) as S[];
    }

    /**
     * Get the probability assigned to a specific state.
     *
     * @param state - The state to query
     * @returns Probability between 0 and 1, or 0 if state not found
     */
    probability(state: S): number {
        return this.distribution[state] ?? 0;
    }

    /**
     * Perform Bayesian belief update given observation likelihood.
     *
     * Computes: posterior(s) ∝ likelihood(s) × prior(s)
     *
     * The result is automatically normalized to sum to 1.
     * If all likelihoods are 0, returns unchanged distribution.
     *
     * @param likelihood - P(observation | state) for each state
     * @returns New DiscreteBelief representing the posterior
     *
     * @example
     * ```typescript
     * const prior = new DiscreteBelief({ a: 0.5, b: 0.5 });
     * const likelihood = { a: 0.9, b: 0.1 }; // evidence favors 'a'
     * const posterior = prior.update(likelihood);
     * // posterior ≈ { a: 0.9, b: 0.1 }
     * ```
     */
    update(likelihood: Distribution<S>): DiscreteBelief<S> {
        const newDist: Distribution<S> = {} as Distribution<S>;
        let sum = 0;

        for (const state of this.states) {
            newDist[state] =
                this.distribution[state] * (likelihood[state] ?? 0);
            sum += newDist[state];
        }

        if (sum > 0) {
            for (const state of this.states) {
                newDist[state] /= sum;
            }
        }

        return new DiscreteBelief<S>(newDist);
    }

    /**
     * Create an independent deep copy of this belief.
     *
     * Modifications to the copy will not affect the original.
     *
     * @returns New DiscreteBelief with same distribution
     */
    copy(): DiscreteBelief<S> {
        return new DiscreteBelief<S>({ ...this.distribution } as Distribution<S>);
    }
}
