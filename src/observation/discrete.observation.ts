import { Distribution } from '../models/belief.model';
import {
    IObservationModel,
    ObservationMatrix,
} from '../models/observation.model';

/**
 * Discrete observation model implementing the A matrix in Active Inference.
 *
 * This class represents P(o|s) - the probability of receiving observation o
 * given the hidden state s. It's a core component of the agent's generative
 * model used for perception and planning.
 *
 * ## Matrix Structure
 *
 * The observation matrix is organized as: `observation → state → probability`
 *
 * For each observation, you specify the probability of that observation
 * occurring given each possible hidden state. Each column (observation
 * conditional on all states) defines a likelihood function.
 *
 * ## Usage in Active Inference
 *
 * The observation model enables:
 * - **Perception**: Bayesian belief updates when observations are received
 * - **Ambiguity computation**: Expected uncertainty about observations
 * - **Policy evaluation**: Predicting what observations actions will cause
 *
 * ## Sensor Accuracy
 *
 * The observation matrix encodes sensor reliability:
 * - Perfect sensor: P(o|s) = 1 for matching state, 0 otherwise
 * - Noisy sensor: Some probability mass spreads to wrong observations
 * - Ambiguous sensor: Multiple states produce similar observations
 *
 * @typeParam O - Union type of possible observation names
 * @typeParam S - Union type of possible state names
 *
 * @example
 * ```typescript
 * // 90% accurate sensor for detecting danger
 * const observation = new DiscreteObservation({
 *   see_safe: { safe: 0.9, danger: 0.1 },    // see_safe mostly in safe state
 *   see_danger: { safe: 0.1, danger: 0.9 }   // see_danger mostly in danger state
 * });
 *
 * // Use for Bayesian update
 * const likelihood = observation.getLikelihood('see_danger');
 * const posterior = belief.update(likelihood);
 * ```
 *
 * @see {@link IObservationModel} - Interface this class implements
 * @see {@link ObservationMatrix} - Type definition for the matrix structure
 */
export class DiscreteObservation<
    O extends string = string,
    S extends string = string,
> implements IObservationModel<O, S>
{
    /**
     * Create a discrete observation model from an observation matrix.
     *
     * @param matrix - Observation matrix defining P(o|s) for all observation-state pairs.
     *                 Structure: observation → state → probability
     *
     * @example
     * ```typescript
     * const observation = new DiscreteObservation({
     *   bright: { light_on: 0.95, light_off: 0.05 },
     *   dim: { light_on: 0.05, light_off: 0.95 }
     * });
     * ```
     */
    constructor(public matrix: ObservationMatrix<O, S>) {}

    /**
     * List of all possible observations in this model.
     *
     * Observations are extracted from the keys of the observation matrix.
     *
     * @returns Array of observation names
     *
     * @example
     * ```typescript
     * observation.observations; // ['bright', 'dim']
     * ```
     */
    get observations(): O[] {
        return Object.keys(this.matrix) as O[];
    }

    /**
     * List of all possible hidden states in this model.
     *
     * States are extracted from the first observation's state mappings.
     * Assumes all observations have the same state space.
     *
     * @returns Array of state names
     *
     * @example
     * ```typescript
     * observation.states; // ['light_on', 'light_off']
     * ```
     */
    get states(): S[] {
        const firstObs = this.observations[0];
        return Object.keys(this.matrix[firstObs] || {}) as S[];
    }

    /**
     * Get the likelihood function for a specific observation.
     *
     * Returns P(o|s) for all states - this is the key function for
     * Bayesian belief updates during perception.
     *
     * When the agent receives an observation, it uses this likelihood
     * to update its beliefs: P(s|o) ∝ P(o|s) × P(s)
     *
     * @param observation - The observation received
     * @returns Distribution mapping each state to its likelihood
     *
     * @example
     * ```typescript
     * const likelihood = observation.getLikelihood('see_reward');
     * // { good_state: 0.8, bad_state: 0.1 }
     *
     * // Use for belief update
     * const posterior = belief.update(likelihood);
     * ```
     */
    getLikelihood(observation: O): Distribution<S> {
        return this.matrix[observation] ?? ({} as Distribution<S>);
    }

    /**
     * Get the probability of a specific observation given a state.
     *
     * Returns P(o|s) - the probability of observing o when the
     * true hidden state is s.
     *
     * @param observation - The observation
     * @param state - The hidden state
     * @returns Probability between 0 and 1
     *
     * @example
     * ```typescript
     * observation.probability('see_safe', 'danger'); // 0.1 (unlikely)
     * observation.probability('see_safe', 'safe');   // 0.9 (likely)
     * ```
     */
    probability(observation: O, state: S): number {
        return this.matrix[observation]?.[state] ?? 0;
    }
}
