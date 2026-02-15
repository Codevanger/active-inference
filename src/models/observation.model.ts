import { Distribution } from './belief.model';

/**
 * Observation likelihood matrix (A matrix in Active Inference notation).
 * Defines P(o|s) - probability of observing o given hidden state s.
 *
 * Structure: observation → state → probability
 *
 * The observation model captures how hidden states generate observable
 * outcomes. Since the agent cannot directly perceive the true state,
 * it must infer it from observations using this mapping.
 *
 * @typeParam O - Union type of possible observation names
 * @typeParam S - Union type of possible state names
 *
 * @example
 * ```typescript
 * // A sensor that's 90% accurate
 * const A: ObservationMatrix<'see_light' | 'see_dark', 'light' | 'dark'> = {
 *   see_light: { light: 0.9, dark: 0.1 },  // P(see_light | state)
 *   see_dark: { light: 0.1, dark: 0.9 }    // P(see_dark | state)
 * };
 * ```
 */
export type ObservationMatrix<
    O extends string = string,
    S extends string = string,
> = Record<O, Distribution<S>>;

/**
 * Interface for observation models (likelihood models).
 *
 * In Active Inference, the observation model (A) defines the relationship
 * between hidden states and observations. It serves two purposes:
 *
 * 1. **Perception**: Given an observation, compute the likelihood for
 *    Bayesian belief update (what states could have caused this?)
 *
 * 2. **Prediction**: Given predicted future states, compute expected
 *    observations for evaluating action outcomes
 *
 * The observation model is crucial for:
 * - Bayesian inference during perception
 * - Computing ambiguity (uncertainty about observations)
 * - Evaluating Expected Free Energy for action selection
 *
 * @typeParam O - Union type of possible observation names
 * @typeParam S - Union type of possible state names
 */
export interface IObservationModel<
    O extends string = string,
    S extends string = string,
> {
    /**
     * List of all possible observations the agent can receive.
     */
    readonly observations: O[];

    /**
     * Get likelihood function for Bayesian update.
     * Returns P(observation | state) for all states.
     *
     * This is the key function for perception - it tells
     * the agent how likely each state is given what it observed.
     *
     * @param observation - The observation received
     * @returns Distribution over states given the observation
     *
     * @example
     * ```typescript
     * const likelihood = model.getLikelihood('see_reward');
     * // { good_state: 0.9, bad_state: 0.1 }
     * ```
     */
    getLikelihood(observation: O): Distribution<S>;

    /**
     * Get probability of a specific observation given a state.
     * Returns P(observation | state).
     *
     * @param observation - The observation
     * @param state - The hidden state
     * @returns Probability between 0 and 1
     */
    probability(observation: O, state: S): number;

    /**
     * Update model from experience (optional).
     * Learnable models refine their parameters after each observation.
     *
     * @param observation - The observation received
     * @param posteriorBelief - Posterior belief over states
     */
    learn?(observation: O, posteriorBelief: Distribution<S>): void;
}
