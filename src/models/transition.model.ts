import { Belief, Distribution } from './belief.model';

/**
 * State transition matrix (B matrix in Active Inference notation).
 * Defines P(s'|s, a) - probability of transitioning to state s'
 * given current state s and action a.
 *
 * Structure: action → current_state → next_state → probability
 *
 * @typeParam A - Union type of possible action names
 * @typeParam S - Union type of possible state names
 *
 * @example
 * ```typescript
 * const B: TransitionMatrix<'move' | 'stay', 'left' | 'right'> = {
 *   move: {
 *     left: { left: 0.1, right: 0.9 },   // move from left → mostly goes right
 *     right: { left: 0.9, right: 0.1 }   // move from right → mostly goes left
 *   },
 *   stay: {
 *     left: { left: 1.0, right: 0.0 },   // stay at left
 *     right: { left: 0.0, right: 1.0 }   // stay at right
 *   }
 * };
 * ```
 */
export type TransitionMatrix<
    A extends string = string,
    S extends string = string,
> = Record<A, Record<S, Distribution<S>>>;

/**
 * Interface for state transition models.
 *
 * In Active Inference, the transition model (B) captures the agent's
 * beliefs about how the world state changes in response to actions.
 * It's a key component of the generative model used for planning.
 *
 * The transition model enables:
 * - Predicting future states given current beliefs and actions
 * - Evaluating action sequences (policies) by simulating outcomes
 * - Computing Expected Free Energy for action selection
 *
 * @typeParam A - Union type of possible action names
 * @typeParam S - Union type of possible state names
 *
 * @example
 * ```typescript
 * class MyTransition implements ITransitionModel<'up' | 'down', 'top' | 'bottom'> {
 *   get actions() { return ['up', 'down'] as const; }
 *   predict(belief, action) { // ... return predicted belief }
 * }
 * ```
 */
export interface ITransitionModel<
    A extends string = string,
    S extends string = string,
> {
    /**
     * List of all possible actions the agent can take.
     */
    readonly actions: A[];

    /**
     * Predict the belief state after taking an action.
     * Computes: P(s') = Σ_s P(s'|s, a) × P(s)
     *
     * This is belief propagation through the transition model,
     * used during planning to simulate future states.
     *
     * @param belief - Current belief over states
     * @param action - Action to simulate
     * @returns Predicted belief after the action
     */
    predict(belief: Belief<S>, action: A): Belief<S>;

    /**
     * Update model from experience (optional).
     * Learnable models refine their parameters after each transition.
     *
     * @param action - The action taken
     * @param priorBelief - Belief before the action
     * @param posteriorBelief - Belief after observing the outcome
     */
    learn?(action: A, priorBelief: Distribution<S>, posteriorBelief: Distribution<S>): void;
}
