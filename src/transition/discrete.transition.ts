import { Distribution } from '../models/belief.model';
import type { Belief } from '../models/belief.model';
import { ITransitionModel, TransitionMatrix } from '../models/transition.model';
import { DiscreteBelief } from '../beliefs/discrete.belief';

/**
 * Discrete state transition model implementing the B matrix in Active Inference.
 *
 * This class represents P(s'|s, a) - the probability of transitioning to
 * a new state s' given the current state s and action a. It's a core
 * component of the agent's generative model used for planning.
 *
 * ## Matrix Structure
 *
 * The transition matrix is organized as: `action → current_state → next_state → probability`
 *
 * For each action, you specify how each current state transitions to next states.
 * Each row (current state) should sum to 1.0 (valid probability distribution).
 *
 * ## Usage in Active Inference
 *
 * The transition model enables:
 * - **Planning**: Simulating future states to evaluate action sequences
 * - **Policy evaluation**: Computing Expected Free Energy over time horizons
 * - **State prediction**: Propagating beliefs through time
 *
 * @typeParam A - Union type of possible action names
 * @typeParam S - Union type of possible state names
 *
 * @example
 * ```typescript
 * // Light switch model: actions affect the light state
 * const transition = new DiscreteTransition({
 *   turn_on: {
 *     dark: { dark: 0.1, light: 0.9 },   // turn_on in dark → 90% light
 *     light: { dark: 0.0, light: 1.0 }   // turn_on in light → stays light
 *   },
 *   turn_off: {
 *     dark: { dark: 1.0, light: 0.0 },   // turn_off in dark → stays dark
 *     light: { dark: 0.9, light: 0.1 }   // turn_off in light → 90% dark
 *   }
 * });
 *
 * // Predict state after action
 * const currentBelief = new DiscreteBelief({ dark: 1.0, light: 0.0 });
 * const predicted = transition.predict(currentBelief, 'turn_on');
 * // predicted: { dark: 0.1, light: 0.9 }
 * ```
 *
 * @see {@link ITransitionModel} - Interface this class implements
 * @see {@link TransitionMatrix} - Type definition for the matrix structure
 */
export class DiscreteTransition<
    A extends string = string,
    S extends string = string,
> implements ITransitionModel<A, S>
{
    /**
     * Create a discrete transition model from a transition matrix.
     *
     * @param matrix - Transition matrix defining P(s'|s, a) for all state-action pairs.
     *                 Structure: action → current_state → next_state → probability
     *
     * @example
     * ```typescript
     * const transition = new DiscreteTransition({
     *   move_left: {
     *     left: { left: 1.0, right: 0.0 },
     *     right: { left: 0.8, right: 0.2 }
     *   },
     *   move_right: {
     *     left: { left: 0.2, right: 0.8 },
     *     right: { left: 0.0, right: 1.0 }
     *   }
     * });
     * ```
     */
    constructor(public matrix: TransitionMatrix<A, S>) {}

    /**
     * List of all possible actions in this model.
     *
     * Actions are extracted from the keys of the transition matrix.
     *
     * @returns Array of action names
     *
     * @example
     * ```typescript
     * transition.actions; // ['move_left', 'move_right', 'stay']
     * ```
     */
    get actions(): A[] {
        return Object.keys(this.matrix) as A[];
    }

    /**
     * List of all possible states in this model.
     *
     * States are extracted from the first action's state mappings.
     * Assumes all actions have the same state space.
     *
     * @returns Array of state names
     *
     * @example
     * ```typescript
     * transition.states; // ['left', 'center', 'right']
     * ```
     */
    get states(): S[] {
        const firstAction = this.actions[0];
        return Object.keys(this.matrix[firstAction] || {}) as S[];
    }

    /**
     * Get the transition distribution for a specific state-action pair.
     *
     * Returns P(s'|s, a) - the probability distribution over next states
     * given the current state and action.
     *
     * @param state - The current state
     * @param action - The action being taken
     * @returns Distribution over next states
     *
     * @example
     * ```typescript
     * const dist = transition.getTransition('left', 'move_right');
     * // { left: 0.2, right: 0.8 }
     * ```
     */
    getTransition(state: S, action: A): Distribution<S> {
        return this.matrix[action]?.[state] ?? ({} as Distribution<S>);
    }

    /**
     * Predict the belief state after taking an action.
     *
     * Performs belief propagation through the transition model:
     * P(s') = Σ_s P(s'|s, a) × P(s)
     *
     * This is the core function used during planning to simulate
     * what the agent's beliefs would be after taking an action.
     *
     * @param belief - Current belief distribution over states
     * @param action - Action to simulate
     * @returns Predicted belief after the action
     *
     * @example
     * ```typescript
     * const current = new DiscreteBelief({ left: 0.8, right: 0.2 });
     * const predicted = transition.predict(current, 'move_right');
     * // Belief shifts toward 'right' based on transition probabilities
     * ```
     */
    predict(belief: Belief<S>, action: A): Belief<S> {
        const newDist: Distribution<S> = {} as Distribution<S>;

        for (const state of this.states) {
            newDist[state] = 0;
        }

        for (const currentState of belief.states) {
            const transition = this.getTransition(currentState, action);
            const currentProb = belief.probability(currentState);

            for (const nextState of Object.keys(transition) as S[]) {
                newDist[nextState] += transition[nextState] * currentProb;
            }
        }

        return new DiscreteBelief<S>(newDist);
    }
}
