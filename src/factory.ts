import { Belief, Preferences } from './models/belief.model';
import { ITransitionModel } from './models/transition.model';
import { IObservationModel } from './models/observation.model';
import { Agent, Habits } from './models/agent.model';
import { Random } from './helpers/math.helpers';

/**
 * Configuration object for creating an Active Inference agent.
 *
 * This interface defines all the components needed to instantiate an agent
 * with its generative model and behavioral parameters.
 *
 * ## Required Components
 *
 * The generative model consists of:
 * - **belief**: Initial prior over hidden states (D matrix)
 * - **transitionModel**: State dynamics P(s'|s,a) (B matrix)
 * - **observationModel**: Observation likelihood P(o|s) (A matrix)
 * - **preferences**: Preferred observations as log probabilities (C vector)
 *
 * ## Optional Parameters
 *
 * Behavioral parameters that tune agent behavior:
 * - **seed**: For reproducible random behavior
 * - **planningHorizon**: How far ahead to plan (default: 1)
 * - **precision**: Action selection temperature (default: 1)
 * - **habits**: Prior action preferences (E matrix)
 *
 * @typeParam A - Union type of possible action names
 * @typeParam O - Union type of possible observation names
 * @typeParam S - Union type of possible state names
 *
 * @example
 * ```typescript
 * const config: AgentConfig<'left' | 'right', 'see_goal' | 'see_wall', 'at_goal' | 'at_start'> = {
 *   belief: new DiscreteBelief({ at_goal: 0.1, at_start: 0.9 }),
 *   transitionModel: myTransitions,
 *   observationModel: myObservations,
 *   preferences: { see_goal: 0, see_wall: -2 },
 *   planningHorizon: 3,
 *   precision: 4
 * };
 * ```
 */
export interface AgentConfig<
    A extends string = string,
    O extends string = string,
    S extends string = string,
> {
    /**
     * Initial belief distribution over hidden states.
     *
     * This is the agent's prior - what it believes about the world
     * before receiving any observations. Can be uncertain (spread
     * across states) or confident (concentrated on one state).
     */
    belief: Belief<S>;

    /**
     * State transition model defining P(s'|s, a).
     *
     * Encodes how the agent believes actions affect world state.
     * Used during planning to simulate future states.
     */
    transitionModel: ITransitionModel<A, S>;

    /**
     * Observation model defining P(o|s).
     *
     * Encodes how hidden states generate observations.
     * Used for Bayesian belief updates and computing ambiguity.
     */
    observationModel: IObservationModel<O, S>;

    /**
     * Preferred observations expressed as log probabilities.
     *
     * Higher values = more preferred. Typically:
     * - 0 for neutral/desired observations
     * - Negative for undesired observations (e.g., -5 for pain)
     *
     * These preferences define the agent's "goals" - what observations
     * it will act to make more likely.
     */
    preferences: Preferences<O>;

    /**
     * Random seed for reproducible behavior.
     *
     * When set, the agent's stochastic action selection will be
     * deterministic given the same sequence of observations.
     * Useful for testing and debugging.
     */
    seed?: number;

    /**
     * Planning horizon - number of time steps to look ahead.
     *
     * - 1 = greedy/reactive (only considers immediate outcomes)
     * - 2+ = planning (considers future consequences)
     *
     * Higher values enable better long-term decisions but increase
     * computation exponentially (actions^horizon policies to evaluate).
     *
     * @default 1
     */
    planningHorizon?: number;

    /**
     * Precision parameter (β) for action selection.
     *
     * Controls the "temperature" of the softmax over Expected Free Energy:
     * - β = 0: Uniform random action selection
     * - β → ∞: Deterministic selection of best action
     * - β = 1: Standard softmax (balanced exploration/exploitation)
     *
     * @default 1
     */
    precision?: number;

    /**
     * Habitual action preferences (E matrix in Active Inference).
     *
     * Biases action selection independently of Expected Free Energy.
     * Higher values make actions more likely to be selected regardless
     * of their predicted outcomes.
     *
     * Useful for modeling:
     * - Learned motor habits
     * - Default behaviors
     * - Action priors from experience
     */
    habits?: Partial<Habits<A>>;
}

/**
 * Factory function to create an Active Inference agent.
 *
 * This is the recommended way to instantiate agents, as it provides
 * a clean interface with sensible defaults for optional parameters.
 *
 * ## Type Inference
 *
 * TypeScript will automatically infer the type parameters from your
 * configuration objects, providing full type safety for actions,
 * observations, and states throughout your code.
 *
 * @typeParam A - Union type of possible action names (inferred from transitionModel)
 * @typeParam O - Union type of possible observation names (inferred from observationModel)
 * @typeParam S - Union type of possible state names (inferred from belief)
 *
 * @param config - Agent configuration object
 * @returns Configured Active Inference agent ready for use
 *
 * @example
 * ```typescript
 * // Create a simple agent
 * const agent = createAgent({
 *   belief: new DiscreteBelief({ safe: 0.5, danger: 0.5 }),
 *   transitionModel: new DiscreteTransition({
 *     stay: { safe: { safe: 1, danger: 0 }, danger: { safe: 0, danger: 1 } },
 *     flee: { safe: { safe: 0.9, danger: 0.1 }, danger: { safe: 0.7, danger: 0.3 } }
 *   }),
 *   observationModel: new DiscreteObservation({
 *     calm: { safe: 0.9, danger: 0.1 },
 *     alarm: { safe: 0.1, danger: 0.9 }
 *   }),
 *   preferences: { calm: 0, alarm: -5 },
 *   planningHorizon: 2,
 *   precision: 4,
 *   seed: 42  // For reproducibility
 * });
 *
 * // Use the agent
 * const action = agent.step('alarm');  // Types are inferred!
 * // action is typed as 'stay' | 'flee'
 * ```
 *
 * @see {@link Agent} - The agent class this creates
 * @see {@link AgentConfig} - Configuration interface
 */
export function createAgent<
    A extends string = string,
    O extends string = string,
    S extends string = string,
>(config: AgentConfig<A, O, S>): Agent<A, O, S> {
    const random =
        config.seed !== undefined ? new Random(config.seed) : new Random();

    return new Agent<A, O, S>(
        config.belief,
        config.transitionModel,
        config.observationModel,
        config.preferences,
        random,
        config.planningHorizon ?? 1,
        config.precision ?? 1,
        config.habits ?? {},
    );
}
