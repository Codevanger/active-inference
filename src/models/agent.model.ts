import { Belief, Distribution, Preferences } from './belief.model';
import { ITransitionModel } from './transition.model';
import { IObservationModel } from './observation.model';
import { LinearAlgebra, Random } from '../helpers/math.helpers';
import { isLearnable } from './learnable.model';
import type { DirichletObservation } from '../observation/dirichlet.observation';
import type { DirichletTransition } from '../transition/dirichlet.transition';

/**
 * Prior probability over actions, representing habitual action tendencies.
 * In Active Inference notation, this is the E matrix.
 *
 * Habits bias action selection independently of Expected Free Energy.
 * They model learned action preferences or "default" behaviors.
 *
 * @typeParam A - Union type of possible action names
 *
 * @example
 * ```typescript
 * // Agent has a habit of staying still
 * const habits: Habits<'move' | 'stay'> = {
 *   move: 0.3,
 *   stay: 0.7
 * };
 * ```
 */
export type Habits<A extends string = string> = Record<A, number>;

/**
 * Active Inference agent implementing the Free Energy Principle.
 *
 * This agent perceives the world through observations, maintains beliefs
 * about hidden states, and selects actions to minimize Expected Free Energy.
 *
 * ## Core Concepts
 *
 * **Generative Model**: The agent has an internal model of how the world works:
 * - Transition model (B): How states change given actions
 * - Observation model (A): How states generate observations
 * - Preferences (C): What observations the agent "wants" to experience
 *
 * **Perception**: When the agent observes something, it updates its beliefs
 * using Bayesian inference: P(state|obs) ∝ P(obs|state) × P(state)
 *
 * **Action Selection**: The agent evaluates possible action sequences (policies)
 * by computing Expected Free Energy, which balances:
 * - **Risk**: Avoiding unpreferred observations
 * - **Ambiguity**: Seeking informative states
 *
 * ## Key Parameters
 *
 * - `planningHorizon`: How many steps ahead to plan (1 = greedy/reactive)
 * - `precision` (β): Temperature for action selection (higher = more deterministic)
 * - `habits` (E): Prior preferences over actions independent of goals
 *
 * @typeParam A - Union type of possible action names
 * @typeParam O - Union type of possible observation names
 * @typeParam S - Union type of possible state names
 *
 * @example
 * ```typescript
 * const agent = createAgent({
 *   belief: new DiscreteBelief({ safe: 0.5, danger: 0.5 }),
 *   transitionModel: myTransitions,
 *   observationModel: myObservations,
 *   preferences: { good: 0, bad: -5 },
 *   planningHorizon: 2,
 *   precision: 4
 * });
 *
 * // Perception-action loop
 * while (running) {
 *   const observation = environment.getObservation();
 *   const action = agent.step(observation);
 *   environment.execute(action);
 * }
 * ```
 *
 * @see {@link https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf | Friston (2010) - The Free Energy Principle}
 */
export class Agent<
    A extends string = string,
    O extends string = string,
    S extends string = string,
> {
    private _belief: Belief<S>;
    private _random: Random;
    private _planningHorizon: number;
    private _precision: number;
    private _habits: Partial<Habits<A>>;
    private _previousBelief: Belief<S> | null = null;
    private _previousAction: A | null = null;

    /**
     * Create a new Active Inference agent.
     *
     * @param belief - Initial belief over hidden states
     * @param transitionModel - Model of state transitions P(s'|s,a)
     * @param observationModel - Model of observations P(o|s)
     * @param preferences - Preferred observations (log probabilities)
     * @param random - Random number generator (optional, for reproducibility)
     * @param planningHorizon - Steps to plan ahead (default: 1)
     * @param precision - Action selection temperature (default: 1)
     * @param habits - Prior over actions (default: uniform)
     */
    constructor(
        belief: Belief<S>,
        private transitionModel: ITransitionModel<A, S>,
        private observationModel: IObservationModel<O, S>,
        private preferences: Preferences<O>,
        random?: Random,
        planningHorizon: number = 1,
        precision: number = 1,
        habits: Partial<Habits<A>> = {},
    ) {
        this._belief = belief.copy();
        this._random = random ?? new Random();
        this._planningHorizon = Math.max(1, Math.floor(planningHorizon));
        this._precision = Math.max(0, precision);
        this._habits = habits;
    }

    private get resolvedPreferences(): Preferences<O> {
        return this.preferences;
    }

    /**
     * Most likely hidden state (Maximum A Posteriori estimate).
     */
    get state(): S {
        return this._belief.argmax();
    }

    /**
     * Uncertainty in the agent's beliefs (Shannon entropy in nats).
     * Higher values indicate more uncertainty about the current state.
     */
    get uncertainty(): number {
        return this._belief.entropy();
    }

    /**
     * Update beliefs based on a new observation (perception).
     *
     * Performs Bayesian inference:
     * posterior ∝ likelihood × prior
     * P(s|o) ∝ P(o|s) × P(s)
     *
     * This is the "perception" step of the Active Inference loop,
     * where the agent updates its model of the world state.
     *
     * @param observation - The observation received from the environment
     *
     * @example
     * ```typescript
     * agent.observe('see_reward');
     * console.log(agent.belief.argmax()); // Most likely state after observation
     * ```
     */
    observe(observation: O): void {
        const likelihood = this.observationModel.getLikelihood(observation);
        this._belief = this._belief.update(likelihood);
    }

    /**
     * Select an action by minimizing Expected Free Energy.
     *
     * The agent:
     * 1. Generates all possible policies (action sequences) up to the planning horizon
     * 2. Evaluates each policy's Expected Free Energy: G(π) = ambiguity + risk
     * 3. Computes policy probabilities: P(π) ∝ E(π) × exp(-β × G(π))
     * 4. Samples a policy and returns its first action
     *
     * Expected Free Energy (G) combines:
     * - **Ambiguity**: Expected uncertainty about observations (epistemic)
     * - **Risk**: Expected deviation from preferred observations (pragmatic)
     *
     * @returns The selected action to execute
     *
     * @example
     * ```typescript
     * const action = agent.act();
     * environment.execute(action);
     * ```
     */
    act(): A {
        const policies = this.generatePolicies(this._planningHorizon);

        const policyEFEs: number[] = [];
        for (const policy of policies) {
            policyEFEs.push(this.evaluatePolicy(policy));
        }

        let policyProbs = LinearAlgebra.softmin(policyEFEs, this._precision);

        if (Object.keys(this._habits).length > 0) {
            const combined = policyProbs.map((p, i) => {
                return p * this.getPolicyHabit(policies[i]);
            });
            policyProbs = LinearAlgebra.normalize(combined);
        }

        const idx = this.sampleIndex(policyProbs);
        return policies[idx][0];
    }

    /**
     * Complete perception-action cycle: observe then act.
     *
     * Convenience method that combines observe() and act() into
     * a single call, representing one full cycle of the Active
     * Inference loop.
     *
     * @param observation - The observation received from the environment
     * @returns The selected action to execute
     *
     * @example
     * ```typescript
     * // Main loop
     * let obs = environment.reset();
     * while (!done) {
     *   const action = agent.step(obs);
     *   obs = environment.execute(action);
     * }
     * ```
     */
    step(observation: O): A {
        this.observe(observation);
        this.updateModels(observation);

        const action = this.act();

        this._previousBelief = this._belief;
        this._previousAction = action;

        return action;
    }

    /**
     * Export current belief as a plain object for serialization.
     *
     * Useful for:
     * - Saving agent state to storage
     * - Transferring beliefs between agents
     * - Debugging/visualization
     *
     * @returns Plain object mapping states to probabilities
     *
     * @example
     * ```typescript
     * const saved = agent.exportBelief();
     * localStorage.setItem('belief', JSON.stringify(saved));
     *
     * // Later: restore
     * const loaded = JSON.parse(localStorage.getItem('belief'));
     * const newAgent = createAgent({
     *   belief: new DiscreteBelief(loaded),
     *   // ... other config
     * });
     * ```
     */
    exportBelief(): Distribution<S> {
        const result: Distribution<S> = {} as Distribution<S>;
        for (const state of this._belief.states) {
            result[state] = this._belief.probability(state);
        }
        return result;
    }

    /**
     * Variational Free Energy of the current belief state.
     *
     * F = -H(Q) + E_Q[H(o|s)]
     *   = negative_entropy + ambiguity
     *
     * This is a measure of "surprise" or model-data mismatch.
     * The Free Energy Principle states that agents act to minimize
     * this quantity over time.
     *
     * Note: This is VFE (perception), not EFE (action selection).
     *
     * @returns Variational Free Energy (can be negative)
     */
    get freeEnergy(): number {
        return -this._belief.entropy() + this.computeAmbiguity(this._belief);
    }

    /**
     * Update learnable models from the current observation and belief.
     *
     * Called after observe() so the posterior belief is available.
     * - A-learning: update observation model with (observation, posterior)
     * - B-learning: update transition model with (previous_action, previous_belief, posterior)
     */
    private updateModels(observation: O): void {
        const posteriorDist = this.exportBelief();

        // A-matrix: P(o|s)
        if (isLearnable(this.observationModel)) {
            (this.observationModel as unknown as DirichletObservation<O, S>).learn(
                observation,
                posteriorDist,
            );
        }

        // B-matrix: P(s'|s,a) — only after at least one action
        if (
            isLearnable(this.transitionModel) &&
            this._previousAction !== null &&
            this._previousBelief !== null
        ) {
            const prevDist = {} as Distribution<S>;
            for (const state of this._previousBelief.states) {
                prevDist[state] = this._previousBelief.probability(state);
            }

            (this.transitionModel as unknown as DirichletTransition<A, S>).learn(
                this._previousAction,
                prevDist,
                posteriorDist,
            );
        }
    }

    /**
     * Generate all possible policies (action sequences) of given depth.
     * For depth=2 with actions [a,b]: [[a,a], [a,b], [b,a], [b,b]]
     */
    private generatePolicies(depth: number): A[][] {
        const actions = this.transitionModel.actions;
        if (depth <= 1) {
            return actions.map((a) => [a]);
        }
        const policies: A[][] = [];
        const subPolicies = this.generatePolicies(depth - 1);
        for (const action of actions) {
            for (const sub of subPolicies) {
                policies.push([action, ...sub]);
            }
        }
        return policies;
    }

    /**
     * Evaluate a policy by computing its Expected Free Energy.
     * G(π) = Σ_τ G(a_τ | Q_τ) where Q_τ is the predicted belief at time τ
     */
    private evaluatePolicy(policy: A[]): number {
        let totalEFE = 0;
        let currentBelief = this._belief;
        for (const action of policy) {
            const predicted = this.transitionModel.predict(currentBelief, action);
            totalEFE += this.computeAmbiguity(predicted) + this.computeRisk(predicted);
            currentBelief = predicted;
        }
        return totalEFE;
    }

    /**
     * Compute ambiguity term of Expected Free Energy.
     *
     * Ambiguity = E_Q[H(o|s)] = -Σ_s Q(s) Σ_o P(o|s) log P(o|s)
     *
     * High ambiguity means the agent is uncertain about what
     * observations to expect - the state-observation mapping is noisy.
     * Minimizing ambiguity drives epistemic/exploratory behavior.
     *
     * @param predictedBelief - Predicted belief state
     * @returns Ambiguity (non-negative)
     */
    private computeAmbiguity(predictedBelief: Belief<S>): number {
        let ambiguity = 0;

        for (const state of predictedBelief.states) {
            const stateProb = predictedBelief.probability(state);

            for (const obs of this.observationModel.observations) {
                const obsProb = this.observationModel.probability(obs, state);

                if (obsProb > 0 && stateProb > 0) {
                    ambiguity -= stateProb * obsProb * Math.log(obsProb);
                }
            }
        }

        return ambiguity;
    }

    /**
     * Compute risk term of Expected Free Energy.
     *
     * Risk = -E_Q[log P(o)] = -Σ_o Q(o) log C(o)
     * where Q(o) = Σ_s P(o|s)Q(s) and C(o) = preferred observations
     *
     * High risk means expected observations are far from preferences.
     * Minimizing risk drives pragmatic/goal-directed behavior.
     *
     * @param predictedBelief - Predicted belief state
     * @returns Risk (higher = worse outcomes expected)
     */
    private computeRisk(predictedBelief: Belief<S>): number {
        let risk = 0;
        const prefs = this.resolvedPreferences;

        for (const obs of this.observationModel.observations) {
            let expectedObsProb = 0;

            for (const state of predictedBelief.states) {
                expectedObsProb +=
                    this.observationModel.probability(obs, state) *
                    predictedBelief.probability(state);
            }

            const preferredLogProb = prefs[obs] ?? -10;

            if (expectedObsProb > 0) {
                risk -= expectedObsProb * preferredLogProb;
            }
        }

        return risk;
    }

    /**
     * Sample an index from a probability distribution.
     */
    private sampleIndex(probs: number[]): number {
        const rand = this._random.next();
        let cumulative = 0;

        for (let i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (rand < cumulative) {
                return i;
            }
        }

        return probs.length - 1;
    }

    /**
     * Get habit prior for a policy (product of action habits).
     */
    private getPolicyHabit(policy: A[]): number {
        if (Object.keys(this._habits).length === 0) {
            return 1;
        }
        return policy.reduce((prior, action) => {
            return prior * (this._habits[action] ?? 1);
        }, 1);
    }
}
