import { Belief, Distribution, Preferences } from './belief.model';
import { ITransitionModel } from './transition.model';
import { IObservationModel } from './observation.model';
import { LinearAlgebra, Random } from '../helpers/math.helpers';

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
    private _beamWidth: number;
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
     * @param beamWidth - Max policies to keep at each planning depth (default: 0 = no limit)
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
        beamWidth: number = 0,
    ) {
        this._belief = belief.copy();
        this._random = random ?? new Random();
        this._planningHorizon = Math.max(1, Math.floor(planningHorizon));
        this._precision = Math.max(0, precision);
        this._habits = habits;
        this._beamWidth = Math.max(0, Math.floor(beamWidth));
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
     * Replace the agent's belief with a new distribution.
     *
     * Useful for fully observable environments where the state
     * is known directly from the observation.
     *
     * @param belief - New belief distribution
     */
    resetBelief(belief: Belief<S>): void {
        this._belief = belief.copy();
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
        const actions = this.transitionModel.actions;

        // Initialize beams: one per action
        let beams: { policy: A[]; efe: number; belief: Belief<S> }[] = [];
        for (const action of actions) {
            const predicted = this.transitionModel.predict(this._belief, action);
            const efe = this.computeAmbiguity(predicted) + this.computeRisk(predicted);
            beams.push({ policy: [action], efe, belief: predicted });
        }

        // Extend beams for remaining horizon steps
        for (let depth = 1; depth < this._planningHorizon; depth++) {
            const nextBeams: typeof beams = [];
            for (const beam of beams) {
                for (const action of actions) {
                    const predicted = this.transitionModel.predict(beam.belief, action);
                    const efe = this.computeAmbiguity(predicted) + this.computeRisk(predicted);
                    nextBeams.push({
                        policy: [...beam.policy, action],
                        efe: beam.efe + efe,
                        belief: predicted,
                    });
                }
            }

            if (this._beamWidth > 0 && nextBeams.length > this._beamWidth) {
                nextBeams.sort((a, b) => a.efe - b.efe);
                beams = nextBeams.slice(0, this._beamWidth);
            } else {
                beams = nextBeams;
            }
        }

        const policies = beams.map((b) => b.policy);
        const policyEFEs = beams.map((b) => b.efe);

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

        this.observationModel.learn?.(observation, posteriorDist);

        if (this._previousAction !== null && this._previousBelief !== null) {
            const prevDist = {} as Distribution<S>;
            for (const state of this._previousBelief.states) {
                prevDist[state] = this._previousBelief.probability(state);
            }

            if (this.transitionModel.learn) {
                this.transitionModel.learn(this._previousAction, prevDist, posteriorDist);
            }
        }
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
