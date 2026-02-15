import { LinearAlgebra, Random } from '../helpers/math.helpers';

/**
 * Prior probability over actions, representing habitual action tendencies.
 * In Active Inference notation, this is the E matrix.
 *
 * @typeParam A - Union type of possible action names
 */
export type Habits<A extends string = string> = Record<A, number>;

/**
 * Active Inference agent implementing the Free Energy Principle.
 *
 * Generic over belief type B, enabling both discrete (categorical)
 * and continuous (Gaussian) Active Inference with a single class.
 *
 * The `computeEFE` callback defines how Expected Free Energy is computed
 * for a predicted belief. For discrete models this is ambiguity + risk;
 * for Gaussian models it is −C(E[y]).
 *
 * @typeParam A - Union type of possible action names
 * @typeParam B - Belief type (must have copy())
 * @typeParam O - Observation type
 *
 * @example
 * ```typescript
 * const agent = createAgent({ belief, transitionModel, observationModel, preferences });
 * const action = agent.step(observation);
 * ```
 */
export class Agent<
    A extends string = string,
    B extends { copy(): B } = any,
    O = any,
> {
    private _belief: B;
    private _random: Random;
    private _planningHorizon: number;
    private _precision: number;
    private _habits: Partial<Habits<A>>;
    private _beamWidth: number;
    private _previousBelief: B | null = null;
    private _previousAction: A | null = null;

    constructor(
        belief: B,
        private transitionModel: { readonly actions: A[]; predict(belief: B, action: A): B },
        private observationModel: { update(belief: B, observation: O): B },
        private _computeEFE: (predicted: B) => number,
        random?: Random,
        planningHorizon: number = 1,
        precision: number = 1,
        habits: Partial<Habits<A>> = {},
        beamWidth: number = 0,
        private _afterObserve?: (
            observation: O,
            belief: B,
            previousAction: A | null,
            previousBelief: B | null,
        ) => void,
    ) {
        this._belief = belief.copy();
        this._random = random ?? new Random();
        this._planningHorizon = Math.max(1, Math.floor(planningHorizon));
        this._precision = Math.max(0, precision);
        this._habits = habits;
        this._beamWidth = Math.max(0, Math.floor(beamWidth));
    }

    /**
     * Current belief (returns a copy).
     */
    get belief(): B {
        return this._belief.copy();
    }

    /**
     * Update beliefs given an observation.
     * Delegates to the observation model.
     */
    observe(observation: O): void {
        this._belief = this.observationModel.update(this._belief, observation);
    }

    /**
     * Replace the agent's belief.
     */
    resetBelief(belief: B): void {
        this._belief = belief.copy();
    }

    /**
     * Select an action by minimizing Expected Free Energy.
     *
     * 1. Enumerate all action sequences up to planningHorizon
     * 2. Sum per-step EFE via the computeEFE callback
     * 3. Softmin → sample first action
     */
    act(): A {
        const actions = this.transitionModel.actions;

        let beams: { policy: A[]; efe: number; belief: B }[] = [];
        for (const action of actions) {
            const predicted = this.transitionModel.predict(this._belief, action);
            const efe = this._computeEFE(predicted);
            beams.push({ policy: [action], efe, belief: predicted });
        }

        for (let depth = 1; depth < this._planningHorizon; depth++) {
            const nextBeams: typeof beams = [];
            for (const beam of beams) {
                for (const action of actions) {
                    const predicted = this.transitionModel.predict(
                        beam.belief,
                        action,
                    );
                    const efe = this._computeEFE(predicted);
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
     * Complete perception-action cycle: observe, learn, act.
     */
    step(observation: O): A {
        this.observe(observation);
        this._afterObserve?.(
            observation,
            this._belief,
            this._previousAction,
            this._previousBelief,
        );

        const action = this.act();

        this._previousBelief = this._belief;
        this._previousAction = action;

        return action;
    }

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

    private getPolicyHabit(policy: A[]): number {
        if (Object.keys(this._habits).length === 0) {
            return 1;
        }
        return policy.reduce((prior, action) => {
            return prior * (this._habits[action] ?? 1);
        }, 1);
    }
}
