import { GaussianBelief } from '../beliefs/gaussian.belief';
import { GaussianTransition } from '../transition/gaussian.transition';
import { GaussianObservation } from '../observation/gaussian.observation';
import { LinearAlgebra, Random } from '../helpers/math.helpers';

/**
 * Preference function over predicted observation means.
 * Returns a log-preference (0 = neutral, negative = undesired).
 *
 * This generalises the standard Gaussian preference C(o) = −½(o−o*)²/σ²
 * to arbitrary shapes, enabling asymmetric costs like in the Trolley Problem.
 */
export type GaussianPreferenceFn = (mean: number) => number;

/**
 * Continuous Active Inference agent with Gaussian beliefs.
 *
 * Maintains a 1-D Gaussian belief N(μ, σ²) over a hidden continuous state,
 * selects among discrete actions by minimising Expected Free Energy via
 * beam search (same algorithm as the discrete Agent), and updates beliefs
 * using a Kalman filter.
 *
 * ## EFE Decomposition
 *
 * - **Risk**: −C(E[y]) where C is the preference function and E[y] is
 *   the predicted observation mean.
 * - **Ambiguity**: ½ log(2πe · R) — constant for fixed observation noise,
 *   so it does not affect action selection.
 *
 * @example
 * ```typescript
 * const agent = new GaussianAgent(
 *   new GaussianBelief(1, 0.1),
 *   transition,
 *   observation,
 *   (mean) => mean >= 0 ? 0 : -mean * mean,
 *   new Random(42),
 *   3,  // planningHorizon
 *   4,  // precision
 * );
 *
 * const action = agent.act(); // 'pull' or 'pass'
 * ```
 */
export class GaussianAgent<A extends string = string> {
    private _belief: GaussianBelief;
    private _random: Random;
    private _planningHorizon: number;
    private _precision: number;

    constructor(
        belief: GaussianBelief,
        private transitionModel: GaussianTransition<A>,
        private observationModel: GaussianObservation,
        private preferences: GaussianPreferenceFn,
        random?: Random,
        planningHorizon: number = 1,
        precision: number = 1,
    ) {
        this._belief = belief.copy();
        this._random = random ?? new Random();
        this._planningHorizon = Math.max(1, Math.floor(planningHorizon));
        this._precision = Math.max(0, precision);
    }

    /**
     * Most-likely hidden state (belief mean).
     */
    get state(): number {
        return this._belief.mean;
    }

    /**
     * Uncertainty in the agent's belief (variance).
     */
    get uncertainty(): number {
        return this._belief.variance;
    }

    /**
     * Update beliefs given a scalar observation (Kalman filter).
     */
    observe(observation: number): void {
        this._belief = this.observationModel.update(this._belief, observation);
    }

    /**
     * Replace the agent's belief.
     */
    resetBelief(belief: GaussianBelief): void {
        this._belief = belief.copy();
    }

    /**
     * Export current belief as { mean, variance }.
     */
    exportBelief(): { mean: number; variance: number } {
        return { mean: this._belief.mean, variance: this._belief.variance };
    }

    /**
     * Select an action by minimising Expected Free Energy.
     *
     * Uses the same beam-search policy enumeration as the discrete Agent:
     * 1. Enumerate all action sequences up to planningHorizon
     * 2. Sum per-step EFE (risk only — ambiguity is constant)
     * 3. Softmin → sample first action
     */
    act(): A {
        const actions = this.transitionModel.actions;

        let beams: { policy: A[]; efe: number; belief: GaussianBelief }[] = [];
        for (const action of actions) {
            const predicted = this.transitionModel.predict(
                this._belief,
                action,
            );
            const efe = this.computeRisk(predicted);
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
                    const efe = this.computeRisk(predicted);
                    nextBeams.push({
                        policy: [...beam.policy, action],
                        efe: beam.efe + efe,
                        belief: predicted,
                    });
                }
            }
            beams = nextBeams;
        }

        const policyEFEs = beams.map((b) => b.efe);
        const policyProbs = LinearAlgebra.softmin(policyEFEs, this._precision);

        const idx = this.sampleIndex(policyProbs);
        return beams[idx].policy[0];
    }

    /**
     * Observe then act (one full cycle).
     */
    step(observation: number): A {
        this.observe(observation);
        return this.act();
    }

    /**
     * Risk = −C(E[y]) where C is the preference function.
     */
    private computeRisk(predicted: GaussianBelief): number {
        const obsMean =
            this.observationModel.expectedObservation(predicted);
        return -this.preferences(obsMean);
    }

    private sampleIndex(probs: number[]): number {
        const rand = this._random.next();
        let cumulative = 0;
        for (let i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (rand < cumulative) return i;
        }
        return probs.length - 1;
    }
}
