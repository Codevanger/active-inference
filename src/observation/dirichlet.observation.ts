import { Distribution } from '../models/belief.model';
import {
    IObservationModel,
    ObservationMatrix,
} from '../models/observation.model';
import { ILearnable } from '../models/learnable.model';

/**
 * Dirichlet concentration parameters for observation model.
 * Same structure as ObservationMatrix: observation → state → concentration.
 *
 * Each value is a positive pseudo-count. Higher values encode stronger
 * prior beliefs about the observation-state mapping.
 */
export type ObservationConcentrations<
    O extends string = string,
    S extends string = string,
> = Record<O, Record<S, number>>;

/**
 * Learnable observation model using Dirichlet concentrations.
 *
 * Instead of a fixed A matrix, this model maintains Dirichlet pseudo-counts
 * from which the probability matrix P(o|s) is derived by normalization:
 *
 *   P(o|s) = a[o][s] / Σ_o' a[o'][s]
 *
 * After each observation, concentrations are updated using the posterior
 * belief about states (Dirichlet-categorical conjugate update):
 *
 *   a[o*][s] += Q(s)   for the observed o*
 *
 * @typeParam O - Union type of possible observation names
 * @typeParam S - Union type of possible state names
 *
 * @example
 * ```typescript
 * // Weak prior: agent is uncertain about observation-state mapping
 * const obs = new DirichletObservation({
 *   see_safe:   { safe: 2, danger: 1 },
 *   see_danger: { safe: 1, danger: 2 },
 * });
 *
 * // Strong prior: agent has confident beliefs (equivalent to scale * probabilities)
 * const obs = new DirichletObservation({
 *   see_safe:   { safe: 90, danger: 10 },
 *   see_danger: { safe: 10, danger: 90 },
 * });
 * ```
 */
export class DirichletObservation<
    O extends string = string,
    S extends string = string,
> implements IObservationModel<O, S>, ILearnable
{
    readonly learnable: true = true;

    private _matrix: ObservationMatrix<O, S> | null = null;

    /**
     * @param concentrations - Dirichlet pseudo-counts a[o][s].
     *   Each value must be > 0. Higher values encode stronger prior beliefs.
     */
    constructor(public concentrations: ObservationConcentrations<O, S>) {
        // Deep copy to avoid aliasing
        this.concentrations = {} as ObservationConcentrations<O, S>;
        for (const obs of Object.keys(concentrations) as O[]) {
            this.concentrations[obs] = { ...concentrations[obs] };
        }
    }

    get observations(): O[] {
        return Object.keys(this.concentrations) as O[];
    }

    get states(): S[] {
        const firstObs = this.observations[0];
        return Object.keys(this.concentrations[firstObs] || {}) as S[];
    }

    /**
     * Normalized probability matrix derived from concentrations.
     * Lazily computed and cached; invalidated on learn().
     *
     * Column-wise normalization (per state s):
     *   P(o|s) = a[o][s] / Σ_o' a[o'][s]
     */
    get matrix(): ObservationMatrix<O, S> {
        if (this._matrix === null) {
            this._matrix = this.normalize();
        }
        return this._matrix;
    }

    getLikelihood(observation: O): Distribution<S> {
        return this.matrix[observation] ?? ({} as Distribution<S>);
    }

    probability(observation: O, state: S): number {
        return this.matrix[observation]?.[state] ?? 0;
    }

    /**
     * Update concentrations from an observation and posterior belief.
     *
     * Dirichlet-categorical conjugate update:
     *   a[o*][s] += posteriorBelief[s]   for the observed o*
     *
     * The belief-weighting handles state uncertainty: if the agent
     * is 80% sure it's in state A, the count for (o*, A) increases
     * by 0.8 and (o*, B) by 0.2.
     *
     * @param observation - The observation that was received
     * @param posteriorBelief - Posterior belief distribution over states
     */
    learn(observation: O, posteriorBelief: Distribution<S>): void {
        for (const state of this.states) {
            this.concentrations[observation][state] +=
                posteriorBelief[state] ?? 0;
        }
        this._matrix = null;
    }

    /**
     * Column-wise normalization of concentrations.
     * For each state s: P(o|s) = a[o][s] / Σ_o' a[o'][s]
     */
    private normalize(): ObservationMatrix<O, S> {
        const matrix = {} as ObservationMatrix<O, S>;
        const observations = this.observations;
        const states = this.states;

        const colSums: Record<string, number> = {};
        for (const s of states) {
            colSums[s] = 0;
            for (const o of observations) {
                colSums[s] += this.concentrations[o][s];
            }
        }

        for (const o of observations) {
            matrix[o] = {} as Distribution<S>;
            for (const s of states) {
                matrix[o][s] =
                    colSums[s] > 0
                        ? this.concentrations[o][s] / colSums[s]
                        : 0;
            }
        }

        return matrix;
    }
}
