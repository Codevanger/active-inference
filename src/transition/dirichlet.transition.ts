import { Distribution } from '../models/belief.model';
import type { Belief } from '../models/belief.model';
import { ITransitionModel, TransitionMatrix } from '../models/transition.model';
import { DiscreteBelief } from '../beliefs/discrete.belief';

/**
 * Dirichlet concentration parameters for transition model.
 * Same structure as TransitionMatrix: action → current_state → next_state → concentration.
 */
export type TransitionConcentrations<
    A extends string = string,
    S extends string = string,
> = Record<A, Record<S, Record<S, number>>>;

/**
 * Learnable transition model using Dirichlet concentrations.
 *
 * Instead of a fixed B matrix, this model maintains Dirichlet pseudo-counts
 * from which the probability matrix P(s'|s,a) is derived by normalization:
 *
 *   P(s'|s,a) = b[a][s][s'] / Σ_s'' b[a][s][s'']
 *
 * After each state transition, concentrations are updated using the
 * outer product of prior and posterior beliefs:
 *
 *   b[a][s][s'] += Q_prior(s) × Q_posterior(s')
 *
 * @typeParam A - Union type of possible action names
 * @typeParam S - Union type of possible state names
 *
 * @example
 * ```typescript
 * const transition = new DirichletTransition({
 *   move: {
 *     here: { here: 1, there: 5 },  // move from here → likely end up there
 *     there: { here: 1, there: 5 },
 *   },
 *   stay: {
 *     here: { here: 5, there: 1 },
 *     there: { here: 1, there: 5 },
 *   },
 * });
 * ```
 */
export class DirichletTransition<
    A extends string = string,
    S extends string = string,
> implements ITransitionModel<A, S>
{

    private _matrix: TransitionMatrix<A, S> | null = null;

    /**
     * @param concentrations - Dirichlet pseudo-counts b[a][s][s'].
     *   Structure: action → current_state → next_state → count.
     *   Each value must be > 0.
     */
    constructor(public concentrations: TransitionConcentrations<A, S>) {
        // Deep copy to avoid aliasing
        this.concentrations = {} as TransitionConcentrations<A, S>;
        for (const a of Object.keys(concentrations) as A[]) {
            this.concentrations[a] = {} as Record<S, Record<S, number>>;
            for (const s of Object.keys(concentrations[a]) as S[]) {
                this.concentrations[a][s] = { ...concentrations[a][s] };
            }
        }
    }

    get actions(): A[] {
        return Object.keys(this.concentrations) as A[];
    }

    get states(): S[] {
        const firstAction = this.actions[0];
        return Object.keys(this.concentrations[firstAction] || {}) as S[];
    }

    /**
     * Normalized probability matrix derived from concentrations.
     * Lazily computed and cached; invalidated on learn().
     *
     * Row-wise normalization (per action a, per current state s):
     *   P(s'|s,a) = b[a][s][s'] / Σ_s'' b[a][s][s'']
     */
    get matrix(): TransitionMatrix<A, S> {
        if (this._matrix === null) {
            this._matrix = this.normalize();
        }
        return this._matrix;
    }

    getTransition(state: S, action: A): Distribution<S> {
        return this.matrix[action]?.[state] ?? ({} as Distribution<S>);
    }

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

    /**
     * Update concentrations from a state transition.
     *
     * Uses outer product of prior and posterior beliefs:
     *   b[a][s][s'] += Q_prior(s) × Q_posterior(s')
     *
     * This encodes the agent's best estimate of the transition
     * that occurred, weighted by uncertainty in both states.
     *
     * @param action - The action that was taken
     * @param priorBelief - Belief distribution before the action
     * @param posteriorBelief - Belief distribution after observing the outcome
     */
    learn(
        action: A,
        priorBelief: Distribution<S>,
        posteriorBelief: Distribution<S>,
    ): void {
        for (const s of this.states) {
            for (const sPrime of this.states) {
                this.concentrations[action][s][sPrime] +=
                    (priorBelief[s] ?? 0) * (posteriorBelief[sPrime] ?? 0);
            }
        }
        this._matrix = null;
    }

    /**
     * Row-wise normalization of concentrations.
     * For each (action, current_state):
     *   P(s'|s,a) = b[a][s][s'] / Σ_s'' b[a][s][s'']
     */
    private normalize(): TransitionMatrix<A, S> {
        const matrix = {} as TransitionMatrix<A, S>;

        for (const a of this.actions) {
            matrix[a] = {} as Record<S, Distribution<S>>;
            for (const s of this.states) {
                matrix[a][s] = {} as Distribution<S>;
                let rowSum = 0;
                for (const sPrime of this.states) {
                    rowSum += this.concentrations[a][s][sPrime];
                }
                for (const sPrime of this.states) {
                    matrix[a][s][sPrime] =
                        rowSum > 0
                            ? this.concentrations[a][s][sPrime] / rowSum
                            : 0;
                }
            }
        }

        return matrix;
    }
}
