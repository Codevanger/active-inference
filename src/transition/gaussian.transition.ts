import { GaussianBelief } from '../beliefs/gaussian.belief';

/**
 * Configuration for a single action's effect on continuous state.
 *
 * @param fn - Deterministic transition function: μ' = fn(μ)
 * @param noise - Process noise variance added after transition
 */
export interface GaussianActionModel {
    fn: (x: number) => number;
    noise: number;
}

/**
 * Maps each discrete action to its continuous state transition.
 */
export type GaussianTransitionConfig<A extends string = string> = Record<
    A,
    GaussianActionModel
>;

/**
 * State transition model for continuous (Gaussian) Active Inference
 * with discrete actions.
 *
 * Each action maps to a transition function and process noise.
 * Belief propagation uses first-order approximation:
 *   μ' = fn(μ)
 *   σ²' = fn'(μ)² · σ² + noise
 *
 * where fn'(μ) is approximated via finite differences.
 *
 * @example
 * ```typescript
 * const transition = new GaussianTransition({
 *   accelerate: { fn: x => x + 1, noise: 0.1 },
 *   brake:      { fn: x => x * 0.5, noise: 0.05 },
 * });
 *
 * const predicted = transition.predict(belief, 'accelerate');
 * ```
 */
export class GaussianTransition<A extends string = string> {
    private readonly config: GaussianTransitionConfig<A>;
    private readonly _actions: A[];

    constructor(config: GaussianTransitionConfig<A>) {
        this.config = config;
        this._actions = Object.keys(config) as A[];
    }

    get actions(): A[] {
        return this._actions;
    }

    /**
     * Predict belief after taking an action.
     *
     * Uses first-order (Extended Kalman) approximation for
     * nonlinear transition functions.
     */
    predict(belief: GaussianBelief, action: A): GaussianBelief {
        const model = this.config[action];
        const newMean = model.fn(belief.mean);

        // Numerical derivative via central finite difference
        const h = 1e-5;
        const derivative =
            (model.fn(belief.mean + h) - model.fn(belief.mean - h)) / (2 * h);

        const newVariance =
            derivative * derivative * belief.variance + model.noise;

        return new GaussianBelief(newMean, newVariance);
    }
}
