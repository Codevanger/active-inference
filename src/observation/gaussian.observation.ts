import { GaussianBelief } from '../beliefs/gaussian.belief';

/**
 * Configuration for a linear Gaussian observation model.
 *
 * @param scale - Observation gain: y = scale · x + bias + noise
 * @param bias - Constant offset (default 0)
 * @param noise - Observation noise variance
 */
export interface GaussianObservationConfig {
    scale: number;
    bias?: number;
    noise: number;
}

/**
 * Linear Gaussian observation model: y = C·x + b + ε, ε ~ N(0, R).
 *
 * Provides Kalman-filter belief updates and predicted observation
 * statistics for Expected Free Energy computation.
 *
 * @example
 * ```typescript
 * const obs = new GaussianObservation({ scale: 1, noise: 0.1 });
 *
 * // Kalman update
 * const posterior = obs.update(prior, 3.2);
 *
 * // Predicted observation for EFE
 * const expectedObs = obs.expectedObservation(belief);
 * ```
 */
export class GaussianObservation {
    readonly scale: number;
    readonly bias: number;
    readonly noise: number;

    constructor(config: GaussianObservationConfig) {
        this.scale = config.scale;
        this.bias = config.bias ?? 0;
        this.noise = config.noise;
    }

    /**
     * Kalman-filter update: incorporate a scalar observation.
     *
     * K = σ² · c / (c² · σ² + R)
     * μ_post = μ + K · (y − c·μ − b)
     * σ²_post = (1 − K·c) · σ²
     */
    update(belief: GaussianBelief, observation: number): GaussianBelief {
        const c = this.scale;
        const predictedObs = c * belief.mean + this.bias;
        const S = c * c * belief.variance + this.noise;
        const K = (c * belief.variance) / S;

        return new GaussianBelief(
            belief.mean + K * (observation - predictedObs),
            (1 - K * c) * belief.variance,
        );
    }

    /**
     * Expected observation mean given a belief: E[y] = c·μ + b.
     */
    expectedObservation(belief: GaussianBelief): number {
        return this.scale * belief.mean + this.bias;
    }
}
