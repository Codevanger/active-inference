/**
 * A 1-dimensional Gaussian (Normal) belief over a continuous hidden state.
 *
 * Represents the agent's uncertainty about a single real-valued state
 * as a Normal distribution N(μ, σ²).
 *
 * Used in continuous Active Inference where states live on the real line
 * rather than in a finite set.
 *
 * @example
 * ```typescript
 * const belief = new GaussianBelief(2.0, 0.5);
 * console.log(belief.mean);     // 2.0
 * console.log(belief.variance); // 0.5
 * console.log(belief.entropy()); // ≈ 1.07
 * ```
 */
export class GaussianBelief {
    constructor(
        public readonly mean: number,
        public readonly variance: number,
    ) {}

    /**
     * Shannon entropy of the Gaussian in nats.
     * H = ½ log(2πeσ²)
     */
    entropy(): number {
        return 0.5 * Math.log(2 * Math.PI * Math.E * this.variance);
    }

    /**
     * Create an independent copy.
     */
    copy(): GaussianBelief {
        return new GaussianBelief(this.mean, this.variance);
    }
}
