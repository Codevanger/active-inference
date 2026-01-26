/**
 * Seeded pseudo-random number generator using the Mulberry32 algorithm.
 *
 * Provides reproducible random sequences when initialized with the same seed.
 * This is essential for testing and debugging Active Inference agents,
 * as it allows deterministic replay of stochastic action selection.
 *
 * Mulberry32 is a fast, high-quality 32-bit PRNG with a period of 2^32.
 *
 * @example
 * ```typescript
 * const rng = new Random(42);
 * console.log(rng.next()); // 0.8817... (always same for seed 42)
 * console.log(rng.next()); // 0.3951...
 *
 * // Reset to get same sequence again
 * rng.reset(42);
 * console.log(rng.next()); // 0.8817... (same as before)
 * ```
 */
export class Random {
    private state: number;

    /**
     * Create a new random number generator.
     *
     * @param seed - Initial seed value. Defaults to current timestamp
     *               for non-reproducible behavior.
     */
    constructor(seed: number = Date.now()) {
        this.state = seed;
    }

    /**
     * Generate the next random number in [0, 1).
     *
     * Works like Math.random() but with deterministic sequence
     * based on the seed.
     *
     * @returns Random number between 0 (inclusive) and 1 (exclusive)
     */
    next(): number {
        this.state |= 0;
        this.state = (this.state + 0x6d2b79f5) | 0;
        let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }

    /**
     * Generate a random number from standard normal distribution N(0, 1).
     *
     * Uses the Box-Muller transform to convert uniform random numbers
     * into normally distributed values.
     *
     * @returns Random number from standard normal distribution
     *
     * @example
     * ```typescript
     * const rng = new Random(42);
     * const samples = Array.from({ length: 1000 }, () => rng.gaussian());
     * // samples will have mean ≈ 0 and std ≈ 1
     * ```
     */
    gaussian(): number {
        const u1 = this.next();
        const u2 = this.next();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    /**
     * Reset the generator to a new seed.
     *
     * Useful for replaying random sequences from a known state.
     *
     * @param seed - New seed value
     */
    reset(seed: number): void {
        this.state = seed;
    }
}

/**
 * Linear algebra utilities for Active Inference computations.
 *
 * Provides common operations needed for probability manipulation,
 * including softmax/softmin for converting values to probabilities.
 */
export class LinearAlgebra {
    /**
     * Softmin function - converts values to probabilities inversely.
     *
     * Lower values get higher probabilities. Used in Active Inference
     * to convert Expected Free Energy (where lower is better) into
     * action probabilities.
     *
     * Formula: P(i) = exp(-β × x_i) / Σ exp(-β × x_j)
     *
     * @param arr - Array of values to convert
     * @param beta - Precision/temperature parameter (default: 1).
     *               Higher β = more deterministic (concentrates on minimum).
     *               β = 0 gives uniform distribution.
     * @returns Normalized probability distribution
     *
     * @example
     * ```typescript
     * const efe = [2.5, 1.0, 3.0];  // Policy EFEs (lower is better)
     * const probs = LinearAlgebra.softmin(efe, 4);
     * // probs ≈ [0.05, 0.93, 0.02] - strongly prefers policy with EFE=1.0
     * ```
     */
    static softmin(arr: Array<number>, beta: number = 1): Array<number> {
        return this.softmax(arr.map((x) => -x), beta);
    }

    /**
     * Softmax function - converts values to probabilities.
     *
     * Higher values get higher probabilities. Standard operation for
     * converting logits or preferences into a probability distribution.
     *
     * Formula: P(i) = exp(β × x_i) / Σ exp(β × x_j)
     *
     * Uses numerical stability trick of subtracting max before exp.
     *
     * @param arr - Array of values to convert
     * @param beta - Precision/temperature parameter (default: 1).
     *               Higher β = more deterministic (concentrates on maximum).
     *               β = 0 gives uniform distribution.
     * @returns Normalized probability distribution
     *
     * @example
     * ```typescript
     * const logits = [1.0, 2.0, 0.5];
     * const probs = LinearAlgebra.softmax(logits);
     * // probs ≈ [0.24, 0.67, 0.09]
     * ```
     */
    static softmax(arr: Array<number>, beta: number = 1): Array<number> {
        const max = Math.max(...arr);
        const exp = arr.map((x) => Math.exp(beta * (x - max)));
        const sum = exp.reduce((a, b) => a + b);

        return exp.map((x) => x / sum);
    }

    /**
     * Normalize an array to sum to 1.
     *
     * @param arr - Array of non-negative values
     * @returns Normalized array summing to 1
     *
     * @example
     * ```typescript
     * LinearAlgebra.normalize([2, 3, 5]); // [0.2, 0.3, 0.5]
     * ```
     */
    static normalize(arr: number[]): number[] {
        const sum = arr.reduce((a, b) => a + b, 0);
        return arr.map((x) => x / sum);
    }

    /**
     * Compute dot product of two arrays.
     *
     * @param a - First array
     * @param b - Second array (must be same length as a)
     * @returns Sum of element-wise products
     *
     * @example
     * ```typescript
     * LinearAlgebra.dotProduct([1, 2, 3], [4, 5, 6]); // 32
     * ```
     */
    static dotProduct(a: number[], b: number[]): number {
        return a.reduce((sum, x, i) => sum + x * b[i], 0);
    }
}
