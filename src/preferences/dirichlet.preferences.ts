import { ILearnable } from '../models/learnable.model';

/**
 * Dirichlet concentration parameters for preferences.
 * Maps each observation to a positive pseudo-count.
 */
export type PreferenceConcentrations<O extends string = string> = Record<
    O,
    number
>;

/**
 * Learnable preferences using Dirichlet concentrations.
 *
 * Instead of fixed log-probability preferences, this class maintains
 * Dirichlet pseudo-counts from which log-preferences are derived:
 *
 *   P_preferred(o) = c[o] / Σ_o' c[o']
 *   preferences[o] = log(P_preferred(o))
 *
 * Higher concentration → higher preferred probability → less negative log → more preferred.
 *
 * After each observation, concentrations can be reinforced:
 *   c[o*] += 1
 *
 * This allows preferences to drift based on experience.
 *
 * @typeParam O - Union type of possible observation names
 *
 * @example
 * ```typescript
 * // Strongly prefer reward over no_reward
 * const prefs = new DirichletPreferences({
 *   reward: 10,      // log(10/11) ≈ -0.095
 *   no_reward: 1,    // log(1/11) ≈ -2.398
 * });
 * ```
 */
export class DirichletPreferences<O extends string = string>
    implements ILearnable
{
    readonly learnable: true = true;

    private _preferences: Record<O, number> | null = null;

    /**
     * @param concentrations - Pseudo-counts c[o] for each observation.
     *   Higher values indicate more preferred observations.
     *   All values must be > 0.
     */
    constructor(public concentrations: PreferenceConcentrations<O>) {
        this.concentrations = { ...concentrations };
    }

    get observations(): O[] {
        return Object.keys(this.concentrations) as O[];
    }

    /**
     * Log-preference values derived from concentrations.
     *
     *   P(o) = c[o] / Σ_o' c[o']
     *   preferences[o] = log(P(o))
     *
     * Returns a plain Record<O, number> compatible with Preferences<O>.
     */
    get preferences(): Record<O, number> {
        if (this._preferences === null) {
            this._preferences = this.normalize();
        }
        return this._preferences;
    }

    /**
     * Reinforce an observation's preference.
     *
     * @param observation - The observation to reinforce
     * @param amount - Amount to add to the concentration (default: 1)
     */
    learn(observation: O, amount: number = 1): void {
        this.concentrations[observation] += amount;
        this._preferences = null;
    }

    private normalize(): Record<O, number> {
        const result = {} as Record<O, number>;
        let sum = 0;

        for (const o of this.observations) {
            sum += this.concentrations[o];
        }

        for (const o of this.observations) {
            const p = this.concentrations[o] / sum;
            result[o] = Math.log(Math.max(p, 1e-16));
        }

        return result;
    }
}
