import { Belief, Distribution, Preferences } from './models/belief.model';
import { ITransitionModel } from './models/transition.model';
import { IObservationModel } from './models/observation.model';
import { Agent, Habits } from './models/agent.model';
import { GaussianBelief } from './beliefs/gaussian.belief';
import { GaussianTransition } from './transition/gaussian.transition';
import { GaussianObservation } from './observation/gaussian.observation';
import { Random } from './helpers/math.helpers';

/**
 * Preference function over predicted observation means (Gaussian models).
 * Returns a log-preference (0 = neutral, negative = undesired).
 */
export type GaussianPreferenceFn = (mean: number) => number;

// ── Discrete EFE helpers (exported for advanced use) ─────────────

/**
 * Ambiguity = E_Q[H(o|s)] = −Σ_s Q(s) Σ_o P(o|s) log P(o|s)
 */
export function computeAmbiguity<
    O extends string,
    S extends string,
>(
    predictedBelief: Belief<S>,
    observationModel: IObservationModel<O, S>,
): number {
    let ambiguity = 0;
    for (const state of predictedBelief.states) {
        const stateProb = predictedBelief.probability(state);
        for (const obs of observationModel.observations) {
            const obsProb = observationModel.probability(obs, state);
            if (obsProb > 0 && stateProb > 0) {
                ambiguity -= stateProb * obsProb * Math.log(obsProb);
            }
        }
    }
    return ambiguity;
}

/**
 * Risk = −Σ_o Q(o) log C(o)
 */
export function computeRisk<
    O extends string,
    S extends string,
>(
    predictedBelief: Belief<S>,
    observationModel: IObservationModel<O, S>,
    preferences: Preferences<O>,
): number {
    let risk = 0;
    for (const obs of observationModel.observations) {
        let expectedObsProb = 0;
        for (const state of predictedBelief.states) {
            expectedObsProb +=
                observationModel.probability(obs, state) *
                predictedBelief.probability(state);
        }
        const preferredLogProb = preferences[obs] ?? -10;
        if (expectedObsProb > 0) {
            risk -= expectedObsProb * preferredLogProb;
        }
    }
    return risk;
}

/**
 * Export a discrete belief as a plain Distribution object.
 */
export function exportBelief<S extends string>(belief: Belief<S>): Distribution<S> {
    const result = {} as Distribution<S>;
    for (const state of belief.states) {
        result[state] = belief.probability(state);
    }
    return result;
}

// ── Discrete Agent factory ───────────────────────────────────────

/**
 * Configuration object for creating a discrete Active Inference agent.
 */
export interface AgentConfig<
    A extends string = string,
    O extends string = string,
    S extends string = string,
> {
    belief: Belief<S>;
    transitionModel: ITransitionModel<A, S>;
    observationModel: IObservationModel<O, S>;
    preferences: Preferences<O>;
    seed?: number;
    planningHorizon?: number;
    precision?: number;
    habits?: Partial<Habits<A>>;
    beamWidth?: number;
}

/**
 * Create a discrete Active Inference agent.
 *
 * EFE = ambiguity + risk, computed from the observation model and preferences.
 * Learning (Dirichlet updates) is handled automatically in step().
 */
export function createAgent<
    A extends string = string,
    O extends string = string,
    S extends string = string,
>(config: AgentConfig<A, O, S>): Agent<A, Belief<S>, O> {
    const random =
        config.seed !== undefined ? new Random(config.seed) : new Random();

    const computeEFE = (predicted: Belief<S>): number =>
        computeAmbiguity(predicted, config.observationModel) +
        computeRisk(predicted, config.observationModel, config.preferences);

    const afterObserve = (
        observation: O,
        belief: Belief<S>,
        previousAction: A | null,
        previousBelief: Belief<S> | null,
    ): void => {
        const posteriorDist = exportBelief(belief);
        config.observationModel.learn?.(observation, posteriorDist);

        if (previousAction !== null && previousBelief !== null) {
            const prevDist = exportBelief(previousBelief);
            config.transitionModel.learn?.(previousAction, prevDist, posteriorDist);
        }
    };

    return new Agent<A, Belief<S>, O>(
        config.belief,
        config.transitionModel,
        config.observationModel,
        computeEFE,
        random,
        config.planningHorizon ?? 1,
        config.precision ?? 1,
        config.habits ?? {},
        config.beamWidth ?? 0,
        afterObserve,
    );
}

// ── Gaussian Agent factory ───────────────────────────────────────

/**
 * Configuration for creating a Gaussian Active Inference agent.
 */
export interface GaussianAgentConfig<A extends string = string> {
    belief: GaussianBelief;
    transitionModel: GaussianTransition<A>;
    observationModel: GaussianObservation;
    preferences: GaussianPreferenceFn;
    seed?: number;
    planningHorizon?: number;
    precision?: number;
}

/**
 * Create a continuous (Gaussian) Active Inference agent.
 *
 * EFE = −C(E[y]) where C is the preference function.
 * Ambiguity is constant for fixed observation noise and does not affect action selection.
 */
export function createGaussianAgent<A extends string = string>(
    config: GaussianAgentConfig<A>,
): Agent<A, GaussianBelief, number> {
    const random =
        config.seed !== undefined ? new Random(config.seed) : new Random();

    const computeEFE = (predicted: GaussianBelief): number => {
        const obsMean = config.observationModel.expectedObservation(predicted);
        return -config.preferences(obsMean);
    };

    return new Agent<A, GaussianBelief, number>(
        config.belief,
        config.transitionModel,
        config.observationModel,
        computeEFE,
        random,
        config.planningHorizon ?? 1,
        config.precision ?? 1,
    );
}
