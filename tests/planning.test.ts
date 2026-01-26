import { describe, it, expect } from 'vitest';
import { createAgent } from '../src/factory';
import { DiscreteBelief } from '../src/beliefs/discrete.belief';
import { DiscreteTransition } from '../src/transition/discrete.transition';
import { DiscreteObservation } from '../src/observation/discrete.observation';
import {
    VERY_HIGH_CONFIDENCE,
    VERY_LOW_CONFIDENCE,
    CERTAIN,
    NO_PROBABILITY,
    UNIFORM_PRIOR,
    NEUTRAL_PREFERENCE,
    STRONG_PENALTY,
    SAMPLE_SIZE,
} from './constants';

describe('Planning', () => {
    const observationModel = new DiscreteObservation({
        see_a: {
            state_a: VERY_HIGH_CONFIDENCE,
            state_b: VERY_LOW_CONFIDENCE,
        },
        see_b: {
            state_a: VERY_LOW_CONFIDENCE,
            state_b: VERY_HIGH_CONFIDENCE,
        },
    });

    const transitionModel = new DiscreteTransition({
        go_a: {
            state_a: { state_a: CERTAIN, state_b: NO_PROBABILITY },
            state_b: { state_a: CERTAIN, state_b: NO_PROBABILITY },
        },
        go_b: {
            state_a: { state_a: NO_PROBABILITY, state_b: CERTAIN },
            state_b: { state_a: NO_PROBABILITY, state_b: CERTAIN },
        },
    });

    const preferences = {
        see_a: NEUTRAL_PREFERENCE,
        see_b: STRONG_PENALTY,
    };

    describe('planning behavior', () => {
        it('greedy and planning agents prefer go_a (leads to preferred observation)', () => {
            const greedyActions: Record<string, number> = { go_a: 0, go_b: 0 };
            const planningActions: Record<string, number> = { go_a: 0, go_b: 0 };

            for (let i = 0; i < SAMPLE_SIZE; i++) {
                const greedyAgent = createAgent({
                    belief: new DiscreteBelief({
                        state_a: UNIFORM_PRIOR,
                        state_b: UNIFORM_PRIOR,
                    }),
                    transitionModel,
                    observationModel,
                    preferences,
                    planningHorizon: 1,
                    seed: i,
                });

                const planningAgent = createAgent({
                    belief: new DiscreteBelief({
                        state_a: UNIFORM_PRIOR,
                        state_b: UNIFORM_PRIOR,
                    }),
                    transitionModel,
                    observationModel,
                    preferences,
                    planningHorizon: 2,
                    seed: i,
                });

                greedyActions[greedyAgent.act()]++;
                planningActions[planningAgent.act()]++;
            }

            expect(greedyActions.go_a).toBeGreaterThan(greedyActions.go_b);
            expect(planningActions.go_a).toBeGreaterThan(planningActions.go_b);
        });

        it('different horizons produce consistent preferences', () => {
            const horizons = [1, 2, 3, 4];
            const results: Record<number, Record<string, number>> = {};

            for (const horizon of horizons) {
                results[horizon] = { go_a: 0, go_b: 0 };
                for (let i = 0; i < SAMPLE_SIZE; i++) {
                    const agent = createAgent({
                        belief: new DiscreteBelief({
                            state_a: UNIFORM_PRIOR,
                            state_b: UNIFORM_PRIOR,
                        }),
                        transitionModel,
                        observationModel,
                        preferences,
                        planningHorizon: horizon,
                        seed: i,
                    });
                    results[horizon][agent.act()]++;
                }
            }

            for (const horizon of horizons) {
                expect(results[horizon].go_a).toBeGreaterThan(results[horizon].go_b);
            }
        });
    });

    describe('planning advantage', () => {
        const trapObservation = new DiscreteObservation({
            see_start: { start: 1, trap: 0, doom: 0, safe: 0 },
            see_trap: { start: 0, trap: 1, doom: 0, safe: 0 },
            see_doom: { start: 0, trap: 0, doom: 1, safe: 0 },
            see_safe: { start: 0, trap: 0, doom: 0, safe: 1 },
        });

        const trapTransition = new DiscreteTransition({
            go_trap: {
                start: { start: 0, trap: 1, doom: 0, safe: 0 },
                trap: { start: 0, trap: 0, doom: 1, safe: 0 },
                doom: { start: 0, trap: 0, doom: 1, safe: 0 },
                safe: { start: 0, trap: 0, doom: 0, safe: 1 },
            },
            go_safe: {
                start: { start: 0, trap: 0, doom: 0, safe: 1 },
                trap: { start: 0, trap: 0, doom: 0, safe: 1 },
                doom: { start: 0, trap: 0, doom: 1, safe: 0 },
                safe: { start: 0, trap: 0, doom: 0, safe: 1 },
            },
        });

        const trapPreferences = {
            see_start: 0,
            see_trap: 0,
            see_doom: -10,
            see_safe: 0,
        };

        it('planning avoids trap that greedy falls into', () => {
            const greedyResults = { go_trap: 0, go_safe: 0 };
            const planningResults = { go_trap: 0, go_safe: 0 };

            for (let i = 0; i < SAMPLE_SIZE; i++) {
                const greedyAgent = createAgent({
                    belief: new DiscreteBelief({ start: 1, trap: 0, doom: 0, safe: 0 }),
                    transitionModel: trapTransition,
                    observationModel: trapObservation,
                    preferences: trapPreferences,
                    planningHorizon: 1,
                    seed: i,
                });

                const planningAgent = createAgent({
                    belief: new DiscreteBelief({ start: 1, trap: 0, doom: 0, safe: 0 }),
                    transitionModel: trapTransition,
                    observationModel: trapObservation,
                    preferences: trapPreferences,
                    planningHorizon: 2,
                    seed: i,
                });

                greedyResults[greedyAgent.act()]++;
                planningResults[planningAgent.act()]++;
            }

            expect(planningResults.go_safe).toBeGreaterThan(planningResults.go_trap);

            const greedySafeRatio = greedyResults.go_safe / SAMPLE_SIZE;
            const planningSafeRatio = planningResults.go_safe / SAMPLE_SIZE;
            expect(planningSafeRatio).toBeGreaterThan(greedySafeRatio);
        });

        const deepTrapObservation = new DiscreteObservation({
            see_start: { start: 1, mid: 0, trap: 0, doom: 0, safe: 0 },
            see_mid: { start: 0, mid: 1, trap: 0, doom: 0, safe: 0 },
            see_trap: { start: 0, mid: 0, trap: 1, doom: 0, safe: 0 },
            see_doom: { start: 0, mid: 0, trap: 0, doom: 1, safe: 0 },
            see_safe: { start: 0, mid: 0, trap: 0, doom: 0, safe: 1 },
        });

        const deepTrapTransition = new DiscreteTransition({
            go_danger: {
                start: { start: 0, mid: 1, trap: 0, doom: 0, safe: 0 },
                mid: { start: 0, mid: 0, trap: 1, doom: 0, safe: 0 },
                trap: { start: 0, mid: 0, trap: 0, doom: 1, safe: 0 },
                doom: { start: 0, mid: 0, trap: 0, doom: 1, safe: 0 },
                safe: { start: 0, mid: 0, trap: 0, doom: 0, safe: 1 },
            },
            go_safe: {
                start: { start: 0, mid: 0, trap: 0, doom: 0, safe: 1 },
                mid: { start: 0, mid: 0, trap: 0, doom: 0, safe: 1 },
                trap: { start: 0, mid: 0, trap: 0, doom: 0, safe: 1 },
                doom: { start: 0, mid: 0, trap: 0, doom: 1, safe: 0 },
                safe: { start: 0, mid: 0, trap: 0, doom: 0, safe: 1 },
            },
        });

        const deepTrapPreferences = {
            see_start: 0,
            see_mid: 0,
            see_trap: 0,
            see_doom: -10,
            see_safe: 0,
        };

        it('horizon=3 sees deeper trap that horizon=2 misses', () => {
            const horizon2Results = { go_danger: 0, go_safe: 0 };
            const horizon3Results = { go_danger: 0, go_safe: 0 };

            for (let i = 0; i < SAMPLE_SIZE; i++) {
                const agent2 = createAgent({
                    belief: new DiscreteBelief({ start: 1, mid: 0, trap: 0, doom: 0, safe: 0 }),
                    transitionModel: deepTrapTransition,
                    observationModel: deepTrapObservation,
                    preferences: deepTrapPreferences,
                    planningHorizon: 2,
                    seed: i,
                });

                const agent3 = createAgent({
                    belief: new DiscreteBelief({ start: 1, mid: 0, trap: 0, doom: 0, safe: 0 }),
                    transitionModel: deepTrapTransition,
                    observationModel: deepTrapObservation,
                    preferences: deepTrapPreferences,
                    planningHorizon: 3,
                    seed: i,
                });

                horizon2Results[agent2.act()]++;
                horizon3Results[agent3.act()]++;
            }

            const h2SafeRatio = horizon2Results.go_safe / SAMPLE_SIZE;
            const h3SafeRatio = horizon3Results.go_safe / SAMPLE_SIZE;
            expect(h3SafeRatio).toBeGreaterThan(h2SafeRatio);
        });
    });
});
