import { describe, it, expect } from 'vitest';
import { createAgent } from '../src/factory';
import { DiscreteBelief } from '../src/beliefs/discrete.belief';
import { DiscreteTransition } from '../src/transition/discrete.transition';
import { DiscreteObservation } from '../src/observation/discrete.observation';
import {
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    CERTAIN,
    NO_PROBABILITY,
    UNIFORM_PRIOR,
    NEUTRAL_PREFERENCE,
    MODERATE_PENALTY,
    HIGH_PROBABILITY_THRESHOLD,
    SEED_A,
    SEED_B,
    SEED_C,
    SEED_D,
    SEED_E,
    SAMPLE_SIZE,
    SMALL_SAMPLE_SIZE,
    MEDIUM_SAMPLE_SIZE,
} from './constants';

describe('Agent', () => {
    const observationModel = new DiscreteObservation({
        see_a: { state_a: HIGH_CONFIDENCE, state_b: LOW_CONFIDENCE },
        see_b: { state_a: LOW_CONFIDENCE, state_b: HIGH_CONFIDENCE },
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
        stay: {
            state_a: { state_a: CERTAIN, state_b: NO_PROBABILITY },
            state_b: { state_a: NO_PROBABILITY, state_b: CERTAIN },
        },
    });

    const preferences = {
        see_a: NEUTRAL_PREFERENCE,
        see_b: MODERATE_PENALTY,
    };

    const priorBeliefs = { state_a: UNIFORM_PRIOR, state_b: UNIFORM_PRIOR };

    const makeAgent = (seed?: number) =>
        createAgent({
            belief: new DiscreteBelief(priorBeliefs),
            transitionModel,
            observationModel,
            preferences,
            seed,
        });

    describe('observe', () => {
        it('updates belief based on observation', () => {
            const agent = makeAgent();

            agent.observe('see_a');

            expect(agent.exportBelief().state_a).toBeGreaterThan(
                HIGH_PROBABILITY_THRESHOLD,
            );
        });

        it('multiple observations strengthen belief', () => {
            const agent = makeAgent();

            agent.observe('see_a');
            const afterOne = agent.exportBelief().state_a;

            agent.observe('see_a');
            const afterTwo = agent.exportBelief().state_a;

            expect(afterTwo).toBeGreaterThan(afterOne);
        });
    });

    describe('act', () => {
        it('selects action from available actions', () => {
            const agent = makeAgent();
            const action = agent.act();
            expect(['go_a', 'go_b', 'stay']).toContain(action);
        });

        it('prefers actions leading to preferred observations', () => {
            const actions: Record<string, number> = { go_a: 0, go_b: 0, stay: 0 };
            for (let i = 0; i < SAMPLE_SIZE; i++) {
                const testAgent = makeAgent(i);
                testAgent.observe('see_b');
                actions[testAgent.act()]++;
            }

            expect(actions.go_a).toBeGreaterThan(actions.go_b);
        });
    });

    describe('step', () => {
        it('combines observe and act', () => {
            const agent1 = makeAgent(SEED_A);
            const agent2 = makeAgent(SEED_A);

            const actionStep = agent1.step('see_a');

            agent2.observe('see_a');
            const actionSeparate = agent2.act();

            expect(actionStep).toBe(actionSeparate);
            expect(agent1.exportBelief().state_a).toBe(
                agent2.exportBelief().state_a,
            );
        });
    });

    describe('freeEnergy', () => {
        it('increases when uncertainty decreases', () => {
            const agent = makeAgent();

            const feBefore = agent.freeEnergy;
            agent.observe('see_a');
            const feAfter = agent.freeEnergy;

            expect(feAfter).toBeGreaterThan(feBefore);
        });
    });

    describe('seed', () => {
        it('same seed produces same actions', () => {
            const agent1 = makeAgent(SEED_B);
            const agent2 = makeAgent(SEED_B);

            for (let i = 0; i < SMALL_SAMPLE_SIZE; i++) {
                expect(agent1.act()).toBe(agent2.act());
            }
        });

        it('different seeds produce different sequences', () => {
            const agent1 = makeAgent(SEED_C);
            const agent2 = makeAgent(SEED_D);

            const actions1 = Array.from({ length: MEDIUM_SAMPLE_SIZE }, () =>
                agent1.act(),
            );
            const actions2 = Array.from({ length: MEDIUM_SAMPLE_SIZE }, () =>
                agent2.act(),
            );

            expect(actions1.join()).not.toBe(actions2.join());
        });

        it('all seeds produce valid actions', () => {
            const seeds = [SEED_A, SEED_B, SEED_C, SEED_D, SEED_E];

            for (const seed of seeds) {
                const agent = makeAgent(seed);
                const action = agent.act();
                expect(['go_a', 'go_b', 'stay']).toContain(action);
            }
        });
    });

    describe('exportBelief', () => {
        it('exports current belief distribution', () => {
            const agent = createAgent({
                belief: new DiscreteBelief({ state_a: 0.7, state_b: 0.3 }),
                transitionModel,
                observationModel,
                preferences,
            });

            const exported = agent.exportBelief();

            expect(exported.state_a).toBeCloseTo(0.7);
            expect(exported.state_b).toBeCloseTo(0.3);
        });

        it('exported belief can create new agent with same state', () => {
            const agent1 = makeAgent(SEED_A);
            agent1.observe('see_a');
            const exported = agent1.exportBelief();

            const agent2 = createAgent({
                belief: new DiscreteBelief(exported),
                transitionModel,
                observationModel,
                preferences,
                seed: SEED_A,
            });

            const belief1 = agent1.exportBelief();
            const belief2 = agent2.exportBelief();

            expect(belief1.state_a).toBeCloseTo(belief2.state_a);
            expect(belief1.state_b).toBeCloseTo(belief2.state_b);
        });

        it('exported belief reflects observations', () => {
            const agent = makeAgent();

            const beforeObs = agent.exportBelief();
            agent.observe('see_a');
            const afterObs = agent.exportBelief();

            expect(afterObs.state_a).toBeGreaterThan(beforeObs.state_a);
        });

        it('new agent from exported belief behaves identically', () => {
            const agent1 = createAgent({
                belief: new DiscreteBelief({ state_a: 0.8, state_b: 0.2 }),
                transitionModel,
                observationModel,
                preferences,
                seed: SEED_A,
            });

            const exported = agent1.exportBelief();

            const agent2 = createAgent({
                belief: new DiscreteBelief(exported),
                transitionModel,
                observationModel,
                preferences,
                seed: SEED_A,
            });

            expect(agent1.act()).toBe(agent2.act());
        });
    });
});
