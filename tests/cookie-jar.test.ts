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
    SMALL_SAMPLE_SIZE,
} from './constants';

describe('Cookie Jar', () => {
    const observationMatrix = {
        see_cookie: { cookie_left: HIGH_CONFIDENCE, cookie_right: LOW_CONFIDENCE },
        see_empty: { cookie_left: LOW_CONFIDENCE, cookie_right: HIGH_CONFIDENCE },
    };

    const transitionMatrix = {
        go_left: {
            cookie_left: { cookie_left: CERTAIN, cookie_right: NO_PROBABILITY },
            cookie_right: { cookie_left: NO_PROBABILITY, cookie_right: CERTAIN },
        },
        go_right: {
            cookie_left: { cookie_left: CERTAIN, cookie_right: NO_PROBABILITY },
            cookie_right: { cookie_left: NO_PROBABILITY, cookie_right: CERTAIN },
        },
        stay: {
            cookie_left: { cookie_left: CERTAIN, cookie_right: NO_PROBABILITY },
            cookie_right: { cookie_left: NO_PROBABILITY, cookie_right: CERTAIN },
        },
    };

    const preferences = {
        see_cookie: NEUTRAL_PREFERENCE,
        see_empty: MODERATE_PENALTY,
    };

    const priorBeliefs = {
        cookie_left: UNIFORM_PRIOR,
        cookie_right: UNIFORM_PRIOR,
    };

    const makeAgent = (seed?: number) =>
        createAgent({
            belief: new DiscreteBelief(priorBeliefs),
            transitionModel: new DiscreteTransition(transitionMatrix),
            observationModel: new DiscreteObservation(observationMatrix),
            preferences,
            seed,
        });

    it('creates an agent', () => {
        const agent = makeAgent();
        expect(agent).toBeDefined();
        expect(Object.keys(agent.exportBelief())).toContain('cookie_left');
        expect(Object.keys(agent.exportBelief())).toContain('cookie_right');
    });

    it('updates belief after observing see_cookie', () => {
        const agent = makeAgent();
        expect(agent.exportBelief().cookie_left).toBeCloseTo(UNIFORM_PRIOR);
        agent.observe('see_cookie');
        expect(agent.exportBelief().cookie_left).toBeGreaterThan(
            HIGH_PROBABILITY_THRESHOLD,
        );
    });

    it('updates belief after observing see_empty', () => {
        const agent = makeAgent();
        agent.observe('see_empty');
        expect(agent.exportBelief().cookie_right).toBeGreaterThan(
            HIGH_PROBABILITY_THRESHOLD,
        );
    });

    it('selects an action', () => {
        const agent = makeAgent();
        const action = agent.act();
        expect(['go_left', 'go_right', 'stay']).toContain(action);
    });

    it('step() = observe + act', () => {
        const agent = makeAgent();
        const action = agent.step('see_cookie');

        expect(['go_left', 'go_right', 'stay']).toContain(action);
        expect(agent.exportBelief().cookie_left).toBeGreaterThan(
            UNIFORM_PRIOR,
        );
    });

    it('state returns most likely state', () => {
        const agent = makeAgent();
        agent.observe('see_cookie');
        expect(agent.state).toBe('cookie_left');
    });

    it('seed makes results deterministic', () => {
        const agent1 = makeAgent(SEED_A);
        const agent2 = makeAgent(SEED_A);

        const actions1 = [agent1.act(), agent1.act(), agent1.act()];
        const actions2 = [agent2.act(), agent2.act(), agent2.act()];

        expect(actions1).toEqual(actions2);
    });

    it('different seeds produce different results', () => {
        const agent1 = makeAgent(SEED_C);
        const agent2 = makeAgent(SEED_D);

        const actions1: string[] = [];
        const actions2: string[] = [];

        for (let i = 0; i < SMALL_SAMPLE_SIZE; i++) {
            actions1.push(agent1.act());
            actions2.push(agent2.act());
        }

        expect(actions1.join()).not.toBe(actions2.join());
    });

    it('all seeds work correctly', () => {
        const seeds = [SEED_A, SEED_B, SEED_C, SEED_D, SEED_E];

        for (const seed of seeds) {
            const agent = makeAgent(seed);
            expect(agent.act()).toBeDefined();
            expect(agent.state).toBeDefined();
        }
    });
});
