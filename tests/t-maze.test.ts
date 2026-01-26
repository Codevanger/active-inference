import { describe, it, expect } from 'vitest';
import { createAgent } from '../src/factory';
import { DiscreteBelief } from '../src/beliefs/discrete.belief';
import { DiscreteTransition } from '../src/transition/discrete.transition';
import { DiscreteObservation } from '../src/observation/discrete.observation';
import {
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    VERY_HIGH_CONFIDENCE,
    VERY_LOW_CONFIDENCE,
    CERTAIN,
    NO_PROBABILITY,
    UNIFORM_PRIOR,
    NEUTRAL_PREFERENCE,
    STRONG_PENALTY,
    HIGH_PROBABILITY_THRESHOLD,
    AMBIGUOUS_LOW,
    AMBIGUOUS_HIGH,
    SEED_A,
    SEED_B,
    SEED_C,
    SEED_D,
    SEED_E,
    SAMPLE_SIZE,
} from './constants';

describe('T-Maze', () => {
    const observationModel = new DiscreteObservation({
        cue_left: {
            center_left: HIGH_CONFIDENCE,
            center_right: LOW_CONFIDENCE,
            left_left: NO_PROBABILITY,
            left_right: NO_PROBABILITY,
            right_left: NO_PROBABILITY,
            right_right: NO_PROBABILITY,
        },
        cue_right: {
            center_left: LOW_CONFIDENCE,
            center_right: HIGH_CONFIDENCE,
            left_left: NO_PROBABILITY,
            left_right: NO_PROBABILITY,
            right_left: NO_PROBABILITY,
            right_right: NO_PROBABILITY,
        },
        reward: {
            center_left: NO_PROBABILITY,
            center_right: NO_PROBABILITY,
            left_left: VERY_HIGH_CONFIDENCE,
            left_right: VERY_LOW_CONFIDENCE,
            right_left: VERY_LOW_CONFIDENCE,
            right_right: VERY_HIGH_CONFIDENCE,
        },
        no_reward: {
            center_left: NO_PROBABILITY,
            center_right: NO_PROBABILITY,
            left_left: VERY_LOW_CONFIDENCE,
            left_right: VERY_HIGH_CONFIDENCE,
            right_left: VERY_HIGH_CONFIDENCE,
            right_right: VERY_LOW_CONFIDENCE,
        },
    });

    const transitionModel = new DiscreteTransition({
        go_left: {
            center_left: {
                center_left: NO_PROBABILITY,
                center_right: NO_PROBABILITY,
                left_left: CERTAIN,
                left_right: NO_PROBABILITY,
                right_left: NO_PROBABILITY,
                right_right: NO_PROBABILITY,
            },
            center_right: {
                center_left: NO_PROBABILITY,
                center_right: NO_PROBABILITY,
                left_left: NO_PROBABILITY,
                left_right: CERTAIN,
                right_left: NO_PROBABILITY,
                right_right: NO_PROBABILITY,
            },
            left_left: {
                center_left: NO_PROBABILITY,
                center_right: NO_PROBABILITY,
                left_left: CERTAIN,
                left_right: NO_PROBABILITY,
                right_left: NO_PROBABILITY,
                right_right: NO_PROBABILITY,
            },
            left_right: {
                center_left: NO_PROBABILITY,
                center_right: NO_PROBABILITY,
                left_left: NO_PROBABILITY,
                left_right: CERTAIN,
                right_left: NO_PROBABILITY,
                right_right: NO_PROBABILITY,
            },
            right_left: {
                center_left: NO_PROBABILITY,
                center_right: NO_PROBABILITY,
                left_left: CERTAIN,
                left_right: NO_PROBABILITY,
                right_left: NO_PROBABILITY,
                right_right: NO_PROBABILITY,
            },
            right_right: {
                center_left: NO_PROBABILITY,
                center_right: NO_PROBABILITY,
                left_left: NO_PROBABILITY,
                left_right: CERTAIN,
                right_left: NO_PROBABILITY,
                right_right: NO_PROBABILITY,
            },
        },
        go_right: {
            center_left: {
                center_left: NO_PROBABILITY,
                center_right: NO_PROBABILITY,
                left_left: NO_PROBABILITY,
                left_right: NO_PROBABILITY,
                right_left: CERTAIN,
                right_right: NO_PROBABILITY,
            },
            center_right: {
                center_left: NO_PROBABILITY,
                center_right: NO_PROBABILITY,
                left_left: NO_PROBABILITY,
                left_right: NO_PROBABILITY,
                right_left: NO_PROBABILITY,
                right_right: CERTAIN,
            },
            left_left: {
                center_left: NO_PROBABILITY,
                center_right: NO_PROBABILITY,
                left_left: NO_PROBABILITY,
                left_right: NO_PROBABILITY,
                right_left: CERTAIN,
                right_right: NO_PROBABILITY,
            },
            left_right: {
                center_left: NO_PROBABILITY,
                center_right: NO_PROBABILITY,
                left_left: NO_PROBABILITY,
                left_right: NO_PROBABILITY,
                right_left: NO_PROBABILITY,
                right_right: CERTAIN,
            },
            right_left: {
                center_left: NO_PROBABILITY,
                center_right: NO_PROBABILITY,
                left_left: NO_PROBABILITY,
                left_right: NO_PROBABILITY,
                right_left: CERTAIN,
                right_right: NO_PROBABILITY,
            },
            right_right: {
                center_left: NO_PROBABILITY,
                center_right: NO_PROBABILITY,
                left_left: NO_PROBABILITY,
                left_right: NO_PROBABILITY,
                right_left: NO_PROBABILITY,
                right_right: CERTAIN,
            },
        },
    });

    const preferences = {
        cue_left: NEUTRAL_PREFERENCE,
        cue_right: NEUTRAL_PREFERENCE,
        reward: NEUTRAL_PREFERENCE,
        no_reward: STRONG_PENALTY,
    };

    const makeAgent = (seed?: number) =>
        createAgent({
            belief: new DiscreteBelief({
                center_left: UNIFORM_PRIOR,
                center_right: UNIFORM_PRIOR,
                left_left: NO_PROBABILITY,
                left_right: NO_PROBABILITY,
                right_left: NO_PROBABILITY,
                right_right: NO_PROBABILITY,
            }),
            transitionModel,
            observationModel,
            preferences,
            seed,
        });

    it('agent goes left after cue_left', () => {
        const actions: Record<string, number> = { go_left: 0, go_right: 0 };

        for (let i = 0; i < SAMPLE_SIZE; i++) {
            const agent = makeAgent(i);
            agent.observe('cue_left');
            actions[agent.act()]++;
        }

        expect(actions.go_left).toBeGreaterThan(actions.go_right);
    });

    it('agent goes right after cue_right', () => {
        const actions: Record<string, number> = { go_left: 0, go_right: 0 };

        for (let i = 0; i < SAMPLE_SIZE; i++) {
            const agent = makeAgent(i);
            agent.observe('cue_right');
            actions[agent.act()]++;
        }

        expect(actions.go_right).toBeGreaterThan(actions.go_left);
    });

    it('agent updates belief based on cue', () => {
        const agent = makeAgent();
        expect(agent.exportBelief().center_left).toBeCloseTo(UNIFORM_PRIOR);

        agent.observe('cue_left');

        expect(agent.exportBelief().center_left).toBeGreaterThan(
            HIGH_PROBABILITY_THRESHOLD,
        );
    });

    it('multiple cues strengthen confidence', () => {
        const agent = makeAgent();

        agent.observe('cue_left');
        const afterOne = agent.exportBelief().center_left;

        agent.observe('cue_left');
        const afterTwo = agent.exportBelief().center_left;

        expect(afterTwo).toBeGreaterThan(afterOne);
    });

    it('contradictory cues create uncertainty', () => {
        const agent = makeAgent();

        agent.observe('cue_left');
        agent.observe('cue_right');

        const prob = agent.exportBelief().center_left;
        expect(prob).toBeGreaterThan(AMBIGUOUS_LOW);
        expect(prob).toBeLessThan(AMBIGUOUS_HIGH);
    });

    it('all seeds produce correct behavior', () => {
        const seeds = [SEED_A, SEED_B, SEED_C, SEED_D, SEED_E];

        for (const seed of seeds) {
            const agent = makeAgent(seed);
            agent.observe('cue_left');

            const action = agent.act();
            expect(['go_left', 'go_right']).toContain(action);

            expect(agent.exportBelief().center_left).toBeGreaterThan(
                HIGH_PROBABILITY_THRESHOLD,
            );
        }
    });
});
