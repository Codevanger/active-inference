import { describe, it, expect } from 'vitest';
import { createAgent } from '../src/factory';
import { DiscreteBelief } from '../src/beliefs/discrete.belief';
import { DirichletObservation } from '../src/observation/dirichlet.observation';
import { DirichletTransition } from '../src/transition/dirichlet.transition';
import { DirichletPreferences } from '../src/preferences/dirichlet.preferences';
import { DiscreteTransition } from '../src/transition/discrete.transition';
import { DiscreteObservation } from '../src/observation/discrete.observation';
import {
    SEED_A,
    UNIFORM_PRIOR,
    HIGH_PROBABILITY_THRESHOLD,
} from './constants';

describe('DirichletObservation', () => {
    it('normalizes concentrations into probabilities', () => {
        const obs = new DirichletObservation({
            see_safe: { safe: 9, danger: 1 },
            see_danger: { safe: 1, danger: 9 },
        });

        // Column-wise: for state 'safe', sum = 9+1=10
        expect(obs.probability('see_safe', 'safe')).toBeCloseTo(0.9);
        expect(obs.probability('see_danger', 'safe')).toBeCloseTo(0.1);

        // For state 'danger', sum = 1+9=10
        expect(obs.probability('see_safe', 'danger')).toBeCloseTo(0.1);
        expect(obs.probability('see_danger', 'danger')).toBeCloseTo(0.9);
    });

    it('returns observations and states', () => {
        const obs = new DirichletObservation({
            see_safe: { safe: 1, danger: 1 },
            see_danger: { safe: 1, danger: 1 },
        });

        expect(obs.observations).toEqual(['see_safe', 'see_danger']);
        expect(obs.states).toEqual(['safe', 'danger']);
    });

    it('getLikelihood returns distribution over states', () => {
        const obs = new DirichletObservation({
            see_safe: { safe: 9, danger: 1 },
            see_danger: { safe: 1, danger: 9 },
        });

        const likelihood = obs.getLikelihood('see_safe');
        expect(likelihood.safe).toBeCloseTo(0.9);
        expect(likelihood.danger).toBeCloseTo(0.1);
    });

    it('updates concentrations on learn', () => {
        const obs = new DirichletObservation({
            see_safe: { safe: 5, danger: 5 },
            see_danger: { safe: 5, danger: 5 },
        });

        // Before learning: uniform P(o|s) = 0.5
        expect(obs.probability('see_safe', 'safe')).toBeCloseTo(0.5);

        // Learn: observed 'see_safe' while believing mostly in 'safe'
        obs.learn('see_safe', { safe: 0.9, danger: 0.1 });

        // After learning: P(see_safe|safe) should increase
        expect(obs.probability('see_safe', 'safe')).toBeGreaterThan(0.5);
        // P(see_safe|danger) should change less
        expect(obs.concentrations['see_safe'].safe).toBeCloseTo(5.9);
        expect(obs.concentrations['see_safe'].danger).toBeCloseTo(5.1);
    });

    it('repeated learning strengthens associations', () => {
        const obs = new DirichletObservation({
            see_safe: { safe: 1, danger: 1 },
            see_danger: { safe: 1, danger: 1 },
        });

        // Start uniform
        expect(obs.probability('see_safe', 'safe')).toBeCloseTo(0.5);

        // Learn 10 times: see_safe correlates with safe
        for (let i = 0; i < 10; i++) {
            obs.learn('see_safe', { safe: 1, danger: 0 });
        }

        // Now P(see_safe|safe) should be high
        expect(obs.probability('see_safe', 'safe')).toBeGreaterThan(0.8);
    });

    it('does not mutate original concentrations object', () => {
        const original = {
            a: { x: 5, y: 5 },
            b: { x: 5, y: 5 },
        };
        const obs = new DirichletObservation(original);
        obs.learn('a', { x: 1, y: 0 });

        // Original should be unchanged
        expect(original.a.x).toBe(5);
    });
});

describe('DirichletTransition', () => {
    it('normalizes concentrations into probabilities', () => {
        const trans = new DirichletTransition({
            move: {
                here: { here: 1, there: 9 },
                there: { here: 9, there: 1 },
            },
        });

        // Row-wise: for (move, here), sum = 1+9=10
        expect(trans.matrix.move.here.there).toBeCloseTo(0.9);
        expect(trans.matrix.move.here.here).toBeCloseTo(0.1);
    });

    it('returns actions and states', () => {
        const trans = new DirichletTransition({
            go: {
                a: { a: 1, b: 1 },
                b: { a: 1, b: 1 },
            },
            stay: {
                a: { a: 1, b: 1 },
                b: { a: 1, b: 1 },
            },
        });

        expect(trans.actions).toEqual(['go', 'stay']);
        expect(trans.states).toEqual(['a', 'b']);
    });

    it('predict propagates beliefs through transition', () => {
        const trans = new DirichletTransition({
            move: {
                here: { here: 1, there: 9 },
                there: { here: 1, there: 9 },
            },
        });

        const belief = new DiscreteBelief({ here: 1.0, there: 0.0 });
        const predicted = trans.predict(belief, 'move');

        expect(predicted.probability('there')).toBeGreaterThan(0.8);
    });

    it('updates concentrations on learn', () => {
        const trans = new DirichletTransition({
            move: {
                a: { a: 5, b: 5 },
                b: { a: 5, b: 5 },
            },
        });

        // Before: uniform transitions
        expect(trans.matrix.move.a.b).toBeCloseTo(0.5);

        // Learn: took 'move' from state a, ended in state b
        trans.learn('move', { a: 1, b: 0 }, { a: 0, b: 1 });

        // After: P(b|a, move) should increase
        expect(trans.matrix.move.a.b).toBeGreaterThan(0.5);
        // Concentration for (move, a→b) should be 5 + 1*1 = 6
        expect(trans.concentrations.move.a.b).toBeCloseTo(6);
        // Concentration for (move, a→a) should be 5 + 1*0 = 5
        expect(trans.concentrations.move.a.a).toBeCloseTo(5);
    });

    it('does not mutate original concentrations object', () => {
        const original = {
            go: {
                x: { x: 5, y: 5 },
                y: { x: 5, y: 5 },
            },
        };
        const trans = new DirichletTransition(original);
        trans.learn('go', { x: 1, y: 0 }, { x: 0, y: 1 });

        expect(original.go.x.y).toBe(5);
    });
});

describe('DirichletPreferences', () => {
    it('derives log-preferences from concentrations', () => {
        const prefs = new DirichletPreferences({
            reward: 10,
            no_reward: 1,
        });

        const p = prefs.preferences;

        // reward should be less negative (more preferred)
        expect(p.reward).toBeGreaterThan(p.no_reward);
        // Both should be negative (log of probability < 1)
        expect(p.reward).toBeLessThan(0);
        expect(p.no_reward).toBeLessThan(0);
        // Check actual values: log(10/11) and log(1/11)
        expect(p.reward).toBeCloseTo(Math.log(10 / 11));
        expect(p.no_reward).toBeCloseTo(Math.log(1 / 11));
    });

    it('updates concentrations on learn', () => {
        const prefs = new DirichletPreferences({
            reward: 5,
            no_reward: 5,
        });

        // Before: equal preferences
        const before = prefs.preferences;
        expect(before.reward).toBeCloseTo(before.no_reward);

        // Learn: reinforce reward
        prefs.learn('reward');

        // After: reward should be more preferred
        const after = prefs.preferences;
        expect(after.reward).toBeGreaterThan(after.no_reward);
        expect(prefs.concentrations.reward).toBe(6);
    });

    it('learn with custom amount', () => {
        const prefs = new DirichletPreferences({
            good: 10,
            bad: 10,
        });

        prefs.learn('good', 5);
        expect(prefs.concentrations.good).toBe(15);
        expect(prefs.concentrations.bad).toBe(10);
    });

    it('returns observations', () => {
        const prefs = new DirichletPreferences({ a: 1, b: 2, c: 3 });
        expect(prefs.observations).toEqual(['a', 'b', 'c']);
    });
});

describe('Agent with learning', () => {
    // Simple environment: agent learns observation and transition models
    const makeLearnableAgent = (seed?: number) => {
        const observationModel = new DirichletObservation({
            see_safe: { safe: 5, danger: 5 },
            see_danger: { safe: 5, danger: 5 },
        });

        const transitionModel = new DirichletTransition({
            flee: {
                safe: { safe: 5, danger: 5 },
                danger: { safe: 5, danger: 5 },
            },
            stay: {
                safe: { safe: 5, danger: 5 },
                danger: { safe: 5, danger: 5 },
            },
        });

        const agent = createAgent({
            belief: new DiscreteBelief({
                safe: UNIFORM_PRIOR,
                danger: UNIFORM_PRIOR,
            }),
            transitionModel,
            observationModel,
            preferences: { see_safe: 0, see_danger: -5 },
            seed,
        });

        return { agent, observationModel, transitionModel };
    };

    it('observation model learns during step()', () => {
        const { agent, observationModel } = makeLearnableAgent(SEED_A);

        const beforeConc =
            observationModel.concentrations['see_safe'].safe;

        agent.step('see_safe');

        // Concentrations should have increased
        expect(observationModel.concentrations['see_safe'].safe).toBeGreaterThan(
            beforeConc,
        );
    });

    it('transition model learns after second step', () => {
        const { agent, transitionModel } = makeLearnableAgent(SEED_A);

        // First step: no B-learning (no previous action)
        agent.step('see_safe');

        const beforeConc = transitionModel.concentrations.flee.safe.safe +
            transitionModel.concentrations.flee.safe.danger +
            transitionModel.concentrations.flee.danger.safe +
            transitionModel.concentrations.flee.danger.danger +
            transitionModel.concentrations.stay.safe.safe +
            transitionModel.concentrations.stay.safe.danger +
            transitionModel.concentrations.stay.danger.safe +
            transitionModel.concentrations.stay.danger.danger;

        // Second step: B-learning should happen
        agent.step('see_safe');

        const afterConc = transitionModel.concentrations.flee.safe.safe +
            transitionModel.concentrations.flee.safe.danger +
            transitionModel.concentrations.flee.danger.safe +
            transitionModel.concentrations.flee.danger.danger +
            transitionModel.concentrations.stay.safe.safe +
            transitionModel.concentrations.stay.safe.danger +
            transitionModel.concentrations.stay.danger.safe +
            transitionModel.concentrations.stay.danger.danger;

        expect(afterConc).toBeGreaterThan(beforeConc);
    });

    it('preference learning via DirichletPreferences.learn()', () => {
        const prefs = new DirichletPreferences({
            see_safe: 5,
            see_danger: 5,
        });

        expect(prefs.concentrations.see_safe).toBe(5);
        prefs.learn('see_safe');
        expect(prefs.concentrations.see_safe).toBe(6);
    });

    it('non-learnable models are not affected by step()', () => {
        // Use standard Discrete models (not learnable)
        const observationModel = new DiscreteObservation({
            see_safe: { safe: 0.9, danger: 0.1 },
            see_danger: { safe: 0.1, danger: 0.9 },
        });

        const transitionModel = new DiscreteTransition({
            flee: {
                safe: { safe: 0.9, danger: 0.1 },
                danger: { safe: 0.5, danger: 0.5 },
            },
            stay: {
                safe: { safe: 0.9, danger: 0.1 },
                danger: { safe: 0.1, danger: 0.9 },
            },
        });

        const agent = createAgent({
            belief: new DiscreteBelief({
                safe: UNIFORM_PRIOR,
                danger: UNIFORM_PRIOR,
            }),
            transitionModel,
            observationModel,
            preferences: { see_safe: 0, see_danger: -5 },
            seed: SEED_A,
        });

        // Matrix should stay fixed
        const matrixBefore = JSON.stringify(observationModel.matrix);
        agent.step('see_safe');
        agent.step('see_safe');
        const matrixAfter = JSON.stringify(observationModel.matrix);

        expect(matrixAfter).toBe(matrixBefore);
    });

    it('observation model improves with experience', () => {
        // Start with weak uniform prior (knows nothing about observations)
        const obs = new DirichletObservation({
            see_safe: { safe: 1, danger: 1 },
            see_danger: { safe: 1, danger: 1 },
        });

        // Before learning: uniform
        expect(obs.probability('see_safe', 'safe')).toBeCloseTo(0.5);

        // Simulate learning: agent repeatedly sees 'see_safe' while in state 'safe'
        for (let i = 0; i < 20; i++) {
            obs.learn('see_safe', { safe: 0.9, danger: 0.1 });
        }

        // After learning: model should reflect the true observation-state mapping
        expect(obs.probability('see_safe', 'safe')).toBeGreaterThan(
            HIGH_PROBABILITY_THRESHOLD,
        );
    });
});
