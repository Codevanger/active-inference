import { describe, it, expect } from 'vitest';
import { createAgent, computeAmbiguity } from '../src/factory';
import { DiscreteBelief } from '../src/beliefs/discrete.belief';
import { DiscreteTransition } from '../src/transition/discrete.transition';
import { DiscreteObservation } from '../src/observation/discrete.observation';
import {
    VERY_HIGH_CONFIDENCE,
    VERY_LOW_CONFIDENCE,
    NEAR_CERTAIN,
    NEAR_IMPOSSIBLE,
    CERTAIN,
    NO_PROBABILITY,
    UNIFORM_PRIOR,
    NEUTRAL_PREFERENCE,
    WEAK_PENALTY,
    STRONG_PENALTY,
    VERY_STRONG_PENALTY,
    SEED_A,
    SEED_B,
    SEED_C,
    SEED_D,
    SEED_E,
    SAMPLE_SIZE,
    SIMULATION_STEPS,
} from './constants';

describe('Free Energy Principle', () => {
    const observationModel = new DiscreteObservation({
        light_on: {
            switch_on: VERY_HIGH_CONFIDENCE,
            switch_off: VERY_LOW_CONFIDENCE,
        },
        light_off: {
            switch_on: VERY_LOW_CONFIDENCE,
            switch_off: VERY_HIGH_CONFIDENCE,
        },
    });

    const transitionModel = new DiscreteTransition({
        turn_on: {
            switch_on: { switch_on: CERTAIN, switch_off: NO_PROBABILITY },
            switch_off: { switch_on: CERTAIN, switch_off: NO_PROBABILITY },
        },
        turn_off: {
            switch_on: { switch_on: NO_PROBABILITY, switch_off: CERTAIN },
            switch_off: { switch_on: NO_PROBABILITY, switch_off: CERTAIN },
        },
    });

    describe('Variational Free Energy', () => {
        it('confident agent has higher VFE than uncertain agent', () => {
            const confidentAgent = createAgent({
                belief: new DiscreteBelief({
                    switch_on: NEAR_CERTAIN,
                    switch_off: NEAR_IMPOSSIBLE,
                }),
                transitionModel,
                observationModel,
                preferences: { light_on: NEUTRAL_PREFERENCE, light_off: WEAK_PENALTY },
            });

            const uncertainAgent = createAgent({
                belief: new DiscreteBelief({
                    switch_on: UNIFORM_PRIOR,
                    switch_off: UNIFORM_PRIOR,
                }),
                transitionModel,
                observationModel,
                preferences: { light_on: NEUTRAL_PREFERENCE, light_off: WEAK_PENALTY },
            });

            const confidentVFE =
                -confidentAgent.belief.entropy() +
                computeAmbiguity(confidentAgent.belief, observationModel);
            const uncertainVFE =
                -uncertainAgent.belief.entropy() +
                computeAmbiguity(uncertainAgent.belief, observationModel);

            expect(confidentVFE).toBeGreaterThan(uncertainVFE);
        });

        it('observation reduces belief entropy', () => {
            const agent = createAgent({
                belief: new DiscreteBelief({
                    switch_on: UNIFORM_PRIOR,
                    switch_off: UNIFORM_PRIOR,
                }),
                transitionModel,
                observationModel,
                preferences: { light_on: NEUTRAL_PREFERENCE, light_off: WEAK_PENALTY },
            });

            const entropyBefore = agent.belief.entropy();
            agent.observe('light_on');
            const entropyAfter = agent.belief.entropy();

            expect(entropyAfter).toBeLessThan(entropyBefore);
        });
    });

    describe('Expected Free Energy (EFE)', () => {
        it('agent prefers actions with lower EFE', () => {
            const preferences = {
                light_on: NEUTRAL_PREFERENCE,
                light_off: STRONG_PENALTY,
            };

            const actions: Record<string, number> = { turn_on: 0, turn_off: 0 };
            for (let i = 0; i < SAMPLE_SIZE; i++) {
                const testAgent = createAgent({
                    belief: new DiscreteBelief({
                        switch_on: UNIFORM_PRIOR,
                        switch_off: UNIFORM_PRIOR,
                    }),
                    transitionModel,
                    observationModel,
                    preferences,
                    seed: i,
                });
                actions[testAgent.act()]++;
            }

            expect(actions.turn_on).toBeGreaterThan(actions.turn_off);
        });

        it('stronger preferences increase bias toward preferred actions', () => {
            const weakPreferences = {
                light_on: NEUTRAL_PREFERENCE,
                light_off: WEAK_PENALTY,
            };
            const strongPreferences = {
                light_on: NEUTRAL_PREFERENCE,
                light_off: VERY_STRONG_PENALTY,
            };

            let weakTurnOn = 0;
            let strongTurnOn = 0;

            for (let i = 0; i < SAMPLE_SIZE; i++) {
                const weakAgent = createAgent({
                    belief: new DiscreteBelief({
                        switch_on: UNIFORM_PRIOR,
                        switch_off: UNIFORM_PRIOR,
                    }),
                    transitionModel,
                    observationModel,
                    preferences: weakPreferences,
                    seed: i,
                });

                const strongAgent = createAgent({
                    belief: new DiscreteBelief({
                        switch_on: UNIFORM_PRIOR,
                        switch_off: UNIFORM_PRIOR,
                    }),
                    transitionModel,
                    observationModel,
                    preferences: strongPreferences,
                    seed: i,
                });

                if (weakAgent.act() === 'turn_on') weakTurnOn++;
                if (strongAgent.act() === 'turn_on') strongTurnOn++;
            }

            expect(strongTurnOn).toBeGreaterThanOrEqual(weakTurnOn);
        });
    });

    describe('Perception-Action Loop', () => {
        it('agent converges to preferred state', () => {
            const seeds = [SEED_A, SEED_B, SEED_C, SEED_D, SEED_E];
            let successCount = 0;

            for (const seed of seeds) {
                const agent = createAgent({
                    belief: new DiscreteBelief({
                        switch_on: UNIFORM_PRIOR,
                        switch_off: UNIFORM_PRIOR,
                    }),
                    transitionModel,
                    observationModel,
                    preferences: { light_on: NEUTRAL_PREFERENCE, light_off: STRONG_PENALTY },
                    seed,
                });

                let state = 'switch_off';

                for (let i = 0; i < SIMULATION_STEPS; i++) {
                    const observation =
                        state === 'switch_on' ? 'light_on' : 'light_off';
                    const action = agent.step(observation);

                    if (action === 'turn_on') state = 'switch_on';
                    if (action === 'turn_off') state = 'switch_off';
                }

                if (state === 'switch_on') successCount++;
            }

            expect(successCount).toBeGreaterThanOrEqual(seeds.length - 1);
        });
    });
});
