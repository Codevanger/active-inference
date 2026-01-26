import { describe, it, expect } from 'vitest';
import { DiscreteBelief } from '../src/beliefs/discrete.belief';
import {
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    UNIFORM_PRIOR,
    CERTAIN,
    NO_PROBABILITY,
    SEED_A,
    SEED_B,
    SEED_C,
    SEED_D,
    SEED_E,
} from './constants';

describe('DiscreteBelief', () => {
    describe('probability', () => {
        it('returns probability of a state', () => {
            const belief = new DiscreteBelief({ a: 0.7, b: 0.3 });
            expect(belief.probability('a')).toBeCloseTo(0.7);
            expect(belief.probability('b')).toBeCloseTo(0.3);
        });

        it('returns 0 for unknown state', () => {
            const belief = new DiscreteBelief({ a: CERTAIN });
            expect(belief.probability('unknown')).toBe(NO_PROBABILITY);
        });
    });

    describe('argmax', () => {
        it('returns most likely state', () => {
            const belief = new DiscreteBelief({ a: 0.1, b: 0.6, c: 0.3 });
            expect(belief.argmax()).toBe('b');
        });

        it('works with equal probabilities', () => {
            const belief = new DiscreteBelief({
                a: UNIFORM_PRIOR,
                b: UNIFORM_PRIOR,
            });
            expect(['a', 'b']).toContain(belief.argmax());
        });
    });

    describe('entropy', () => {
        it('uniform distribution has maximum entropy', () => {
            const uniform = new DiscreteBelief({
                a: UNIFORM_PRIOR,
                b: UNIFORM_PRIOR,
            });
            const peaked = new DiscreteBelief({
                a: HIGH_CONFIDENCE,
                b: LOW_CONFIDENCE,
            });
            expect(uniform.entropy()).toBeGreaterThan(peaked.entropy());
        });

        it('deterministic distribution has zero entropy', () => {
            const deterministic = new DiscreteBelief({
                a: CERTAIN,
                b: NO_PROBABILITY,
            });
            expect(deterministic.entropy()).toBeCloseTo(0);
        });

        it('entropy >= 0', () => {
            const belief = new DiscreteBelief({ a: 0.3, b: 0.7 });
            expect(belief.entropy()).toBeGreaterThanOrEqual(0);
        });
    });

    describe('kl', () => {
        it('KL(P || P) = 0', () => {
            const p = new DiscreteBelief({ a: 0.3, b: 0.7 });
            expect(p.kl(p)).toBeCloseTo(0);
        });

        it('KL(P || Q) >= 0 (Gibbs inequality)', () => {
            const p = new DiscreteBelief({ a: 0.3, b: 0.7 });
            const q = new DiscreteBelief({ a: UNIFORM_PRIOR, b: UNIFORM_PRIOR });
            expect(p.kl(q)).toBeGreaterThanOrEqual(0);
        });

        it('KL(P || Q) != KL(Q || P) (asymmetry)', () => {
            const p = new DiscreteBelief({ a: 0.8, b: 0.2 });
            const q = new DiscreteBelief({ a: UNIFORM_PRIOR, b: UNIFORM_PRIOR });
            expect(p.kl(q)).not.toBeCloseTo(q.kl(p));
        });
    });

    describe('update (Bayesian inference)', () => {
        it('updates belief by likelihood', () => {
            const prior = new DiscreteBelief({
                a: UNIFORM_PRIOR,
                b: UNIFORM_PRIOR,
            });
            const likelihood = { a: HIGH_CONFIDENCE, b: LOW_CONFIDENCE };
            const posterior = prior.update(likelihood);

            expect(posterior.probability('a')).toBeGreaterThan(UNIFORM_PRIOR);
            expect(posterior.probability('b')).toBeLessThan(UNIFORM_PRIOR);
        });

        it('posterior is proportional to prior * likelihood', () => {
            const prior = new DiscreteBelief({ a: 0.6, b: 0.4 });
            const likelihood = { a: 0.8, b: 0.2 };
            const posterior = prior.update(likelihood);

            expect(posterior.probability('a')).toBeCloseTo(0.48 / 0.56);
        });

        it('probabilities sum to 1 after update', () => {
            const prior = new DiscreteBelief({ a: 0.3, b: 0.7 });
            const likelihood = { a: UNIFORM_PRIOR, b: UNIFORM_PRIOR };
            const posterior = prior.update(likelihood);
            const sum = posterior.probability('a') + posterior.probability('b');

            expect(sum).toBeCloseTo(CERTAIN);
        });
    });

    describe('copy', () => {
        it('creates an independent copy', () => {
            const original = new DiscreteBelief({
                a: UNIFORM_PRIOR,
                b: UNIFORM_PRIOR,
            });
            const copy = original.copy();

            copy.distribution.a = CERTAIN;
            copy.distribution.b = NO_PROBABILITY;

            expect(original.probability('a')).toBeCloseTo(UNIFORM_PRIOR);
        });
    });
});
