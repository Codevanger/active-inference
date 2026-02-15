/**
 * Interface for models that support Dirichlet concentration-based learning.
 *
 * Models implementing this interface maintain Dirichlet concentration
 * parameters (pseudo-counts) that are updated from experience.
 * The underlying probability matrices are derived by normalizing
 * concentrations.
 */
export interface ILearnable {
    readonly learnable: true;
}

/**
 * Type guard to check if a model supports learning.
 */
export function isLearnable(obj: unknown): obj is ILearnable {
    return (
        typeof obj === 'object' &&
        obj !== null &&
        'learnable' in obj &&
        (obj as ILearnable).learnable === true
    );
}
