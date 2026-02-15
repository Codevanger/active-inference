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
