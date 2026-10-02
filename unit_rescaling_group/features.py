"""Build model features and change physical units.

U lists the mass, length, and time powers of each input in its columns.
v gives the output's powers. Each row of exponents defines one monomial.
A monomial is a product of input powers, such as rho * V**2 * A.
"""

from itertools import product

import numpy as np

DRAG_U = np.array([[1, 1, 0, 0, 0], [-3, -1, 1, 2, 1], [0, -1, -1, 0, 0]])
CLASS_U = DRAG_U[:, [0, 1, 2, 4]]
FORCE_V = np.array([1, 1, -2])


def allowed(U, exponents, v):
    """Find monomials whose units match the output: U @ exponent == v."""
    return np.all(np.asarray(exponents) @ U.T == v, axis=1)


def dictionary(task, wrong_count=250):
    """Return U, v, and feature exponents, with allowed features first.

    wrong_count is the number of added features whose units do not match v.
    """
    U, v = (DRAG_U, FORCE_V) if task == "regression" else (CLASS_U, np.zeros(3, dtype=int))
    candidates = np.array(list(product(range(-2, 4), repeat=U.shape[1])))
    mask = allowed(U, candidates, v)
    correct = candidates[mask]
    wrong = np.unique(candidates[~mask & (np.abs(candidates).sum(axis=1) <= 6)], axis=0)
    wrong = wrong[np.random.default_rng(1729).permutation(len(wrong))]
    if not 0 <= wrong_count <= len(wrong):
        raise ValueError(f"wrong_count must be between 0 and {len(wrong)}")
    return U, v, np.vstack((correct, wrong[:wrong_count]))


def design(X, exponents, chunk_size=8192):
    """Return a matrix with one row per input and one column per monomial.

    Monomials may have negative powers. Complex inputs are used for averaging.
    """
    X = np.asarray(X)
    if not np.iscomplexobj(X) and np.any(X <= 0):
        raise ValueError("Real Laurent inputs must be strictly positive")
    out = np.empty((len(X), len(exponents)), dtype=np.result_type(X, float))
    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        out[start:stop] = np.exp(np.log(X[start:stop]) @ np.asarray(exponents).T)
    return out


def change_units(X, U, g):
    """Apply mass, length, and time factors g: x_i becomes g**(-u_i) x_i.

    Use one three-element g for all rows, or a separate g for each row.
    """
    return np.asarray(X) * np.exp(-np.log(g) @ U)


def output_factor(v, g):
    """Return the factor that converts outputs back to their original units."""
    return np.exp(np.log(g) @ v)


def rescaled_design(Phi, U, exponents, v, g):
    """Rescale feature matrix Phi and convert outputs back to original units.

    This gives g**v Phi(g.x), so training targets do not need to change.
    """
    frequencies = v[None, :] - np.asarray(exponents) @ U.T
    return Phi * np.exp(np.log(g) @ frequencies.T)
