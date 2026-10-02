"""Symmetry projections and averaging."""

from itertools import combinations, product

import numpy as np

from features import allowed, change_units, output_factor
from models import fit_linear


def weyl_project(coefficients, U, exponents, v):
    """Set coefficients to zero when their feature units do not match v."""
    return np.where(allowed(U, exponents, v), coefficients, 0.0)


def empirical_l2(Phi, coefficients, mask):
    """Fit allowed features to baseline predictions on separate inputs.

    Phi is the feature matrix after unit changes. Use 80% of rows to fit and
    20% to choose the penalty. Use predictions as targets, without labels.
    For classification, fit the baseline logits.
    """
    pseudo = Phi @ coefficients
    cut = int(0.8 * len(Phi))
    retained, penalty = fit_linear(Phi[:cut, mask], pseudo[:cut],
                                   Phi[cut:, mask], pseudo[cut:])
    result = np.zeros_like(coefficients)
    result[mask] = retained
    return result, penalty


def frame_sections(U):
    """Choose the four input triples used for frame averaging in Eq. (20)."""
    return [pivot for pivot in combinations(range(U.shape[1]), U.shape[0])
            if abs(round(np.linalg.det(U[:, pivot]))) == 1]


def frame_average(forward, X, U, v):
    """Set each chosen triple to one, predict, convert back, and average."""
    predictions = []
    for pivot in frame_sections(U):
        log_g = np.log(X[:, pivot]) @ np.linalg.inv(U[:, pivot])
        g = np.exp(log_g)
        predictions.append(output_factor(v, g) * forward(change_units(X, U, g)))
    return np.mean(predictions, axis=0)


def frame_polynomial(coefficients, exponents, U, v):
    """Return monomials and coefficients for the same frame average."""
    sections = frame_sections(U)
    combined = {}
    for pivot in sections:
        inverse = np.rint(np.linalg.inv(U[:, pivot].T)).astype(int)
        selector = np.eye(U.shape[1], dtype=int)[list(pivot)]
        mapped = exponents - (exponents @ U.T - v) @ inverse @ selector
        for exponent, coefficient in zip(mapped, coefficients / len(sections)):
            key = tuple(exponent)
            combined[key] = combined.get(key, 0.0) + float(coefficient)
    keys = sorted(combined)
    return np.asarray(keys, dtype=int), np.array([combined[key] for key in keys])


def paired_phases(generator, count):
    """Draw random phase triples and pair each theta with -theta."""
    if count <= 0 or count % 2:
        raise ValueError("The number of paired model evaluations must be positive and even")
    theta = generator.uniform(0, 2*np.pi, size=(count//2, 3))
    return np.stack((theta, -theta), axis=1).reshape(count, 3)


def grid_phases(grid):
    """Make an evenly spaced phase grid for the three unit factors."""
    return 2*np.pi * np.array(list(product(*(range(n) for n in grid)))) / np.array(grid)


def grid_is_exact(U, exponents, v, grid):
    """Check that the phase grid cancels every feature with incorrect units."""
    frequencies = v - exponents @ U.T
    wrong = np.any(frequencies != 0, axis=1)
    return not np.any(wrong & np.all(frequencies % np.asarray(grid) == 0, axis=1))


def torus_values(forward, X, U, v, phases):
    """Evaluate complex unit changes, including the output correction.

    Average these values to approximate the Weyl projection.
    """
    return np.array([np.exp(1j*theta @ v) * forward(X*np.exp(-1j*theta @ U))
                     for theta in phases])
