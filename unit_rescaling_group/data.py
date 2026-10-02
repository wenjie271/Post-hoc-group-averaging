"""Generate data for drag-force regression and Reynolds classification."""

import hashlib
import json

import numpy as np


def rng(study, task, seed, split, purpose="data", *, unit_stream=False,
        grid_cell="NA", severity="NA", unit_system_id="NA", independent_test=False,
        training_units=None, replica=None):
    """Create repeatable random draws for one part of a study.

    Keep the original seed labels below so renamed commands give the same data.
    """
    experiment = {"compare-methods": "u1", "check-projection": "u2", "vary-size": "u4"}[study]
    if independent_test:
        experiment += "_independent_test_sets"
    if replica is not None:
        original_training = {"fixed_units": "canonical", "small_unit_shifts": "unit_randomized"}
        grid_cell = f"training:{original_training[training_units]}:replica:{replica}"
    fields = dict(master_seed=20260715, profile="v4-full" if unit_stream else "full",
                  experiment=experiment, task=task, model_seed=seed, data_seed=seed,
                  grid_cell=grid_cell, split=split, severity=severity,
                  unit_system_id=unit_system_id, phase_replica="NA", purpose=purpose)
    digest = hashlib.sha256(json.dumps(list(fields.items()), separators=(",", ":")).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big") % 2**32)


def sigmoid(logits):
    """Convert logits to probabilities without overflow."""
    logits = np.asarray(logits, dtype=float)
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1 / (1 + np.exp(-logits[positive]))
    expx = np.exp(logits[~positive])
    result[~positive] = expx / (1 + expx)
    return result


def drag_force(n, generator, *, ood=False, noise=0.01):
    """Generate force data with inputs rho, mu, V, A, ell.

    These are density, viscosity, speed, area, and length. Return true forces
    (y_star), noisy training targets (y_observed), and Re and q=A/ell**2.
    Set ood=True to sample Re and q outside their training ranges.
    """
    rho, V, ell = 10.0 ** generator.uniform(-0.5, 0.5, size=(3, n))
    if ood:
        sides = generator.integers(0, 2, size=(2, n))
        log_re = np.where(sides[0] == 0, generator.uniform(-1.5, -1.0, n), generator.uniform(1.0, 1.5, n))
        log_q = np.where(sides[1] == 0, generator.uniform(-0.7, -0.4, n), generator.uniform(0.4, 0.7, n))
    else:
        log_re = generator.uniform(-0.75, 0.75, n)
        log_q = generator.uniform(-0.3, 0.3, n)
    Re, q = 10.0**log_re, 10.0**log_q
    mu, A = rho * V * ell / Re, q * ell**2
    X = np.column_stack((rho, mu, V, A, ell))
    y_star = rho * V**2 * A * (0.7 + 0.2 * Re + 0.15 / Re)
    y_observed = y_star * (1 + noise * generator.normal(size=n))
    return dict(X=X, y_star=y_star, y_observed=y_observed, Re=Re, q=q)


def reynolds(n, generator, *, ood=False):
    """Generate binary labels and true logits from the Reynolds number Re.

    Input columns: density rho, viscosity mu, speed V, length ell.
    Set ood=True to sample Re outside its training range.
    """
    rho, V, ell = 10.0 ** generator.uniform(-0.5, 0.5, size=(3, n))
    if ood:
        side = generator.integers(0, 2, n)
        log_re = np.where(side == 0, generator.uniform(-1.5, -1.0, n), generator.uniform(1.0, 1.5, n))
    else:
        log_re = generator.uniform(-0.75, 0.75, n)
    Re = 10.0**log_re
    mu = rho * V * ell / Re
    X = np.column_stack((rho, mu, V, ell))
    logits_star = 2.5 * (Re - 1 / Re)
    labels = generator.binomial(1, sigmoid(logits_star), size=n)
    return dict(X=X, logits_star=logits_star, labels=labels, Re=Re)


GENERATORS = {"regression": drag_force, "classification": reynolds}
