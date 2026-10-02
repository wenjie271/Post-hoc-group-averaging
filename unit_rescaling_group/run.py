"""Run the unit-rescaling studies from Tables 2, 6, and 7."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from data import GENERATORS, rng
from features import allowed, rescaled_design, design, dictionary
from models import bernoulli_kl, fit_linear, nrmse
from projections import (empirical_l2, frame_polynomial, grid_phases, grid_is_exact,
                         paired_phases, torus_values, weyl_project)

TRAINING_UNITS = ("fixed_units", "small_unit_shifts")
TEST_UNITS = {"fixed_units": 0.0, "small_unit_shifts": 1.0, "large_unit_shifts": 2.0}


def physical_data(study, task, seed, split, n, *, independent_test=False):
    """Draw physical inputs and targets before any unit changes."""
    return GENERATORS[task](n, rng(study, task, seed, split, independent_test=independent_test))


def row_units(study, task, seed, split, n, training_units, grid_cell="NA"):
    """Choose one unit system per training, validation, or calibration row."""
    if training_units == "fixed_units":
        return np.ones((n, 3))
    generator = rng(study, task, seed, split, f"{split}_units",
                    unit_stream=True, grid_cell=grid_cell)
    return 10.0**generator.uniform(-1, 1, size=(n, 3))


def test_units(study, task, seed, severity, count):
    """Choose shared test units with log10 factors in [-severity, severity]."""
    if severity == 0:
        return [(0, np.ones(3))]
    result = []
    for unit_id in range(1, count+1):
        generator = rng(study, task, seed, "heldout_test_global", "heldout_test_units",
                        unit_stream=True, severity=severity, unit_system_id=unit_id)
        result.append((unit_id, 10.0**generator.uniform(-severity, severity, size=3)))
    return result


def fitting_data(study, task, seed, train_rows, validation_rows, calibration_rows):
    return {split: physical_data(study, task, seed, split, n)
            for split, n in [("train", train_rows), ("validation", validation_rows),
                             ("calibration", calibration_rows)]}


def fitting_designs(data, study, task, seed, training_units, U, v, exponents):
    """Build features in the units used for training and calibration."""
    result = {}
    for split, samples in data.items():
        # Share calibration units across training sizes and feature counts.
        cell = "paired_across_N_W" if study == "vary-size" and split == "calibration" else "NA"
        g = row_units(study, task, seed, split, len(samples["X"]), training_units, cell)
        result[split] = rescaled_design(design(samples["X"], exponents), U, exponents, v, g)
    return result


def fit_models(task, data, Phi, U, v, exponents, n):
    """Fit the baseline and invariant model, then apply the three symmetry methods."""
    target = "y_observed" if task == "regression" else "labels"
    mask = allowed(U, exponents, v)
    beta, penalty = fit_linear(Phi["train"][:n], data["train"][target][:n],
                               Phi["validation"], data["validation"][target], task=task)
    direct, direct_penalty = fit_linear(Phi["train"][:n, mask], data["train"][target][:n],
                                       Phi["validation"][:, mask], data["validation"][target], task=task)
    direct_full = np.zeros_like(beta)
    direct_full[mask] = direct
    empirical, empirical_penalty = empirical_l2(Phi["calibration"], beta, mask)
    models = {"baseline": (exponents, beta),
              "weyl_projection": (exponents, weyl_project(beta, U, exponents, v)),
              "empirical_l2": (exponents, empirical),
              "invariant_regression": (exponents, direct_full),
              "frame_average": frame_polynomial(beta, exponents, U, v)}
    penalties = dict(baseline=penalty, weyl_projection=penalty, frame_average=penalty,
                     invariant_regression=direct_penalty, empirical_l2=empirical_penalty)
    return models, penalties


def evaluate(models, penalties, tests, study, task, seed, training_units, U, v, n, wrong, unit_count):
    """Score every method after converting predictions back to original units."""
    rows = []
    for condition, samples in tests.items():
        units = test_units(study, task, seed, TEST_UNITS[condition], unit_count)
        log_g = np.array([np.log(g) for _, g in units])
        for model, (exponents, beta) in models.items():
            Phi = design(samples["X"], exponents)
            factors = np.exp(log_g @ (v - exponents @ U.T).T)
            predictions = Phi @ (beta[:, None] * factors.T)
            for column, (unit_id, _) in enumerate(units):
                value = (nrmse(predictions[:, column], samples["y_star"]) if task == "regression"
                         else bernoulli_kl(predictions[:, column], samples["logits_star"]))
                rows.append(dict(study=study, task=task, seed=seed,
                                 training_units=training_units, train_size=n, non_invariant_features=wrong,
                                 method=model, test_units=condition, unit_system_id=unit_id,
                                 metric="nrmse" if task == "regression" else "bernoulli_kl",
                                 value=value, penalty=penalties[model]))
    return rows


def prediction_study(study, seeds, quick=False):
    """Compare the five methods (Table 2) or vary training and feature counts (Table 7)."""
    compare_methods = study == "compare-methods"
    tasks = ("regression", "classification") if compare_methods else ("regression",)
    sizes = [2000] if compare_methods else [50, 100, 300, 1000, 3000]
    wrong_counts = [250] if compare_methods else [0, 25, 100, 250, 500]
    validation_rows, test_rows, calibration_rows, unit_count = 2000, 10000, 10000, 16
    if quick:
        sizes = [256] if compare_methods else [50, 300]
        wrong_counts = [250] if compare_methods else [0, 25]
        validation_rows, test_rows, calibration_rows, unit_count = 256, 256, 512, 2
    rows = []
    for seed in range(seeds):
        for task in tasks:
            print(f"{study}: seed {seed}, {task}", flush=True)
            data = fitting_data(study, task, seed, max(sizes), validation_rows, calibration_rows)
            # Use a separate physical test sample for each table column.
            tests = {condition: physical_data(study, task, seed, f"ID_independent_severity_{r:g}",
                                              test_rows, independent_test=True)
                     for condition, r in TEST_UNITS.items()}
            for wrong in wrong_counts:
                U, v, exponents = dictionary(task, wrong)
                for training_units in TRAINING_UNITS:
                    Phi = fitting_designs(data, study, task, seed, training_units, U, v, exponents)
                    for n in sizes:
                        models, penalties = fit_models(task, data, Phi, U, v, exponents, n)
                        rows.extend(evaluate(models, penalties, tests, study, task, seed,
                                             training_units, U, v, n, wrong, unit_count))
    return rows


def check_projection(seeds, quick=False):
    """Compare numerical averaging with exact coefficient filtering (Table 6)."""
    rows = []
    train_rows, validation_rows, test_rows, evaluation_rows = 2000, 2000, 10000, 512
    counts, replicas = [2, 4, 8, 16, 32, 64, 128, 256], 3
    if quick:
        train_rows, validation_rows, test_rows, evaluation_rows = 256, 256, 256, 32
        counts, replicas = [2, 8], 1
    for seed in range(seeds):
        for task in ("regression", "classification"):
            # Use the same baseline fits as the method comparison.
            data = fitting_data("compare-methods", task, seed, train_rows, validation_rows, 512)
            X = physical_data("compare-methods", task, seed, "ID", test_rows)["X"][:evaluation_rows]
            U, v, exponents = dictionary(task)
            grid = (6, 16, 7) if task == "regression" else (7, 13, 7)
            assert grid_is_exact(U, exponents, v, grid)
            for training_units in TRAINING_UNITS:
                print(f"check-projection: seed {seed}, {task}, {training_units}", flush=True)
                Phi = fitting_designs(data, "compare-methods", task, seed, training_units, U, v, exponents)
                target = "y_observed" if task == "regression" else "labels"
                beta, _ = fit_linear(Phi["train"], data["train"][target],
                                     Phi["validation"], data["validation"][target], task=task)
                forward = lambda z: design(z, exponents) @ beta
                exact = design(X, exponents) @ weyl_project(beta, U, exponents, v)
                reference_norm = max(np.linalg.norm(exact), np.finfo(float).tiny)
                estimate = torus_values(forward, X, U, v, grid_phases(grid)).mean(axis=0)
                rows.append(dict(task=task, seed=seed, training_units=training_units,
                                 method="deterministic_grid", evaluations=int(np.prod(grid)), phase_sequence=0,
                                 relative_error=float(np.linalg.norm(estimate-exact)/reference_norm)))
                for replica in range(replicas):
                    generator = rng("check-projection", task, seed, "held_out_ID", "phases", unit_stream=True,
                                    training_units=training_units, replica=replica)
                    phases = paired_phases(generator, max(counts))
                    values = torus_values(forward, X, U, v, phases)
                    for count in counts:
                        estimate = values[:count].mean(axis=0)
                        rows.append(dict(task=task, seed=seed, training_units=training_units,
                                         method="paired_monte_carlo", evaluations=count, phase_sequence=replica,
                                         relative_error=float(np.linalg.norm(estimate-exact)/reference_norm)))
    return rows


def summarize(rows, study):
    """Summarize projection errors or average prediction errors over units and seeds."""
    if study == "check-projection":
        groups = defaultdict(list)
        for row in rows:
            # Each task uses a different grid size.
            count = row["evaluations"] if row["method"] == "paired_monte_carlo" else "varies_by_task"
            groups[(row["method"], count)].append(row["relative_error"])
        return [dict(method=key[0], evaluations=key[1], mean=float(np.mean(values)),
                     median=float(np.median(values)), maximum=float(np.max(values)), comparisons=len(values))
                for key, values in groups.items()]
    keys = ["task", "training_units", "train_size", "non_invariant_features", "method", "test_units", "metric"]
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[tuple(row[k] for k in keys)][row["seed"]].append(row["value"])
    summary = []
    for key, by_seed in groups.items():
        values = [float(np.mean(unit_values)) for unit_values in by_seed.values()]
        summary.append(dict(zip(keys, key), mean=float(np.mean(values)),
                            sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                            seeds=len(values)))
    return summary


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["compare-methods", "check-projection", "vary-size"],
                        help="compare methods (Table 2), check numerical projection (Table 6), "
                             "or vary training size and feature count (Table 7)")
    parser.add_argument("--quick", action="store_true", help="run a small test instead of the full experiment")
    parser.add_argument("--seeds", type=int,
                        help="number of seeds (default: 5 for check-projection, 10 otherwise; 1 for --quick)")
    parser.add_argument("--output", type=Path, help="folder for rows.csv and summary.csv (default: results/STUDY)")
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else (1 if args.quick else (5 if args.study == "check-projection" else 10))
    if seeds < 1:
        parser.error("--seeds must be positive")
    output = args.output or Path("results") / (args.study + ("_quick" if args.quick else ""))
    output.mkdir(parents=True, exist_ok=True)
    rows = (check_projection(seeds, args.quick) if args.study == "check-projection"
            else prediction_study(args.study, seeds, args.quick))
    summary = summarize(rows, args.study)
    write_csv(output / "rows.csv", rows)
    write_csv(output / "summary.csv", summary)
    print(f"Saved {len(rows)} evaluations and {len(summary)} summaries in {output}")
