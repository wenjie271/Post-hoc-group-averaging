# Unit-rescaling experiments

Code for Section 4.5.2 and Appendix C of the paper (PDF version 42).
All code is in `unit_rescaling_group/`, which runs on its own.

## Files

- [`unit_rescaling_group/data.py`](unit_rescaling_group/data.py): generate force-regression and Reynolds-classification data.
- [`unit_rescaling_group/features.py`](unit_rescaling_group/features.py): build features and change units.
- [`unit_rescaling_group/models.py`](unit_rescaling_group/models.py): fit models and measure errors.
- [`unit_rescaling_group/projections.py`](unit_rescaling_group/projections.py): apply the projection and averaging methods.
- [`unit_rescaling_group/run.py`](unit_rescaling_group/run.py): run the studies and save results.
- [`unit_rescaling_group/requirements.txt`](unit_rescaling_group/requirements.txt): package dependencies.

## Run

From the repository root, enter the code folder, install the packages, and try a small test:

```bash
cd unit_rescaling_group
python -m pip install -r requirements.txt
python run.py compare-methods --quick
```

From the same code folder, run the full studies:

```bash
python run.py compare-methods    # Table 2: compare the five prediction methods
python run.py check-projection   # Table 6: compare numerical and exact projection
python run.py vary-size          # Table 7: vary training size and feature count
```

Add `--quick` for a small test, `--seeds 1` for one seed, or `--output PATH`
to choose where results go. For example, `compare-methods` saves to
`results/compare-methods`; its quick run saves to `results/compare-methods_quick`.
These paths are inside `unit_rescaling_group/` when using the commands above.
Quick runs check the code with less data; use full runs for the paper tables.

## Methods

- **Baseline:** fit ridge or logistic regression using features with both correct and incorrect physical units.
- **Weyl projection:** zero the baseline coefficients of features with incorrect units.
- **Empirical L2 projection:** fit allowed features to baseline predictions on separate inputs, without labels.
- **Invariant regression:** train a new model using only features with correct units.
- **Frame average:** set each of the four input triples in Eq. (20) to one, predict, convert back, and average.

“Correct units” means force units for regression and dimensionless features
for classification. The name “Invariant regression” follows the paper;
its force predictions scale with the output units.
For classification, projections and averaging act on logits before converting
them to probabilities.

Training uses fixed units or random unit factors between 0.1 and 10.
Tests also include larger shifts, with factors between 0.01 and 100.
Random factors are sampled uniformly on a log scale. Predictions are converted
back to the original units before scoring.

## Results

`rows.csv` contains individual results; `summary.csv` contains averages.

For `compare-methods` and `vary-size`:

- `training_units` and `test_units` are `fixed_units`, `small_unit_shifts`, or `large_unit_shifts`. Training uses the first two.
- `train_size` is the number of training examples, N in the paper.
- `non_invariant_features` is the number of features with incorrect units, W in the paper.
- `method` names the fitted or projected model.
- `value` is NRMSE for regression or Bernoulli KL for classification; lower is better.
- `mean` and `sample_sd` give the mean and sample standard deviation over seeds, after averaging over test unit systems within each seed.

Full runs use ten seeds and 16 unit systems per transformed test condition.
Each test condition uses a separate physical sample, so even invariant models
can have slightly different errors across table columns.

`compare-methods` uses 2,000 training examples and 250 non-invariant features.
`vary-size` tests N = 50, 100, 300, 1000, 3000 and W = 0, 25, 100, 250, 500.
For Table 7, select `non_invariant_features=500` and `train_size` 50 or 3000.

`check-projection` compares a fixed phase grid and paired Monte Carlo averages
with exact coefficient filtering. It uses five seeds for each task and training
setting. `relative_error` measures the difference from the exact projection;
`evaluations` counts model calls and `phase_sequence` identifies a random phase
sequence. Its summary gives the mean, median, and maximum relative error.
For Table 6, use the fixed-grid row and Monte Carlo counts 32, 128, and 256.
The fixed-grid count is marked `varies_by_task`: 672 for regression and 637 for
classification.

## Notes

Adapted from `src/unit_weyl/` and `scripts/` in the original repository.
The seed labels in `data.py` preserve the original random draws.
Tested with Python 3.14.2 and the packages in `requirements.txt`.
The full method comparison and numerical projection check reproduced
Tables 2 and 6. One full seed of the training-size grid was also checked
against the original results, within `1e-10` relative error.
