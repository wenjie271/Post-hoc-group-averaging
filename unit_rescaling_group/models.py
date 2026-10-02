"""Fit linear models and measure prediction error."""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss

from data import sigmoid

PENALTIES = [10.0**power for power in range(-10, 3)]


def nrmse(prediction, target):
    """Return root mean squared error divided by the target's root mean square."""
    return float(np.sqrt(np.sum(np.abs(prediction - target)**2)
                         / (np.sum(np.abs(target)**2) + np.finfo(float).eps)))


def bernoulli_kl(logits, true_logits):
    """Measure the mean KL divergence from true to predicted probabilities."""
    p = np.clip(sigmoid(true_logits), 1e-15, 1 - 1e-15)
    q = np.clip(sigmoid(logits), 1e-15, 1 - 1e-15)
    return float(np.mean(p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))))


def fit_linear(Phi, y, validation, y_validation, *, task="regression", penalties=PENALTIES):
    """Fit a model and choose its penalty using validation data.

    Scale each feature by its training root mean square (RMS), without centering.
    Fit no intercept. Return coefficients in the original scale and the penalty.
    Ridge uses summed squared loss; logistic uses mean log loss + lambda*||w||²/2.
    """
    scales = np.sqrt(np.mean(np.asarray(Phi)**2, axis=0))
    X, V = Phi / scales, validation / scales
    candidates = []
    for penalty in penalties:
        if task == "regression":
            model = Ridge(alpha=penalty, fit_intercept=False).fit(X, y)
            score = nrmse(model.predict(V), y_validation)
            coefficients = model.coef_.copy()
        else:
            model = LogisticRegression(C=1/(penalty*len(y)), l1_ratio=0,
                                       fit_intercept=False, solver="lbfgs", max_iter=5000).fit(X, y)
            score = log_loss(y_validation, sigmoid(model.decision_function(V)))
            coefficients = model.coef_[0].copy()
        candidates.append((score, penalty, coefficients))
    _, penalty, coefficients = min(candidates, key=lambda item: (item[0], item[1]))
    return coefficients / scales, float(penalty)
