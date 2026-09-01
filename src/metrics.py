"""Evaluation metrics used by the released forecasting and interval-fusion code."""

import numpy as np


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


def interval_metrics(upper, lower, observed):
    """Return PICP, PINAW, and a compact combined F metric."""
    upper = np.asarray(upper)
    lower = np.asarray(lower)
    observed = np.asarray(observed)

    picp = np.mean((upper >= observed) & (lower <= observed))
    data_range = np.ptp(observed)
    pinaw = np.mean(upper - lower) / (data_range + 1.0e-12)
    f_metric = (2.0 * picp / (pinaw + 1.0e-6)) / (
        picp + 1.0 / (pinaw + 1.0e-6) + 1.0e-12
    )
    return picp, pinaw, f_metric


def high_flow_metrics(upper, lower, observed, flood_threshold):
    """
    Return high-flow PICP, high-flow PINAW, high-flow F, and SIS.

    SIS follows the study implementation: high-flow values are transformed
    with tanh before calculating the mean out-of-interval distance.
    """
    upper = np.asarray(upper)
    lower = np.asarray(lower)
    observed = np.asarray(observed)

    mask = observed >= flood_threshold
    if not np.any(mask):
        return np.nan, np.nan, np.nan, np.nan

    upper_h = upper[mask]
    lower_h = lower[mask]
    observed_h = observed[mask]

    picp = np.mean((upper_h >= observed_h) & (lower_h <= observed_h))
    data_range = np.ptp(observed_h)
    pinaw = np.mean(upper_h - lower_h) / (data_range + 1.0e-12)
    f_metric = (2.0 * picp / (pinaw + 1.0e-6)) / (
        picp + 1.0 / (pinaw + 1.0e-6) + 1.0e-12
    )

    upper_t = np.tanh(upper_h)
    lower_t = np.tanh(lower_h)
    observed_t = np.tanh(observed_h)

    sis = np.zeros_like(observed_t, dtype=float)
    valid = upper_t >= lower_t

    above = valid & (upper_t < observed_t)
    below = valid & (lower_t > observed_t)
    inside = valid & (lower_t <= observed_t) & (observed_t <= upper_t)
    invalid = ~valid

    sis[above] = observed_t[above] - upper_t[above]
    sis[below] = lower_t[below] - observed_t[below]
    sis[inside] = 0.0
    sis[invalid] = 10.0

    return picp, pinaw, f_metric, float(np.mean(sis))
