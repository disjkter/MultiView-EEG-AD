"""Transferability metrics for lightweight source-model auditing."""

from __future__ import annotations

import math

import numpy as np


def _standardize_features(features: np.ndarray) -> np.ndarray:
    """Return finite, column-standardized features shaped [N, D]."""

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("features must have shape [N, D] with N >= 2")
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = (x - mean) / np.maximum(std, 1e-8)
    return np.nan_to_num(x, copy=False)


def logme_score(features: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> float:
    """Estimate normalized LogME for a binary or multi-class target task.

    The implementation follows Bayesian linear-regression evidence updates and
    averages one-vs-rest evidence over classes.
    """

    x = _standardize_features(features)
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (x.shape[0],):
        raise ValueError("labels must have shape [N]")
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("LogME requires at least two target classes")

    # Economy SVD keeps the computation practical when D is much larger than N.
    u, singular_values, vh = np.linalg.svd(x, full_matrices=False)
    eigenvalues = singular_values**2
    rank = len(eigenvalues)
    n_samples, n_features = x.shape
    scores: list[float] = []

    for class_id in classes:
        target = (y == class_id).astype(np.float64)
        projected = u.T @ target
        target_norm_sq = float(target @ target)
        alpha = 1.0
        beta = 1.0

        for _ in range(max_iter):
            denom = alpha + beta * eigenvalues
            gamma = float(np.sum(beta * eigenvalues / np.maximum(denom, 1e-12)))
            coefficients = beta * (vh.T @ (projected / np.maximum(denom, 1e-12)))
            weight_norm_sq = float(coefficients @ coefficients)
            prediction = x @ coefficients
            residual_sq = float(np.sum((target - prediction) ** 2))
            new_alpha = gamma / max(weight_norm_sq, 1e-12)
            new_beta = max(n_samples - gamma, 1e-8) / max(residual_sq, 1e-12)
            if abs(new_alpha - alpha) / max(alpha, 1e-12) < 1e-5 and abs(
                new_beta - beta
            ) / max(beta, 1e-12) < 1e-5:
                alpha, beta = new_alpha, new_beta
                break
            alpha, beta = new_alpha, new_beta

        denom = alpha + beta * eigenvalues
        coefficients = beta * (vh.T @ (projected / np.maximum(denom, 1e-12)))
        residual_sq = float(np.sum((target - x @ coefficients) ** 2))
        weight_norm_sq = float(coefficients @ coefficients)
        log_determinant = float(np.log(np.maximum(denom, 1e-12)).sum())
        if n_features > rank:
            log_determinant += float((n_features - rank) * math.log(max(alpha, 1e-12)))
        evidence = (
            n_features * math.log(max(alpha, 1e-12))
            + n_samples * math.log(max(beta, 1e-12))
            - beta * residual_sq
            - alpha * weight_norm_sq
            - log_determinant
            - n_samples * math.log(2.0 * math.pi)
        ) / 2.0
        scores.append(evidence / n_samples)
    return float(np.mean(scores))


def _coding_rate(features: np.ndarray, epsilon: float) -> float:
    """Compute the coding-rate surrogate used by TransRate."""

    x = _standardize_features(features)
    n_samples, n_features = x.shape
    gram = x @ x.T if n_samples <= n_features else x.T @ x
    scale = n_features / max(n_samples * epsilon, 1e-12)
    sign, logdet = np.linalg.slogdet(np.eye(gram.shape[0]) + scale * gram)
    if sign <= 0:
        raise FloatingPointError("coding-rate matrix is not positive definite")
    return float(0.5 * logdet)


def transrate_score(features: np.ndarray, labels: np.ndarray, epsilon: float = 1e-4) -> float:
    """Return TransRate: global coding rate minus class-conditional rates."""

    x = np.asarray(features)
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (x.shape[0],):
        raise ValueError("labels must have shape [N]")
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("TransRate requires at least two target classes")
    total = _coding_rate(x, epsilon)
    conditional = 0.0
    for class_id in classes:
        class_x = x[y == class_id]
        if len(class_x) < 2:
            continue
        conditional += len(class_x) / len(x) * _coding_rate(class_x, epsilon)
    return float(total - conditional)


def leep_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Compute Log Expected Empirical Prediction from source probabilities."""

    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or probs.shape[0] != len(y):
        raise ValueError("probabilities must have shape [N, source_classes]")
    probs = np.clip(probs, 1e-12, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    target_classes = np.unique(y)
    joint = np.zeros((len(target_classes), probs.shape[1]), dtype=np.float64)
    for target_idx, class_id in enumerate(target_classes):
        joint[target_idx] = probs[y == class_id].sum(axis=0)
    joint /= len(y)
    source_marginal = joint.sum(axis=0, keepdims=True)
    target_given_source = joint / np.maximum(source_marginal, 1e-12)
    class_to_index = {int(class_id): idx for idx, class_id in enumerate(target_classes)}
    empirical_prediction = probs @ target_given_source.T
    selected = np.asarray(
        [empirical_prediction[idx, class_to_index[int(label)]] for idx, label in enumerate(y)]
    )
    return float(np.log(np.maximum(selected, 1e-12)).mean())


def conditional_prototype_distance(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
) -> float:
    """Return mean cosine distance between matching class prototypes."""

    source_x = np.asarray(source_features, dtype=np.float64)
    target_x = np.asarray(target_features, dtype=np.float64)
    source_y = np.asarray(source_labels, dtype=np.int64)
    target_y = np.asarray(target_labels, dtype=np.int64)
    shared_classes = sorted(set(source_y.tolist()).intersection(target_y.tolist()))
    if len(shared_classes) < 2:
        raise ValueError("prototype distance requires at least two shared classes")
    distances: list[float] = []
    for class_id in shared_classes:
        source_proto = source_x[source_y == class_id].mean(axis=0)
        target_proto = target_x[target_y == class_id].mean(axis=0)
        denom = np.linalg.norm(source_proto) * np.linalg.norm(target_proto)
        cosine = float(source_proto @ target_proto / max(denom, 1e-12))
        distances.append(1.0 - float(np.clip(cosine, -1.0, 1.0)))
    return float(np.mean(distances))


def rank_aggregate(rows: list[dict[str, float]]) -> list[float]:
    """Aggregate heterogeneous metrics into a [0, 1] source-ranking score."""

    if not rows:
        return []
    if len(rows) == 1:
        return [1.0]
    high_is_good = ("logme", "transrate", "leep", "gradient_cosine")
    low_is_good = (
        "graph_prototype_distance",
        "temporal_prototype_distance",
        "fused_prototype_distance",
    )
    score = np.zeros(len(rows), dtype=np.float64)
    used = 0
    for key in (*high_is_good, *low_is_good):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            continue
        tolerance = max(1e-6, 1e-3 * max(1.0, abs(float(values.mean()))))
        if float(values.max() - values.min()) < tolerance:
            continue
        order = np.argsort(values)
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        if len(values) > 1:
            ranks /= len(values) - 1
        else:
            ranks[:] = 1.0
        if key in low_is_good:
            ranks = 1.0 - ranks
        score += ranks
        used += 1
    if used == 0:
        raise ValueError("no finite transferability metrics to aggregate")
    return (score / used).tolist()
