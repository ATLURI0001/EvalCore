from __future__ import annotations

import typing as t

import numpy as np


def fbeta_score(tp, fp, fn, beta=1.0):
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision == 0 and recall == 0:
        return 0.0

    beta_squared = beta**2
    fbeta = (
        (1 + beta_squared)
        * (precision * recall)
        / ((beta_squared * precision) + recall)
    )

    return fbeta


def normalize_similarity_text(text: t.Optional[str]) -> str:
    """Normalize None, empty, or whitespace-only text to a single-space fallback."""
    if not text or not text.strip():
        return " "
    return text


def safe_cosine_similarity(
    embedding_1: t.Any,
    embedding_2: t.Any,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> float:
    """Compute bounded cosine similarity between two embeddings with zero-norm and NaN guards."""
    vec_1 = np.asarray(embedding_1, dtype=float).reshape(-1)
    vec_2 = np.asarray(embedding_2, dtype=float).reshape(-1)

    if vec_1.size == 0 or vec_2.size == 0:
        return 0.0

    if vec_1.size != vec_2.size:
        raise ValueError(f"Embedding dimension mismatch: {vec_1.size} != {vec_2.size}")

    if not (np.all(np.isfinite(vec_1)) and np.all(np.isfinite(vec_2))):
        return 0.0

    norm_1 = float(np.linalg.norm(vec_1))
    norm_2 = float(np.linalg.norm(vec_2))
    if norm_1 == 0.0 or norm_2 == 0.0:
        return 0.0

    similarity = float(np.dot(vec_1 / norm_1, vec_2 / norm_2))
    return float(np.clip(similarity, min_value, max_value))
