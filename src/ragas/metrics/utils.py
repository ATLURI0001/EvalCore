from __future__ import annotations

import math
import typing as t

import numpy as np


def _validate_non_negative_number(value: t.Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a non-negative real number, got {type(value).__name__}")
    num = float(value)
    if not math.isfinite(num) or num < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return num


def fbeta_score(tp, fp, fn, beta=1.0):
    tp_val = _validate_non_negative_number(tp, "tp")
    fp_val = _validate_non_negative_number(fp, "fp")
    fn_val = _validate_non_negative_number(fn, "fn")
    beta_val = _validate_non_negative_number(beta, "beta")
    if beta_val == 0.0:
        raise ValueError("beta must be strictly positive")

    if tp_val + fp_val == 0:
        precision = 0.0
    else:
        precision = tp_val / (tp_val + fp_val)

    if tp_val + fn_val == 0:
        recall = 0.0
    else:
        recall = tp_val / (tp_val + fn_val)

    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta_squared = beta_val**2
    fbeta = (
        (1.0 + beta_squared)
        * (precision * recall)
        / ((beta_squared * precision) + recall)
    )

    return fbeta


def normalize_similarity_text(text: t.Optional[str]) -> str:
    """Normalize None, empty, or whitespace-only text to a single-space fallback."""
    if text is None:
        return " "
    text_str = str(text)
    if not text_str.strip():
        return " "
    return text_str


def safe_cosine_similarity(
    embedding_1: t.Any,
    embedding_2: t.Any,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> float:
    """Compute bounded cosine similarity between two embeddings with zero-norm and NaN guards."""
    if (
        isinstance(min_value, bool)
        or isinstance(max_value, bool)
        or not math.isfinite(min_value)
        or not math.isfinite(max_value)
        or min_value > max_value
    ):
        raise ValueError(f"Invalid similarity bounds: [{min_value!r}, {max_value!r}]")

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
