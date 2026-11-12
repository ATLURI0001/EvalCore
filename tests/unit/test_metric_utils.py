import math

import pytest

from ragas.metrics.utils import fbeta_score, normalize_similarity_text, safe_cosine_similarity


def test_fbeta_score_valid_and_zero_cases():
    assert fbeta_score(2, 1, 1) == pytest.approx(2 / 3)
    assert fbeta_score(0, 0, 0) == 0.0
    assert fbeta_score(0, 2, 3, beta=2.0) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tp": True, "fp": 0, "fn": 0},
        {"tp": -1, "fp": 1, "fn": 0},
        {"tp": 1, "fp": float("nan"), "fn": 0},
        {"tp": 1, "fp": 0, "fn": 0, "beta": 0},
        {"tp": 1, "fp": 0, "fn": 0, "beta": -1.0},
        {"tp": 1, "fp": 0, "fn": 0, "beta": math.inf},
        {"tp": 1, "fp": 0, "fn": 0, "beta": False},
    ],
)
def test_fbeta_score_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        fbeta_score(**kwargs)


def test_normalize_similarity_text_handles_none_whitespace_and_non_str():
    assert normalize_similarity_text(None) == " "
    assert normalize_similarity_text("   \t\n") == " "
    assert normalize_similarity_text("hello") == "hello"
    assert normalize_similarity_text(123) == "123"  # type: ignore[arg-type]


def test_safe_cosine_similarity_bounds_validation():
    assert safe_cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Invalid similarity bounds"):
        safe_cosine_similarity([1.0, 0.0], [1.0, 0.0], min_value=0.9, max_value=0.1)
