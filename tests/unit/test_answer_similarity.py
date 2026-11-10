from __future__ import annotations

import math
import typing as t

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbedding, BaseRagasEmbeddings
from ragas.metrics._answer_similarity import AnswerSimilarity, SemanticSimilarity
from ragas.metrics.collections import (
    SemanticSimilarity as CollectionsSemanticSimilarity,
)
from ragas.metrics.utils import normalize_similarity_text, safe_cosine_similarity
from ragas.run_config import RunConfig


class MappingModernEmbedding(BaseRagasEmbedding):
    """Deterministic modern embedding test double backed by a text-to-vector map."""

    def __init__(self, mapping: t.Dict[str, t.List[float]]):
        super().__init__()
        self.mapping = mapping
        self.calls: t.List[str] = []

    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        self.calls.append(text)
        return self.mapping[text]

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        return self.embed_text(text, **kwargs)


class MappingLegacyEmbedding(BaseRagasEmbeddings):
    """Legacy async embedding test double."""

    def __init__(self, mapping: t.Dict[str, t.List[float]]):
        super().__init__()
        self.mapping = mapping
        self.calls: t.List[str] = []
        self.set_run_config(RunConfig())

    async def aembed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return [self.mapping[t_] for t_ in texts]

    async def aembed_query(self, text: str) -> t.List[float]:
        self.calls.append(text)
        return self.mapping[text]

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return [self.mapping[t_] for t_ in texts]

    def embed_query(self, text: str) -> t.List[float]:
        self.calls.append(text)
        return self.mapping[text]


def test_normalize_similarity_text():
    assert normalize_similarity_text(None) == " "
    assert normalize_similarity_text("") == " "
    assert normalize_similarity_text("   \t\n  ") == " "
    assert normalize_similarity_text("Paris") == "Paris"
    assert normalize_similarity_text("  Paris  ") == "  Paris  "


def test_safe_cosine_similarity_clamps_floating_point_overshoot():
    # Orthogonal/collinear floats that commonly produce > 1.0 under IEEE-754 dot products
    vec = [1.0, 1.0, 1.0]
    score = safe_cosine_similarity(vec, vec)
    assert 0.0 <= score <= 1.0
    assert math.isclose(score, 1.0, rel_tol=1e-12)

    # Tiny magnitude collinear vectors
    tiny = [1e-12, 2e-12, 3e-12]
    assert safe_cosine_similarity(tiny, tiny) <= 1.0


def test_safe_cosine_similarity_zero_negative_nan_and_shape_guards():
    assert safe_cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0
    assert safe_cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert safe_cosine_similarity([], []) == 0.0
    assert safe_cosine_similarity([1.0, float("nan")], [1.0, 2.0]) == 0.0
    assert safe_cosine_similarity([1.0, float("inf")], [1.0, 2.0]) == 0.0
    # Opposing embeddings clamp cleanly to 0.0 for similarity score range
    assert safe_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == 0.0

    # 2D single-row embedding shape (1, D)
    assert math.isclose(
        safe_cosine_similarity([[1.0, 0.0]], [[1.0, 0.0]]), 1.0, rel_tol=1e-12
    )

    with pytest.raises(ValueError, match="dimension mismatch"):
        safe_cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


@pytest.mark.asyncio
async def test_legacy_semantic_similarity_handles_whitespace_and_zero_norms():
    emb = MappingModernEmbedding(
        {
            " ": [0.0, 0.0, 0.0],
            "non-empty": [1.0, 0.0, 0.0],
        }
    )
    metric = SemanticSimilarity(embeddings=emb)
    sample = SingleTurnSample(reference="   \n\t ", response="non-empty")
    score = await metric._single_turn_ascore(sample, None)
    assert score == 0.0
    assert emb.calls == [" ", "non-empty"]


@pytest.mark.asyncio
async def test_legacy_answer_similarity_threshold_including_zero():
    emb = MappingLegacyEmbedding(
        {
            "ref": [1.0, 0.0],
            "close": [0.9, 0.43588989],
            "far": [0.0, 1.0],
        }
    )
    metric_high = AnswerSimilarity(embeddings=emb, threshold=0.95)
    assert (
        await metric_high._single_turn_ascore(
            SingleTurnSample(reference="ref", response="close"), None
        )
        == 0.0
    )

    metric_low = AnswerSimilarity(embeddings=emb, threshold=0.8)
    assert (
        await metric_low._single_turn_ascore(
            SingleTurnSample(reference="ref", response="close"), None
        )
        == 1.0
    )

    metric_zero_threshold = AnswerSimilarity(embeddings=emb, threshold=0.0)
    assert (
        await metric_zero_threshold._single_turn_ascore(
            SingleTurnSample(reference="ref", response="far"), None
        )
        == 1.0
    )


@pytest.mark.asyncio
async def test_collections_semantic_similarity_guards_and_clamping():
    emb = MappingModernEmbedding(
        {
            " ": [0.0, 0.0, 0.0],
            "a": [1.0, 1.0, 1.0],
            "b": [-1.0, -1.0, -1.0],
        }
    )
    metric = CollectionsSemanticSimilarity(embeddings=emb)

    res_same = await metric.ascore(reference="a", response="a")
    assert res_same.value <= 1.0
    assert math.isclose(res_same.value, 1.0, rel_tol=1e-12)

    res_opp = await metric.ascore(reference="a", response="b")
    assert res_opp.value == 0.0

    res_blank = await metric.ascore(reference="", response="   ")
    assert res_blank.value == 0.0

    metric_zero = CollectionsSemanticSimilarity(embeddings=emb, threshold=0.0)
    res_threshold_zero = await metric_zero.ascore(reference="a", response="b")
    assert res_threshold_zero.value == 1.0
