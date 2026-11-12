from ragas.metrics.validators import (
    DiscreteValidator,
    NumericValidator,
    RankingValidator,
    get_metric_type_name,
    get_validator_for_allowed_values,
)


class _SampleNumeric(NumericValidator):
    name = "num"

    def __init__(self, allowed):
        self.allowed_values = allowed


class _SampleRanking(RankingValidator):
    name = "rank"

    def __init__(self, allowed):
        self.allowed_values = allowed


def test_numeric_validator_rejects_booleans_and_non_finite():
    v_tuple = _SampleNumeric((0.0, 1.0))
    assert v_tuple.validate_result_value(0.5) is None
    assert v_tuple.validate_result_value(True) is not None
    assert v_tuple.validate_result_value(False) is not None
    assert v_tuple.validate_result_value(float("nan")) is not None
    assert v_tuple.validate_result_value(float("inf")) is not None

    v_range = _SampleNumeric(range(0, 5))
    assert v_range.validate_result_value(2) is None
    assert v_range.validate_result_value(2.0) is None
    assert v_range.validate_result_value(2.5) is not None
    assert v_range.validate_result_value(True) is not None


def test_bool_allowed_values_does_not_resolve_as_ranking_metric():
    assert get_validator_for_allowed_values(True) is DiscreteValidator  # type: ignore[arg-type]
    assert get_metric_type_name(True) == "CustomMetric"  # type: ignore[arg-type]
    assert _SampleRanking(True).validate_result_value("ignored") is None  # type: ignore[arg-type]
