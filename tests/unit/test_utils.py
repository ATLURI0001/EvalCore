import asyncio
import os
import tempfile

import pytest

from ragas.utils import (
    DistributionDriftResult,
    async_to_sync,
    batched,
    check_if_sum_is_close,
    compute_distribution_drift,
    compute_ks_2sample,
    compute_wasserstein_1d,
    convert_row_v1_to_v2,
    convert_row_v2_to_v1,
    create_nano_id,
    get_from_dict,
    get_or_init,
    get_required_columns_v1,
    get_test_directory,
    is_nan,
    safe_nanmean,
)


@pytest.mark.parametrize(
    ["values", "close_to", "num_places"],
    [
        [[0.1, 0.2, 0.3], 0.6, 1],
        [[0.8, 0.1, 0.1], 1.0, 1],
        [[0.94, 0.03, 0.03], 1.0, 2],
        [[0.3948, 0.3948, 0.2104], 1.0, 4],
        [[10.19, 10.19, 10.19], 30.57, 2],
    ],
)
def test_check_if_sum_is_close(values, close_to, num_places):
    assert check_if_sum_is_close(values, close_to, num_places)


def test_check_if_sum_is_close_nan_and_infinities():
    import math
    import numpy as np

    assert not check_if_sum_is_close([0.1, float("nan")], 0.6, 1)
    assert not check_if_sum_is_close([0.1, float("inf")], 0.6, 1)
    assert not check_if_sum_is_close([0.1, True], 0.6, 1)
    assert not check_if_sum_is_close([0.1, 0.5], float("nan"), 1)

    with pytest.raises(ValueError):
        check_if_sum_is_close([0.1, 0.2], 0.3, -1)


def test_safe_nanmean_handles_empty_nans_and_invalid_types():
    import numpy as np

    assert np.isnan(safe_nanmean([]))
    assert np.isnan(safe_nanmean([float("nan"), float("nan")]))
    assert np.isnan(safe_nanmean([True, False]))
    assert np.isnan(safe_nanmean(["invalid", "strings"]))
    assert safe_nanmean([1.0, 2.0, float("nan"), 3.0]) == 2.0


def test_is_nan_resilience():
    import numpy as np

    assert is_nan(float("nan")) is True
    assert is_nan(np.nan) is True
    assert is_nan(0.0) is False
    assert is_nan(100) is False
    assert is_nan(True) is False
    assert is_nan(False) is False
    assert is_nan(None) is False
    assert is_nan("not a number") is False
    assert is_nan([1, 2]) is False
    assert is_nan(np.array([1, 2])) is False


data_dict = {
    "something": {"nested": {"key": "value"}},
    "other": {"key": "value"},
    "key": "value",
    "another_key": "value",
    "nested_key": {"key": "value"},
}


@pytest.mark.parametrize(
    ["data_dict", "key", "expected"],
    [
        (data_dict, "something.nested.key", "value"),
        (data_dict, "other.key", "value"),
        (data_dict, "something.not_there_in_key", None),
        (data_dict, "something.nested.not_here", None),
        ({"items": [{"name": "first"}, {"name": "second"}]}, "items.1.name", "second"),
        ({"items": ["a", "b"]}, "items.5", None),
        (None, "key", None),
        ("not_a_dict", "key", None),
        (data_dict, "", None),
        (data_dict, None, None),
    ],
)
def test_get_from_dict(data_dict, key, expected):
    assert get_from_dict(data_dict, key) == expected


@pytest.mark.parametrize(
    ["camel_case_string", "expected"],
    [
        ("myVariableName", "my_variable_name"),
        ("CamelCaseString", "camel_case_string"),
        ("AnotherCamelCaseString", "another_camel_case_string"),
    ],
)
def test_camel_to_snake(camel_case_string, expected):
    from ragas.utils import camel_to_snake, num_tokens_from_string

    assert camel_to_snake(camel_case_string) == expected
    assert camel_to_snake("") == ""
    assert num_tokens_from_string("") == 0
    with pytest.raises(TypeError):
        camel_to_snake(12345)


class TestBatched:
    # Test cases for the `batched` function
    @pytest.mark.parametrize(
        "iterable, n, expected",
        [
            ("ABCDEFG", 3, [("A", "B", "C"), ("D", "E", "F"), ("G",)]),
            ([1, 2, 3, 4, 5, 6, 7], 2, [(1, 2), (3, 4), (5, 6), (7,)]),
            (range(5), 5, [(0, 1, 2, 3, 4)]),
            (["a", "b", "c", "d"], 1, [("a",), ("b",), ("c",), ("d",)]),
            ([], 3, []),  # Edge case: empty iterable
        ],
    )
    def test_batched(self, iterable, n: int, expected):
        result = list(batched(iterable, n))
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_batched_invalid_n(self):
        """Test that `batched` raises ValueError if n < 1 or is a boolean/invalid type."""
        with pytest.raises(ValueError, match="n must be at least one"):
            list(batched("ABCDEFG", 0))  # n = 0 should raise ValueError
        with pytest.raises(ValueError, match="n must be at least one"):
            list(batched("ABCDEFG", True))  # booleans rejected
        with pytest.raises(ValueError, match="n must be at least one"):
            list(batched("ABCDEFG", "3"))  # non-integer string rejected

    @pytest.mark.parametrize(
        "iterable, n, expected_type",
        [
            ("ABCDEFG", 3, str),
            ([1, 2, 3], 2, int),
            (["x", "y", "z"], 1, str),
        ],
    )
    def test_batched_output_type(self, iterable, n, expected_type: type):
        """Test that items in each batch maintain the original data type."""
        result = list(batched(iterable, n))
        for batch in result:
            assert all(isinstance(item, expected_type) for item in batch)


class TestCreateNanoId:
    """Test cases for the create_nano_id function."""

    def test_create_nano_id_default_size(self):
        """Test that create_nano_id generates IDs of default size (12)."""
        nano_id = create_nano_id()
        assert len(nano_id) == 12
        assert nano_id.isalnum()

    def test_create_nano_id_custom_size(self):
        """Test that create_nano_id respects custom size parameter."""
        for size in [5, 8, 16, 20]:
            nano_id = create_nano_id(size=size)
            assert len(nano_id) == size
            assert nano_id.isalnum()

    def test_create_nano_id_uniqueness(self):
        """Test that create_nano_id generates unique IDs."""
        ids = set()
        for _ in range(100):
            nano_id = create_nano_id()
            assert nano_id not in ids, "Generated duplicate ID"
            ids.add(nano_id)

    def test_create_nano_id_alphanumeric(self):
        """Test that create_nano_id only uses alphanumeric characters."""
        nano_id = create_nano_id(size=50)  # Larger size for better coverage
        for char in nano_id:
            assert char.isalnum(), f"Non-alphanumeric character found: {char}"

    def test_create_nano_id_exact_length_guarantee(self):
        """Test that create_nano_id guarantees exact requested length even for sizes > 22."""
        for size in [1, 5, 12, 22, 25, 32, 64]:
            nid = create_nano_id(size=size)
            assert len(nid) == size
            assert nid.isalnum()

    def test_create_nano_id_invalid_size_raises(self):
        """Test that invalid sizes raise ValueError."""
        import pytest
        for invalid in [0, -1, -10, True, False, "12"]:
            with pytest.raises(ValueError):
                create_nano_id(size=invalid)

    def test_create_nano_id_custom_alphabet(self):
        """Test that create_nano_id supports and validates custom alphabets."""
        import pytest
        custom = "0123456789ABCDEF"
        nid = create_nano_id(size=16, alphabet=custom)
        assert len(nid) == 16
        assert all(c in custom for c in nid)

        for invalid in ["", "a", "aaa", 123, None, []]:
            if invalid is not None:
                with pytest.raises(ValueError):
                    create_nano_id(size=10, alphabet=invalid)


class TestAsyncToSync:
    """Test cases for the async_to_sync function."""

    def test_async_to_sync_basic(self):
        """Test basic async to sync conversion."""

        async def async_add(a, b):
            await asyncio.sleep(0.001)  # Small delay to make it truly async
            return a + b

        sync_add = async_to_sync(async_add)
        result = sync_add(3, 4)
        assert result == 7

    def test_async_to_sync_with_kwargs(self):
        """Test async to sync conversion with keyword arguments."""

        async def async_multiply(x, multiplier=2):
            await asyncio.sleep(0.001)
            return x * multiplier

        sync_multiply = async_to_sync(async_multiply)
        result = sync_multiply(5, multiplier=3)
        assert result == 15

    def test_async_to_sync_exception_handling(self):
        """Test that exceptions in async functions are properly propagated."""

        async def async_error():
            await asyncio.sleep(0.001)
            raise ValueError("Test error")

        sync_error = async_to_sync(async_error)
        with pytest.raises(ValueError, match="Test error"):
            sync_error()

    def test_async_to_sync_return_types(self):
        """Test that return types are preserved."""

        async def async_return_dict():
            await asyncio.sleep(0.001)
            return {"key": "value", "number": 42}

        sync_return_dict = async_to_sync(async_return_dict)
        result = sync_return_dict()
        expected = {"key": "value", "number": 42}
        assert isinstance(result, dict) and result == expected


class TestGetTestDirectory:
    """Test cases for the get_test_directory function."""

    def test_get_test_directory_exists(self):
        """Test that get_test_directory creates a directory that exists."""
        test_dir = get_test_directory()
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)

    def test_get_test_directory_in_temp(self):
        """Test that test directory is created in system temp directory."""
        test_dir = get_test_directory()
        temp_root = tempfile.gettempdir()
        assert test_dir.startswith(temp_root)

    def test_get_test_directory_unique(self):
        """Test that get_test_directory creates unique directories."""
        dirs = set()
        for _ in range(5):
            test_dir = get_test_directory()
            assert test_dir not in dirs, "Generated duplicate directory path"
            dirs.add(test_dir)

    def test_get_test_directory_naming_pattern(self):
        """Test that test directory follows expected naming pattern."""
        test_dir = get_test_directory()
        dir_name = os.path.basename(test_dir)
        assert dir_name.startswith("ragas_test_")
        # The suffix should be the nano_id, which is alphanumeric
        suffix = dir_name[len("ragas_test_") :]
        assert suffix.isalnum()

    def test_get_test_directory_writable(self):
        """Test that the created test directory is writable."""
        test_dir = get_test_directory()
        # Try to create a file in the directory
        test_file = os.path.join(test_dir, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        # Verify file was created and has correct content
        assert os.path.exists(test_file)
        with open(test_file, "r") as f:
            content = f.read()
        assert content == "test content"


class TestRowConversionAndGetOrInit:
    """Test cases for row schema conversion and get_or_init helpers."""

    def test_get_or_init_behavior_and_guards(self):
        d = {"existing": 42}
        assert get_or_init(d, "existing", lambda: 100) == 42
        assert get_or_init(d, "missing", lambda: [1, 2, 3]) == [1, 2, 3]

        with pytest.raises(TypeError, match="must be a dict"):
            get_or_init(None, "k", lambda: 1)

        with pytest.raises(TypeError, match="must be a callable"):
            get_or_init({}, "k", 123)

    def test_convert_row_v1_to_v2(self):
        v1_row = {
            "question": "What is RAG?",
            "contexts": ["RAG is retrieval augmented generation."],
            "answer": "RAG enhances LLMs with external data.",
            "ground_truth": "Retrieval Augmented Generation.",
            "extra_field": "preserved or ignored",
        }
        v2_row = convert_row_v1_to_v2(v1_row)
        assert v2_row["user_input"] == "What is RAG?"
        assert v2_row["retrieved_contexts"] == ["RAG is retrieval augmented generation."]
        assert v2_row["response"] == "RAG enhances LLMs with external data."
        assert v2_row["reference"] == "Retrieval Augmented Generation."
        assert "extra_field" not in v2_row

        with pytest.raises(TypeError, match="row must be a dict"):
            convert_row_v1_to_v2("not a dict")

    def test_convert_row_v2_to_v1(self):
        v2_row = {
            "user_input": "What is RAG?",
            "retrieved_contexts": ["RAG is retrieval augmented generation."],
            "response": "RAG enhances LLMs with external data.",
            "reference": "Retrieval Augmented Generation.",
        }
        v1_row = convert_row_v2_to_v1(v2_row)
        assert v1_row["question"] == "What is RAG?"
        assert v1_row["contexts"] == ["RAG is retrieval augmented generation."]
        assert v1_row["answer"] == "RAG enhances LLMs with external data."
        assert v1_row["ground_truth"] == "Retrieval Augmented Generation."

        with pytest.raises(TypeError, match="row must be a dict"):
            convert_row_v2_to_v1(None)

    def test_get_required_columns_v1_safe_fallback(self):
        assert get_required_columns_v1(None) == []
        assert get_required_columns_v1(object()) == []


class TestDistributionDriftMetrics:
    def test_identical_distributions(self):
        sample = [0.12, 0.45, 0.78, 0.92, 0.33, 0.55]
        result = compute_distribution_drift(sample, sample, alpha=0.05)
        assert isinstance(result, DistributionDriftResult)
        assert result.ks_statistic == 0.0
        assert result.ks_p_value == 1.0
        assert result.wasserstein_distance == 0.0
        assert result.sample1_size == len(sample)
        assert result.sample2_size == len(sample)
        assert not result.is_drift_detected
        assert result.alpha == 0.05

    def test_disjoint_distributions_detects_drift(self):
        s1 = [0.1, 0.2, 0.15, 0.25, 0.18, 0.22]
        s2 = [0.85, 0.9, 0.88, 0.95, 0.92, 0.89]
        result = compute_distribution_drift(s1, s2, alpha=0.01)
        assert result.ks_statistic == 1.0
        assert result.ks_p_value < 0.01
        assert result.wasserstein_distance > 0.6
        assert result.is_drift_detected
        assert result.alpha == 0.01

    def test_sanitizes_nans_booleans_and_invalids(self):
        s1 = [0.2, float("nan"), True, False, "invalid", None, 0.4, 0.6]
        s2 = [0.25, 0.45, float("inf"), None, 0.65]
        result = compute_distribution_drift(s1, s2)
        assert result.sample1_size == 3  # [0.2, 0.4, 0.6]
        assert result.sample2_size == 3  # [0.25, 0.45, 0.65]
        assert 0.0 <= result.ks_statistic <= 1.0
        assert 0.0 <= result.ks_p_value <= 1.0
        assert result.wasserstein_distance >= 0.0

    def test_insufficient_samples_raises(self):
        with pytest.raises(ValueError, match="must contain at least 2 valid numeric values"):
            compute_distribution_drift([1.0], [1.0, 2.0])

        with pytest.raises(ValueError, match="must contain at least 2 valid numeric values"):
            compute_distribution_drift([1.0, 2.0], [float("nan"), True])

    def test_invalid_alpha_raises(self):
        s = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="alpha must be a float strictly between 0 and 1"):
            compute_distribution_drift(s, s, alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be a float strictly between 0 and 1"):
            compute_distribution_drift(s, s, alpha=1.0)

        with pytest.raises(ValueError, match="alpha must be a float strictly between 0 and 1"):
            compute_distribution_drift(s, s, alpha=True)

    def test_standalone_ks_and_wasserstein(self):
        ks_stat, ks_p = compute_ks_2sample([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        assert 0.0 <= ks_stat <= 1.0
        assert 0.0 <= ks_p <= 1.0

        w_dist = compute_wasserstein_1d([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        assert abs(w_dist - 1.0) < 1e-6


