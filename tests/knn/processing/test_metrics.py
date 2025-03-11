import numpy as np
import pytest

from src.knn.processing.metrics import accuracy_score, f1_score


class TestAccuracyScore:

    @pytest.mark.parametrize(
        ["y_true", "y_pred", "expected"],
        [
            ([1, 0, 1, 1, 0], [1, 0, 1, 1, 0], 1.0),
            ([1, 0, 1, 1, 0], [0, 1, 0, 0, 1], 0.0),
            ([1, 0, 1, 1, 0], [1, 0, 0, 1, 1], 3 / 5),
            ([2, 0, 1, 2, 1, 0], [2, 0, 1, 1, 2, 0], 4 / 6),
        ],
    )
    def test_different_accuracy(self, y_true, y_pred, expected):
        assert accuracy_score(np.array(y_pred), np.array(y_true)) == expected

    def test_empty_input(self):
        with pytest.raises(ValueError):
            accuracy_score(np.array([]), np.array([]))

    def test_mismatched_lengths(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0])
        with pytest.raises(ValueError):
            accuracy_score(y_pred, y_true)


class TestFScore:
    @pytest.mark.parametrize(
        "y_true, y_pred, expected_f1",
        [
            ([1, 0, 1, 1, 0], [1, 0, 1, 1, 0], 1.0),
            ([1, 1, 1, 0, 0], [0, 0, 0, 1, 1], 0.0),
            ([1, 0, 1, 1, 0], [1, 0, 0, 1, 1], 0.6667),
            ([2, 0, 1, 2, 1, 0], [2, 0, 1, 1, 2, 0], 0.6667),
        ],
    )
    def test_f1_score(self, y_true, y_pred, expected_f1):
        assert f1_score(np.array(y_pred), np.array(y_true)) == pytest.approx(expected_f1, rel=1e-3)

    @pytest.mark.parametrize(
        "y_true, y_pred, expected_exception",
        [([], [], ValueError), ([1, 0, 1], [1, 0], ValueError)],
    )
    def test_f1_score_exceptions(self, y_true, y_pred, expected_exception):
        with pytest.raises(expected_exception):
            f1_score(np.array(y_true), np.array(y_pred))
