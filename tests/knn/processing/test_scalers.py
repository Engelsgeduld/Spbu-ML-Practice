import numpy as np
import pytest

from src.knn.processing.scalers import MinMaxScaler, RobustScaler


class TestMinMaxScaler:
    @pytest.fixture
    def scaler(self):
        return MinMaxScaler()

    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                np.array([[1], [2], [3], [4], [5]]),
                np.array([[0.0], [0.25], [0.5], [0.75], [1.0]]),
            ),
            (
                np.array([[-5], [0], [5], [10]]),
                np.array([[0.0], [0.33], [0.67], [1.0]]),
            ),
            (np.array([[5], [5], [5]]), np.array([[0], [0], [0]])),
        ],
    )
    def test_different_scaling(self, scaler, data, expected):
        transformed = scaler.fit_transform(data)
        np.testing.assert_almost_equal(transformed, expected, decimal=2)

    def test_single_feature_input(self, scaler):
        data = np.array([[10], [20], [30]])
        transformed = scaler.fit_transform(data)
        assert transformed.min() == 0 and transformed.max() == 1

    def test_unfitted_exception(self, scaler):
        data = np.array([[5], [5], [5]])
        with pytest.raises(ValueError):
            scaler.transform(data)

    def test_empty_input(self, scaler):
        X = np.array([])
        with pytest.raises(ValueError):
            scaler.fit_transform(X)


class TestRobustScaler:
    @pytest.fixture
    def scaler(self):
        return RobustScaler()

    @pytest.mark.parametrize(
        "X, median, first_quantile, third_quantile",
        [
            (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]), 0, -0.5, 0.5),
            (
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [100, 100, 100]]),
                0,
                -0.5,
                0.5,
            ),
        ],
    )
    def test_different_scaling(self, scaler, X, median, first_quantile, third_quantile):
        X_scaled = scaler.fit_transform(X)
        assert np.allclose(np.median(X_scaled, axis=0), median)
        assert np.allclose(np.percentile(X_scaled, 25, axis=0), first_quantile, atol=1e-2)
        assert np.allclose(np.percentile(X_scaled, 75, axis=0), third_quantile, atol=1e-2)

    def test_empty_input(self, scaler):
        X = np.array([])
        with pytest.raises(ValueError):
            scaler.fit_transform(X)

    def test_multidimensional_data(self, scaler):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.shape == X.shape
        assert np.allclose(np.median(X_scaled, axis=0), 0)
