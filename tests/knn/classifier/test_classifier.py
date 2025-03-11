import numpy as np
import pytest
from scipy.spatial.distance import euclidean

from src.knn.classifier.knn_classifier import KNNClassifier


class TestKNNClassifier:
    @pytest.fixture
    def knn(self):
        return KNNClassifier(k=3, leaf_size=2, metric=euclidean)

    @pytest.fixture
    def sample_data(self):
        X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 5], [6, 6]])
        y_train = np.array([0, 0, 1, 1, 1])
        X_test = np.array([[2, 2], [4, 5]])
        return X_train, y_train, X_test

    @pytest.mark.parametrize("count", [10, 100, 500, 1000])
    def test_predict_proba_equal_probability(self, count):
        features = np.random.rand(count, 2)
        target = np.random.permutation([1] * (count // 2) + [0] * (count // 2))
        x_test = np.random.rand(100, 2)
        classifier = KNNClassifier(count, 2, euclidean)
        classifier.fit(features, target)
        result = classifier.predict_proba(x_test)
        assert np.allclose(result.mean(axis=0), [0.5, 0.5])
        np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-6)

    def test_already_existed_point(self, knn):
        train = np.array([(0, 1), (1, 0), (0, 0), (1, 1)])
        target = np.array([0, 1, 0, 1])
        x = np.array([(0, 1)])
        knn.fit(train, target)
        assert np.array_equal(knn.predict_proba(x), np.array([[1, 0]]))

    def test_predict(self, knn, sample_data):
        X_train, y_train, X_test = sample_data
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, knn, sample_data):
        X_train, y_train, X_test = sample_data
        knn.fit(X_train, y_train)
        proba = knn.predict_proba(X_test)

        assert proba.shape == (len(X_test), len(set(y_train)))
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1)
