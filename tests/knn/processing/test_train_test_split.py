import numpy as np
import pytest

from src.knn.processing.train_test_split import train_test_split


class TestTrainTestSplit:
    def test_basic_split(self):
        X = np.arange(10).reshape((10, 1))
        y = np.arange(10)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        assert len(X_train) == 8
        assert len(X_test) == 2
        assert len(y_train) == 8
        assert len(y_test) == 2

    def test_different_test_sizes(self):
        X = np.arange(20).reshape((20, 1))
        y = np.arange(20)

        for test_size in [0.1, 0.2, 0.5]:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            assert len(X_test) == int(len(X) * test_size)
            assert len(y_test) == int(len(y) * test_size)

    def test_random_state(self):
        X = np.arange(50).reshape((50, 1))
        y = np.arange(50)
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_seed=42)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_seed=42)

        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(X_test1, X_test2)
        assert np.array_equal(y_train1, y_train2)
        assert np.array_equal(y_test1, y_test2)

    def test_small_dataset(self):
        X = np.array([[1], [2], [3]])
        y = np.array([0, 1, 0])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_seed=1)

        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
