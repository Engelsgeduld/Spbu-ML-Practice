import numpy as np
import pytest
from scipy.spatial.distance import euclidean

from src.knn.kd_tree.kdtree import KDTree


class DefaultFounder:
    def __init__(self, X):
        self.X = X

    def k_neighbors(self, points, k):
        def one_point_find(fixed):
            res = sorted([(euclidean(fixed, p), p) for p in self.X], key=lambda x: x[0])[:k]
            return [np.array(pair[1]) for pair in res]

        return [one_point_find(point) for point in points]


class TestKDTree:
    @pytest.fixture
    def get_dataset(self):
        point_dim = list(range(2, 30))
        train_dataset = [np.random.rand(200, dim) for dim in point_dim]
        test_dataset = [np.random.rand(30, dim) for dim in point_dim]
        return train_dataset, test_dataset

    def test_k_neighbors_search(self, get_dataset):
        train_dataset, test_dataset = get_dataset
        for train, test in zip(train_dataset, test_dataset):
            tree = KDTree(train, 2, euclidean)
            tree_res = tree.query(test, 5)
            default = DefaultFounder(train)
            default_res = default.k_neighbors(test, 5)
            assert np.array_equiv(np.sort(tree_res, axis=1), np.sort(default_res, axis=1))

    @pytest.mark.parametrize(
        "points",
        [
            [1, 2, 3, 4, 5],
            [(1, 1), (1, 1), (2, 2)],
            [(1, 1), (1, 1, 1)],
            {1: 2, 3: 4},
            "123",
            (1, 2),
        ],
    )
    def test_validate_points(self, points):
        with pytest.raises(ValueError):
            tree = KDTree(points, 2, euclidean)
