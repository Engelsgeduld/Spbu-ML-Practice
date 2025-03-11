from typing import Callable, Optional

import numpy as np

from src.knn.kd_tree.kdtree import KDTree


class KNNClassifier:
    def __init__(self, k: int, leaf_size: int, metric: Callable):
        self.targets: Optional[np._typing.NDArray] = None
        self.classifier: Optional[dict] = None
        self.model: Optional[KDTree] = None
        self.k = k
        self.leaf_size = leaf_size
        self.metric = metric

    def fit(self, features: np._typing.NDArray, targets: np._typing.NDArray) -> None:
        if len(features) != len(targets):
            raise ValueError("Features and targets must be same lenght")
        self.model = KDTree(features, self.leaf_size, self.metric)
        self.classifier = dict((tuple(pair[0]), pair[1]) for pair in zip(features.tolist(), targets.tolist()))
        self.targets = targets

    def _predict_proba(self, data: np._typing.NDArray) -> list:
        if self.model is None or self.classifier is None or self.targets is None:
            raise ValueError("Model unfitted")
        probability = []
        for point in data:
            point = tuple(point)
            if point in self.classifier:
                probability.append(
                    (
                        np.unique(self.targets),
                        (self.classifier[point] == np.unique(self.targets)).astype(int),
                    )
                )
            else:
                result = self.model.query([point], self.k)
                target_result = np.array([self.classifier[tuple(neighbors.tolist())] for neighbors in result[0]])
                counts = np.array([(target_result == val).sum() for val in np.unique(self.targets)])
                probability.append((self.targets, counts / len(result[0])))
        return probability

    def predict_proba(self, data: np._typing.NDArray) -> np._typing.ArrayLike:
        results = self._predict_proba(data)
        print(results)
        return np.array([result[1] for result in results])

    def predict(self, data: np._typing.NDArray) -> np._typing.ArrayLike:
        results = self._predict_proba(data)
        return np.array([result[0][np.argmax(result[1])] for result in results])
