from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from src.knn.kd_tree.heap import Heap
from src.knn.knn_typing import PointsContainer, PointType


@dataclass
class Leaf:
    points: PointsContainer


@dataclass
class Node:
    key: PointType
    axis: int
    left: Optional["Node"] | Leaf = None
    right: Optional["Node"] | Leaf = None


class KDTree:
    def __init__(self, points: PointsContainer, leaf_size: int, metric: Callable):
        self.leaf_size = leaf_size
        self.root = self._build_tree(points)
        self.metric = metric

    @staticmethod
    def _validate_points(points: PointsContainer) -> np._typing.NDArray:
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("Points should be unique")
        try:
            valid_points = np.array(points)
        except ValueError:
            raise ValueError("Points container should be Sequence and all points must have same arity")
        if len(valid_points.shape) != 2:
            raise ValueError("Points should be from R^(m x n)")
        return valid_points

    def _build_tree(self, points: PointsContainer) -> Node | Leaf:
        def build_tree_recursion(points: np._typing.NDArray) -> Node | Leaf:
            if len(points) < self.leaf_size * 2 + 1:
                return Leaf(points)
            axis = int(np.argmax(points.std(axis=0)))
            sorted_x = points[points[:, axis].argsort()]
            current_root = Node(sorted_x[len(points) // 2], axis)
            current_root.left = build_tree_recursion(np.array([sorted_x[i] for i in range(len(points) // 2)]))
            current_root.right = build_tree_recursion(
                np.array([sorted_x[i] for i in range(len(points) // 2 + 1, len(points))])
            )
            return current_root

        if self.leaf_size <= 0:
            raise ValueError("Leaf size must be positive")

        valid_points = self._validate_points(points)
        self.dim: int = valid_points.shape[1]
        return build_tree_recursion(valid_points)

    def _find_k_neighbors(self, fixed_point: PointType, k: int) -> PointsContainer:
        heap = Heap(k)

        def _find_recursion(curr_node: Optional[Node | Leaf]) -> None:
            if curr_node is None:
                return
            if isinstance(curr_node, Leaf):
                distances = [self.metric(point, fixed_point) for point in curr_node.points]
                for pair in zip(distances, curr_node.points):
                    heap.push(pair)
                return
            distance = self.metric(curr_node.key, fixed_point)
            if -distance > heap.get_max()[0]:
                heap.push((distance, curr_node.key))
            if fixed_point[curr_node.axis] < curr_node.key[curr_node.axis]:
                _find_recursion(curr_node.left)
                if np.abs(curr_node.key[curr_node.axis] - fixed_point[curr_node.axis]) < -heap.get_max()[0]:
                    _find_recursion(curr_node.right)
            else:
                _find_recursion(curr_node.right)
                if np.abs(curr_node.key[curr_node.axis] - fixed_point[curr_node.axis]) < -heap.get_max()[0]:
                    _find_recursion(curr_node.left)

        _find_recursion(self.root)
        return np.array([pair[2] for pair in heap.get_all_elements()])

    def query(self, points: PointsContainer, k: int) -> PointsContainer:
        if k <= 0:
            raise ValueError("Number of neighbors must be positive")
        valid_points = self._validate_points(points)
        if valid_points.shape[1] != self.dim:
            raise ValueError("Incorrect points arity")
        return np.array([self._find_k_neighbors(point, k) for point in valid_points])
