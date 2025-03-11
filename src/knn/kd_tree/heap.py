import heapq
from typing import Optional

import numpy as np

from src.knn.knn_typing import PointType


class Heap:
    def __init__(self, size: int):
        self.id = 1
        self.size = size
        self.heap: list[tuple[float, int, Optional[PointType]]] = [(-np.inf, 1, None)]
        heapq.heapify(self.heap)

    def push(self, addition: tuple[float, PointType]) -> None:
        changed_addition = (-addition[0], self.id, addition[1])
        if len(self.heap) < self.size:
            heapq.heappush(self.heap, changed_addition)
        elif changed_addition[0] > self.heap[0][0]:
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, changed_addition)
        self.id += 1

    def get_max(self) -> tuple[float, int, Optional[PointType]]:
        return self.heap[0]

    def get_all_elements(self) -> list[tuple[float, int, Optional[PointType]]]:
        return self.heap
