from abc import ABCMeta
from typing import Optional

import numpy as np


class AbstractScaler(metaclass=ABCMeta):
    def fit(self, data: np._typing.NDArray) -> None:
        raise NotImplementedError()

    def transform(self, data: np._typing.NDArray) -> np._typing.NDArray:
        raise NotImplementedError()

    def fit_transform(self, data: np._typing.NDArray) -> np._typing.NDArray:
        self.fit(data)
        return self.transform(data)


class MinMaxScaler(AbstractScaler):
    def __init__(self) -> None:
        self.data_min: Optional[float] = None
        self.data_max: Optional[float] = None

    def fit(self, data: np._typing.NDArray) -> None:
        if len(data) == 0:
            raise ValueError("Empty input data")
        self.data_min = data.min(axis=0)
        self.data_max = data.max(axis=0)

    def transform(self, data: np._typing.NDArray) -> np._typing.NDArray:
        if self.data_min is None or self.data_max is None:
            raise ValueError("Scaler unfitted")
        if self.data_min == self.data_max:
            return np.zeros(shape=data.shape)
        return (data - self.data_min) / (self.data_max - self.data_min)


class RobustScaler(AbstractScaler):
    def __init__(self) -> None:
        self.median: Optional[float] = None
        self.iqr: Optional[float] = None

    def fit(self, data: np._typing.NDArray) -> None:
        if len(data) == 0:
            raise ValueError("Empty input data")
        self.median = np.median(data, axis=0)
        self.iqr = np.quantile(data, q=0.75, axis=0) - np.quantile(data, q=0.25, axis=0)

    def transform(self, data: np._typing.NDArray) -> np._typing.NDArray:
        if self.median is None or self.iqr is None:
            raise ValueError("Scaler unfitted")
        return (data - self.median) / self.iqr
