from typing import Optional

import numpy as np


def train_test_split(
    X: np._typing.NDArray,
    Y: np._typing.NDArray,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_seed: Optional[int] = None,
) -> tuple[np._typing.NDArray, np._typing.NDArray, np._typing.NDArray, np._typing.NDArray]:
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be between 0 and 1")

    if len(X) != len(Y):
        raise ValueError("X and Y must be of the same length")

    if random_seed is not None:
        np.random.seed(random_seed)

    indices = np.arange(len(X))

    if shuffle:
        np.random.shuffle(indices)

    split_idx = int(len(indices) * (1 - test_size))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]

    return X_train, X_test, Y_train, Y_test
