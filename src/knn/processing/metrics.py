import numpy as np


def confusion_matrix(y_pred: np._typing.NDArray, y_true: np._typing.NDArray) -> np._typing.NDArray:
    if len(y_pred) == 0 or len(y_true) == 0:
        raise ValueError("Empty y_pred or y_true")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    if unique_labels.size == 1:
        unique_labels = np.array([unique_labels[0], unique_labels[0] + 1])

    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    for true, pred in zip(y_true, y_pred):
        matrix[label_map[true], label_map[pred]] += 1

    return matrix


def accuracy_score(y_pred: np._typing.NDArray, y_true: np._typing.NDArray) -> float:
    if len(y_pred) == 0 or len(y_true) == 0:
        raise ValueError("Empty y_pred or y_true")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return np.sum(y_pred == y_true) / len(y_true)


def f1_score(
    y_pred: np._typing.NDArray,
    y_true: np._typing.NDArray,
    zero_division: float = 0.0,
) -> float:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        if tp == 0 or (tp + fp + fn) == 0:
            return zero_division
    else:
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        if np.any(tp == 0) or np.any((tp + fn + fp) == 0):
            return zero_division
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1.mean()
