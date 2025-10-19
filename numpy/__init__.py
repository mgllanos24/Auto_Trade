"""Lightweight subset of the :mod:`numpy` API used in the kata tests.

This module implements a very small, list backed ``ndarray`` type together
with the handful of numerical helpers exercised by the unit tests.  It is not
feature complete – the goal is merely to provide deterministic behaviour for
the project modules without depending on the real NumPy package which is not
available in the execution environment.
"""
from __future__ import annotations

import math
from typing import Iterable, Iterator, List, Sequence, Tuple, Union

Number = Union[int, float]


def _coerce_sequence(seq: Iterable[object]) -> List[object]:
    if isinstance(seq, ndarray):
        return list(seq._data)
    return list(seq)


class ndarray:
    """Minimal list backed array implementation."""

    def __init__(self, data: Iterable[object]) -> None:
        self._data: List[object] = _coerce_sequence(data)

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)

    def __iter__(self) -> Iterator[object]:  # pragma: no cover - trivial
        return iter(self._data)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return ndarray(self._data[item])
        return self._data[item]

    def __setitem__(self, key, value) -> None:  # pragma: no cover - unused
        self._data[key] = value

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (len(self._data),)

    def tolist(self) -> List[object]:  # pragma: no cover - trivial
        return list(self._data)

    def max(self):
        return max(self._data)

    def min(self):  # pragma: no cover - convenience
        return min(self._data)

    def mean(self):
        if not self._data:
            return float("nan")
        total = 0.0
        count = 0
        for value in self._data:
            total += float(value)
            count += 1
        return total / count

    # ------------------------------------------------------------------
    # Arithmetic helpers
    # ------------------------------------------------------------------
    def _binary_op(self, other, op) -> "ndarray":
        if isinstance(other, ndarray):
            other_values = other._data
        else:
            other_values = [other] * len(self._data)
        result = [op(a, b) for a, b in zip(self._data, other_values)]
        return ndarray(result)

    def __add__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other) -> "ndarray":  # pragma: no cover - trivial
        return self.__add__(other)

    def __sub__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other) -> "ndarray":
        return ndarray([other]) - self

    def __mul__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other) -> "ndarray":  # pragma: no cover - trivial
        return self.__mul__(other)

    def __truediv__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other) -> "ndarray":
        return ndarray([other]) / self

    def __neg__(self) -> "ndarray":
        return ndarray(-value for value in self._data)

    def __le__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a <= b)

    def __lt__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a < b)

    def __ge__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a >= b)

    def __gt__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a > b)

    def astype(self, dtype) -> "ndarray":
        if dtype is float:
            return ndarray(float(x) if x is not None else nan for x in self._data)
        if dtype is int:
            return ndarray(int(x) for x in self._data)
        return ndarray(dtype(x) for x in self._data)

    def reshape(self, *_shape) -> "ndarray":  # pragma: no cover - narrow use
        shape = _shape[0] if len(_shape) == 1 else _shape
        if shape == (-1, 1):
            return ndarray([[value] for value in self._data])
        raise ValueError("Unsupported reshape operation in lightweight ndarray")


# ----------------------------------------------------------------------
# Module level helpers mirroring NumPy
# ----------------------------------------------------------------------
nan = float("nan")
inf = float("inf")
pi = math.pi


def array(seq: Iterable[object], dtype=None) -> ndarray:
    arr = ndarray(seq)
    return arr.astype(dtype) if dtype is not None else arr


def asarray(seq: Iterable[object], dtype=None) -> ndarray:
    return array(seq, dtype=dtype)


def arange(start: Number, stop: Number | None = None, step: Number = 1) -> ndarray:
    if stop is None:
        start, stop = 0, start
    values: List[Number] = []
    current = start
    if step == 0:
        raise ValueError("step must be non-zero")
    if step > 0:
        while current < stop:  # pragma: no branch - simple loop
            values.append(current)
            current += step
    else:
        while current > stop:  # pragma: no branch - simple loop
            values.append(current)
            current += step
    return ndarray(values)


def linspace(start: Number, stop: Number, num: int) -> ndarray:
    if num <= 1:
        return ndarray([float(start)])
    step = (stop - start) / (num - 1)
    values = [start + step * i for i in range(num)]
    return ndarray(values)


def sin(values: Union[ndarray, Number]) -> Union[ndarray, float]:
    if isinstance(values, ndarray):
        return ndarray(math.sin(float(v)) for v in values)
    return math.sin(float(values))


def mean(values: Union[Sequence[Number], ndarray]) -> float:
    data = _coerce_sequence(values)
    finite = [float(v) for v in data]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def abs(values: Union[ndarray, Number]) -> Union[ndarray, float]:  # pragma: no shadow
    if isinstance(values, ndarray):
        return ndarray(math.fabs(float(v)) for v in values)
    return math.fabs(float(values))


def sum(values: Union[ndarray, Sequence[Number]]) -> float:
    data = _coerce_sequence(values)
    return float(__builtins__["sum"](data))


def any(values: Union[ndarray, Sequence[object]]) -> bool:  # pragma: no shadow
    data = _coerce_sequence(values)
    return bool(__builtins__["any"](data))


def argmax(values: Union[ndarray, Sequence[Number]]) -> int:
    data = _coerce_sequence(values)
    if not data:
        raise ValueError("argmax of an empty sequence")
    max_idx = 0
    max_val = data[0]
    for idx, val in enumerate(data[1:], start=1):
        if val > max_val:
            max_val = val
            max_idx = idx
    return max_idx


def isnan(value: object) -> bool:
    try:
        return math.isnan(float(value))
    except Exception:
        return False


def isfinite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def sqrt(value: Number) -> float:  # pragma: no cover - convenience
    return math.sqrt(float(value))


def maximum(a: ndarray, b: ndarray) -> ndarray:  # pragma: no cover - unused
    return ndarray(max(x, y) for x, y in zip(a, b))


def minimum(a: ndarray, b: ndarray) -> ndarray:  # pragma: no cover - unused
    return ndarray(min(x, y) for x, y in zip(a, b))


__all__ = [
    "array",
    "asarray",
    "arange",
    "argmax",
    "any",
    "inf",
    "isfinite",
    "isnan",
    "linspace",
    "mean",
    "nan",
    "ndarray",
    "sin",
    "sum",
]

