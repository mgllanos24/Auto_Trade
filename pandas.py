"""Very small, test oriented subset of the :mod:`pandas` API.

The real project depends on NumPy and pandas which are not available in the
execution environment.  The unit tests exercise a constrained subset of the
DataFrame and Series APIs which we re-implement here using plain Python data
structures.  Only the behaviour required by the tests is provided – the
implementation should be considered a compatibility shim rather than a drop-in
replacement for pandas.
"""
from __future__ import annotations

import datetime as _dt
import math
import csv
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np

# Provide a ``__version__`` attribute so libraries performing runtime
# compatibility checks (such as scikit-learn) can operate without failing
# during import.  The value itself is not important for the test shim; any
# valid version string will suffice.
__version__ = "0.1.0"

NaT = None


def _is_nan(value: object) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except Exception:
        return False


def _is_scalar(value: object) -> bool:
    return isinstance(value, (int, float, str, bool)) or value is None


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


class RangeIndex(list):  # pragma: no cover - simple container
    pass


class _TimestampFactory:
    def __call__(self, value=None):
        if value is None:
            return self.now()
        if isinstance(value, _dt.datetime):
            return value
        if isinstance(value, _dt.date):  # pragma: no cover - convenience
            return _dt.datetime.combine(value, _dt.time.min)
        if isinstance(value, (int, float)):
            return _dt.datetime.fromtimestamp(value)
        if isinstance(value, str):
            try:
                return _dt.datetime.fromisoformat(value)
            except ValueError:
                return self.now()
        return value

    @staticmethod
    def now():  # pragma: no cover - trivial
        return _dt.datetime.utcnow()

    @staticmethod
    def utcnow():  # pragma: no cover - trivial
        return _dt.datetime.utcnow()


Timestamp = _TimestampFactory()


def date_range(start: str, periods: int, freq: str = "D") -> List[_dt.datetime]:
    if isinstance(start, str):
        start_dt = _dt.datetime.fromisoformat(start)
    else:  # pragma: no cover - unused
        start_dt = start
    step = _dt.timedelta(days=1 if freq.upper().startswith("D") else 0)
    return [start_dt + step * i for i in range(periods)]


def to_datetime(values, utc: bool = False):  # pragma: no cover - minimal helper
    if isinstance(values, Series):
        converted = [Timestamp(val) for val in values]
        return Series(converted, index=values.index[:], name=values.name)
    if isinstance(values, list):
        return [Timestamp(val) for val in values]
    result = Timestamp(values)
    if utc and isinstance(result, _dt.datetime):
        return result.replace(tzinfo=_dt.timezone.utc)
    return result


class Series:
    def __init__(self, data: Iterable[object], index: Sequence[object] | None = None, name: str | None = None):
        self._data: List[object] = list(data)
        if index is None:
            self.index: List[object] = list(range(len(self._data)))
        else:
            self.index = list(index)
        self.name = name

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)

    def __iter__(self) -> Iterator[object]:  # pragma: no cover - trivial
        return iter(self._data)

    def __getitem__(self, key):
        return self._data[key]

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    # ------------------------------------------------------------------
    # Arithmetic helpers
    # ------------------------------------------------------------------
    def _binary_op(self, other, op) -> "Series":
        if isinstance(other, Series):
            values = other._data
            if len(values) != len(self._data):
                raise ValueError("Series length mismatch")
        else:
            values = [other] * len(self._data)
        result: List[float] = []
        for left, right in zip(self._data, values):
            if _is_nan(left) or _is_nan(right):
                result.append(float("nan"))
            else:
                result.append(op(_to_float(left), _to_float(right)))
        return Series(result, index=self.index[:], name=self.name)

    def __sub__(self, other) -> "Series":
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other) -> "Series":
        if isinstance(other, Series):
            return other.__sub__(self)
        filler = Series([other] * len(self._data), index=self.index[:], name=self.name)
        return filler.__sub__(self)

    def __add__(self, other) -> "Series":
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other) -> "Series":  # pragma: no cover - symmetric
        return self.__add__(other)

    def __mul__(self, other) -> "Series":
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other) -> "Series":  # pragma: no cover - symmetric
        return self.__mul__(other)

    def __truediv__(self, other) -> "Series":
        def _div(a: float, b: float) -> float:
            return a / b if b != 0 else float("nan")

        return self._binary_op(other, _div)

    def __rtruediv__(self, other) -> "Series":
        if isinstance(other, Series):
            return other.__truediv__(self)
        filler = Series([other] * len(self._data), index=self.index[:], name=self.name)
        return filler.__truediv__(self)

    def __neg__(self) -> "Series":
        return Series([-_to_float(value) if not _is_nan(value) else float("nan") for value in self._data], index=self.index[:], name=self.name)

    # ------------------------------------------------------------------
    # Array like helpers
    # ------------------------------------------------------------------
    def copy(self) -> "Series":
        return Series(self._data[:], index=self.index[:], name=self.name)

    def tail(self, n: int) -> "Series":
        if n <= 0:
            return Series([], index=[], name=self.name)
        return Series(self._data[-n:], index=self.index[-n:], name=self.name)

    @property
    def iloc(self):
        return _SeriesILoc(self)

    def to_numpy(self) -> np.ndarray:
        return np.asarray([_to_float(value) for value in self._data])

    def mean(self) -> float:
        values = [_to_float(x) for x in self._data if not _is_nan(x)]
        if not values:
            return float("nan")
        return sum(values) / len(values)

    def max(self) -> float:
        values = [_to_float(x) for x in self._data if not _is_nan(x)]
        return max(values) if values else float("nan")

    def min(self) -> float:
        values = [_to_float(x) for x in self._data if not _is_nan(x)]
        return min(values) if values else float("nan")

    def diff(self) -> "Series":
        result: List[float] = []
        prev = None
        for value in self._data:
            if prev is None or _is_nan(prev) or _is_nan(value):
                result.append(float("nan"))
            else:
                result.append(_to_float(value) - _to_float(prev))
            prev = value
        return Series(result, index=self.index[:], name=self.name)

    def shift(self, periods: int) -> "Series":
        if periods >= 0:
            if periods == 0:
                data = self._data[:]
            elif periods >= len(self._data):
                data = [float("nan")] * len(self._data)
            else:
                filler = [float("nan")] * periods
                data = filler + self._data[:-periods]
        else:  # pragma: no cover - unused
            abs_periods = abs(periods)
            if abs_periods >= len(self._data):
                data = [float("nan")] * len(self._data)
            else:
                tail = self._data[abs_periods:]
                data = tail + [float("nan")] * abs_periods
        return Series(data, index=self.index[:], name=self.name)

    def clip(self, lower=None, upper=None) -> "Series":
        data: List[float] = []
        for value in self._data:
            val = _to_float(value)
            if lower is not None and val < lower:
                val = lower
            if upper is not None and val > upper:
                val = upper
            data.append(val)
        return Series(data, index=self.index[:], name=self.name)

    def abs(self) -> "Series":
        return Series([abs(_to_float(v)) for v in self._data], index=self.index[:], name=self.name)

    def replace(self, to_replace, value) -> "Series":
        targets = to_replace if isinstance(to_replace, (list, tuple, set)) else [to_replace]
        data = [value if elem in targets or (_is_nan(elem) and any(_is_nan(t) for t in targets)) else elem for elem in self._data]
        return Series(data, index=self.index[:], name=self.name)

    def fillna(self, value: float) -> "Series":
        data = [value if _is_nan(elem) else elem for elem in self._data]
        return Series(data, index=self.index[:], name=self.name)

    def isnull(self) -> "Series":
        return Series([_is_nan(elem) for elem in self._data], index=self.index[:], name=self.name)

    def any(self) -> bool:
        return any(bool(val) for val in self._data)

    def rolling(self, window: int, min_periods: int | None = None) -> "Rolling":
        return Rolling(self, window, min_periods)

    def ewm(self, span: int, adjust: bool = False) -> "EWM":
        return EWM(self, span, adjust=adjust)


class _SeriesILoc:
    def __init__(self, series: Series) -> None:
        self._series = series

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self._series))
            data = [self._series._data[i] for i in range(start, stop, step)]
            index = [self._series.index[i] for i in range(start, stop, step)]
            return Series(data, index=index, name=self._series.name)
        idx = key
        if idx < 0:
            idx += len(self._series._data)
        return self._series._data[idx]


class Rolling:
    def __init__(self, series: Series, window: int, min_periods: int | None) -> None:
        self.series = series
        self.window = max(int(window), 1)
        self.min_periods = self.window if min_periods is None else max(int(min_periods), 1)

    def _apply(self, func) -> Series:
        data: List[float] = []
        values = self.series._data
        for idx in range(len(values)):
            start = max(0, idx - self.window + 1)
            window_values = [_to_float(values[i]) for i in range(start, idx + 1) if not _is_nan(values[i])]
            if len(window_values) < self.min_periods:
                data.append(float("nan"))
            else:
                data.append(func(window_values))
        return Series(data, index=self.series.index[:], name=self.series.name)

    def mean(self) -> Series:
        return self._apply(lambda vals: sum(vals) / len(vals))

    def max(self) -> Series:
        return self._apply(lambda vals: max(vals))


class EWM:
    def __init__(self, series: Series, span: int, adjust: bool) -> None:
        self.series = series
        self.span = max(int(span), 1)
        self.adjust = adjust

    def mean(self) -> Series:
        alpha = 2 / (self.span + 1)
        ema = float("nan")
        data: List[float] = []
        for value in self.series._data:
            if _is_nan(value):
                data.append(float("nan"))
                continue
            val = _to_float(value)
            if _is_nan(ema):
                ema = val
            else:
                ema = alpha * val + (1 - alpha) * ema
            data.append(ema)
        return Series(data, index=self.series.index[:], name=self.series.name)


class DataFrame:
    def __init__(self, data: Mapping[str, Iterable[object]], index: Sequence[object] | None = None):
        if not data:
            self._data: Dict[str, Series] = {}
            self.index: List[object] = list(index or [])
            return

        inferred_length = None
        for values in data.values():
            if _is_scalar(values):
                continue
            current_length = len(list(values))
            if inferred_length is None:
                inferred_length = current_length
            elif inferred_length != current_length:
                raise ValueError("All columns must share the same length")
        if inferred_length is None:
            inferred_length = len(index) if index is not None else 0
        if index is None:
            self.index = list(range(inferred_length))
        else:
            self.index = list(index)
            if len(self.index) != inferred_length:
                raise ValueError("Index length does not match data length")

        def _ensure_list(values):
            if _is_scalar(values):
                return [values] * inferred_length
            as_list = list(values)
            if len(as_list) != inferred_length:
                raise ValueError("Column length mismatch")
            return as_list

        self._data = {
            name: Series(_ensure_list(values), index=self.index, name=name)
            for name, values in data.items()
        }

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.index)

    @property
    def columns(self) -> List[str]:
        return list(self._data.keys())

    def __getitem__(self, key):
        if isinstance(key, list):
            return DataFrame({name: self._data[name]._data for name in key}, index=self.index[:])
        return self._data[key].copy()

    def __setitem__(self, key: str, value) -> None:
        if isinstance(value, Series):
            if value.index != self.index:
                raise ValueError("Mismatched index for assignment")
            series = value.copy()
            series.name = key
        else:
            values = list(value)
            if len(values) != len(self.index):
                raise ValueError("Length mismatch when assigning new column")
            series = Series(values, index=self.index[:], name=key)
        self._data[key] = series

    @property
    def empty(self) -> bool:
        return len(self.index) == 0

    def copy(self) -> "DataFrame":
        return DataFrame({name: series._data[:] for name, series in self._data.items()}, index=self.index[:])

    def tail(self, n: int) -> "DataFrame":
        if n <= 0:
            return DataFrame({}, index=[])
        start = max(0, len(self.index) - n)
        return self.iloc[start:]

    def iterrows(self):
        for position, idx in enumerate(self.index):
            row_values = [self._data[col]._data[position] for col in self.columns]
            yield idx, Series(row_values, index=self.columns, name=idx)

    # ------------------------------------------------------------------
    # Mutating helpers used in the project code
    # ------------------------------------------------------------------
    def rename(self, *, columns: Mapping[str, str] | None = None, inplace: bool = False):
        if not columns:
            return None if inplace else self.copy()
        mapped = {}
        for name, series in self._data.items():
            new_name = columns.get(name, name)
            mapped[new_name] = Series(series._data[:], index=self.index[:], name=new_name)
        if inplace:
            self._data = mapped
            return self
        return DataFrame({name: series._data[:] for name, series in mapped.items()}, index=self.index[:])

    def set_index(self, column: str, inplace: bool = False):
        if column not in self._data:
            raise KeyError(column)
        new_index = self._data[column]._data[:]
        remaining = {name: series._data[:] for name, series in self._data.items() if name != column}
        if inplace:
            self.index = list(new_index)
            self._data = {name: Series(values, index=self.index[:], name=name) for name, values in remaining.items()}
            return self
        return DataFrame(remaining, index=list(new_index))

    def isnull(self) -> "DataFrame":
        return DataFrame({name: series.isnull()._data for name, series in self._data.items()}, index=self.index[:])

    def any(self) -> Series:
        data = [self._data[name].any() for name in self.columns]
        return Series(data, index=self.columns, name=None)

    def max(self, axis: int | None = None) -> Series:
        if axis == 1:
            rows: List[float] = []
            for idx in range(len(self.index)):
                values = [self._data[col]._data[idx] for col in self.columns]
                finite = [float(v) for v in values if not _is_nan(v)]
                rows.append(max(finite) if finite else float("nan"))
            return Series(rows, index=self.index[:], name=None)
        data = [self._data[col].max() for col in self.columns]
        return Series(data, index=self.columns, name=None)

    @property
    def iloc(self):
        return _DataFrameILoc(self)


class _DataFrameILoc:
    def __init__(self, frame: DataFrame) -> None:
        self._frame = frame

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self._frame.index))
            new_index = [self._frame.index[i] for i in range(start, stop, step)]
            data = {name: [series._data[i] for i in range(start, stop, step)] for name, series in self._frame._data.items()}
            return DataFrame(data, index=new_index)
        raise TypeError("Only slicing is supported in iloc")


def concat(objs: Sequence[Series | DataFrame], axis: int = 0) -> DataFrame:
    if axis != 1:
        raise ValueError("This lightweight concat only supports axis=1")
    if not objs:
        return DataFrame({}, index=[])
    first = objs[0]
    index = first.index[:] if isinstance(first, Series) else first.index[:]
    data: Dict[str, Iterable[object]] = {}
    for idx, obj in enumerate(objs):
        if isinstance(obj, Series):
            name = obj.name or f"col{idx}"
            data[name] = obj._data[:]
        elif isinstance(obj, DataFrame):
            for name in obj.columns:
                data[name] = obj._data[name]._data[:]
        else:  # pragma: no cover - unused
            raise TypeError("Unsupported object for concat")
    return DataFrame(data, index=index)


@classmethod
def _from_records(cls, records: Sequence[Mapping[str, object]]):
    if not records:
        return cls({}, index=[])
    keys: List[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    columns = {key: [] for key in keys}
    for record in records:
        for key in keys:
            columns[key].append(record.get(key))
    return cls(columns)


DataFrame.from_records = classmethod(_from_records)

del _from_records


def read_csv(filepath_or_buffer, index_col=None, parse_dates=False) -> DataFrame:
    close_after = False
    if hasattr(filepath_or_buffer, "read"):
        handle = filepath_or_buffer
    else:
        handle = open(filepath_or_buffer, "r", newline="", encoding="utf-8")
        close_after = True
    try:
        reader = csv.reader(handle)
        rows = list(reader)
    finally:
        if close_after:
            handle.close()

    if not rows:
        return DataFrame({}, index=[])

    header = rows[0]
    data_rows = rows[1:]

    def _normalise_column(column):
        if isinstance(column, int):
            return header[column]
        return column

    parse_targets: List[str] = []
    if parse_dates:
        if parse_dates is True:
            if index_col is not None:
                parse_targets = [_normalise_column(index_col)]
            else:
                parse_targets = header[:]
        elif isinstance(parse_dates, (list, tuple, set)):
            parse_targets = [_normalise_column(col) for col in parse_dates]
        else:
            parse_targets = [_normalise_column(parse_dates)]

    columns = {name: [] for name in header}
    for row in data_rows:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        for idx, name in enumerate(header):
            raw_value = row[idx]
            if raw_value == "":
                value = float("nan")
            elif name in parse_targets:
                value = to_datetime(raw_value)
            else:
                value = raw_value
            columns[name].append(value)

    index_values: List[object] | None = None
    if index_col is not None:
        index_name = _normalise_column(index_col)
        if index_name not in columns:
            raise KeyError(index_name)
        index_values = columns.pop(index_name)
    return DataFrame(columns, index=index_values)


__all__ = [
    "DataFrame",
    "NaT",
    "RangeIndex",
    "Series",
    "Timestamp",
    "concat",
    "date_range",
    "read_csv",
    "to_datetime",
]

