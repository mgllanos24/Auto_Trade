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
import sys
import traceback
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

import csv

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


class Index(list):  # pragma: no cover - simple container
    def __init__(
        self,
        data: Iterable[object] | None = None,
        name: str | None = None,
        tz: _dt.tzinfo | None = None,
    ):
        if data is None:
            values: List[object] = []
        else:
            values = list(data)
        super().__init__(values)
        self.name = name

        if tz is None:
            inferred_tz = None
            for value in values:
                tzinfo = getattr(value, "tzinfo", None)
                if tzinfo is not None:
                    inferred_tz = tzinfo
                    break
            tz = inferred_tz
        self._tz = tz

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return Index(result, name=self.name, tz=self._tz)
        return result

    def copy(self) -> "Index":
        copied = Index(self, name=self.name, tz=self._tz)
        return copied

    def _debug_state(self) -> Dict[str, object]:  # pragma: no cover - debugging helper
        sample = list(self[:5])
        sample_types = sorted({type(value).__name__ for value in sample}) if sample else []
        return {
            "name": self.name,
            "tz": self._tz,
            "tz_type": type(self._tz).__name__ if self._tz is not None else None,
            "length": len(self),
            "sample": sample,
            "sample_types": sample_types,
        }

    def __getattr__(self, attr):
        debug_payload = self._debug_state()
        debug_payload["requested_attr"] = attr
        debug_payload["stack"] = "".join(traceback.format_stack(limit=6))
        print(f"[pandas.Index DEBUG] Missing attribute lookup: {debug_payload}")
        raise AttributeError(f"'Index' object has no attribute '{attr}'")

    @property
    def tz(self):  # pragma: no cover - minimal compatibility shim
        return self._tz

    def tz_localize(self, tz):  # pragma: no cover - compatibility helper with debug output
        debug_details: Dict[str, object] = {
            **self._debug_state(),
            "incoming_tz": tz,
            "has_existing_tz": self._tz is not None,
        }
        print(f"[pandas.Index DEBUG] tz_localize invoked: {debug_details}")

        tzinfo = tz
        if isinstance(tz, str):
            if tz.upper() == "UTC":
                tzinfo = _dt.timezone.utc
            else:
                try:  # pragma: no cover - depends on optional zoneinfo
                    from zoneinfo import ZoneInfo

                    tzinfo = ZoneInfo(tz)
                except Exception:
                    print(
                        "[pandas.Index DEBUG] Unable to resolve timezone string; "
                        "preserving original values"
                    )
                    tzinfo = None

        if tzinfo is None:
            print(
                "[pandas.Index DEBUG] No timezone information available after tz_localize"
            )
            return self.copy()

        localized_values: List[object] = []
        for value in self:
            if isinstance(value, _dt.datetime) and value.tzinfo is None:
                localized_values.append(value.replace(tzinfo=tzinfo))
            else:
                localized_values.append(value)

        result = Index(localized_values, name=self.name, tz=tzinfo)
        print(
            "[pandas.Index DEBUG] tz_localize returning new index:",
            {"name": result.name, "tz": result._tz, "length": len(result)},
        )
        return result


class RangeIndex(Index):  # pragma: no cover - simple container
    def __init__(self, data: Iterable[object] | int | None = None, name: str | None = None):
        if data is None:
            values = []
        elif isinstance(data, int):
            values = range(data)
        else:
            values = data
        super().__init__(values, name=name)


def _ensure_index(values: Sequence[object] | Index, *, name: str | None = None) -> Index:
    if isinstance(values, Index):
        return values.copy()
    inferred_name = name if name is not None else getattr(values, "name", None)
    return Index(values, name=inferred_name)


class _TimestampFactory:
    _UNIT_MULTIPLIERS = {
        "s": 1,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
        "m": 60,
        "h": 3600,
        "d": 86400,
    }

    def __call__(self, value=None, *, unit: str | None = None):
        if unit is not None:
            if unit not in self._UNIT_MULTIPLIERS:
                raise ValueError(f"Unsupported unit '{unit}'")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                scaled_value = value
            else:
                scaled_value = numeric_value * self._UNIT_MULTIPLIERS[unit]
            value = scaled_value
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


def to_datetime(values, utc: bool = False, unit: str | None = None):  # pragma: no cover - minimal helper
    def _unit_to_seconds(value):
        if unit is None:
            return value
        multipliers = {
            "s": 1,
            "ms": 1e-3,
            "us": 1e-6,
            "ns": 1e-9,
            "m": 60,
            "h": 3600,
            "d": 86400,
        }
        if unit not in multipliers:
            raise ValueError(f"Unsupported unit '{unit}'")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return value
        return numeric * multipliers[unit]

    def _convert(value):
        scaled = _unit_to_seconds(value)
        ts = Timestamp(scaled)
        if utc and isinstance(ts, _dt.datetime):
            return ts.replace(tzinfo=_dt.timezone.utc)
        return ts

    if isinstance(values, Series):
        converted = [_convert(val) for val in values]
        return Series(converted, index=values.index[:], name=values.name)
    if isinstance(values, list):
        return [_convert(val) for val in values]
    return _convert(values)


class Series:
    def __init__(self, data: Iterable[object], index: Sequence[object] | None = None, name: str | None = None):
        self._data: List[object] = list(data)
        if index is None:
            self.index = range(len(self._data))
        else:
            self.index = index
        self.name = name

    @property
    def index(self) -> Index:
        return self._index

    @index.setter
    def index(self, new_index) -> None:
        ensured = _ensure_index(new_index)
        if len(ensured) != len(self._data):
            raise ValueError("Index length does not match data length")
        self._index = ensured

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)

    def __iter__(self) -> Iterator[object]:  # pragma: no cover - trivial
        return iter(self._data)

    def __getitem__(self, key):
        if isinstance(key, slice):  # pragma: no cover - simple container slicing
            return Series(self._data[key], index=self.index[key], name=self.name)

        if isinstance(key, int):  # pragma: no cover - trivial positional access
            return self._data[key]

        if key in self.index:
            position = self.index.index(key)
            return self._data[position]

        raise KeyError(key)

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    @property
    def values(self):
        """Return the underlying data as a NumPy array."""

        def _is_numeric(value: object) -> bool:
            return isinstance(value, (int, float)) or _is_nan(value)

        if all(_is_numeric(value) for value in self._data):
            return np.array([_to_float(value) for value in self._data], dtype=float)
        return np.array(self._data, dtype=object)

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
            index = _ensure_index([self._series.index[i] for i in range(start, stop, step)], name=self._series.index.name)
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
            self.index = index or []
            self._columns_name: str | None = None
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
            self.index = range(inferred_length)
        else:
            ensured_index = _ensure_index(index)
            if len(ensured_index) != inferred_length:
                raise ValueError("Index length does not match data length")
            self.index = ensured_index

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
        self._columns_name: str | None = None

    @property
    def index(self) -> Index:
        return self._index

    @index.setter
    def index(self, new_index) -> None:
        self._index = _ensure_index(new_index)

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.index)

    @property
    def columns(self) -> Index:
        return Index(self._data.keys(), name=self._columns_name)

    @columns.setter
    def columns(self, new_columns) -> None:
        ensured = _ensure_index(new_columns)
        if len(ensured) != len(self._data):
            raise ValueError("Length mismatch when setting columns")
        existing_series = list(self._data.values())
        renamed: Dict[str, Series] = {}
        for name, series in zip(ensured, existing_series):
            renamed[name] = Series(series._data[:], index=self.index[:], name=name)
        self._data = renamed
        self._columns_name = ensured.name

    def __getitem__(self, key):
        if isinstance(key, list):
            frame = DataFrame({name: self._data[name]._data for name in key}, index=self.index[:])
            frame._columns_name = self._columns_name
            return frame
        return self._data[key].copy()

    @property
    def values(self):
        """Return the frame data as a 2D NumPy array."""
        if not self._data:
            return np.empty((0, 0))
        columns = list(self.columns)
        data = []
        for row_idx in range(len(self.index)):
            data.append([self._data[col]._data[row_idx] for col in columns])
        try:
            return np.array(data, dtype=float)
        except (TypeError, ValueError):
            return np.array(data, dtype=object)

    def iterrows(self):
        column_names = list(self.columns)
        for position, label in enumerate(self.index):
            values = [self._data[name]._data[position] for name in column_names]
            yield label, Series(values, index=column_names, name=label)

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
        frame = DataFrame({name: series._data[:] for name, series in self._data.items()}, index=self.index[:])
        frame._columns_name = self._columns_name
        return frame

    def tail(self, n: int) -> "DataFrame":
        if n <= 0:
            frame = DataFrame({}, index=[])
            frame._columns_name = self._columns_name
            return frame
        start = max(0, len(self.index) - n)
        result = self.iloc[start:]
        result._columns_name = self._columns_name
        return result

    # ------------------------------------------------------------------
    # Mutating helpers used in the project code
    # ------------------------------------------------------------------
    def sort_index(
        self,
        axis: int = 0,
        ascending: bool = True,
        inplace: bool = False,
    ) -> "DataFrame" | None:
        """Return a new frame with the index sorted."""

        if axis not in (0, "index"):
            raise NotImplementedError("This shim only supports sorting the index")

        ordered_positions = sorted(
            enumerate(self.index),
            key=lambda item: item[1],
            reverse=not ascending,
        )

        new_index_values = [label for _, label in ordered_positions]
        new_index = _ensure_index(new_index_values, name=self.index.name)

        new_columns = {}
        for name, series in self._data.items():
            new_columns[name] = [series._data[pos] for pos, _ in ordered_positions]

        if inplace:
            self.index = new_index
            for name, values in new_columns.items():
                self._data[name] = Series(values, index=self.index[:], name=name)
            return None

        frame = DataFrame(new_columns, index=new_index)
        frame._columns_name = self._columns_name
        return frame

    def rename(self, *, columns: Mapping[str, str] | None = None, inplace: bool = False):
        if not columns:
            return None if inplace else self.copy()
        mapped = {}
        for name, series in self._data.items():
            new_name = columns.get(name, name)
            mapped[new_name] = Series(series._data[:], index=self.index[:], name=new_name)
        if inplace:
            self._data = mapped
            self._columns_name = self._columns_name
            return self
        frame = DataFrame({name: series._data[:] for name, series in mapped.items()}, index=self.index[:])
        frame._columns_name = self._columns_name
        return frame

    def set_index(self, column: str, inplace: bool = False):
        if column not in self._data:
            raise KeyError(column)
        new_index = self._data[column]._data[:]
        remaining = {name: series._data[:] for name, series in self._data.items() if name != column}
        if inplace:
            self.index = _ensure_index(new_index)
            self._data = {name: Series(values, index=self.index[:], name=name) for name, values in remaining.items()}
            return self
        frame = DataFrame(remaining, index=list(new_index))
        frame._columns_name = self._columns_name
        return frame

    def isnull(self) -> "DataFrame":
        frame = DataFrame({name: series.isnull()._data for name, series in self._data.items()}, index=self.index[:])
        frame._columns_name = self._columns_name
        return frame

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
        column_index = self.columns
        data = [self._data[col].max() for col in column_index]
        return Series(data, index=column_index, name=None)

    @property
    def iloc(self):
        return _DataFrameILoc(self)


class _DataFrameILoc:
    def __init__(self, frame: DataFrame) -> None:
        self._frame = frame

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self._frame.index))
            new_index = _ensure_index([self._frame.index[i] for i in range(start, stop, step)], name=self._frame.index.name)
            data = {name: [series._data[i] for i in range(start, stop, step)] for name, series in self._frame._data.items()}
            frame = DataFrame(data, index=new_index)
            frame._columns_name = self._frame._columns_name
            return frame
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
    frame = DataFrame(data, index=index)
    if isinstance(first, DataFrame):
        frame._columns_name = first._columns_name
    else:
        frame._columns_name = None
    return frame


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


def _coerce_value(value: str | None) -> object:
    """Best-effort conversion of CSV field values.

    The lightweight shim mirrors pandas' behaviour by interpreting blank values
    as ``NaN`` and attempting numeric conversion before falling back to the raw
    string.  The heuristics are intentionally conservative but sufficient for
    the data processed in the tests and application code.
    """

    if value in ("", None):
        return float("nan")
    try:
        # Try integer conversion first so we do not lose precision for whole
        # numbers.  If that fails, attempt a float conversion.
        as_int = int(value)
        if str(as_int) == value:
            return as_int
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return value


def read_csv(path: str, index_col: str | int | None = None, **kwargs) -> "DataFrame":
    """Parse a CSV file into a :class:`DataFrame`.

    Only a very small subset of pandas' ``read_csv`` API is implemented –
    enough for the application code to load watchlists and historical price
    data.  Unsupported keyword arguments raise ``NotImplementedError`` so that
    unexpected usages surface clearly during testing.
    """

    unsupported = set(kwargs) - {"encoding", "newline"}
    if unsupported:
        raise NotImplementedError(f"Unsupported arguments: {', '.join(sorted(unsupported))}")

    encoding = kwargs.get("encoding", "utf-8")
    newline = kwargs.get("newline", "")

    with open(path, "r", encoding=encoding, newline=newline) as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        records: List[Dict[str, object]] = []
        for row in reader:
            converted = {key: _coerce_value(value) for key, value in row.items()}
            records.append(converted)

    frame = DataFrame.from_records(records)

    if index_col is None:
        return frame

    if isinstance(index_col, int):
        try:
            column_name = fieldnames[index_col]
        except (TypeError, IndexError):  # pragma: no cover - defensive
            raise KeyError(index_col)
    else:
        column_name = index_col

    if column_name not in frame.columns:
        raise KeyError(column_name)

    return frame.set_index(column_name, inplace=False)


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

