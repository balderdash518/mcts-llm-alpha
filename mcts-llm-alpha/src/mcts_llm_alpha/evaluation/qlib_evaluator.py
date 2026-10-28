"""Qlib market data: per-date index membership, forward-return labels, factor matrices."""

from collections import OrderedDict

import numpy as np

from ..formula import to_qlib, check, FormulaError
from .protocol import EvaluationError, measure


def custom_operators():
    """Paper operators that Qlib lacks natively (Sin, Cos, Tanh)."""
    from qlib.data.ops import NpElemOperator

    def make(name, func):
        def __init__(self, feature):
            NpElemOperator.__init__(self, feature, func)
        return type(name, (NpElemOperator,), {"__init__": __init__})

    return [make("Sin", "sin"), make("Cos", "cos"), make("Tanh", "tanh")]


class QlibMarket:
    """Loads labels and factors for named periods (``train`` drives the search;
    ``test`` is only read after mining to report out-of-sample statistics)."""

    def __init__(self, provider_uri, universe, periods, horizon=10, fields=None,
                 min_days=50, min_coverage=0.3, cache_size=64, initialize=True, region="cn"):
        if initialize:
            import qlib
            qlib.init(provider_uri=str(provider_uri), region=region, custom_ops=custom_operators(),
                      kernels=1, expression_cache=None, dataset_cache=None)
        from qlib.data import D
        self.D = D
        self.universe, self.periods, self.horizon = universe, dict(periods), int(horizon)
        self.fields = tuple(fields) if fields else None
        self.min_days, self.min_coverage = min_days, min_coverage
        self._labels, self._factors, self.cache_size = {}, OrderedDict(), cache_size

    def label_expression(self):
        # Buy and sell at the close: return over the next ``horizon`` trading days.
        return f"Ref($close, -{self.horizon}) / $close - 1"

    def _load(self, expression, period):
        start, end = self.periods[period]
        frame = self.D.features(self.D.instruments(self.universe), [expression],
                                start_time=start, end_time=end, freq="day")
        if frame.empty:
            raise EvaluationError(f"{period} 区间没有任何数据")
        return frame.iloc[:, 0].unstack(level="instrument").sort_index()

    def label(self, period):
        if period not in self._labels:
            label = self._load(self.label_expression(), period).astype(float)
            # Purge the last ``horizon`` days whose labels need prices after the period.
            label.iloc[-self.horizon:] = np.nan
            label = label.replace([np.inf, -np.inf], np.nan)
            self._labels[period] = label
        return self._labels[period]

    def factor(self, formula, period):
        key = (formula, period)
        if key in self._factors:
            self._factors.move_to_end(key)
            return self._factors[key]
        try:
            check(formula, self.fields) if self.fields else check(formula)
            expression = to_qlib(formula)
        except FormulaError as error:
            raise EvaluationError(str(error)) from None
        label = self.label(period)
        try:
            frame = self._load(expression, period)
        except EvaluationError:
            raise
        except Exception as error:
            raise EvaluationError(f"Qlib 计算失败：{type(error).__name__}: {str(error)[:300]}") from None
        frame = frame.reindex(index=label.index, columns=label.columns).astype(float)
        self._factors[key] = frame
        if len(self._factors) > self.cache_size:
            self._factors.popitem(last=False)
        return frame

    def measure(self, formula, period="train", repository=None):
        return measure(formula, self.factor(formula, period), self.label(period), repository,
                       self.min_days, self.min_coverage)
