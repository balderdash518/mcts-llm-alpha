"""Factor metrics, the effective alpha repository and the paper's admission rules.

Factors are date x instrument matrices aligned with the label matrix.

- Effectiveness: RankIC, mean daily cross-sectional Spearman correlation with
  the forward ``horizon``-day close-to-close return.
- Stability: RankIR = mean(daily RankIC) / std(daily RankIC).
- Turnover: average daily change of holdings, sum_i |w_t,i - w_t-1,i|, where w is
  the demeaned factor scaled to unit gross exposure (range 0-2). The paper names
  the metric and its 1.6 cap without giving the formula; this is our reading.
- Diversity: maximum |mean daily Spearman correlation| with repository alphas.

Dimension scores follow Eq. (3): R = share of reference alphas strictly better
on that metric, score = 10 * (1 - R). The reference is the effective repository.
The paper does not define the empty-repository case; until the first admission
we rank against the tree nodes evaluated before the current tree (MetricPool),
and only a brand-new run's first tree sees R = 0 everywhere.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DIMENSIONS = ("Effectiveness", "Stability", "Turnover", "Diversity", "Overfitting")
METRIC_OF = {"Effectiveness": "rank_ic", "Stability": "rank_ir",
             "Turnover": "turnover", "Diversity": "max_corr"}
LOWER_IS_BETTER = {"turnover", "max_corr"}
METRIC_TEXT = {
    "rank_ic": "RankIC", "rank_ir": "RankIR", "turnover": "日换手率",
    "max_corr": "最大相关", "ic": "IC", "coverage": "覆盖率",
}


class EvaluationError(RuntimeError):
    """The formula could not be measured; never treated as a weak signal."""


def _rowwise_corr(x, y, min_obs):
    mask = np.isfinite(x) & np.isfinite(y)
    count = mask.sum(axis=1)
    x = np.where(mask, x, np.nan)
    y = np.where(mask, y, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-missing dates become NaN
        xc = x - np.nanmean(x, axis=1, keepdims=True)
        yc = y - np.nanmean(y, axis=1, keepdims=True)
        cov = np.nansum(xc * yc, axis=1)
        var = np.nansum(xc * xc, axis=1) * np.nansum(yc * yc, axis=1)
        corr = cov / np.sqrt(var)
    corr[(count < min_obs) | ~(var > 1e-24)] = np.nan
    return corr


def cross_sectional_rank(matrix):
    """Percentile rank within each date; missing values stay missing."""
    frame = matrix if isinstance(matrix, pd.DataFrame) else pd.DataFrame(matrix)
    return frame.rank(axis=1, pct=True).to_numpy(dtype=np.float32)


def daily_rank_ic(factor, label, min_obs=10):
    f = factor.to_numpy(dtype=float)
    y = label.to_numpy(dtype=float)
    mask = np.isfinite(f) & np.isfinite(y)
    fr = pd.DataFrame(np.where(mask, f, np.nan)).rank(axis=1).to_numpy()
    yr = pd.DataFrame(np.where(mask, y, np.nan)).rank(axis=1).to_numpy()
    return pd.Series(_rowwise_corr(fr, yr, min_obs), index=factor.index).dropna()


def daily_ic(factor, label, min_obs=10):
    return pd.Series(_rowwise_corr(factor.to_numpy(dtype=float), label.to_numpy(dtype=float),
                                   min_obs), index=factor.index).dropna()


def information_ratio(series):
    std = float(np.std(series, ddof=0)) if len(series) else 0.0
    return float(np.mean(series) / std) if std > 1e-12 else 0.0


def daily_turnover(factor):
    values = factor.to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-missing dates
        demeaned = values - np.nanmean(values, axis=1, keepdims=True)
    gross = np.nansum(np.abs(demeaned), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = np.where(gross > 1e-12, demeaned / gross, np.nan)
    weights = np.nan_to_num(weights, nan=0.0)
    valid = gross[:, 0] > 1e-12
    pairs = valid[1:] & valid[:-1]
    if not pairs.any():
        return float("nan")
    change = np.abs(weights[1:] - weights[:-1]).sum(axis=1)
    return float(change[pairs].mean())


def rank_correlation(ranked_a, ranked_b, min_obs=10):
    """|mean daily correlation| between two cross-sectionally ranked factors."""
    daily = _rowwise_corr(ranked_a.astype(float), ranked_b.astype(float), min_obs)
    daily = daily[np.isfinite(daily)]
    return float(abs(daily.mean())) if len(daily) else 0.0


@dataclass
class Measurement:
    formula: str
    metrics: dict
    ranked: np.ndarray = field(repr=False)
    correlations: dict = field(default_factory=dict)


def measure(formula, factor, label, repository=None, min_days=50, min_coverage=0.3):
    """Compute raw metrics for an aligned factor matrix against the label."""
    values = factor.to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    factor = pd.DataFrame(values, index=factor.index, columns=factor.columns)
    available = np.isfinite(label.to_numpy(dtype=float))
    coverage = float((np.isfinite(values) & available).sum() / max(available.sum(), 1))
    if coverage < min_coverage:
        raise EvaluationError(f'因子在评估区间的有效值覆盖率只有 {coverage:.1%}，'
                              f'低于 {min_coverage:.0%}（窗口过长或出现除零/对数非正）')
    rank_ic_series = daily_rank_ic(factor, label)
    if len(rank_ic_series) < min_days:
        raise EvaluationError(f'只有 {len(rank_ic_series)} 个交易日能计算截面相关，'
                              f'少于 {min_days} 日（因子可能在截面上为常数）')
    ranked = cross_sectional_rank(factor)
    metrics = {
        "rank_ic": float(rank_ic_series.mean()),
        "rank_ir": information_ratio(rank_ic_series),
        "ic": float(daily_ic(factor, label).mean()),
        "turnover": daily_turnover(factor),
        "coverage": coverage,
    }
    correlations = {}
    if repository is not None:
        correlations = repository.correlations_with(ranked)
    metrics["max_corr"] = max(correlations.values(), default=0.0)
    for name, value in metrics.items():
        if not np.isfinite(value):
            raise EvaluationError(f'指标 {METRIC_TEXT.get(name, name)} 无法计算（非有限值）')
    return Measurement(formula, metrics, ranked, correlations)


@dataclass
class RepositoryAlpha:
    attempt_id: str
    formula: str
    metrics: dict
    ranked: np.ndarray = field(repr=False)
    tree: int = 0
    label: str = ""
    summary: str = ""
    description: str = ""


class RankingReference:
    """Eq. (3) relative ranking against ``metric_values``."""

    def metric_values(self, metric):
        raise NotImplementedError

    def relative_rank(self, metric, value):
        """Share of reference alphas strictly better than ``value``."""
        values = self.metric_values(metric)
        if not values:
            return 0.0
        better = sum(v < value for v in values) if metric in LOWER_IS_BETTER else \
            sum(v > value for v in values)
        return better / len(values)

    def scores(self, metrics):
        return {dim: round(10.0 * (1.0 - self.relative_rank(metric, metrics[metric])), 4)
                for dim, metric in METRIC_OF.items()}


class MetricPool(RankingReference):
    """Cold-start reference: raw metrics of previously evaluated candidates."""

    def __init__(self, metrics):
        self.rows = [m for m in metrics if m and all(k in m for k in METRIC_OF.values())]

    def __len__(self):
        return len(self.rows)

    def metric_values(self, metric):
        return [m[metric] for m in self.rows]


class FactorRepository(RankingReference):
    """Effective alpha repository R with incremental pairwise correlations."""

    def __init__(self):
        self.alphas = []
        self._corr = {}

    def __len__(self):
        return len(self.alphas)

    def __iter__(self):
        return iter(self.alphas)

    def contains(self, formula):
        key = "".join(formula.split())
        return any("".join(a.formula.split()) == key for a in self.alphas)

    def correlations_with(self, ranked):
        return {a.attempt_id: rank_correlation(ranked, a.ranked) for a in self.alphas}

    def metric_values(self, metric):
        if metric == "max_corr":
            return [max((self._corr[a.attempt_id].get(b.attempt_id, 0.0)
                         for b in self.alphas if b is not a), default=0.0) for a in self.alphas]
        return [a.metrics[metric] for a in self.alphas]

    def add(self, alpha, correlations=None):
        correlations = dict(correlations or {})
        for other in self.alphas:
            if other.attempt_id not in correlations:
                correlations[other.attempt_id] = rank_correlation(alpha.ranked, other.ranked)
        self._corr[alpha.attempt_id] = correlations
        for other_id, value in correlations.items():
            self._corr.setdefault(other_id, {})[alpha.attempt_id] = value
        self.alphas.append(alpha)

    def snapshot_ids(self):
        return [a.attempt_id for a in self.alphas]


@dataclass
class EffectiveCriteria:
    """Appendix F.1 effective alpha check."""
    rank_ic_min: float = 0.015
    rank_ir_min: float = 0.3
    relative_rank_max: float = 0.95
    turnover_max: float = 1.6
    max_correlation: float = 0.8
    # A 0.95 rank bound only resolves once 1 / (1 - 0.95) = 20 alphas exist; below that
    # it would demand beating existing alphas outright. Set to 1 for the literal rule.
    relative_rank_min_size: int = 20

    def failures(self, metrics, repository):
        reasons = []
        if metrics["rank_ic"] < self.rank_ic_min:
            reasons.append(f'RankIC {metrics["rank_ic"]:.4f} < {self.rank_ic_min}')
        if metrics["rank_ir"] < self.rank_ir_min:
            reasons.append(f'RankIR {metrics["rank_ir"]:.3f} < {self.rank_ir_min}')
        for metric in ("rank_ic", "rank_ir"):
            if len(repository) < self.relative_rank_min_size:
                break
            rank = repository.relative_rank(metric, metrics[metric])
            if rank > self.relative_rank_max:
                reasons.append(f'R_{METRIC_TEXT[metric]} {rank:.2f} > {self.relative_rank_max}')
        if metrics["turnover"] > self.turnover_max:
            reasons.append(f'日换手率 {metrics["turnover"]:.3f} > {self.turnover_max}')
        if metrics["max_corr"] >= self.max_correlation:
            reasons.append(f'与有效因子库最大相关 {metrics["max_corr"]:.3f} >= {self.max_correlation}')
        return reasons
