"""
Расчёт торговых метрик по кривой эквити и списку сделок.

Все метрики считаются из двух источников:
    * equity_curve -- pd.Series эквити по закрытию каждого бара;
    * trades_df    -- DataFrame завершённых сделок (BacktestResult.trades_df()).

Годовая нормализация (Sharpe/Sortino/CAGR) определяется параметром
periods_per_year; если он не задан -- оценивается по медианному шагу индекса.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365.25 * 24 * 3600

#: Критерии оптимизации для GUI: подпись -> поле PerformanceMetrics
CRITERIA: dict[str, str] = {
    "Максимальный баланс (Net Profit)": "net_profit",
    "Максимальная доходность, %": "total_return_pct",
    "Max Sharpe Ratio": "sharpe_ratio",
    "Max Sortino Ratio": "sortino_ratio",
    "Max Profit Factor": "profit_factor",
    "Max Win Rate, %": "win_rate_pct",
    "Max Recovery Factor": "recovery_factor",
    "Max матожидание (Expectancy)": "expectancy",
    "Complex Criterion max (комплексный)": "complex_criterion",
}


@dataclass
class PerformanceMetrics:
    """Контейнер торговых метрик одного прогона."""

    # Капитал
    initial_equity: float = 0.0
    final_equity: float = 0.0
    net_profit: float = 0.0
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0

    # Риск
    max_drawdown: float = 0.0          # в деньгах
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0     # в барах
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0          # CAGR% / MaxDD%
    annualized_volatility_pct: float = 0.0
    recovery_factor: float = 0.0       # net_profit / max_drawdown
    exposure_pct: float = 0.0          # % баров с открытой позицией

    # Статистика сделок
    total_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0            # матожидание прибыли на сделку
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trade_duration_bars: float = 0.0
    total_commission: float = 0.0

    # Комплексный критерий (аналог «Complex Criterion max» в тестере MT5):
    # доходность * profit factor (с потолком) * sqrt(числа сделок) / (1 + просадка%).
    # Поощряет прибыльность, статистическую значимость и малую просадку.
    complex_criterion: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __str__(self) -> str:
        d = self.to_dict()
        width = max(len(k) for k in d)
        lines = ["=== Метрики стратегии ==="]
        for key, value in d.items():
            if isinstance(value, float):
                lines.append(f"{key:<{width}} : {value:,.4f}")
            else:
                lines.append(f"{key:<{width}} : {value}")
        return "\n".join(lines)


def _estimate_periods_per_year(index: pd.Index) -> int:
    """Оценка числа торговых периодов в году по фактической плотности баров.

    Считается как (число баров - 1) / (календарная длительность данных в годах):
    выходные и праздники, когда рынок закрыт, автоматически уменьшают оценку.
    Оценка по медианному шагу (например, 15 минут -> ~35000) завышала бы
    periods_per_year и вместе с ним Sharpe/Sortino на внутридневных данных.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return 252
    span_seconds = (index[-1] - index[0]).total_seconds()
    if span_seconds <= 0:
        return 252
    return max(int(round((len(index) - 1) * SECONDS_PER_YEAR / span_seconds)), 1)


def max_drawdown_abs_pct(values: np.ndarray) -> tuple[float, float]:
    """Макс. просадка в деньгах и в % (только numpy, без длительности).

    Длительность просадки требует Python-цикла по всем барам и не входит ни
    в один критерий оптимизации -- лёгкий путь её не считает.
    """
    running_max = np.maximum.accumulate(values)
    drawdown = running_max - values
    dd_abs = float(drawdown.max()) if len(drawdown) else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct_series = np.where(running_max > 0, drawdown / running_max, 0.0)
    dd_pct = float(dd_pct_series.max() * 100.0) if len(values) else 0.0
    return dd_abs, dd_pct


def max_drawdown_info(equity: pd.Series) -> tuple[float, float, int]:
    """Возвращает (макс. просадка в деньгах, в %, длительность в барах)."""
    values = equity.to_numpy(dtype=np.float64)
    dd_abs, dd_pct = max_drawdown_abs_pct(values)

    # Длительность: максимальное число баров между обновлениями максимума
    running_max = np.maximum.accumulate(values)
    duration = 0
    current = 0
    for value, peak in zip(values, running_max):
        current = 0 if value >= peak else current + 1
        duration = max(duration, current)
    return dd_abs, dd_pct, duration


def sharpe_ratio(
    returns: np.ndarray, periods_per_year: int, risk_free_rate: float = 0.0
) -> float:
    """Годовой коэффициент Шарпа по ряду доходностей за период."""
    if len(returns) < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    std = excess.std(ddof=1)
    # std сравнивается с масштабом доходностей: у константного ряда std
    # не точный 0, а ~1e-18 из-за плавающей точки -- без защиты Sharpe
    # взлетел бы до бессмысленных величин
    if math.isnan(std) or std <= abs(float(excess.mean())) * 1e-9:
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray, periods_per_year: int, risk_free_rate: float = 0.0
) -> float:
    """Годовой коэффициент Сортино (штрафует только отрицательную волатильность)."""
    if len(returns) < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    downside = excess[excess < 0]
    if len(downside) == 0:
        return math.inf if excess.mean() > 0 else 0.0
    downside_std = math.sqrt(float(np.mean(downside**2)))
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * math.sqrt(periods_per_year))


def exposure_pct(trades_df: pd.DataFrame, n_bars: int) -> float:
    """Доля баров с открытой позицией, % (объединение интервалов сделок).

    Интервалы [entry_bar, exit_bar) сливаются перед суммированием: частичные
    закрытия дают несколько сделок с пересекающимися интервалами, и простая
    сумма длительностей завысила бы экспозицию. Без экспозиции Sharpe двух
    стратегий с разным временем в рынке несравним.
    """
    if n_bars <= 0 or trades_df.empty:
        return 0.0
    intervals = sorted(
        zip(
            trades_df["entry_bar"].to_numpy(dtype=np.int64),
            trades_df["exit_bar"].to_numpy(dtype=np.int64),
        )
    )
    covered = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            covered += cur_end - cur_start
            cur_start, cur_end = start, end
    covered += cur_end - cur_start
    return min(covered / n_bars * 100.0, 100.0)


def _max_consecutive(mask: np.ndarray) -> int:
    """Максимальная длина непрерывной серии True в булевом массиве."""
    best = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def compute_metrics(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: Optional[int] = None,
) -> PerformanceMetrics:
    """Считает полный набор метрик по эквити и сделкам.

    Параметры
    ---------
    equity_curve:     эквити по закрытию каждого бара (DatetimeIndex).
    trades_df:        DataFrame сделок с колонками pnl, duration_bars, commission.
    risk_free_rate:   годовая безрисковая ставка (для Sharpe/Sortino).
    periods_per_year: число периодов в году; None -- оценить по шагу индекса.
    """
    m = PerformanceMetrics()
    if equity_curve.empty:
        return m

    ppy = periods_per_year or _estimate_periods_per_year(equity_curve.index)

    # --- Капитал ------------------------------------------------------- #
    m.initial_equity = float(equity_curve.iloc[0])
    m.final_equity = float(equity_curve.iloc[-1])
    m.net_profit = m.final_equity - m.initial_equity
    if m.initial_equity > 0:
        m.total_return_pct = (m.final_equity / m.initial_equity - 1.0) * 100.0

    years = len(equity_curve) / ppy
    if years > 0 and m.initial_equity > 0 and m.final_equity > 0:
        m.cagr_pct = ((m.final_equity / m.initial_equity) ** (1.0 / years) - 1.0) * 100.0

    # --- Риск ---------------------------------------------------------- #
    m.max_drawdown, m.max_drawdown_pct, m.max_drawdown_duration = max_drawdown_info(
        equity_curve
    )
    returns = equity_curve.pct_change().dropna().to_numpy(dtype=np.float64)
    m.sharpe_ratio = sharpe_ratio(returns, ppy, risk_free_rate)
    m.sortino_ratio = sortino_ratio(returns, ppy, risk_free_rate)
    if len(returns) >= 2:
        m.annualized_volatility_pct = float(
            returns.std(ddof=1) * math.sqrt(ppy) * 100.0
        )
    if m.max_drawdown > 0:
        m.recovery_factor = m.net_profit / m.max_drawdown
    if m.max_drawdown_pct > 0:
        m.calmar_ratio = m.cagr_pct / m.max_drawdown_pct

    # --- Сделки -------------------------------------------------------- #
    m.total_trades = len(trades_df)
    m.exposure_pct = exposure_pct(trades_df, len(equity_curve))
    if m.total_trades == 0:
        m.complex_criterion = m.total_return_pct - m.max_drawdown_pct
        return m

    pnl = trades_df["pnl"].to_numpy(dtype=np.float64)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    m.win_rate_pct = len(wins) / m.total_trades * 100.0
    gross_profit = float(wins.sum())
    gross_loss = float(np.abs(losses).sum())
    if gross_loss > 0:
        m.profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        m.profit_factor = math.inf

    m.expectancy = float(pnl.mean())
    m.avg_win = float(wins.mean()) if len(wins) else 0.0
    m.avg_loss = float(losses.mean()) if len(losses) else 0.0
    m.largest_win = float(wins.max()) if len(wins) else 0.0
    m.largest_loss = float(losses.min()) if len(losses) else 0.0
    m.max_consecutive_wins = _max_consecutive(pnl > 0)
    m.max_consecutive_losses = _max_consecutive(pnl <= 0)
    m.avg_trade_duration_bars = float(trades_df["duration_bars"].mean())
    m.total_commission = float(trades_df["commission"].sum())

    # --- Комплексный критерий ------------------------------------------ #
    if m.net_profit > 0:
        pf_capped = min(m.profit_factor, 3.0)  # потолок против пары удачных сделок
        m.complex_criterion = (
            m.total_return_pct
            * pf_capped
            * math.sqrt(m.total_trades)
            / (1.0 + m.max_drawdown_pct)
        )
    else:
        # Убыточные прогоны ранжируем по доходности с штрафом за просадку
        m.complex_criterion = m.total_return_pct - m.max_drawdown_pct
    return m


# ---------------------------------------------------------------------- #
# Лёгкий путь для оптимизации: одно значение метрики без лишних расчётов
# ---------------------------------------------------------------------- #
#: метрики, которым нужна статистика по сделкам (pnl), а не только эквити
_TRADE_LEVEL_METRICS = frozenset(
    {"profit_factor", "win_rate_pct", "expectancy", "complex_criterion"}
)
#: метрики с быстрым путём; остальное считается полным compute_metrics
_FAST_METRICS = frozenset(
    {
        "net_profit", "total_return_pct", "sharpe_ratio", "sortino_ratio",
        "recovery_factor",
    }
    | _TRADE_LEVEL_METRICS
)


def metric_needs_trades(metric: str) -> bool:
    """Нужна ли metric статистика по сделкам (для ленивой сборки trades_df)."""
    return metric not in _FAST_METRICS or metric in _TRADE_LEVEL_METRICS


def compute_optimization_metric(
    equity_curve: pd.Series,
    trades_df: Optional[pd.DataFrame],
    metric: str,
    risk_free_rate: float = 0.0,
    periods_per_year: Optional[int] = None,
) -> float:
    """Одно значение metric для оптимизатора -- без дорогих метрик, не входящих
    ни в один критерий (длительность просадки, серии сделок, экспозиция).

    Результат численно совпадает с getattr(compute_metrics(...), metric)
    (проверяется в tests/test_metrics.py). Для редких метрик вне быстрого
    набора честно считается полный compute_metrics.

    trades_df можно передать None, если metric не относится к сделкам
    (metric_needs_trades(metric) == False) -- тогда DataFrame не нужен вовсе.
    """
    if metric not in _FAST_METRICS:
        return float(getattr(
            compute_metrics(equity_curve, trades_df if trades_df is not None
                            else pd.DataFrame(), risk_free_rate, periods_per_year),
            metric,
        ))
    if equity_curve.empty:
        return 0.0  # как PerformanceMetrics() по умолчанию

    initial = float(equity_curve.iloc[0])
    final = float(equity_curve.iloc[-1])
    net_profit = final - initial
    total_return_pct = (final / initial - 1.0) * 100.0 if initial > 0 else 0.0

    if metric == "net_profit":
        return net_profit
    if metric == "total_return_pct":
        return total_return_pct

    if metric in ("sharpe_ratio", "sortino_ratio"):
        ppy = periods_per_year or _estimate_periods_per_year(equity_curve.index)
        returns = equity_curve.pct_change().dropna().to_numpy(dtype=np.float64)
        if metric == "sharpe_ratio":
            return sharpe_ratio(returns, ppy, risk_free_rate)
        return sortino_ratio(returns, ppy, risk_free_rate)

    if metric == "recovery_factor":
        dd_abs, _ = max_drawdown_abs_pct(equity_curve.to_numpy(dtype=np.float64))
        return net_profit / dd_abs if dd_abs > 0 else 0.0

    # --- метрики уровня сделок ---------------------------------------- #
    total_trades = 0 if trades_df is None else len(trades_df)
    if total_trades == 0:
        if metric == "complex_criterion":
            _, dd_pct = max_drawdown_abs_pct(equity_curve.to_numpy(dtype=np.float64))
            return total_return_pct - dd_pct
        return 0.0  # win_rate / profit_factor / expectancy по умолчанию

    pnl = trades_df["pnl"].to_numpy(dtype=np.float64)
    if metric == "win_rate_pct":
        return float((pnl > 0).sum()) / total_trades * 100.0
    if metric == "expectancy":
        return float(pnl.mean())

    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(np.abs(pnl[pnl <= 0]).sum())
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0
    if metric == "profit_factor":
        return profit_factor

    # complex_criterion
    if net_profit > 0:
        _, dd_pct = max_drawdown_abs_pct(equity_curve.to_numpy(dtype=np.float64))
        pf_capped = min(profit_factor, 3.0)
        return (
            total_return_pct * pf_capped * math.sqrt(total_trades)
            / (1.0 + dd_pct)
        )
    _, dd_pct = max_drawdown_abs_pct(equity_curve.to_numpy(dtype=np.float64))
    return total_return_pct - dd_pct
