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
    recovery_factor: float = 0.0       # net_profit / max_drawdown

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
    """Оценка числа торговых периодов в году по медианному шагу индекса."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return 252
    step = index.to_series().diff().median()
    if step is pd.NaT or step.total_seconds() <= 0:
        return 252
    return max(int(round(SECONDS_PER_YEAR / step.total_seconds())), 1)


def max_drawdown_info(equity: pd.Series) -> tuple[float, float, int]:
    """Возвращает (макс. просадка в деньгах, в %, длительность в барах)."""
    values = equity.to_numpy(dtype=np.float64)
    running_max = np.maximum.accumulate(values)
    drawdown = running_max - values

    dd_abs = float(drawdown.max()) if len(drawdown) else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct_series = np.where(running_max > 0, drawdown / running_max, 0.0)
    dd_pct = float(dd_pct_series.max() * 100.0) if len(values) else 0.0

    # Длительность: максимальное число баров между обновлениями максимума
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
    if std == 0 or math.isnan(std):
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
    if m.max_drawdown > 0:
        m.recovery_factor = m.net_profit / m.max_drawdown

    # --- Сделки -------------------------------------------------------- #
    m.total_trades = len(trades_df)
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
