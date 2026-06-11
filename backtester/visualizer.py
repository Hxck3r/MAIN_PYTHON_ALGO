"""
Интерактивная визуализация результатов (Plotly).

Возможности:
    * plot_backtest          -- цена (свечи) с точками входа/выхода,
                                кривая эквити и просадка;
    * plot_optimization      -- гистограмма распределения score по испытаниям;
    * plot_param_heatmap     -- тепловая карта score по двум параметрам
                                (для результатов Grid Search).

Каждая функция возвращает go.Figure; show=True открывает график в браузере,
save_path сохраняет интерактивный HTML-отчёт.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .backtest_engine import BacktestResult
from .optimizer import OptimizationResult


def _finish(fig: go.Figure, show: bool, save_path: Optional[str | Path]) -> go.Figure:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs="cdn")
    if show:
        fig.show()
    return fig


def plot_backtest(
    result: BacktestResult,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str | Path] = None,
) -> go.Figure:
    """График цены со сделками + эквити + просадка (три панели)."""
    data = result.data
    trades = result.trades_df()
    equity = result.equity_curve

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.27, 0.18],
        subplot_titles=("Цена и сделки", "Эквити", "Просадка"),
    )

    # --- Панель 1: свечи ------------------------------------------------ #
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Цена",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    # --- Точки входа/выхода ---------------------------------------------#
    if not trades.empty:
        longs = trades[trades["side"] == "BUY"]
        shorts = trades[trades["side"] == "SELL"]
        marker_sets = [
            (longs["entry_time"], longs["entry_price"], "triangle-up",
             "#2e7d32", "Вход long"),
            (shorts["entry_time"], shorts["entry_price"], "triangle-down",
             "#c62828", "Вход short"),
        ]
        for x, y, symbol, color, name in marker_sets:
            if len(x):
                fig.add_trace(
                    go.Scatter(
                        x=x, y=y, mode="markers", name=name,
                        marker=dict(symbol=symbol, size=11, color=color,
                                    line=dict(width=1, color="white")),
                    ),
                    row=1, col=1,
                )
        # Выходы: крест, цвет по результату сделки
        fig.add_trace(
            go.Scatter(
                x=trades["exit_time"],
                y=trades["exit_price"],
                mode="markers",
                name="Выход",
                marker=dict(
                    symbol="x",
                    size=9,
                    color=["#2e7d32" if p > 0 else "#c62828" for p in trades["pnl"]],
                ),
                text=[
                    f"PnL: {p:,.2f} ({r})"
                    for p, r in zip(trades["pnl"], trades["exit_reason"])
                ],
                hovertemplate="%{x}<br>Цена: %{y:.4f}<br>%{text}<extra></extra>",
            ),
            row=1, col=1,
        )

    # --- Панель 2: эквити ------------------------------------------------#
    fig.add_trace(
        go.Scatter(
            x=equity.index, y=equity.values, name="Эквити",
            line=dict(color="#1565c0", width=1.5),
        ),
        row=2, col=1,
    )
    fig.add_hline(
        y=result.initial_cash, line_dash="dot", line_color="gray", row=2, col=1
    )

    # --- Панель 3: просадка ----------------------------------------------#
    running_max = equity.cummax()
    drawdown_pct = (equity / running_max - 1.0) * 100.0
    fig.add_trace(
        go.Scatter(
            x=drawdown_pct.index, y=drawdown_pct.values, name="Просадка, %",
            fill="tozeroy", line=dict(color="#ef5350", width=1),
        ),
        row=3, col=1,
    )

    fig.update_layout(
        title=title
        or f"{result.strategy_name} {result.strategy_params} | "
           f"Net Profit: {result.net_profit:,.2f}",
        xaxis_rangeslider_visible=False,
        height=850,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=110),
    )
    return _finish(fig, show, save_path)


def plot_optimization(
    opt_result: OptimizationResult,
    bins: int = 40,
    show: bool = True,
    save_path: Optional[str | Path] = None,
) -> go.Figure:
    """Гистограмма распределения score по всем испытаниям оптимизации."""
    df = opt_result.history_df()
    scores = df["score"].replace([float("inf"), -float("inf")], pd.NA).dropna()

    fig = go.Figure(
        go.Histogram(x=scores, nbinsx=bins, marker_color="#1565c0", name="score")
    )
    if len(scores):
        fig.add_vline(
            x=opt_result.best_score,
            line_color="#2e7d32",
            line_dash="dash",
            annotation_text=f"лучший: {opt_result.best_score:.3f}",
        )
    fig.update_layout(
        title=f"Распределение результатов | {opt_result.method} "
              f"({len(df)} испытаний)",
        xaxis_title="score",
        yaxis_title="число испытаний",
        height=500,
    )
    return _finish(fig, show, save_path)


def plot_param_heatmap(
    opt_result: OptimizationResult,
    x_param: str,
    y_param: str,
    show: bool = True,
    save_path: Optional[str | Path] = None,
) -> go.Figure:
    """Тепловая карта score по двум параметрам (Grid Search).

    Если для пары (x, y) есть несколько испытаний (другие параметры
    варьировались), берётся максимум score.
    """
    df = opt_result.history_df()
    for param in (x_param, y_param):
        if param not in df.columns:
            raise ValueError(f"Параметр '{param}' отсутствует в истории оптимизации")

    pivot = df.pivot_table(index=y_param, columns=x_param, values="score", aggfunc="max")
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdYlGn",
            colorbar=dict(title="score"),
        )
    )
    fig.update_layout(
        title=f"Карта score: {x_param} x {y_param} | {opt_result.method}",
        xaxis_title=x_param,
        yaxis_title=y_param,
        height=550,
    )
    return _finish(fig, show, save_path)
