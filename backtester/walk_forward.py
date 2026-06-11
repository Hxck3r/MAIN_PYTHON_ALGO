"""
Walk-Forward Analysis (WFA): скользящие окна оптимизации.

Идея: история режется на последовательные окна. В каждом окне параметры
оптимизируются на in-sample участке (IS), затем лучшие параметры
прогоняются на следующем, не виденном оптимизатором out-of-sample
участке (OOS). Все OOS-прогоны сшиваются в одну кривую эквити с
компаундингом (каждое окно стартует с капитала, на котором закончилось
предыдущее) -- это честная симуляция реальной торговли с периодической
переоптимизацией.

Режимы окон:
    * rolling  -- IS-окно фиксированной длины скользит вперёд на step;
    * anchored -- начало IS закреплено на старте периода, окно растёт.

Ключевая метрика устойчивости -- Walk-Forward Efficiency (WFE, по Пардо):
отношение скорости заработка на OOS к скорости заработка на IS
(доходность, нормированная на число баров), в процентах.
    * WFE ~ 100%  -- стратегия на новых данных работает как на истории;
    * WFE 50-70%  -- умеренная деградация (обычно приемлемо);
    * WFE <= 0    -- оптимизация подгоняет шум, система неустойчива.

Дополнительно считаются: доля прибыльных окон, разброс OOS-доходностей
(консистентность), стабильность оптимальных параметров между окнами
(коэффициент вариации) и полный набор метрик по сшитой OOS-кривой.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

import numpy as np
import pandas as pd

from .backtest_engine import BacktestEngine
from .base_strategy import BaseStrategy
from .metrics import PerformanceMetrics, compute_metrics
from .optimizer import (
    FloatParam,
    GeneticOptimizer,
    GridSearchOptimizer,
    IntParam,
    make_backtest_objective,
)

#: (сделано, всего) -- сквозной прогресс по всем окнам
ProgressCallback = Callable[[int, int], None]
#: текстовый статус для GUI ("Окно 3/10: оптимизация...")
StatusCallback = Callable[[str], None]

#: минимум баров в IS/OOS срезе, чтобы окно имело смысл
MIN_IS_BARS = 50
MIN_OOS_BARS = 10


# ---------------------------------------------------------------------- #
# Конфигурация
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class WalkForwardConfig:
    """Настройки разбиения истории на окна.

    Параметры
    ---------
    start, end:       границы всего периода анализа (end -- исключительно).
    in_sample_days:   длина IS-участка (календарных дней).
    out_sample_days:  длина OOS-участка (календарных дней).
    step_days:        смещение окон; 0 -> равно out_sample_days
                      (классический WFA: OOS-участки стыкуются без зазоров).
                      Число окон выводится из дат и смещения.
    anchored:         якорный режим -- начало IS зафиксировано на start,
                      IS-окно растёт с каждым шагом.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    in_sample_days: int
    out_sample_days: int
    step_days: int = 0
    anchored: bool = False

    @property
    def effective_step_days(self) -> int:
        return self.step_days if self.step_days > 0 else self.out_sample_days

    def validate(self) -> None:
        if self.in_sample_days <= 0 or self.out_sample_days <= 0:
            raise ValueError("Длины IS и OOS должны быть положительными")
        if self.effective_step_days < self.out_sample_days:
            raise ValueError(
                "Смещение меньше длины OOS: OOS-участки перекрылись бы, "
                "и сшитая кривая считала бы одни и те же бары дважды"
            )
        if self.end <= self.start:
            raise ValueError("Конец периода должен быть позже начала")


def window_bounds(
    config: WalkForwardConfig,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Границы окон: список (is_start, is_end, oos_start, oos_end).

    Все интервалы полуоткрытые [start, end), OOS начинается ровно там,
    где кончается IS. В список попадают только окна, чей OOS целиком
    помещается в период -- последний неполный хвост отбрасывается.
    """
    config.validate()
    step = pd.Timedelta(days=config.effective_step_days)
    is_len = pd.Timedelta(days=config.in_sample_days)
    oos_len = pd.Timedelta(days=config.out_sample_days)

    bounds = []
    i = 0
    while True:
        if config.anchored:
            is_start = config.start
            is_end = config.start + is_len + step * i
        else:
            is_start = config.start + step * i
            is_end = is_start + is_len
        oos_end = is_end + oos_len
        if oos_end > config.end:
            break
        bounds.append((is_start, is_end, is_end, oos_end))
        i += 1
    return bounds


# ---------------------------------------------------------------------- #
# Результаты
# ---------------------------------------------------------------------- #
@dataclass
class WindowResult:
    """Итог одного окна: оптимизация на IS + прогон лучших параметров на OOS."""

    index: int                              # номер окна, с 1
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    is_bars: int
    oos_bars: int
    best_params: dict[str, Any]
    is_score: float                         # лучший score критерия на IS
    is_stats: PerformanceMetrics            # бэктест IS с лучшими параметрами
    oos_stats: PerformanceMetrics
    oos_equity: pd.Series
    oos_trades: pd.DataFrame
    efficiency: float                       # WFE окна, %; NaN если IS <= 0
    valid: bool                             # нашлись ли валидные параметры

    @property
    def is_days(self) -> float:
        return (self.is_end - self.is_start).total_seconds() / 86_400.0

    @property
    def oos_days(self) -> float:
        return (self.oos_end - self.oos_start).total_seconds() / 86_400.0


@dataclass
class WalkForwardResult:
    """Полный итог WFA: окна + сшитая OOS-кривая + метрики устойчивости."""

    config: WalkForwardConfig
    windows: list[WindowResult]
    stitched_equity: pd.Series              # OOS-кривая с компаундингом
    oos_trades: pd.DataFrame                # все OOS-сделки (+колонка window)
    oos_stats: PerformanceMetrics           # метрики по сшитой кривой
    wfe_pct: float                          # агрегированная WFE, %
    pct_profitable_windows: float           # доля окон с OOS-прибылью, %
    consistency_std: float                  # std OOS-доходностей окон, %
    param_stability: dict[str, float]       # параметр -> коэф. вариации, %
    metric: str = ""                        # имя критерия оптимизации
    skipped_windows: int = 0                # окон пропущено (мало баров)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def summary_df(self) -> pd.DataFrame:
        """По строке на окно: даты, параметры, IS/OOS результаты."""
        rows = []
        for w in self.windows:
            row: dict[str, Any] = {
                "window": w.index,
                "is_start": w.is_start,
                "is_end": w.is_end,
                "oos_start": w.oos_start,
                "oos_end": w.oos_end,
                "is_score": w.is_score,
                "is_return_pct": w.is_stats.total_return_pct,
                "oos_return_pct": w.oos_stats.total_return_pct,
                "oos_trades": w.oos_stats.total_trades,
                "oos_sharpe": w.oos_stats.sharpe_ratio,
                "oos_max_dd_pct": w.oos_stats.max_drawdown_pct,
                "efficiency_pct": w.efficiency,
            }
            row.update({f"p_{k}": v for k, v in w.best_params.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        wfe = "n/a" if math.isnan(self.wfe_pct) else f"{self.wfe_pct:.1f}%"
        return (
            f"=== Walk-Forward Analysis ===\n"
            f"Окон: {self.n_windows} (пропущено {self.skipped_windows})\n"
            f"WFE: {wfe}\n"
            f"Прибыльных окон: {self.pct_profitable_windows:.0f}%\n"
            f"OOS доходность: {self.oos_stats.total_return_pct:.2f}%\n"
            f"OOS Sharpe: {self.oos_stats.sharpe_ratio:.2f}, "
            f"MaxDD: {self.oos_stats.max_drawdown_pct:.2f}%"
        )


# ---------------------------------------------------------------------- #
# Вспомогательное
# ---------------------------------------------------------------------- #
def grid_to_space(grid: dict[str, list]) -> dict[str, IntParam | FloatParam]:
    """Сетка значений -> пространство поиска для генетического алгоритма."""
    space: dict[str, IntParam | FloatParam] = {}
    for name, values in grid.items():
        if all(isinstance(v, (int, np.integer)) for v in values):
            step = int(values[1] - values[0]) if len(values) > 1 else 1
            space[name] = IntParam(int(min(values)), int(max(values)), max(step, 1))
        else:
            space[name] = FloatParam(float(min(values)), float(max(values)))
    return space


def _per_bar_return(return_pct: float, n_bars: int) -> float:
    """Доходность на бар -- общая шкала для IS и OOS разной длины."""
    return return_pct / n_bars if n_bars > 0 else math.nan


def _window_efficiency(w_is_pct: float, is_bars: int,
                       w_oos_pct: float, oos_bars: int) -> float:
    """WFE окна: (OOS-доходность на бар) / (IS-доходность на бар) * 100."""
    is_speed = _per_bar_return(w_is_pct, is_bars)
    oos_speed = _per_bar_return(w_oos_pct, oos_bars)
    if not math.isfinite(is_speed) or is_speed <= 1e-12:
        return math.nan
    return oos_speed / is_speed * 100.0


def _flat_equity(index: pd.Index, level: float) -> pd.Series:
    """Плоская эквити (окно без сделок / без валидных параметров)."""
    return pd.Series(np.full(len(index), level), index=index, name="equity")


def _param_stability(windows: list[WindowResult]) -> dict[str, float]:
    """Коэффициент вариации (std/|mean|, %) каждого параметра между окнами.

    Чем меньше, тем стабильнее оптимум: если лучшие параметры скачут от
    окна к окну, оптимизация ловит шум, а не структуру рынка.
    """
    valid = [w for w in windows if w.valid and w.best_params]
    if len(valid) < 2:
        return {}
    stability: dict[str, float] = {}
    names = valid[0].best_params.keys()
    for name in names:
        values = np.array(
            [float(w.best_params.get(name, np.nan)) for w in valid], dtype=float
        )
        values = values[np.isfinite(values)]
        if len(values) < 2:
            continue
        mean = values.mean()
        stability[name] = (
            float(values.std(ddof=1) / abs(mean) * 100.0)
            if abs(mean) > 1e-12 else math.nan
        )
    return stability


# ---------------------------------------------------------------------- #
# Основной цикл WFA
# ---------------------------------------------------------------------- #
def run_walk_forward(
    data: pd.DataFrame,
    strategy_cls: Type[BaseStrategy],
    config: WalkForwardConfig,
    grid: dict[str, list],
    metric: str = "sharpe_ratio",
    fixed_params: Optional[dict[str, Any]] = None,
    engine_kwargs: Optional[dict[str, Any]] = None,
    min_trades: int = 5,
    use_genetic: bool = False,
    population_size: int = 30,
    generations: int = 15,
    seed: int = 42,
    n_jobs: int = 1,
    progress_callback: Optional[ProgressCallback] = None,
    status_callback: Optional[StatusCallback] = None,
) -> WalkForwardResult:
    """Запускает полный walk-forward цикл.

    Для каждого окна: оптимизация (Grid/GA) на IS-срезе по критерию
    metric -> бэктест лучших параметров на IS (для WFE) и на OOS.
    OOS-прогоны компаундируются: капитал окна = финальная эквити
    предыдущего окна, поэтому сшитая кривая непрерывна по капиталу.

    n_jobs: число процессов для параллельной оптимизации
    (0 -> авто: ядра - 1; 1 -> без параллельности).
    Остальные параметры аналогичны make_backtest_objective
    + WalkForwardConfig.
    """
    if not grid:
        raise ValueError("Пустая сетка параметров: нечего оптимизировать")
    bounds = window_bounds(config)
    if not bounds:
        raise ValueError(
            "Не помещается ни одно окно: уменьшите длины IS/OOS "
            "или расширьте период"
        )

    fixed = dict(fixed_params or {})
    kwargs = dict(engine_kwargs or {})
    initial_cash = float(kwargs.get("cash", 10_000.0))

    # Пул процессов один на весь WFA: данные пересылаются воркерам один
    # раз, для каждого окна меняются только границы IS-среза
    evaluator = None
    if n_jobs != 1:
        from .parallel import ParallelEvaluator

        evaluator = ParallelEvaluator(
            data=data,
            strategy_cls=strategy_cls,
            metric=metric,
            fixed_params=fixed,
            engine_kwargs=kwargs,
            min_trades=min_trades,
            n_jobs=n_jobs,
        )

    # Сквозной прогресс: оценка «испытаний» на окно одинакова для всех окон
    inner_total = (
        generations if use_genetic
        else int(np.prod([len(v) for v in grid.values()]))
    )
    total_units = len(bounds) * inner_total

    def report(window_idx: int, done: int) -> None:
        if progress_callback is not None:
            progress_callback(window_idx * inner_total + done, total_units)

    def status(text: str) -> None:
        if status_callback is not None:
            status_callback(text)

    windows: list[WindowResult] = []
    equity_parts: list[pd.Series] = []
    trades_parts: list[pd.DataFrame] = []
    running_cash = initial_cash
    skipped = 0

    try:
        for k, (is_start, is_end, oos_start, oos_end) in enumerate(bounds):
            is_df = data[(data.index >= is_start) & (data.index < is_end)]
            oos_df = data[(data.index >= oos_start) & (data.index < oos_end)]
            if len(is_df) < MIN_IS_BARS or len(oos_df) < MIN_OOS_BARS:
                skipped += 1
                report(k, inner_total)
                continue

            status(
                f"Окно {k + 1}/{len(bounds)}: оптимизация IS "
                f"({is_start:%Y-%m-%d} — {is_end:%Y-%m-%d})"
            )

            # --- 1) оптимизация на in-sample ------------------------------ #
            objective = make_backtest_objective(
                data=is_df,
                strategy_cls=strategy_cls,
                metric=metric,
                fixed_params=fixed,
                engine_kwargs=kwargs,
                min_trades=min_trades,
            )
            if evaluator is not None:
                # воркеры пула сами вырежут IS-срез из полных данных
                evaluator.set_bounds(is_start, is_end)
            inner_progress = lambda done, _total, _k=k: report(_k, done)  # noqa: E731
            if use_genetic:
                optimizer: GridSearchOptimizer | GeneticOptimizer = GeneticOptimizer(
                    objective,
                    param_space=grid_to_space(grid),
                    population_size=population_size,
                    generations=generations,
                    seed=seed + k,            # свой поток случайности на окно
                    progress_callback=inner_progress,
                    batch_evaluator=evaluator,
                )
            else:
                optimizer = GridSearchOptimizer(
                    objective, param_grid=grid,
                    progress_callback=inner_progress,
                    batch_evaluator=evaluator,
                )
            opt = optimizer.optimize()
            valid = bool(opt.best_params) and math.isfinite(opt.best_score)

            # --- 2) прогон лучших параметров: IS (для WFE) и OOS ---------- #
            if valid:
                status(f"Окно {k + 1}/{len(bounds)}: прогон OOS")
                params = {**fixed, **opt.best_params}
                is_result = BacktestEngine(
                    data=is_df, strategy_cls=strategy_cls,
                    strategy_params=params, **kwargs,
                ).run()
                is_stats = is_result.stats()

                # Компаундинг: OOS стартует с капитала конца предыдущего окна
                oos_kwargs = {**kwargs, "cash": running_cash}
                oos_result = BacktestEngine(
                    data=oos_df, strategy_cls=strategy_cls,
                    strategy_params=params, **oos_kwargs,
                ).run()
                oos_stats = oos_result.stats()
                oos_equity = oos_result.equity_curve
                oos_trades = oos_result.trades_df()
                running_cash = float(oos_equity.iloc[-1])
            else:
                # Параметры не нашлись (всё отфильтровано min_trades) --
                # окно «торгует в кэше»: плоская эквити, без сделок
                is_stats = PerformanceMetrics()
                oos_stats = PerformanceMetrics()
                oos_equity = _flat_equity(oos_df.index, running_cash)
                oos_trades = pd.DataFrame()

            efficiency = _window_efficiency(
                is_stats.total_return_pct, len(is_df),
                oos_stats.total_return_pct, len(oos_df),
            ) if valid else math.nan

            windows.append(WindowResult(
                index=len(windows) + 1,
                is_start=is_start, is_end=is_end,
                oos_start=oos_start, oos_end=oos_end,
                is_bars=len(is_df), oos_bars=len(oos_df),
                best_params=dict(opt.best_params) if valid else {},
                is_score=opt.best_score if valid else math.nan,
                is_stats=is_stats, oos_stats=oos_stats,
                oos_equity=oos_equity, oos_trades=oos_trades,
                efficiency=efficiency, valid=valid,
            ))
            equity_parts.append(oos_equity)
            if not oos_trades.empty:
                oos_trades = oos_trades.assign(window=len(windows))
                trades_parts.append(oos_trades)
            report(k, inner_total)
    finally:
        if evaluator is not None:
            evaluator.close()

    if not windows:
        raise ValueError(
            "Все окна пропущены: в срезах слишком мало баров "
            "(проверьте даты и плотность данных)"
        )

    # --- Сшивка и метрики устойчивости ------------------------------------ #
    stitched = pd.concat(equity_parts).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    all_trades = (
        pd.concat(trades_parts, ignore_index=True)
        if trades_parts else pd.DataFrame()
    )
    oos_stats_total = compute_metrics(stitched, all_trades)

    valid_windows = [w for w in windows if w.valid]
    if valid_windows:
        oos_speed = _per_bar_return(
            sum(w.oos_stats.total_return_pct for w in valid_windows),
            sum(w.oos_bars for w in valid_windows),
        )
        is_speed = _per_bar_return(
            sum(w.is_stats.total_return_pct for w in valid_windows),
            sum(w.is_bars for w in valid_windows),
        )
        wfe = (
            oos_speed / is_speed * 100.0
            if math.isfinite(is_speed) and is_speed > 1e-12 else math.nan
        )
        profitable = sum(
            1 for w in valid_windows if w.oos_stats.net_profit > 0
        ) / len(valid_windows) * 100.0
        returns = np.array(
            [w.oos_stats.total_return_pct for w in valid_windows], dtype=float
        )
        consistency = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    else:
        wfe, profitable, consistency = math.nan, 0.0, 0.0

    status("Walk-Forward завершён")
    return WalkForwardResult(
        config=config,
        windows=windows,
        stitched_equity=stitched,
        oos_trades=all_trades,
        oos_stats=oos_stats_total,
        wfe_pct=wfe,
        pct_profitable_windows=profitable,
        consistency_std=consistency,
        param_stability=_param_stability(windows),
        metric=metric,
        skipped_windows=skipped,
    )
