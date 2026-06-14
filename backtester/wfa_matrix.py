"""
Матричный WFO-анализ (WFA Cluster Analysis): перебор длин окон IS/OOS.

Одиночный walk-forward отвечает на вопрос «работает ли стратегия при
ЭТИХ длинах окон». Но сами длины IS/OOS -- такие же произвольные
параметры, как и параметры стратегии: удачный WFE при IS=180/OOS=45
ничего не значит, если при IS=160/OOS=40 всё рушится. Матричный WFO
прогоняет полноценный walk-forward для каждой комбинации длин из двух
списков (IS × OOS) и строит матрицу агрегатных метрик.

Что ищем -- ПЛАТО (кластер) стабильности: широкую область соседних
ячеек матрицы с похоже хорошими результатами. Если плато есть, стратегия
устойчива к выбору длин окон (робастна); если хорош лишь одинокий пик
среди провалов -- исходный WFA-результат был, скорее всего, артефактом
конкретного нарезания истории (переподгонка к окну).

Ключевые сводные числа (по Пардо и практике StrategyQuant Walk-Forward
Matrix):
    * avg_wfe_pct          -- средняя Walk-Forward Efficiency по ячейкам;
    * pct_profitable_cells -- доля комбинаций с прибыльной OOS-кривой;
    * target_cv_pct        -- коэффициент вариации целевой метрики по
                              ячейкам (чувствительность к длинам окон;
                              низкий -- широкое плато, высокий -- хрупко);
    * robust_cell          -- центр лучшего плато: рекомендуемые длины
                              IS/OOS (максимум среднего по окрестности,
                              а не глобальный пик -- он переоптимизирован).

Производительность (рассчитано на слабый CPU):
    * единый пул процессов на всю матрицу -- история пересылается воркерам
      ОДИН раз, а не на каждую ячейку (см. ParallelEvaluator);
    * в последовательном режиме -- общий кэш IS-оценок между ячейками:
      совпадающие IS-срезы (частые при разной длине OOS) не считаются
      повторно;
    * ячейки, куда не помещается минимум окон, отсеиваются ДО прогона;
    * полные результаты ячеек по умолчанию не хранятся (память) -- только
      скалярные метрики;
    * сквозной прогресс по всем бэктестам матрицы для отзывчивого GUI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Optional, Type

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy
from .walk_forward import (
    ProgressCallback,
    StatusCallback,
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward,
    window_bounds,
)

#: скалярные поля ячейки, доступные для отображения матрицей.
#: имя -> (человекочитаемая подпись, «больше -- лучше»)
MATRIX_FIELDS: dict[str, tuple[str, bool]] = {
    "wfe_pct": ("Walk-Forward Efficiency, %", True),
    "oos_return_pct": ("OOS доходность (сшитая), %", True),
    "oos_sharpe": ("OOS Sharpe", True),
    "oos_max_dd_pct": ("OOS макс. просадка, %", False),
    "pct_profitable_windows": ("Прибыльных окон, %", True),
    "metric_retention_pct": ("Удержание критерия IS→OOS, %", True),
    "oos_return_tstat": ("t-стат OOS-доходностей", True),
    "is_oos_rank_corr": ("Spearman IS↔OOS", True),
    "avg_plateau_pct": ("Плато параметров, %", True),
    "oos_dsr": ("OOS Deflated Sharpe", True),
    "n_windows": ("Число окон", True),
}


# ---------------------------------------------------------------------- #
# Построение сетки длин окон
# ---------------------------------------------------------------------- #
def span_values(lo: int, hi: int, count: int, geometric: bool = True) -> tuple[int, ...]:
    """count целых длин окна в [lo, hi]: геометрически или линейно.

    Геометрическая шкала (по умолчанию) равномерно покрывает РАЗНЫЕ
    масштабы окон (60, 90, 135, ... а не 60, 70, 80) -- именно так длины
    окон и влияют на WFA. Значения округляются до целых и дедуплицируются,
    поэтому фактическое число может быть меньше count.
    """
    if lo <= 0 or hi <= 0:
        raise ValueError("Длины окон должны быть положительными")
    if hi < lo:
        lo, hi = hi, lo
    if count <= 1 or lo == hi:
        return (int(lo),) if lo == hi else (int(lo), int(hi))
    if geometric:
        raw = np.geomspace(lo, hi, count)
    else:
        raw = np.linspace(lo, hi, count)
    return tuple(sorted({int(round(v)) for v in raw}))


# ---------------------------------------------------------------------- #
# Конфигурация
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class WfaMatrixConfig:
    """Сетка длин окон IS/OOS для матричного WFO.

    Параметры
    ---------
    start, end:       границы всего периода (end -- исключительно).
    is_days_list:     перебираемые длины IS-окна (календарных дней).
    oos_days_list:    перебираемые длины OOS-окна.
    anchored:         якорный режим IS (начало закреплено, окно растёт).
    step_days:        смещение окон; 0 -> равно длине OOS ячейки (классика).
                      Общий step для всех ячеек; если меньше OOS какой-то
                      ячейки, такая ячейка отсеивается (перекрытие OOS).
    """

    start: pd.Timestamp
    end: pd.Timestamp
    is_days_list: tuple[int, ...]
    oos_days_list: tuple[int, ...]
    anchored: bool = False
    step_days: int = 0

    @classmethod
    def from_ranges(
        cls,
        start: pd.Timestamp,
        end: pd.Timestamp,
        is_range: tuple[int, int, int],
        oos_range: tuple[int, int, int],
        anchored: bool = False,
        step_days: int = 0,
        geometric: bool = True,
    ) -> "WfaMatrixConfig":
        """Конфигурация из диапазонов (min, max, count) для IS и OOS."""
        return cls(
            start=start,
            end=end,
            is_days_list=span_values(*is_range, geometric=geometric),
            oos_days_list=span_values(*oos_range, geometric=geometric),
            anchored=anchored,
            step_days=step_days,
        )

    def validate(self) -> None:
        if self.end <= self.start:
            raise ValueError("Конец периода должен быть позже начала")
        if not self.is_days_list or not self.oos_days_list:
            raise ValueError("Списки длин IS и OOS не должны быть пустыми")
        if any(v <= 0 for v in self.is_days_list + self.oos_days_list):
            raise ValueError("Длины окон должны быть положительными")

    def cells(self) -> list[tuple[int, int]]:
        """Все комбинации (is_days, oos_days) в порядке is×oos."""
        return [
            (is_d, oos_d)
            for is_d in sorted(set(self.is_days_list))
            for oos_d in sorted(set(self.oos_days_list))
        ]

    def window_config(self, is_days: int, oos_days: int) -> WalkForwardConfig:
        """WalkForwardConfig для одной ячейки матрицы."""
        return WalkForwardConfig(
            start=self.start,
            end=self.end,
            in_sample_days=is_days,
            out_sample_days=oos_days,
            step_days=self.step_days,
            anchored=self.anchored,
        )


# ---------------------------------------------------------------------- #
# Результаты
# ---------------------------------------------------------------------- #
@dataclass
class WfaMatrixCell:
    """Итог одной ячейки матрицы (один полный walk-forward) -- только скаляры.

    full результат (WalkForwardResult) хранится лишь при keep_results=True:
    он тяжёлый (эквити, сделки), а матрица из десятков ячеек на слабой
    машине съела бы много памяти.
    """

    is_days: int
    oos_days: int
    ran: bool = False                       # ячейка реально прогналась
    reason: str = ""                        # почему пропущена (если ran=False)
    n_windows: int = 0
    valid_windows: int = 0
    skipped_windows: int = 0
    wfe_pct: float = math.nan
    oos_return_pct: float = math.nan
    oos_sharpe: float = math.nan
    oos_max_dd_pct: float = math.nan
    pct_profitable_windows: float = math.nan
    metric_retention_pct: float = math.nan
    oos_return_tstat: float = math.nan
    is_oos_rank_corr: float = math.nan
    avg_plateau_pct: float = math.nan
    oos_dsr: float = math.nan
    result: Optional[WalkForwardResult] = None

    def get(self, field_name: str) -> float:
        """Значение поля для матрицы (n_windows -> float)."""
        return float(getattr(self, field_name, math.nan))


@dataclass
class WfaMatrixStability:
    """Сводка устойчивости по всей матрице."""

    n_cells: int                            # всего ячеек в сетке
    n_ran: int                              # реально прогналось
    avg_wfe_pct: float                      # средняя WFE по прогнанным
    pct_profitable_cells: float             # доля ячеек с OOS-прибылью, %
    target_mean: float                      # среднее целевой метрики
    target_cv_pct: float                    # коэф. вариации целевой, %
    plateau_pct: float                      # «ширина плато» вокруг robust-ячейки, %

    def __str__(self) -> str:
        def fmt(v: float, suffix: str = "") -> str:
            return "n/a" if not math.isfinite(v) else f"{v:.1f}{suffix}"

        return (
            f"ячеек прогнано {self.n_ran}/{self.n_cells}; "
            f"avg WFE {fmt(self.avg_wfe_pct, '%')}; "
            f"прибыльных {fmt(self.pct_profitable_cells, '%')}; "
            f"CV цели {fmt(self.target_cv_pct, '%')}; "
            f"плато {fmt(self.plateau_pct, '%')}"
        )


@dataclass
class WfaMatrixResult:
    """Полный итог матричного WFO: ячейки + матрицы метрик + плато."""

    config: WfaMatrixConfig
    cells: list[WfaMatrixCell]
    target: str = "wfe_pct"                 # метрика, по которой ищем плато
    metric: str = ""                        # критерий оптимизации внутри WFA
    grid: dict[str, list] = field(default_factory=dict)

    # -- доступ к ячейкам --------------------------------------------------- #
    @property
    def ran_cells(self) -> list[WfaMatrixCell]:
        return [c for c in self.cells if c.ran]

    @property
    def is_axis(self) -> list[int]:
        return sorted({c.is_days for c in self.cells})

    @property
    def oos_axis(self) -> list[int]:
        return sorted({c.oos_days for c in self.cells})

    def cell(self, is_days: int, oos_days: int) -> Optional[WfaMatrixCell]:
        for c in self.cells:
            if c.is_days == is_days and c.oos_days == oos_days:
                return c
        return None

    # -- матрицы ------------------------------------------------------------ #
    def matrix(self, field_name: Optional[str] = None) -> pd.DataFrame:
        """Матрица значений поля: строки -- длина IS, столбцы -- длина OOS.

        Непрогнанные ячейки -- NaN. По умолчанию берётся целевая метрика.
        """
        field_name = field_name or self.target
        rows, cols = self.is_axis, self.oos_axis
        mat = pd.DataFrame(np.nan, index=rows, columns=cols, dtype=float)
        mat.index.name = "IS, дн"
        mat.columns.name = "OOS, дн"
        for c in self.cells:
            if c.ran:
                mat.at[c.is_days, c.oos_days] = c.get(field_name)
        return mat

    def _target_array(self) -> tuple[np.ndarray, list[WfaMatrixCell]]:
        """Конечные значения целевой метрики и соответствующие ячейки."""
        vals, cells = [], []
        higher_better = MATRIX_FIELDS.get(self.target, ("", True))[1]
        for c in self.ran_cells:
            v = c.get(self.target)
            if math.isfinite(v):
                # приводим к шкале «больше -- лучше» для выбора оптимума
                vals.append(v if higher_better else -v)
                cells.append(c)
        return np.asarray(vals, dtype=float), cells

    # -- выбор ячеек -------------------------------------------------------- #
    def best_cell(self) -> Optional[WfaMatrixCell]:
        """Глобальный максимум целевой метрики (для справки: вероятный пик)."""
        vals, cells = self._target_array()
        if not len(vals):
            return None
        return cells[int(np.argmax(vals))]

    def robust_cell(self) -> Optional[WfaMatrixCell]:
        """Центр лучшего плато: максимум СРЕДНЕГО по окрестности 3×3.

        Рекомендуемые длины IS/OOS. В отличие от best_cell, устойчив к
        одиночным пикам -- выбирает ячейку, у которой и соседи хороши.
        """
        mat = self.matrix(self.target)
        higher_better = MATRIX_FIELDS.get(self.target, ("", True))[1]
        arr = mat.to_numpy(dtype=float)
        if not higher_better:
            arr = -arr
        rows, cols = mat.index.tolist(), mat.columns.tolist()
        best_score, best_rc = -math.inf, None
        n_r, n_c = arr.shape
        for i in range(n_r):
            for j in range(n_c):
                if not math.isfinite(arr[i, j]):
                    continue
                block = arr[max(i - 1, 0): i + 2, max(j - 1, 0): j + 2]
                finite = block[np.isfinite(block)]
                score = float(finite.mean()) if len(finite) else -math.inf
                if score > best_score:
                    best_score, best_rc = score, (rows[i], cols[j])
        if best_rc is None:
            return None
        return self.cell(*best_rc)

    # -- устойчивость / плато ----------------------------------------------- #
    def stability(self, tol: float = 0.2) -> WfaMatrixStability:
        """Сводные числа робастности по всей матрице.

        plateau_pct: доля ячеек окрестности robust-ячейки (включая её),
        чья целевая метрика держится в пределах tol от значения robust-ячейки.
        Высокое -- широкий стабильный кластер; низкое -- одинокий пик.
        """
        ran = self.ran_cells
        wfes = np.array(
            [c.wfe_pct for c in ran if math.isfinite(c.wfe_pct)], dtype=float
        )
        avg_wfe = float(wfes.mean()) if len(wfes) else math.nan

        profit_total = sum(
            1 for c in ran if math.isfinite(c.oos_return_pct)
        )
        profitable = sum(
            1 for c in ran
            if math.isfinite(c.oos_return_pct) and c.oos_return_pct > 0
        )
        pct_profit = (
            profitable / profit_total * 100.0 if profit_total else math.nan
        )

        vals, _ = self._target_array()
        if len(vals):
            mean = float(vals.mean())
            cv = (
                float(vals.std(ddof=1) / abs(mean) * 100.0)
                if len(vals) > 1 and abs(mean) > 1e-12 else math.nan
            )
        else:
            mean, cv = math.nan, math.nan

        return WfaMatrixStability(
            n_cells=len(self.cells),
            n_ran=len(ran),
            avg_wfe_pct=avg_wfe,
            pct_profitable_cells=pct_profit,
            target_mean=mean,
            target_cv_pct=cv,
            plateau_pct=self._plateau_pct(tol),
        )

    def _plateau_pct(self, tol: float) -> float:
        """Доля соседей robust-ячейки (включая её), близких к её цели, %."""
        robust = self.robust_cell()
        if robust is None:
            return math.nan
        mat = self.matrix(self.target)
        arr = mat.to_numpy(dtype=float)
        rows, cols = mat.index.tolist(), mat.columns.tolist()
        i, j = rows.index(robust.is_days), cols.index(robust.oos_days)
        center = arr[i, j]
        if not math.isfinite(center):
            return math.nan
        block = arr[max(i - 1, 0): i + 2, max(j - 1, 0): j + 2]
        finite = block[np.isfinite(block)]
        if not len(finite):
            return math.nan
        higher_better = MATRIX_FIELDS.get(self.target, ("", True))[1]
        if higher_better:
            good = finite >= center - tol * abs(center)
        else:
            good = finite <= center + tol * abs(center)
        return float(good.sum()) / len(finite) * 100.0

    # -- отчёт -------------------------------------------------------------- #
    def __str__(self) -> str:
        label = MATRIX_FIELDS.get(self.target, (self.target, True))[0]
        lines = [
            "=== Матричный WFO (WFA Cluster Analysis) ===",
            f"Сетка: {len(self.config.is_days_list)} IS × "
            f"{len(self.config.oos_days_list)} OOS = {len(self.cells)} ячеек "
            f"(критерий «{self.metric}», цель плато: {label})",
            str(self.stability()),
        ]
        robust = self.robust_cell()
        best = self.best_cell()
        if robust is not None:
            lines.append(
                f"Рекомендация (центр плато): IS={robust.is_days}д / "
                f"OOS={robust.oos_days}д  "
                f"[WFE={_fmt(robust.wfe_pct, '%')}, "
                f"OOS={_fmt(robust.oos_return_pct, '%')}, "
                f"Sharpe={_fmt(robust.oos_sharpe)}]"
            )
        if best is not None and (
            robust is None
            or (best.is_days, best.oos_days) != (robust.is_days, robust.oos_days)
        ):
            lines.append(
                f"Глобальный пик (возможна переподгонка): IS={best.is_days}д / "
                f"OOS={best.oos_days}д  [{label}={_fmt(best.get(self.target))}]"
            )
        return "\n".join(lines)


def _fmt(value: float, suffix: str = "") -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.2f}{suffix}"


# ---------------------------------------------------------------------- #
# Основной прогон матрицы
# ---------------------------------------------------------------------- #
def run_wfa_matrix(
    data: pd.DataFrame,
    strategy_cls: Type[BaseStrategy],
    matrix_config: WfaMatrixConfig,
    grid: dict[str, list],
    metric: str = "sharpe_ratio",
    target: str = "wfe_pct",
    fixed_params: Optional[dict[str, Any]] = None,
    engine_kwargs: Optional[dict[str, Any]] = None,
    min_trades: int = 5,
    use_genetic: bool = False,
    population_size: int = 30,
    generations: int = 15,
    seed: int = 42,
    n_jobs: int = 1,
    warmup_bars: int = 0,
    min_windows: int = 2,
    keep_results: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    status_callback: Optional[StatusCallback] = None,
) -> WfaMatrixResult:
    """Прогоняет walk-forward для каждой комбинации длин IS/OOS.

    Параметры повторяют run_walk_forward; добавлены:
        matrix_config: сетка длин окон (см. WfaMatrixConfig).
        target:        метрика, по которой ищется плато (ключ MATRIX_FIELDS).
        min_windows:   ячейки с меньшим числом окон отсеиваются (мало точек
                       для оценки устойчивости).
        keep_results:  хранить ли полный WalkForwardResult каждой ячейки
                       (по умолчанию нет -- экономия памяти).

    Производительность: единый пул процессов (n_jobs != 1) или общий кэш
    IS-оценок (последовательный режим) переиспользуются всеми ячейками.
    """
    matrix_config.validate()
    if not grid:
        raise ValueError("Пустая сетка параметров: нечего оптимизировать")
    if target not in MATRIX_FIELDS:
        raise ValueError(
            f"Неизвестная целевая метрика '{target}'. "
            f"Доступны: {', '.join(MATRIX_FIELDS)}"
        )
    fixed = dict(fixed_params or {})
    kwargs = dict(engine_kwargs or {})

    # --- предварительный отбор ячеек + оценка объёма работы ---------------- #
    # window_bounds дёшев -- считаем заранее, чтобы пропустить нежизнеспособные
    # ячейки без прогона и заранее знать сквозной объём прогресса.
    inner_total = (
        generations if use_genetic
        else int(np.prod([len(v) for v in grid.values()]))
    )
    planned: list[tuple[int, int, int]] = []  # (is_days, oos_days, n_bounds)
    skipped: list[WfaMatrixCell] = []
    for is_days, oos_days in matrix_config.cells():
        try:
            n_bounds = len(window_bounds(
                matrix_config.window_config(is_days, oos_days)
            ))
        except ValueError as exc:
            skipped.append(WfaMatrixCell(is_days, oos_days, reason=str(exc)))
            continue
        if n_bounds < min_windows:
            skipped.append(WfaMatrixCell(
                is_days, oos_days,
                reason=f"окон {n_bounds} < {min_windows}",
                n_windows=n_bounds,
            ))
            continue
        planned.append((is_days, oos_days, n_bounds))

    if not planned:
        raise ValueError(
            "Ни одна ячейка матрицы не помещает достаточно окон: "
            "уменьшите длины IS/OOS, ослабьте min_windows или расширьте период"
        )

    grand_total = sum(n * inner_total for _, _, n in planned)
    done_units = 0

    def status(text: str) -> None:
        if status_callback is not None:
            status_callback(text)

    # --- единый пул процессов на всю матрицу ------------------------------- #
    evaluator = None
    is_eval_cache: Optional[dict[tuple, float]] = None
    if n_jobs != 1:
        from .parallel import ParallelEvaluator

        evaluator = ParallelEvaluator(
            data=data, strategy_cls=strategy_cls, metric=metric,
            fixed_params=fixed, engine_kwargs=kwargs,
            min_trades=min_trades, n_jobs=n_jobs,
        )
    else:
        # последовательный режим: кэш IS-оценок общий на все ячейки
        is_eval_cache = {}

    cells: list[WfaMatrixCell] = list(skipped)
    try:
        for k, (is_days, oos_days, n_bounds) in enumerate(planned):
            status(
                f"Ячейка {k + 1}/{len(planned)}: IS={is_days}д, OOS={oos_days}д "
                f"({n_bounds} окон)"
            )
            cell_base = done_units

            def cell_progress(done: int, _total: int, _base: int = cell_base) -> None:
                if progress_callback is not None:
                    progress_callback(_base + done, grand_total)

            try:
                wf = run_walk_forward(
                    data=data, strategy_cls=strategy_cls,
                    config=matrix_config.window_config(is_days, oos_days),
                    grid=grid, metric=metric, fixed_params=fixed,
                    engine_kwargs=kwargs, min_trades=min_trades,
                    use_genetic=use_genetic, population_size=population_size,
                    generations=generations, seed=seed,
                    warmup_bars=warmup_bars,
                    progress_callback=cell_progress,
                    evaluator=evaluator, is_eval_cache=is_eval_cache,
                )
            except ValueError as exc:
                cells.append(WfaMatrixCell(
                    is_days, oos_days, reason=str(exc), n_windows=n_bounds,
                ))
                done_units += n_bounds * inner_total
                if progress_callback is not None:
                    progress_callback(done_units, grand_total)
                continue

            cells.append(_cell_from_result(
                is_days, oos_days, wf, keep_results=keep_results
            ))
            done_units += n_bounds * inner_total
            if progress_callback is not None:
                progress_callback(done_units, grand_total)
    finally:
        if evaluator is not None:
            evaluator.close()

    status("Матричный WFO завершён")
    return WfaMatrixResult(
        config=matrix_config,
        cells=cells,
        target=target,
        metric=metric,
        grid=dict(grid),
    )


def _cell_from_result(
    is_days: int, oos_days: int, wf: WalkForwardResult, keep_results: bool
) -> WfaMatrixCell:
    """Скаляры одной ячейки из полного WalkForwardResult."""
    stats = wf.oos_stats
    dsr = wf.oos_dsr.dsr if wf.oos_dsr is not None else math.nan
    return WfaMatrixCell(
        is_days=is_days,
        oos_days=oos_days,
        ran=True,
        n_windows=wf.n_windows,
        valid_windows=len(wf.valid_windows),
        skipped_windows=wf.skipped_windows,
        wfe_pct=wf.wfe_pct,
        oos_return_pct=stats.total_return_pct,
        oos_sharpe=stats.sharpe_ratio,
        oos_max_dd_pct=stats.max_drawdown_pct,
        pct_profitable_windows=wf.pct_profitable_windows,
        metric_retention_pct=wf.metric_retention_pct,
        oos_return_tstat=wf.oos_return_tstat,
        is_oos_rank_corr=wf.is_oos_rank_corr,
        avg_plateau_pct=wf.avg_plateau_pct,
        oos_dsr=dsr,
        result=wf if keep_results else None,
    )
