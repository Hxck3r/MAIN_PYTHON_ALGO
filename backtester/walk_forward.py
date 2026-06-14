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
from typing import TYPE_CHECKING, Any, Callable, Optional, Type

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # только для аннотаций -- без циклического импорта
    from .statistics import DeflatedSharpeResult

from .backtest_engine import run_backtest
from .base_strategy import BaseStrategy
from .metrics import PerformanceMetrics, compute_metrics
from .optimizer import (
    FloatParam,
    GeneticOptimizer,
    GridSearchOptimizer,
    IntParam,
    StopRequested,
    make_backtest_objective,
)

#: (сделано, всего) -- сквозной прогресс по всем окнам
ProgressCallback = Callable[[int, int], None]
#: текстовый статус для GUI ("Окно 3/10: оптимизация...")
StatusCallback = Callable[[str], None]
#: коллбек прерывания: возвращает True, если расчёт нужно остановить
ShouldStop = Callable[[], bool]

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
    oos_score: float                        # тот же критерий, но на OOS
    is_stats: PerformanceMetrics            # бэктест IS с лучшими параметрами
    oos_stats: PerformanceMetrics
    oos_equity: pd.Series
    oos_trades: pd.DataFrame
    efficiency: float                       # WFE окна, %; NaN если IS <= 0
    valid: bool                             # нашлись ли валидные параметры
    # IS-score каждой испытанной комбинации: ключ -- tuple(sorted(params.items())).
    # Нужен для plateau-check (оценка окрестности оптимума). Для GA покрытие
    # сетки разреженное, поэтому plateau считается в основном для grid search.
    is_scores: dict[tuple, float] = field(default_factory=dict)

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
    grid: dict[str, list] = field(default_factory=dict)  # исходная сетка поиска
    metric_retention_pct: float = math.nan  # медиана OOS-крит. / IS-крит., %
    oos_return_tstat: float = math.nan       # t-стат OOS-доходностей окон (> 0?)
    max_consecutive_losing_windows: int = 0  # самая длинная серия убыточных окон
    is_oos_rank_corr: float = math.nan       # Spearman is_score vs oos_score
    avg_plateau_pct: float = math.nan        # средняя «ширина плато» оптимума, %
    # DSR сшитой OOS-кривой с поправкой на размер перебора в окне (n_trials):
    # значим ли OOS-Sharpe с учётом того, что оптимизатор отсеивал N кандидатов.
    oos_dsr: Optional["DeflatedSharpeResult"] = None

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def valid_windows(self) -> list[WindowResult]:
        return [w for w in self.windows if w.valid]

    # -- Выбор параметров: частота и «лучшие» окна --------------------------- #
    def param_usage(self) -> dict[str, list[tuple[Any, int, float]]]:
        """Частота выбора значений каждого параметра по валидным окнам.

        Для каждого параметра -- список (значение, число_окон, доля_%),
        отсортированный по убыванию частоты. Показывает, на каких значениях
        оптимизатор «останавливается» чаще всего: устойчивый параметр имеет
        ярко выраженную моду, шумный -- размазан по всему диапазону.
        """
        valid = self.valid_windows
        usage: dict[str, list[tuple[Any, int, float]]] = {}
        if not valid:
            return usage
        names = sorted({name for w in valid for name in w.best_params})
        for name in names:
            counts: dict[Any, int] = {}
            present = 0
            for w in valid:
                if name in w.best_params:
                    counts[w.best_params[name]] = counts.get(w.best_params[name], 0) + 1
                    present += 1
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            usage[name] = [
                (val, cnt, cnt / present * 100.0) for val, cnt in ranked
            ]
        return usage

    def consensus_params(self) -> dict[str, Any]:
        """Самое частое (модальное) значение каждого параметра по окнам.

        Покомпонентная мода -- консервативный кандидат на «боевые» параметры:
        она устойчива к выбросам отдельных окон. См. также modal_param_set()
        для самого частого *целого* набора.
        """
        return {name: usage[0][0] for name, usage in self.param_usage().items()}

    def modal_param_set(self) -> tuple[dict[str, Any], int, float]:
        """Самый частый *целый* набор параметров: (набор, число_окон, доля_%).

        В отличие от покомпонентной consensus_params(), это реально
        встречавшаяся в WFA комбинация -- её OOS-поведение уже наблюдалось.
        """
        valid = self.valid_windows
        if not valid:
            return {}, 0, 0.0
        counts: dict[tuple, int] = {}
        for w in valid:
            key = tuple(sorted(w.best_params.items()))
            counts[key] = counts.get(key, 0) + 1
        best_key, best_cnt = max(counts.items(), key=lambda kv: kv[1])
        return dict(best_key), best_cnt, best_cnt / len(valid) * 100.0

    def best_is_window(self) -> Optional[WindowResult]:
        """Окно с лучшим значением критерия на IS (наиболее «выгодная» подгонка)."""
        valid = [w for w in self.valid_windows if math.isfinite(w.is_score)]
        return max(valid, key=lambda w: w.is_score) if valid else None

    def best_oos_window(self) -> Optional[WindowResult]:
        """Окно с лучшим значением критерия на OOS (лучшее на новых данных)."""
        valid = [w for w in self.valid_windows if math.isfinite(w.oos_score)]
        return max(valid, key=lambda w: w.oos_score) if valid else None

    def plateau_per_window(self) -> dict[int, float]:
        """Ширина плато оптимума в каждом валидном окне: {номер_окна -> %}.

        См. _window_plateau_pct: высокий % -- соседи оптимума по сетке почти
        так же хороши (робастно), низкий -- изолированный пик (переподгонка).
        """
        result: dict[int, float] = {}
        for w in self.valid_windows:
            pct = _window_plateau_pct(w.is_scores, w.best_params, self.grid)
            if math.isfinite(pct):
                result[w.index] = pct
        return result

    def param_edge_warnings(self) -> dict[str, str]:
        """Параметры, чья мода упирается в границу сетки поиска.

        Если оптимум систематически садится на минимум/максимум диапазона,
        истинный оптимум, вероятно, лежит ЗА пределами сетки -- диапазон
        стоит расширить, иначе WFA оценивает заведомо урезанную стратегию.
        """
        warnings: dict[str, str] = {}
        if not self.grid:
            return warnings
        consensus = self.consensus_params()
        for name, value in consensus.items():
            values = self.grid.get(name)
            if not values:
                continue
            try:
                lo, hi = min(values), max(values)
            except TypeError:            # категориальные значения -- границ нет
                continue
            if value == lo and lo != hi:
                warnings[name] = f"на нижней границе сетки ({lo})"
            elif value == hi and lo != hi:
                warnings[name] = f"на верхней границе сетки ({hi})"
        return warnings

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

    @staticmethod
    def _fmt_params(params: dict[str, Any]) -> str:
        """Компактная запись набора параметров: a=5, b=1.50."""
        if not params:
            return "—"
        parts = []
        for k, v in sorted(params.items()):
            parts.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
        return ", ".join(parts)

    def param_report(self) -> str:
        """Текстовый разбор выбора параметров: частота, консенсус, лучшие окна.

        Отвечает на главный практический вопрос WFA: «какие параметры брать
        в бой и насколько уверенно их выбирал оптимизатор».
        """
        valid = self.valid_windows
        lines = ["=== Параметры по окнам ==="]
        if not valid:
            lines.append("Нет валидных окон — оптимизатор ничего не выбрал.")
            return "\n".join(lines)

        # Частота выбора значений каждого параметра
        for name, usage in self.param_usage().items():
            shown = ", ".join(
                f"{val:g}×{cnt} ({pct:.0f}%)" if isinstance(val, float)
                else f"{val}×{cnt} ({pct:.0f}%)"
                for val, cnt, pct in usage[:5]
            )
            tail = " …" if len(usage) > 5 else ""
            cv = self.param_stability.get(name)
            cv_txt = (
                f"  CV={cv:.0f}%" if cv is not None and math.isfinite(cv) else ""
            )
            lines.append(f"  {name}: {shown}{tail}{cv_txt}")

        # Покомпонентный консенсус и самый частый целый набор
        lines.append("")
        lines.append(f"Консенсус (мода по параметрам): {self._fmt_params(self.consensus_params())}")
        modal, cnt, pct = self.modal_param_set()
        lines.append(
            f"Самый частый набор: {self._fmt_params(modal)} "
            f"— {cnt}/{len(valid)} окон ({pct:.0f}%)"
        )

        # Лучшие окна на IS и OOS — видно расхождение «подгонка vs реальность»
        bi, bo = self.best_is_window(), self.best_oos_window()
        if bi is not None:
            lines.append(
                f"Лучшее окно на IS:  #{bi.index} {self.metric}={bi.is_score:.3f} "
                f"→ OOS={bi.oos_score:.3f}  [{self._fmt_params(bi.best_params)}]"
            )
        if bo is not None:
            lines.append(
                f"Лучшее окно на OOS: #{bo.index} {self.metric}={bo.oos_score:.3f} "
                f"(на IS={bo.is_score:.3f})  [{self._fmt_params(bo.best_params)}]"
            )

        # Робастность оптимума: предсказуемость IS→OOS и ширина плато
        lines.append("")
        corr_txt = (
            f"{self.is_oos_rank_corr:+.2f}"
            if math.isfinite(self.is_oos_rank_corr) else "n/a"
        )
        lines.append(
            f"Spearman IS↔OOS (предсказуемость ранга): {corr_txt}  "
            f"(>0 — IS-лучшее переносится на OOS; ≤0 — ранжируется шум)"
        )
        plateau_txt = (
            f"{self.avg_plateau_pct:.0f}%"
            if math.isfinite(self.avg_plateau_pct) else "n/a"
        )
        lines.append(
            f"Плато оптимума (соседи по сетке ≈ лучшего): {plateau_txt}  "
            f"(высоко — робастно; низко — изолированный пик/переподгонка)"
        )

        # Предупреждения о параметрах на границе сетки
        edges = self.param_edge_warnings()
        if edges:
            lines.append("")
            lines.append("⚠ Параметры на границе сетки (расширьте диапазон):")
            for name, where in edges.items():
                lines.append(f"  {name}: {where}")
        return "\n".join(lines)

    def __str__(self) -> str:
        wfe = "n/a" if math.isnan(self.wfe_pct) else f"{self.wfe_pct:.1f}%"
        retention = (
            "n/a" if math.isnan(self.metric_retention_pct)
            else f"{self.metric_retention_pct:.0f}%"
        )
        tstat = (
            "n/a" if math.isnan(self.oos_return_tstat)
            else f"{self.oos_return_tstat:.2f}"
        )
        corr = (
            "n/a" if math.isnan(self.is_oos_rank_corr)
            else f"{self.is_oos_rank_corr:+.2f}"
        )
        plateau = (
            "n/a" if math.isnan(self.avg_plateau_pct)
            else f"{self.avg_plateau_pct:.0f}%"
        )
        modal, cnt, pct = self.modal_param_set()
        if self.oos_dsr is None:
            dsr = "n/a"
        else:
            verdict = "значим" if self.oos_dsr.dsr >= 0.95 else "вероятен шум"
            dsr = (
                f"{self.oos_dsr.dsr:.2f} ({verdict}; порог шума SR*="
                f"{self.oos_dsr.expected_max_sr:.3f} при {self.oos_dsr.n_trials} испыт.)"
            )
        return (
            f"=== Walk-Forward Analysis ===\n"
            f"Окон: {self.n_windows} (пропущено {self.skipped_windows})\n"
            f"WFE: {wfe}  |  удержание критерия IS→OOS: {retention}\n"
            f"Прибыльных окон: {self.pct_profitable_windows:.0f}%  "
            f"(макс. серия убыточных подряд: {self.max_consecutive_losing_windows})\n"
            f"t-стат OOS-доходностей окон: {tstat}\n"
            f"Spearman IS↔OOS: {corr}  |  плато оптимума: {plateau}\n"
            f"OOS доходность: {self.oos_stats.total_return_pct:.2f}%\n"
            f"OOS Sharpe: {self.oos_stats.sharpe_ratio:.2f}, "
            f"MaxDD: {self.oos_stats.max_drawdown_pct:.2f}%\n"
            f"OOS Deflated Sharpe: {dsr}\n"
            f"Самый частый набор: {self._fmt_params(modal)} ({pct:.0f}% окон)"
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


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Ранги значений (1..N) со средним рангом для совпадений (как у SciPy)."""
    arr = np.asarray(values, dtype=float)
    sorter = np.argsort(arr, kind="mergesort")
    inv = np.empty(len(arr), dtype=int)
    inv[sorter] = np.arange(len(arr))
    arr_sorted = arr[sorter]
    obs = np.r_[True, arr_sorted[1:] != arr_sorted[:-1]]
    dense = obs.cumsum()[inv]
    counts = np.r_[np.nonzero(obs)[0], len(arr)]
    return 0.5 * (counts[dense] + counts[dense - 1] + 1)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Ранговая корреляция Спирмена (без SciPy). NaN, если данных < 3."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return math.nan
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if denom <= 1e-12:           # нет разброса рангов (все значения равны)
        return math.nan
    return float((rx * ry).sum() / denom)


def _make_cached_objective(
    objective: Callable[[dict[str, Any]], float],
    cache: dict[tuple, float],
    is_start: pd.Timestamp,
    is_end: pd.Timestamp,
) -> Callable[[dict[str, Any]], float]:
    """Обёртка objective с кэшем по (границы IS, набор параметров).

    Матричный WFO перебирает много комбинаций длин IS/OOS, и одни и те же
    IS-срезы регулярно повторяются между ячейками (например, при разной
    длине OOS, но одинаковой длине IS совпадает часть окон). Кэш по
    (is_start, is_end, параметры) исключает повторный прогон одного и того
    же бэктеста в пределах общего словаря cache. Применяется только в
    последовательном режиме: при пуле процессов score считается в воркерах.
    """
    base = (is_start.value, is_end.value)

    def cached(params: dict[str, Any]) -> float:
        key = base + tuple(sorted(params.items()))
        hit = cache.get(key, _CACHE_MISS)
        if hit is not _CACHE_MISS:
            return hit
        value = objective(params)
        cache[key] = value
        return value

    return cached


#: сентинел «промаха» кэша -- отличает отсутствие ключа от закэшированного NaN
_CACHE_MISS = object()


def _window_plateau_pct(
    is_scores: dict[tuple, float],
    best_params: dict[str, Any],
    grid: dict[str, list],
    tol: float = 0.2,
) -> float:
    """Доля соседей оптимума по сетке, сохранивших ~уровень лучшего score, %.

    Берутся непосредственные соседи лучшей точки по каждой оси сетки
    (на один шаг влево/вправо) и считается, у скольких из них IS-score
    остаётся в пределах tol от лучшего (>= best - tol*|best|).

    100% -- оптимум на широком плато (соседи почти так же хороши, параметры
    робастны); низкое значение -- изолированный пик (рядом всё рушится),
    типичный признак переподгонки под конкретный прогон. NaN -- соседей не
    нашлось (например, GA не покрыл сетку плотно).
    """
    if not is_scores or not best_params:
        return math.nan
    best_key = tuple(sorted(best_params.items()))
    best = is_scores.get(best_key)
    if best is None or not math.isfinite(best):
        return math.nan

    axes: dict[str, list] = {}
    for name in best_params:
        values = grid.get(name)
        if not values:
            continue
        try:
            axes[name] = sorted(set(values))
        except TypeError:        # категориальные значения -- «соседство» не задано
            continue

    neighbor_scores: list[float] = []
    for name, axis in axes.items():
        if best_params[name] not in axis:
            continue
        i = axis.index(best_params[name])
        for j in (i - 1, i + 1):
            if 0 <= j < len(axis):
                neighbour = dict(best_params)
                neighbour[name] = axis[j]
                score = is_scores.get(tuple(sorted(neighbour.items())))
                if score is not None:
                    neighbor_scores.append(score)
    if not neighbor_scores:
        return math.nan
    threshold = best - tol * abs(best)
    good = sum(1 for s in neighbor_scores if math.isfinite(s) and s >= threshold)
    return good / len(neighbor_scores) * 100.0


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
    warmup_bars: int = 0,
    progress_callback: Optional[ProgressCallback] = None,
    status_callback: Optional[StatusCallback] = None,
    should_stop: Optional[ShouldStop] = None,
    evaluator: Optional[Any] = None,
    is_eval_cache: Optional[dict[tuple, float]] = None,
) -> WalkForwardResult:
    """Запускает полный walk-forward цикл.

    Для каждого окна: оптимизация (Grid/GA) на IS-срезе по критерию
    metric -> бэктест лучших параметров на IS (для WFE) и на OOS.
    OOS-прогоны компаундируются: капитал окна = финальная эквити
    предыдущего окна, поэтому сшитая кривая непрерывна по капиталу.

    warmup_bars: сколько баров ПЕРЕД началом OOS подать в прогон для
    прогрева индикаторов (торговля на них запрещена, эквити и сделки
    считаются только внутри OOS). Без прогрева первые ~период индикатора
    баров каждого OOS-окна стратегия не торгует -- при коротких окнах это
    систематически искажает OOS-результаты. Разумное значение -- максимум
    из периодов индикаторов стратегии (с запасом).

    n_jobs: число процессов для параллельной оптимизации
    (0 -> авто: ядра - 1; 1 -> без параллельности).

    evaluator: готовый ParallelEvaluator, общий на несколько прогонов WFA
    (для матричного WFO -- пул создаётся один раз, данные пересылаются
    воркерам однократно). Если задан, n_jobs игнорируется, а закрытие пула
    остаётся на стороне вызывающего кода. is_eval_cache: общий словарь
    кэша IS-оценок (только последовательный режим) -- исключает повторный
    прогон совпадающих IS-срезов между ячейками матрицы.

    Остальные параметры аналогичны make_backtest_objective
    + WalkForwardConfig.
    """
    if warmup_bars < 0:
        raise ValueError("warmup_bars не может быть отрицательным")
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
    # раз, для каждого окна меняются только границы IS-среза. Если пул
    # передан извне (матричный WFO), переиспользуем его и не закрываем.
    external_evaluator = evaluator is not None
    if evaluator is None and n_jobs != 1:
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
            if should_stop is not None and should_stop():
                raise StopRequested()
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
            elif is_eval_cache is not None:
                # последовательный режим: кэшируем IS-оценки между ячейками
                objective = _make_cached_objective(
                    objective, is_eval_cache, is_start, is_end
                )
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
                    should_stop=should_stop,
                )
            else:
                optimizer = GridSearchOptimizer(
                    objective, param_grid=grid,
                    progress_callback=inner_progress,
                    batch_evaluator=evaluator,
                    should_stop=should_stop,
                )
            opt = optimizer.optimize()
            valid = bool(opt.best_params) and math.isfinite(opt.best_score)

            # --- 2) прогон лучших параметров: IS (для WFE) и OOS ---------- #
            if valid:
                status(f"Окно {k + 1}/{len(bounds)}: прогон OOS")
                params = {**fixed, **opt.best_params}
                is_result = run_backtest(
                    data=is_df, strategy_cls=strategy_cls,
                    strategy_params=params, **kwargs,
                )
                is_stats = is_result.stats()

                # Компаундинг: OOS стартует с капитала конца предыдущего окна.
                # Прогрев: в прогон подаётся хвост IS (warmup_bars баров),
                # на котором торговля запрещена, -- индикаторы к началу OOS
                # уже прогреты; эквити обрезается до чистого OOS.
                oos_kwargs = {**kwargs, "cash": running_cash}
                oos_pos = int(data.index.searchsorted(oos_start))
                warm_pos = max(oos_pos - warmup_bars, 0)
                actual_warmup = oos_pos - warm_pos
                run_df = data.iloc[warm_pos : oos_pos + len(oos_df)]
                oos_result = run_backtest(
                    data=run_df, strategy_cls=strategy_cls,
                    strategy_params=params, warmup_bars=actual_warmup,
                    **oos_kwargs,
                )
                oos_equity = oos_result.equity_curve.iloc[actual_warmup:]
                oos_trades = oos_result.trades_df()
                oos_stats = compute_metrics(oos_equity, oos_trades)
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

            oos_score = (
                float(getattr(oos_stats, metric, math.nan)) if valid else math.nan
            )
            # score каждой испытанной комбинации (для plateau-check); при
            # совпадении геномов в GA берём первый
            is_scores: dict[tuple, float] = {}
            for params_i, score_i in opt.history:
                key = tuple(sorted(params_i.items()))
                is_scores.setdefault(key, float(score_i))
            windows.append(WindowResult(
                index=len(windows) + 1,
                is_start=is_start, is_end=is_end,
                oos_start=oos_start, oos_end=oos_end,
                is_bars=len(is_df), oos_bars=len(oos_df),
                best_params=dict(opt.best_params) if valid else {},
                is_score=opt.best_score if valid else math.nan,
                oos_score=oos_score,
                is_stats=is_stats, oos_stats=oos_stats,
                oos_equity=oos_equity, oos_trades=oos_trades,
                efficiency=efficiency, valid=valid,
                is_scores=is_scores if valid else {},
            ))
            equity_parts.append(oos_equity)
            if not oos_trades.empty:
                oos_trades = oos_trades.assign(window=len(windows))
                trades_parts.append(oos_trades)
            report(k, inner_total)
    finally:
        if evaluator is not None and not external_evaluator:
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

        # Удержание критерия IS->OOS: насколько просел сам оптимизируемый
        # критерий на новых данных (медианой -- устойчиво к выбросам окон).
        is_scores = np.array(
            [w.is_score for w in valid_windows if math.isfinite(w.is_score)],
            dtype=float,
        )
        oos_scores = np.array(
            [w.oos_score for w in valid_windows if math.isfinite(w.oos_score)],
            dtype=float,
        )
        med_is = float(np.median(is_scores)) if len(is_scores) else math.nan
        med_oos = float(np.median(oos_scores)) if len(oos_scores) else math.nan
        retention = (
            med_oos / med_is * 100.0
            if math.isfinite(med_is) and abs(med_is) > 1e-12 else math.nan
        )

        # t-стат OOS-доходностей окон: значимо ли среднее OOS > 0 (есть ли
        # вообще статистическое преимущество, а не везение пары окон).
        if len(returns) > 1 and returns.std(ddof=1) > 1e-12:
            tstat = float(
                returns.mean() / (returns.std(ddof=1) / math.sqrt(len(returns)))
            )
        else:
            tstat = math.nan

        # Самая длинная серия убыточных OOS-окон подряд (выживаемость капитала)
        max_losing = streak = 0
        for w in valid_windows:
            streak = streak + 1 if w.oos_stats.net_profit <= 0 else 0
            max_losing = max(max_losing, streak)

        # Spearman is_score vs oos_score: предсказывает ли ранг на IS ранг на
        # OOS. ~0/отрицательно -> оптимизация ранжирует шум, IS-«лучшее» не
        # переносится на новые данные.
        rank_corr = _spearman(
            np.array([w.is_score for w in valid_windows], dtype=float),
            np.array([w.oos_score for w in valid_windows], dtype=float),
        )

        # Plateau-check: средняя «ширина плато» оптимума по окнам. Изолированные
        # пики IS-оптимума (низкое плато) -- признак переподгонки.
        plateaus = [
            p for p in (
                _window_plateau_pct(w.is_scores, w.best_params, grid)
                for w in valid_windows
            ) if math.isfinite(p)
        ]
        avg_plateau = float(np.mean(plateaus)) if plateaus else math.nan
    else:
        wfe, profitable, consistency = math.nan, 0.0, 0.0
        retention, tstat, max_losing = math.nan, math.nan, 0
        rank_corr, avg_plateau = math.nan, math.nan

    # Deflated Sharpe сшитой OOS-кривой: значимость OOS-Sharpe с поправкой на
    # размер перебора в окне (сколько кандидатов отсеивал оптимизатор). Для
    # сетки это её мощность, для генетики -- число оценённых геномов.
    from .statistics import deflated_sharpe_ratio

    grid_size = int(np.prod([len(v) for v in grid.values()]))
    trials = population_size * generations if use_genetic else grid_size
    oos_returns = stitched.pct_change().dropna().to_numpy(dtype=np.float64)
    oos_dsr = (
        deflated_sharpe_ratio(
            oos_returns, np.empty(0, dtype=np.float64), n_trials=max(trials, 1)
        )
        if len(oos_returns) >= 2 else None
    )

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
        grid=dict(grid),
        metric_retention_pct=retention,
        oos_return_tstat=tstat,
        max_consecutive_losing_windows=max_losing,
        is_oos_rank_corr=rank_corr,
        avg_plateau_pct=avg_plateau,
        oos_dsr=oos_dsr,
    )
