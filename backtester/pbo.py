"""
Probability of Backtest Overfitting (PBO) через CSCV.

DSR (statistics.py) отвечает «значим ли лучший Sharpe с учётом N испытаний»,
но НЕ отвечает на главный вопрос отбора: «систематически ли конфигурация,
лучшая на in-sample, проваливается на out-of-sample?». Именно это и есть
вероятность переподгонки (PBO), и она вычисляется методом CSCV
(Combinatorially Symmetric Cross-Validation, Bailey, Borwein, Lopez de Prado,
Zhu, 2017).

Идея CSCV. Берём матрицу M (T наблюдений × N конфигураций) побарных
доходностей всех испытаний оптимизации. Режем T строк на S равных смежных
блоков (S чётное). Перебираем все C(S, S/2) способов выбрать половину блоков
как IS, оставшуюся половину -- как OOS:

    * на IS считаем производительность (Sharpe) каждой конфигурации,
      берём лучшую n*;
    * смотрим относительный РАНГ n* среди всех конфигураций на OOS;
    * логит относительного ранга lambda: lambda > 0 -- IS-лучший выше
      OOS-медианы (отбор переносится), lambda <= 0 -- ниже (переподгонка).

PBO = доля комбинаций с lambda <= 0. PBO ~ 0.5 означает, что отбор лучшего
по IS равносилен подбрасыванию монеты на OOS (чистая переподгонка); PBO,
близкая к 0, -- что преимущество переносится на новые данные.

Побочно метод даёт регрессию деградации (OOS-производительность отобранного
против IS) и вероятность убытка отобранного на OOS.

Зависимостей вне numpy/pandas + ядра пакета нет.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Optional, Sequence, Type

import numpy as np
import pandas as pd

from .backtest_engine import run_backtest
from .base_strategy import BaseStrategy


# ---------------------------------------------------------------------- #
# Enabling change: матрица побарных доходностей по испытаниям
# ---------------------------------------------------------------------- #
def unique_configs(
    history: Sequence[tuple[dict[str, Any], float]],
) -> list[dict[str, Any]]:
    """Уникальные наборы параметров из истории оптимизации (без дублей).

    GA переоценивает одни и те же геномы, Grid -- нет; CSCV нужны именно
    различные конфигурации, иначе дубликаты искусственно «уплотняют» ранги.
    Порядок первого появления сохраняется.
    """
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    for params, _ in history:
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            out.append(dict(params))
    return out


def build_trial_returns(
    data: pd.DataFrame,
    strategy_cls: Type[BaseStrategy],
    configs: Sequence[dict[str, Any]],
    *,
    fixed_params: Optional[dict[str, Any]] = None,
    engine_kwargs: Optional[dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Матрица побарных доходностей по каждому испытанию (конфигурации).

    Возвращает DataFrame: индекс -- бары (как у data), колонки -- индексы
    конфигураций 0..N-1, значения -- доходность эквити за бар. Все
    конфигурации прогоняются на одних и тех же данных, поэтому колонки
    выровнены по общему индексу баров -- это и есть вход для CSCV/PBO и
    для канонического DSR (дисперсия Sharpe по испытаниям).

    Дорогой шаг (N полных прогонов); для SignalStrategy автоматически
    используется быстрое numba-ядро (run_backtest сам выбирает движок).

    Конфигурации, на которых стратегия падает (например, недопустимое
    сочетание параметров вроде fast >= slow у SMA-кроссовера), молча
    пропускаются -- в матрицу попадают только успешно прогнанные. Колонки
    нумеруются заново 0..M-1 по успешным прогонам.
    """
    fixed = dict(fixed_params or {})
    kwargs = dict(engine_kwargs or {})
    n = len(configs)
    series: list[pd.Series] = []
    for k, cfg in enumerate(configs):
        try:
            result = run_backtest(
                data=data,
                strategy_cls=strategy_cls,
                strategy_params={**fixed, **cfg},
                **kwargs,
            )
        except Exception:  # недопустимые параметры -- конфиг пропускаем
            if progress_callback is not None:
                progress_callback(k + 1, n)
            continue
        # Доходность эквити по закрытию бара; первый бар -> 0 (нет предыдущего).
        series.append(result.equity_curve.pct_change().fillna(0.0))
        if progress_callback is not None:
            progress_callback(k + 1, n)
    return pd.DataFrame({i: s for i, s in enumerate(series)})


# ---------------------------------------------------------------------- #
# CSCV / PBO
# ---------------------------------------------------------------------- #
@dataclass
class PBOResult:
    """Итог CSCV-анализа переподгонки набора конфигураций."""

    pbo: float                       # вероятность переподгонки [0..1]
    n_splits: int                    # число блоков S
    n_combinations: int              # перебранных комбинаций IS/OOS, C(S, S/2)
    n_trials: int                    # число конфигураций N
    logits: np.ndarray               # lambda по каждой комбинации
    oos_rank_selected: np.ndarray    # относительный OOS-ранг IS-лучшего, (0..1)
    is_perf_selected: np.ndarray     # IS-Sharpe отобранного по комбинациям
    oos_perf_selected: np.ndarray    # OOS-Sharpe отобранного по комбинациям
    prob_oos_loss: float             # доля комбинаций с OOS-Sharpe<=0 у отобранного
    # Наклон регрессии OOS~IS производительности отобранного по комбинациям.
    # Внимание: смещён в минус самой схемой CSCV (IS и OOS -- комплементарные
    # разбиения одной выборки, удачные IS-блоки оставляют менее удачные на
    # OOS), поэтому информативна прежде всего сама PBO, а не знак наклона.
    degradation_slope: float

    def __str__(self) -> str:
        verdict = (
            "переподгонка: отбор лучшего по IS не переносится на OOS"
            if self.pbo >= 0.5
            else "отбор по IS переносится на OOS (приемлемо)"
        )
        return (
            f"=== Probability of Backtest Overfitting (CSCV) ===\n"
            f"PBO: {self.pbo:.3f} -- {verdict}\n"
            f"Блоков S={self.n_splits}, комбинаций {self.n_combinations}, "
            f"конфигураций N={self.n_trials}\n"
            f"P(убыток отобранного на OOS): {self.prob_oos_loss:.3f}; "
            f"наклон деградации OOS~IS: {self.degradation_slope:.3f}"
        )


def _block_sharpe_sufficient_stats(
    matrix: np.ndarray, n_splits: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Достаточные статистики Sharpe по S смежным блокам.

    Возвращает (sum по блокам, sum квадратов по блокам, длина блока).
    Sharpe объединения любых блоков восстанавливается из сумм без повторной
    нарезки данных -- это и делает перебор C(S, S/2) комбинаций дешёвым.
    """
    t = matrix.shape[0]
    block_len = t // n_splits
    if block_len < 2:
        raise ValueError(
            f"слишком мало наблюдений ({t}) на {n_splits} блоков: "
            f"в блоке {block_len} < 2 точек, Sharpe не определён"
        )
    usable = block_len * n_splits  # хвост-остаток отбрасываем для равных блоков
    blocks = matrix[:usable].reshape(n_splits, block_len, matrix.shape[1])
    b_sum = blocks.sum(axis=1)              # (S, N)
    b_sumsq = (blocks ** 2).sum(axis=1)     # (S, N)
    return b_sum, b_sumsq, float(block_len)


def _sets_sharpe(
    masks: np.ndarray, b_sum: np.ndarray, b_sumsq: np.ndarray, block_len: float
) -> np.ndarray:
    """Sharpe каждой конфигурации на наборе блоков, заданном масками.

    masks -- (n_combinations, S) из 0/1; результат -- (n_combinations, N).
    Векторизовано: суммы по выбранным блокам через матричное умножение.
    """
    s = masks @ b_sum                          # (combos, N)
    ss = masks @ b_sumsq                        # (combos, N)
    count = masks.sum(axis=1, keepdims=True) * block_len  # (combos, 1)
    mean = s / count
    var = ss / count - mean ** 2
    var = np.clip(var, 0.0, None)
    std = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 0.0, mean / std, 0.0)
    return sharpe


def _linreg_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Наклон OLS-регрессии y по x (деградация OOS~IS); 0 при вырождении."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    xc = x - x.mean()
    denom = float((xc ** 2).sum())
    if denom <= 0.0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / denom)


def probability_of_backtest_overfitting(
    returns_matrix: Any,
    n_splits: int = 16,
) -> PBOResult:
    """PBO методом CSCV по матрице побарных доходностей конфигураций.

    Параметры
    ---------
    returns_matrix: (T наблюдений × N конфигураций) -- ndarray или DataFrame
                    (например, результат build_trial_returns).
    n_splits:       число блоков S (чётное, >= 2). Каноническое S = 16 даёт
                    C(16, 8) = 12870 комбинаций IS/OOS.

    Возвращает PBOResult: pbo, логиты, OOS-ранги и деградацию.
    """
    matrix = (
        returns_matrix.to_numpy(dtype=np.float64)
        if isinstance(returns_matrix, pd.DataFrame)
        else np.asarray(returns_matrix, dtype=np.float64)
    )
    if matrix.ndim != 2:
        raise ValueError("returns_matrix должна быть 2D (T × N)")
    t, n = matrix.shape
    if n < 2:
        raise ValueError("нужно минимум 2 конфигурации для оценки переподгонки")
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError("n_splits должно быть чётным и >= 2")

    b_sum, b_sumsq, block_len = _block_sharpe_sufficient_stats(matrix, n_splits)

    half = n_splits // 2
    combos = list(combinations(range(n_splits), half))
    n_combos = len(combos)
    is_masks = np.zeros((n_combos, n_splits), dtype=np.float64)
    for i, c in enumerate(combos):
        is_masks[i, list(c)] = 1.0
    oos_masks = 1.0 - is_masks

    is_sr = _sets_sharpe(is_masks, b_sum, b_sumsq, block_len)    # (combos, N)
    oos_sr = _sets_sharpe(oos_masks, b_sum, b_sumsq, block_len)  # (combos, N)

    rows = np.arange(n_combos)
    n_star = np.argmax(is_sr, axis=1)                # лучший по IS в каждой комбинации
    is_sel = is_sr[rows, n_star]
    oos_sel = oos_sr[rows, n_star]

    # Относительный ранг OOS-производительности отобранного среди всех N:
    # доля конфигураций с OOS-Sharpe <= Sharpe(n*), нормированная в (0, 1).
    rank = (oos_sr <= oos_sel[:, None]).sum(axis=1).astype(np.float64)  # 1..N
    omega = np.clip(rank / (n + 1.0), 1e-6, 1.0 - 1e-6)
    logits = np.log(omega / (1.0 - omega))

    pbo = float((logits <= 0.0).mean())
    prob_oos_loss = float((oos_sel <= 0.0).mean())
    slope = _linreg_slope(is_sel, oos_sel)

    return PBOResult(
        pbo=pbo,
        n_splits=n_splits,
        n_combinations=n_combos,
        n_trials=n,
        logits=logits,
        oos_rank_selected=omega,
        is_perf_selected=is_sel,
        oos_perf_selected=oos_sel,
        prob_oos_loss=prob_oos_loss,
        degradation_slope=slope,
    )


def pbo_from_optimization(
    data: pd.DataFrame,
    strategy_cls: Type[BaseStrategy],
    history: Sequence[tuple[dict[str, Any], float]],
    *,
    n_splits: int = 16,
    fixed_params: Optional[dict[str, Any]] = None,
    engine_kwargs: Optional[dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> PBOResult:
    """PBO по результату оптимизации (готовый к отчёту).

    Берёт уникальные конфигурации из OptimizationResult.history, прогоняет
    каждую (build_trial_returns) и оценивает вероятность переподгодки CSCV.
    Дорогой шаг -- N полных прогонов; запускать для финальной диагностики
    оптимизации, не внутри неё.

    Учитываются только испытания с КОНЕЧНЫМ score: оптимизатор помечает
    значением -inf прогоны, которые упали (недопустимые параметры) или не
    прошли фильтр мин. сделок -- такие не были кандидатами и в CSCV не нужны.
    Это и отсекает заведомо плохие комбинации до повторного прогона.
    """
    valid_history = [(p, s) for p, s in history if math.isfinite(s)]
    configs = unique_configs(valid_history or history)
    if len(configs) < 2:
        raise ValueError(
            "в истории меньше 2 различных валидных конфигураций -- CSCV "
            "неприменим (смягчите фильтр мин. сделок или расширьте сетку)"
        )
    matrix = build_trial_returns(
        data,
        strategy_cls,
        configs,
        fixed_params=fixed_params,
        engine_kwargs=engine_kwargs,
        progress_callback=progress_callback,
    )
    if matrix.shape[1] < 2:
        raise ValueError(
            "после прогонов осталось меньше 2 рабочих конфигураций -- "
            "слишком много комбинаций параметров недопустимы"
        )
    return probability_of_backtest_overfitting(matrix, n_splits=n_splits)
