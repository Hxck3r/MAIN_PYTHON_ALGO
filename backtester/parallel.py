"""
Параллельная оценка параметров стратегии (пул процессов).

Бэктест -- чистый Python-цикл по барам, поэтому потоки не помогают (GIL):
ускорение даёт только пул процессов. Узкое место multiprocessing --
пересылка данных, поэтому устроено так:

    * DataFrame с историей передаётся каждому воркеру ОДИН раз при старте
      пула (initializer), а не с каждым испытанием;
    * задача -- лёгкий кортеж (params, границы среза); воркер сам режет
      данные по границам и кэширует срез до смены границ (для WFA, где
      все испытания одного окна используют один и тот же IS-срез);
    * целевая функция строится внутри воркера через make_backtest_objective.

ParallelEvaluator подключается к оптимизаторам через параметр
batch_evaluator: GridSearchOptimizer считает всю сетку через пул,
GeneticOptimizer -- каждое поколение пачкой. Ошибки прогона возвращаются
как NaN и превращаются оптимизатором в штраф (худший результат).
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterator, Optional, Type

import pandas as pd

from .base_strategy import BaseStrategy
from .optimizer import make_backtest_objective

#: контекст воркера (своя копия в каждом процессе пула)
_CTX: dict[str, Any] = {}


def default_jobs() -> int:
    """Число процессов по умолчанию: все ядра минус одно (для GUI)."""
    return max((os.cpu_count() or 2) - 1, 1)


def _init_worker(
    data: pd.DataFrame,
    strategy_cls: Type[BaseStrategy],
    metric: str,
    fixed_params: dict[str, Any],
    engine_kwargs: dict[str, Any],
    min_trades: int,
) -> None:
    """Инициализация процесса пула: данные и настройки сохраняются локально."""
    _CTX.update(
        data=data,
        strategy_cls=strategy_cls,
        metric=metric,
        fixed_params=fixed_params,
        engine_kwargs=engine_kwargs,
        min_trades=min_trades,
        bounds=None,
        objective=None,
    )


def _objective_for(bounds: Optional[tuple]) -> Any:
    """Целевая функция для текущих границ среза (кэшируется до их смены)."""
    if _CTX.get("objective") is None or _CTX.get("bounds") != bounds:
        data = _CTX["data"]
        if bounds is not None:
            start, end = bounds
            data = data[(data.index >= start) & (data.index < end)]
        _CTX["objective"] = make_backtest_objective(
            data=data,
            strategy_cls=_CTX["strategy_cls"],
            metric=_CTX["metric"],
            fixed_params=_CTX["fixed_params"],
            engine_kwargs=_CTX["engine_kwargs"],
            min_trades=_CTX["min_trades"],
        )
        _CTX["bounds"] = bounds
    return _CTX["objective"]


def _eval_task(task: tuple[dict[str, Any], Optional[tuple]]) -> float:
    """Оценка одного набора параметров; ошибки -> NaN (штраф на стороне
    оптимизатора). -inf от фильтра min_trades проходит как есть."""
    params, bounds = task
    try:
        return float(_objective_for(bounds)(params))
    except Exception:
        return math.nan


class ParallelEvaluator:
    """Пул процессов для оценки params -> score.

    Параметры
    ---------
    n_jobs: число процессов; 0 -> авто (ядра - 1); 1 не имеет смысла --
            в этом случае оптимизаторы быстрее работают без пула вовсе.
    Остальные -- как у make_backtest_objective.

    Использование (контекстный менеджер закрывает пул)::

        with ParallelEvaluator(data, MyStrategy, "sharpe_ratio") as ev:
            opt = GridSearchOptimizer(obj, grid, batch_evaluator=ev)
            result = opt.optimize()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_cls: Type[BaseStrategy],
        metric: str,
        fixed_params: Optional[dict[str, Any]] = None,
        engine_kwargs: Optional[dict[str, Any]] = None,
        min_trades: int = 5,
        n_jobs: int = 0,
    ) -> None:
        self.n_jobs = n_jobs if n_jobs > 0 else default_jobs()
        self._bounds: Optional[tuple] = None
        self._executor = ProcessPoolExecutor(
            max_workers=self.n_jobs,
            initializer=_init_worker,
            initargs=(
                data, strategy_cls, metric,
                dict(fixed_params or {}), dict(engine_kwargs or {}),
                min_trades,
            ),
        )

    def set_bounds(
        self, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]
    ) -> None:
        """Ограничивает оценку срезом данных [start, end) -- для окон WFA."""
        self._bounds = (start, end) if start is not None else None

    def evaluate_iter(self, params_list: list[dict[str, Any]]) -> Iterator[float]:
        """Ленивая оценка списка параметров; порядок результатов сохраняется.

        Удобно для прогресс-бара: результаты приходят по мере готовности.
        """
        if not params_list:
            return iter(())
        tasks = [(params, self._bounds) for params in params_list]
        # мелкие чанки -> плавный прогресс, крупные -> меньше накладных
        chunksize = max(1, min(8, len(tasks) // (self.n_jobs * 4) or 1))
        return self._executor.map(_eval_task, tasks, chunksize=chunksize)

    def evaluate(self, params_list: list[dict[str, Any]]) -> list[float]:
        return list(self.evaluate_iter(params_list))

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> ParallelEvaluator:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
