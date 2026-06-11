"""Фоновые потоки для бэктеста и оптимизации.

Расчёты выполняются в QThread, чтобы окно не «замерзало».
Результаты возвращаются в GUI-поток через сигналы Qt.
"""
from __future__ import annotations

from typing import Any, Type

import pandas as pd
from PySide6.QtCore import QThread, Signal

from backtester import (
    BacktestEngine,
    BaseStrategy,
    GeneticOptimizer,
    GridSearchOptimizer,
    WalkForwardConfig,
    make_backtest_objective,
    run_walk_forward,
)
from backtester.parallel import ParallelEvaluator
from backtester.walk_forward import grid_to_space


class BacktestWorker(QThread):
    """Один прогон бэктеста в фоне."""

    finished_ok = Signal(object)   # BacktestResult
    failed = Signal(str)

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_cls: Type[BaseStrategy],
        strategy_params: dict[str, Any],
        engine_kwargs: dict[str, float],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.data = data
        self.strategy_cls = strategy_cls
        self.strategy_params = strategy_params
        self.engine_kwargs = engine_kwargs

    def run(self) -> None:
        try:
            result = BacktestEngine(
                data=self.data,
                strategy_cls=self.strategy_cls,
                strategy_params=self.strategy_params,
                **self.engine_kwargs,
            ).run()
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(result)


class OptimizationWorker(QThread):
    """Оптимизация (Grid Search или генетика) в фоне с прогрессом."""

    finished_ok = Signal(object)   # OptimizationResult
    failed = Signal(str)
    progress = Signal(int, int)    # (сделано, всего)

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_cls: Type[BaseStrategy],
        metric: str,
        fixed_params: dict[str, Any],
        grid: dict[str, list],          # имя -> список значений (старт..стоп)
        engine_kwargs: dict[str, float],
        min_trades: int,
        use_genetic: bool,
        population_size: int = 30,
        generations: int = 15,
        n_jobs: int = 1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.data = data
        self.strategy_cls = strategy_cls
        self.metric = metric
        self.fixed_params = fixed_params
        self.grid = grid
        self.engine_kwargs = engine_kwargs
        self.min_trades = min_trades
        self.use_genetic = use_genetic
        self.population_size = population_size
        self.generations = generations
        self.n_jobs = n_jobs

    def run(self) -> None:
        evaluator = None
        try:
            objective = make_backtest_objective(
                data=self.data,
                strategy_cls=self.strategy_cls,
                metric=self.metric,
                fixed_params=self.fixed_params,
                engine_kwargs=self.engine_kwargs,
                min_trades=self.min_trades,
            )
            if self.n_jobs != 1:
                evaluator = ParallelEvaluator(
                    data=self.data,
                    strategy_cls=self.strategy_cls,
                    metric=self.metric,
                    fixed_params=self.fixed_params,
                    engine_kwargs=self.engine_kwargs,
                    min_trades=self.min_trades,
                    n_jobs=self.n_jobs,
                )
            if self.use_genetic:
                optimizer = GeneticOptimizer(
                    objective,
                    param_space=grid_to_space(self.grid),
                    population_size=self.population_size,
                    generations=self.generations,
                    seed=42,
                    progress_callback=self.progress.emit,
                    batch_evaluator=evaluator,
                )
            else:
                optimizer = GridSearchOptimizer(
                    objective,
                    param_grid=self.grid,
                    progress_callback=self.progress.emit,
                    batch_evaluator=evaluator,
                )
            result = optimizer.optimize()
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        finally:
            if evaluator is not None:
                evaluator.close()
        self.finished_ok.emit(result)


class WalkForwardWorker(QThread):
    """Walk-Forward Analysis в фоне: окна, прогресс и статус через сигналы."""

    finished_ok = Signal(object)   # WalkForwardResult
    failed = Signal(str)
    progress = Signal(int, int)    # (сделано, всего) по всем окнам сквозно
    status = Signal(str)           # "Окно 3/10: оптимизация IS ..."

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_cls: Type[BaseStrategy],
        config: WalkForwardConfig,
        metric: str,
        fixed_params: dict[str, Any],
        grid: dict[str, list],
        engine_kwargs: dict[str, float],
        min_trades: int,
        use_genetic: bool,
        population_size: int = 30,
        generations: int = 15,
        n_jobs: int = 1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.data = data
        self.strategy_cls = strategy_cls
        self.config = config
        self.metric = metric
        self.fixed_params = fixed_params
        self.grid = grid
        self.engine_kwargs = engine_kwargs
        self.min_trades = min_trades
        self.use_genetic = use_genetic
        self.population_size = population_size
        self.generations = generations
        self.n_jobs = n_jobs

    def run(self) -> None:
        try:
            result = run_walk_forward(
                data=self.data,
                strategy_cls=self.strategy_cls,
                config=self.config,
                grid=self.grid,
                metric=self.metric,
                fixed_params=self.fixed_params,
                engine_kwargs=self.engine_kwargs,
                min_trades=self.min_trades,
                use_genetic=self.use_genetic,
                population_size=self.population_size,
                generations=self.generations,
                n_jobs=self.n_jobs,
                progress_callback=self.progress.emit,
                status_callback=self.status.emit,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(result)
