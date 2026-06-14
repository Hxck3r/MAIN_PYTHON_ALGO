"""Фоновые потоки для бэктеста и оптимизации.

Расчёты выполняются в QThread, чтобы окно не «замерзало».
Результаты возвращаются в GUI-поток через сигналы Qt.
"""
from __future__ import annotations

import math
from typing import Any, Type

import pandas as pd
from PySide6.QtCore import QThread, Signal

from backtester import (
    BacktestEngine,
    BacktestResult,
    BaseStrategy,
    GeneticOptimizer,
    GridSearchOptimizer,
    MonteCarloConfig,
    StopRequested,
    WalkForwardConfig,
    WfaMatrixConfig,
    deflated_sharpe_from_optimization,
    make_backtest_objective,
    run_monte_carlo,
    run_walk_forward,
    run_wfa_matrix,
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
    """Оптимизация (Grid Search или генетика) в фоне с прогрессом.

    Остановка: requestInterruption() из GUI-потока -> оптимизатор замечает
    её в точке проверки и поднимает StopRequested -> сигнал stopped.
    """

    finished_ok = Signal(object)   # OptimizationResult
    failed = Signal(str)
    stopped = Signal()             # расчёт прерван пользователем
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
                    should_stop=self.isInterruptionRequested,
                )
            else:
                optimizer = GridSearchOptimizer(
                    objective,
                    param_grid=self.grid,
                    progress_callback=self.progress.emit,
                    batch_evaluator=evaluator,
                    should_stop=self.isInterruptionRequested,
                )
            result = optimizer.optimize()
        except StopRequested:
            self.stopped.emit()
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        finally:
            if evaluator is not None:
                evaluator.close()

        # Deflated Sharpe лучшего результата (в фоне, не блокируя GUI): значим
        # ли найденный Sharpe с поправкой на число перебранных комбинаций.
        if result.best_params and math.isfinite(result.best_score):
            try:
                result.deflated_sharpe = deflated_sharpe_from_optimization(
                    self.data, self.strategy_cls, result.best_params,
                    n_trials=len(result.history),
                    fixed_params=self.fixed_params, engine_kwargs=self.engine_kwargs,
                )
            except Exception:
                result.deflated_sharpe = None
        self.finished_ok.emit(result)


class MonteCarloWorker(QThread):
    """Монте-Карло анализ сделок бэктеста в фоне с прогрессом."""

    finished_ok = Signal(object)   # MonteCarloResult
    failed = Signal(str)
    stopped = Signal()             # расчёт прерван пользователем
    progress = Signal(int, int)    # (симуляций сделано, всего)

    def __init__(
        self,
        result: BacktestResult,
        config: MonteCarloConfig,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.result = result
        self.config = config

    def run(self) -> None:
        try:
            mc = run_monte_carlo(
                self.result, self.config, progress_callback=self.progress.emit,
                should_stop=self.isInterruptionRequested,
            )
        except StopRequested:
            self.stopped.emit()
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(mc)


class WalkForwardWorker(QThread):
    """Walk-Forward Analysis в фоне: окна, прогресс и статус через сигналы."""

    finished_ok = Signal(object)   # WalkForwardResult
    failed = Signal(str)
    stopped = Signal()             # расчёт прерван пользователем
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
                should_stop=self.isInterruptionRequested,
            )
        except StopRequested:
            self.stopped.emit()
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(result)


class WfaMatrixWorker(QThread):
    """Матричный WFO в фоне: перебор длин окон IS/OOS, прогресс и статус."""

    finished_ok = Signal(object)   # WfaMatrixResult
    failed = Signal(str)
    stopped = Signal()             # расчёт прерван пользователем
    progress = Signal(int, int)    # (сделано, всего) по всем бэктестам матрицы
    status = Signal(str)           # "Ячейка 3/9: IS=180д, OOS=45д ..."

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_cls: Type[BaseStrategy],
        matrix_config: WfaMatrixConfig,
        metric: str,
        target: str,
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
        self.matrix_config = matrix_config
        self.metric = metric
        self.target = target
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
            result = run_wfa_matrix(
                data=self.data,
                strategy_cls=self.strategy_cls,
                matrix_config=self.matrix_config,
                grid=self.grid,
                metric=self.metric,
                target=self.target,
                fixed_params=self.fixed_params,
                engine_kwargs=self.engine_kwargs,
                min_trades=self.min_trades,
                use_genetic=self.use_genetic,
                population_size=self.population_size,
                generations=self.generations,
                n_jobs=self.n_jobs,
                progress_callback=self.progress.emit,
                status_callback=self.status.emit,
                should_stop=self.isInterruptionRequested,
            )
        except StopRequested:
            self.stopped.emit()
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(result)
