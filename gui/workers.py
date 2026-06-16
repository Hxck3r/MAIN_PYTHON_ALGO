"""Фоновые потоки для бэктеста и оптимизации.

Расчёты выполняются в QThread, чтобы окно не «замерзало».
Результаты возвращаются в GUI-поток через сигналы Qt.
"""
from __future__ import annotations

import math
from typing import Any, Type

import numpy as np
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
    deflated_sharpe_ratio,
    haircut_sharpe_ratio,
    make_backtest_objective,
    min_track_record_length,
    pbo_from_optimization,
    run_monte_carlo,
    run_walk_forward,
    run_wfa_matrix,
)
from backtester.backtest_engine import run_backtest
from backtester.metrics import _estimate_periods_per_year
from backtester.parallel import ParallelEvaluator
from backtester.walk_forward import grid_to_space


def _optimization_overfit_stats(
    data: pd.DataFrame,
    strategy_cls: Type[BaseStrategy],
    best_params: dict[str, Any],
    n_trials: int,
    fixed_params: dict[str, Any],
    engine_kwargs: dict[str, float],
):
    """DSR + MinTRL + Haircut Sharpe лучшего набора по одному прогону.

    Все три метрики переоптимизации считаются из доходностей ОДНОГО прогона
    best_params, чтобы не гонять бэктест трижды. DSR совпадает с прежним
    deflated_sharpe_from_optimization (тот же путь расчёта).
    """
    result = run_backtest(
        data=data,
        strategy_cls=strategy_cls,
        strategy_params={**fixed_params, **best_params},
        **engine_kwargs,
    )
    returns = result.equity_curve.pct_change().dropna().to_numpy(dtype=np.float64)
    n = max(int(n_trials), 1)
    dsr = deflated_sharpe_ratio(returns, np.empty(0, dtype=np.float64), n_trials=n)
    mtrl = min_track_record_length(returns)
    ppy = _estimate_periods_per_year(result.equity_curve.index)
    hc = haircut_sharpe_ratio(returns, n_trials=n, periods_per_year=ppy)
    return dsr, mtrl, hc


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

        # Метрики переоптимизации лучшего результата (в фоне, не блокируя GUI):
        # DSR (значим ли Sharpe с учётом перебора), MinTRL (хватает ли истории),
        # Haircut Sharpe (сколько «срезается» поправкой на число испытаний).
        if result.best_params and math.isfinite(result.best_score):
            try:
                dsr, mtrl, hc = _optimization_overfit_stats(
                    self.data, self.strategy_cls, result.best_params,
                    len(result.history), self.fixed_params, self.engine_kwargs,
                )
                result.deflated_sharpe = dsr
                result.min_trl = mtrl
                result.haircut = hc
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


class PboWorker(QThread):
    """PBO (вероятность переподгонки) по истории оптимизации в фоне.

    Прогоняет уникальные конфигурации оптимизации (build_trial_returns) и
    оценивает переподгонку методом CSCV. Прогресс -- по числу прогнанных
    конфигураций; остановка -- через requestInterruption().
    """

    finished_ok = Signal(object)   # PBOResult
    failed = Signal(str)
    stopped = Signal()             # расчёт прерван пользователем
    progress = Signal(int, int)    # (конфигов прогнано, всего)

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_cls: Type[BaseStrategy],
        history: list,                  # OptimizationResult.history
        n_splits: int,
        fixed_params: dict[str, Any],
        engine_kwargs: dict[str, float],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.data = data
        self.strategy_cls = strategy_cls
        self.history = history
        self.n_splits = n_splits
        self.fixed_params = fixed_params
        self.engine_kwargs = engine_kwargs

    def run(self) -> None:
        def progress(done: int, total: int) -> None:
            if self.isInterruptionRequested():
                raise StopRequested()
            self.progress.emit(done, total)

        try:
            result = pbo_from_optimization(
                self.data,
                self.strategy_cls,
                self.history,
                n_splits=self.n_splits,
                fixed_params=self.fixed_params,
                engine_kwargs=self.engine_kwargs,
                progress_callback=progress,
            )
        except StopRequested:
            self.stopped.emit()
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(result)


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


# --------------------------------------------------------------------------- #
# Машинное обучение
# --------------------------------------------------------------------------- #
def _feature_config_from_settings(settings: dict[str, Any]):
    """Собрать ``ml.features.FeatureConfig`` из словаря настроек панели.

    Пресет управляет «богатством» признаков — для слабого ПК «Лёгкий» считает
    минимальный набор (быстрее обучение, меньше шанс переобучения на шуме).
    """
    from ml.features.engineering import FeatureConfig

    preset = settings.get("preset", "standard")
    if preset == "light":
        base = dict(
            return_periods=[1, 3, 5], sma_windows=[10, 20], ema_windows=[12],
            use_macd=False, use_bollinger=True, use_atr=True, use_volatility=True,
            use_momentum=True, use_stochastic=False, use_volume=True,
        )
    elif preset == "extended":
        base = dict(
            return_periods=[1, 2, 3, 5, 10, 20], sma_windows=[10, 20, 50, 100],
            ema_windows=[12, 26, 50], use_macd=True, use_bollinger=True,
            use_atr=True, use_volatility=True, use_momentum=True,
            use_stochastic=True, use_volume=True,
        )
    else:  # standard — дефолты FeatureConfig
        base = {}

    # Отбор признаков по умолчанию включаем на «Лёгком» пресете (слабый ПК).
    feature_select = bool(settings.get("feature_select", preset == "light"))

    return FeatureConfig(
        target_mode=settings.get("target_mode", "direction"),
        target_horizon=int(settings.get("target_horizon", 1)),
        target_threshold=float(settings.get("target_threshold", 0.0)),
        use_calendar=bool(settings.get("use_calendar", True)),
        use_regime=bool(settings.get("use_regime", True)),
        feature_select=feature_select,
        **base,
    )


def _train_config_from_settings(settings: dict[str, Any]):
    """Собрать ``ml.trainer.TrainConfig`` из словаря настроек панели."""
    from ml.trainer import TrainConfig

    return TrainConfig(
        model_name=settings.get("model_name", "random_forest"),
        validation=settings.get("validation", "walk_forward"),
        n_splits=int(settings.get("n_splits", 5)),
        embargo=int(settings.get("embargo", 0)),
        balance_classes=bool(settings.get("balance_classes", True)),
        calibrate=bool(settings.get("calibrate", False)),
        use_sample_weights=bool(settings.get("sample_weights", True)),
        compute_mda=bool(settings.get("compute_mda", False)),
    )


class MlTrainingWorker(QThread):
    """Обучение ML-модели в фоне: признаки → purged walk-forward → артефакт.

    Прогресс приходит из ``ml.trainer.train`` колбэком ``(pct, msg)``; кооперативная
    остановка — через ``requestInterruption()`` (колбэк вернёт False → TrainingCancelled).
    Результат — кортеж ``(ModelResult, RunBundle)`` для вкладки «Машинное обучение».
    """

    finished_ok = Signal(object)   # (ModelResult, RunBundle)
    failed = Signal(str)
    stopped = Signal()             # обучение прервано пользователем
    progress = Signal(int, int)    # (процент, 100)
    status = Signal(str)           # текстовый статус этапа

    def __init__(
        self,
        data: pd.DataFrame,
        settings: dict[str, Any],
        source_label: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.data = data
        self.settings = settings
        self.source_label = source_label

    def run(self) -> None:
        try:
            from ml.cache import cached_feature_set
            from ml.experiment import build_bundle
            from ml.trainer import TrainingCancelled, train

            def progress(pct: int, msg: str):
                if self.isInterruptionRequested():
                    return False            # ml.trainer поднимет TrainingCancelled
                self.progress.emit(int(pct), 100)
                self.status.emit(msg)
                return True

            self.status.emit("Расчёт признаков…")
            fcfg = _feature_config_from_settings(self.settings)
            # Кеш матрицы признаков: повторное обучение с теми же признаками не
            # пересчитывает индикаторы (важно на слабом ПК).
            fs = cached_feature_set(self.data, fcfg)

            tcfg = _train_config_from_settings(self.settings)
            # Опциональный подбор гиперпараметров на той же purged-CV.
            if self.settings.get("tune"):
                from dataclasses import replace

                from ml.tuning import tune as tune_hp

                best, _, trial_sharpes = tune_hp(
                    fs, tcfg, int(self.settings.get("tune_trials", 15)), progress=progress
                )
                # Sharpe проб → дефляция (DSR) финальной модели за множественное тестирование.
                tcfg = replace(tcfg, tune_trial_sharpes=trial_sharpes)
                if best:
                    tcfg = replace(tcfg, params=best)

            mr = train(fs, tcfg, progress=progress)

            bundle = build_bundle(mr, fs, self.data, source_path=self.source_label or None)
        except TrainingCancelled:
            self.stopped.emit()
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit((mr, bundle))
