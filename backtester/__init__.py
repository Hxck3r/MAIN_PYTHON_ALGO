"""
Модульный фреймворк для бэктестинга и оптимизации торговых стратегий.

Структура пакета:
    data_loader      -- загрузка, очистка и валидация исторических данных (CSV)
    base_strategy    -- абстрактный базовый класс стратегии (init / next)
    backtest_engine  -- событийное ядро симуляции «бар за баром»
    metrics          -- расчёт торговых метрик по эквити и сделкам
    monte_carlo      -- Монте-Карло анализ сделок (перемешивание/бутстрэп)
    optimizer        -- Grid Search и генетический алгоритм оптимизации
    walk_forward     -- Walk-Forward Analysis (скользящие окна IS/OOS с purge)
    wfa_matrix       -- матричный WFO (перебор длин окон, поиск плато)
    statistics       -- защита от переоптимизации (DSR, MinTRL, haircut, плато)
    pbo              -- вероятность переподгонки (PBO) методом CSCV
    visualizer       -- интерактивная визуализация (Plotly)
"""
from .base_strategy import Bar, BaseStrategy
from .backtest_engine import (
    BacktestEngine,
    BacktestResult,
    Broker,
    ExitReason,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Trade,
)
from .data_loader import DataLoader, DataValidationReport, generate_synthetic_data
from .metrics import PerformanceMetrics, compute_metrics
from .monte_carlo import MonteCarloConfig, MonteCarloResult, run_monte_carlo
from .optimizer import (
    ChoiceParam,
    FloatParam,
    GeneticOptimizer,
    GridSearchOptimizer,
    IntParam,
    OptimizationResult,
    StopRequested,
    make_backtest_objective,
)
from .parallel import ParallelEvaluator, default_jobs
from .statistics import (
    DeflatedSharpeResult,
    HaircutSharpeResult,
    MinTRLResult,
    PermutationTestResult,
    deflated_sharpe_from_optimization,
    deflated_sharpe_ratio,
    effective_number_of_trials,
    expected_max_sharpe,
    haircut_sharpe_ratio,
    min_track_record_length,
    permutation_test,
    permute_bars,
    probabilistic_sharpe_ratio,
    select_plateau_params,
)
from .pbo import (
    PBOResult,
    build_trial_returns,
    pbo_from_optimization,
    probability_of_backtest_overfitting,
    unique_configs,
)
from .walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WindowResult,
    run_walk_forward,
    window_bounds,
)
from .wfa_matrix import (
    MATRIX_FIELDS,
    WfaMatrixCell,
    WfaMatrixConfig,
    WfaMatrixResult,
    WfaMatrixStability,
    run_wfa_matrix,
    span_values,
)

__all__ = [
    "MATRIX_FIELDS",
    "BacktestEngine",
    "BacktestResult",
    "Bar",
    "BaseStrategy",
    "Broker",
    "ChoiceParam",
    "DataLoader",
    "DataValidationReport",
    "DeflatedSharpeResult",
    "ExitReason",
    "FloatParam",
    "GeneticOptimizer",
    "GridSearchOptimizer",
    "HaircutSharpeResult",
    "IntParam",
    "MinTRLResult",
    "MonteCarloConfig",
    "MonteCarloResult",
    "OptimizationResult",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PBOResult",
    "ParallelEvaluator",
    "PerformanceMetrics",
    "PermutationTestResult",
    "Position",
    "StopRequested",
    "Trade",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WfaMatrixCell",
    "WfaMatrixConfig",
    "WfaMatrixResult",
    "WfaMatrixStability",
    "WindowResult",
    "build_trial_returns",
    "compute_metrics",
    "default_jobs",
    "deflated_sharpe_from_optimization",
    "deflated_sharpe_ratio",
    "effective_number_of_trials",
    "expected_max_sharpe",
    "generate_synthetic_data",
    "haircut_sharpe_ratio",
    "make_backtest_objective",
    "min_track_record_length",
    "pbo_from_optimization",
    "permutation_test",
    "permute_bars",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "run_monte_carlo",
    "run_walk_forward",
    "run_wfa_matrix",
    "select_plateau_params",
    "span_values",
    "unique_configs",
    "window_bounds",
]

__version__ = "0.1.0"
