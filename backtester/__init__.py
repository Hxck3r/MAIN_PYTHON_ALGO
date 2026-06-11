"""
Модульный фреймворк для бэктестинга и оптимизации торговых стратегий.

Структура пакета:
    data_loader      -- загрузка, очистка и валидация исторических данных (CSV)
    base_strategy    -- абстрактный базовый класс стратегии (init / next)
    backtest_engine  -- событийное ядро симуляции «бар за баром»
    metrics          -- расчёт торговых метрик по эквити и сделкам
    monte_carlo      -- Монте-Карло анализ сделок (перемешивание/бутстрэп)
    optimizer        -- Grid Search и генетический алгоритм оптимизации
    walk_forward     -- Walk-Forward Analysis (скользящие окна IS/OOS)
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
    make_backtest_objective,
)
from .parallel import ParallelEvaluator, default_jobs
from .walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WindowResult,
    run_walk_forward,
    window_bounds,
)

__all__ = [
    "Bar",
    "BaseStrategy",
    "BacktestEngine",
    "BacktestResult",
    "Broker",
    "ExitReason",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "Trade",
    "DataLoader",
    "DataValidationReport",
    "generate_synthetic_data",
    "PerformanceMetrics",
    "compute_metrics",
    "MonteCarloConfig",
    "MonteCarloResult",
    "run_monte_carlo",
    "ChoiceParam",
    "FloatParam",
    "GeneticOptimizer",
    "GridSearchOptimizer",
    "IntParam",
    "OptimizationResult",
    "make_backtest_objective",
    "ParallelEvaluator",
    "default_jobs",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WindowResult",
    "run_walk_forward",
    "window_bounds",
]

__version__ = "0.1.0"
