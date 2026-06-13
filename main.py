"""
Демонстрация полного цикла работы фреймворка:
    данные -> бэктест -> метрики -> оптимизация -> визуализация.

Запуск:
    python main.py                 # на синтетических данных
    python main.py path/to/data.csv  # на своих данных (в т.ч. экспорт MT5)

Интерактивные HTML-отчёты сохраняются в папку reports/.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

# Консоль Windows по умолчанию не в UTF-8 -- иначе кириллица падает с ошибкой
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from backtester import (
    BacktestEngine,
    DataLoader,
    GeneticOptimizer,
    GridSearchOptimizer,
    IntParam,
    deflated_sharpe_from_optimization,
    generate_synthetic_data,
    make_backtest_objective,
)
from backtester.visualizer import plot_backtest, plot_optimization, plot_param_heatmap
from strategies import SmaCrossStrategy

REPORTS_DIR = Path(__file__).parent / "reports"


def _attach_dsr(result, data: pd.DataFrame) -> None:
    """Считает Deflated Sharpe лучшего результата и кладёт его в result.

    n_trials -- число реально оценённых комбинаций (len(history)). После
    этого result.__str__ автоматически печатает вердикт о значимости.
    """
    if result.best_params and math.isfinite(result.best_score):
        result.deflated_sharpe = deflated_sharpe_from_optimization(
            data, SmaCrossStrategy, result.best_params,
            n_trials=len(result.history), engine_kwargs=ENGINE_KWARGS,
        )

# Общие настройки счёта (применяются и в бэктесте, и в оптимизации)
ENGINE_KWARGS = dict(
    cash=10_000.0,
    commission_pct=0.0005,   # 0.05% за сторону
    slippage_pct=0.0001,     # 0.01% проскальзывание
)


def load_data() -> pd.DataFrame:
    """CSV из аргумента командной строки или синтетика для демонстрации."""
    if len(sys.argv) > 1:
        loader = DataLoader()
        df = loader.load_csv(sys.argv[1])
        print(loader.last_report, "\n")
        return df
    print("CSV не указан -- использую синтетические данные (2000 баров H1)\n")
    return generate_synthetic_data(n_bars=2000, freq="1h", seed=7)


def run_single_backtest(data: pd.DataFrame) -> None:
    print("=" * 60)
    print("1. ОДИНОЧНЫЙ БЭКТЕСТ")
    print("=" * 60)
    engine = BacktestEngine(
        data=data,
        strategy_cls=SmaCrossStrategy,
        strategy_params={"fast_period": 10, "slow_period": 30},
        **ENGINE_KWARGS,
    )
    result = engine.run()
    print(result.stats(), "\n")

    plot_backtest(
        result, show=False, save_path=REPORTS_DIR / "backtest_report.html"
    )
    print(f"Отчёт сохранён: {REPORTS_DIR / 'backtest_report.html'}\n")


def run_grid_search(data: pd.DataFrame) -> None:
    print("=" * 60)
    print("2. ОПТИМИЗАЦИЯ: GRID SEARCH")
    print("=" * 60)
    objective = make_backtest_objective(
        data=data,
        strategy_cls=SmaCrossStrategy,
        metric="sharpe_ratio",
        engine_kwargs=ENGINE_KWARGS,
    )
    grid = GridSearchOptimizer(
        objective,
        param_grid={
            "fast_period": range(5, 30, 5),
            "slow_period": range(30, 110, 20),
        },
        verbose=True,
    )
    result = grid.optimize()
    _attach_dsr(result, data)
    print(result, "\n")

    plot_optimization(
        result, show=False, save_path=REPORTS_DIR / "grid_scores_hist.html"
    )
    plot_param_heatmap(
        result, x_param="fast_period", y_param="slow_period",
        show=False, save_path=REPORTS_DIR / "grid_heatmap.html",
    )
    print(f"Отчёты сохранены: {REPORTS_DIR / 'grid_scores_hist.html'}, "
          f"{REPORTS_DIR / 'grid_heatmap.html'}\n")


def run_genetic(data: pd.DataFrame) -> None:
    print("=" * 60)
    print("3. ОПТИМИЗАЦИЯ: ГЕНЕТИЧЕСКИЙ АЛГОРИТМ")
    print("=" * 60)
    objective = make_backtest_objective(
        data=data,
        strategy_cls=SmaCrossStrategy,
        metric="sharpe_ratio",
        engine_kwargs=ENGINE_KWARGS,
    )
    genetic = GeneticOptimizer(
        objective,
        param_space={
            "fast_period": IntParam(3, 40),
            "slow_period": IntParam(41, 150),
        },
        population_size=20,
        generations=10,
        seed=42,
        verbose=True,
    )
    result = genetic.optimize()
    _attach_dsr(result, data)
    print(result, "\n")

    plot_optimization(
        result, show=False, save_path=REPORTS_DIR / "genetic_scores_hist.html"
    )
    print(f"Отчёт сохранён: {REPORTS_DIR / 'genetic_scores_hist.html'}\n")


def main() -> None:
    data = load_data()
    run_single_backtest(data)
    run_grid_search(data)
    run_genetic(data)
    print("Готово. Откройте HTML-файлы из папки reports/ в браузере.")


if __name__ == "__main__":
    main()
