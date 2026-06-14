"""
Оптимизация параметров стратегии.

Два метода:
    * GridSearchOptimizer -- полный перебор декартова произведения значений;
    * GeneticOptimizer    -- легковесный генетический алгоритм без внешних
      зависимостей (турнирная селекция, равномерный кроссовер, мутация,
      элитизм). При желании легко заменяется на Optuna: достаточно
      реализовать ещё один наследник BaseOptimizer.

Оптимизатор не знает ничего о бэктестере -- он максимизирует произвольную
целевую функцию params -> float. Связка с движком собирается фабрикой
make_backtest_objective(). Такая развязка позволит без изменений ядра
встроить Walk-Forward Analysis (оптимизация на in-sample окнах).
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence, Type

import pandas as pd

from .backtest_engine import run_backtest
from .base_strategy import BaseStrategy
from .metrics import compute_optimization_metric, metric_needs_trades

if TYPE_CHECKING:  # только для аннотаций -- без циклического импорта
    from .statistics import DeflatedSharpeResult

Objective = Callable[[dict[str, Any]], float]


# ---------------------------------------------------------------------- #
# Описание пространства параметров (для генетического алгоритма)
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class IntParam:
    """Целочисленный параметр в диапазоне [low, high] с шагом step."""

    low: int
    high: int
    step: int = 1

    def sample(self, rng: random.Random) -> int:
        n_steps = (self.high - self.low) // self.step
        return self.low + rng.randint(0, n_steps) * self.step

    def mutate(self, value: int, rng: random.Random) -> int:
        span = max((self.high - self.low) // self.step, 1)
        shift = round(rng.gauss(0, max(span * 0.15, 1.0))) * self.step
        return int(min(max(value + shift, self.low), self.high))


@dataclass(frozen=True)
class FloatParam:
    """Вещественный параметр в диапазоне [low, high]."""

    low: float
    high: float
    log: bool = False  # сэмплирование в логарифмической шкале

    def sample(self, rng: random.Random) -> float:
        if self.log:
            return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        return rng.uniform(self.low, self.high)

    def mutate(self, value: float, rng: random.Random) -> float:
        sigma = (self.high - self.low) * 0.15
        return min(max(value + rng.gauss(0, sigma), self.low), self.high)


@dataclass(frozen=True)
class ChoiceParam:
    """Категориальный параметр (выбор из списка значений)."""

    choices: Sequence[Any]

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(list(self.choices))

    def mutate(self, value: Any, rng: random.Random) -> Any:
        return self.sample(rng)


ParamSpec = IntParam | FloatParam | ChoiceParam


# ---------------------------------------------------------------------- #
# Результат оптимизации
# ---------------------------------------------------------------------- #
@dataclass
class OptimizationResult:
    """Итог работы оптимизатора."""

    best_params: dict[str, Any]
    best_score: float
    history: list[tuple[dict[str, Any], float]] = field(default_factory=list)
    method: str = ""
    # DSR лучшего результата с поправкой на размер перебора; заполняется
    # вызывающим кодом (deflated_sharpe_from_optimization) после оптимизации.
    deflated_sharpe: Optional["DeflatedSharpeResult"] = None

    def history_df(self) -> pd.DataFrame:
        """История всех испытаний: колонки параметров + score."""
        rows = [{**params, "score": score} for params, score in self.history]
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        lines = [
            f"=== Оптимизация ({self.method}) ===",
            f"Испытаний: {len(self.history)}",
            f"Лучший score: {self.best_score:.4f}",
            f"Лучшие параметры: {self.best_params}",
        ]
        if self.deflated_sharpe is not None:
            d = self.deflated_sharpe
            verdict = "значим" if d.dsr >= 0.95 else "вероятен артефакт перебора"
            lines.append(
                f"Deflated Sharpe: {d.dsr:.2f} ({verdict}; Sharpe {d.observed_sr:.3f} "
                f"vs порог шума {d.expected_max_sr:.3f} при {d.n_trials} испыт.)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Базовый оптимизатор
# ---------------------------------------------------------------------- #
#: коллбек прогресса: (сделано, всего) -- для прогресс-баров в GUI
ProgressCallback = Callable[[int, int], None]
#: коллбек прерывания: возвращает True, если расчёт нужно остановить
ShouldStop = Callable[[], bool]


class StopRequested(Exception):
    """Кооперативная остановка: пользователь прервал расчёт через should_stop.

    Поднимается оптимизаторами и циклами WFA/WFO в точках проверки
    should_stop(). Не ошибка -- вызывающий код (GUI-воркер) ловит её
    отдельно от Exception и сообщает «остановлено», а не «упало».
    """


class BaseOptimizer(ABC):
    """Общий каркас оптимизаторов: целевая функция + направление поиска.

    batch_evaluator -- опциональный параллельный оценщик (ParallelEvaluator):
    объект с методами evaluate(list[params]) -> list[score] и
    evaluate_iter(...). Если задан, оптимизатор считает испытания пачками
    через пул процессов вместо последовательных вызовов objective.
    """

    def __init__(
        self,
        objective: Objective,
        maximize: bool = True,
        verbose: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        batch_evaluator: Optional[Any] = None,
        should_stop: Optional[ShouldStop] = None,
    ) -> None:
        self.objective = objective
        self.maximize = maximize
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.batch_evaluator = batch_evaluator
        self.should_stop = should_stop

    def _report_progress(self, done: int, total: int) -> None:
        if self.progress_callback is not None:
            self.progress_callback(done, total)

    def _check_stop(self) -> None:
        """Поднимает StopRequested, если запрошена остановка."""
        if self.should_stop is not None and self.should_stop():
            raise StopRequested()

    @abstractmethod
    def optimize(self) -> OptimizationResult:
        """Запускает поиск и возвращает результат."""

    def _evaluate(self, params: dict[str, Any]) -> float:
        """Считает score; ошибки прогона трактуются как наихудший результат."""
        try:
            score = float(self.objective(params))
            if math.isnan(score):
                raise ValueError("objective вернула NaN")
        except Exception as exc:  # стратегия может падать на части параметров
            if self.verbose:
                print(f"  [!] {params}: ошибка ({exc}) -> штраф")
            return -math.inf if self.maximize else math.inf
        return score

    def _sanitize(self, score: Any) -> float:
        """NaN из параллельной оценки (ошибка в воркере) -> штраф."""
        score = float(score)
        if math.isnan(score):
            return -math.inf if self.maximize else math.inf
        return score

    def _is_better(self, score: float, best: float) -> bool:
        return score > best if self.maximize else score < best


# ---------------------------------------------------------------------- #
# Полный перебор
# ---------------------------------------------------------------------- #
class GridSearchOptimizer(BaseOptimizer):
    """Полный перебор по сетке параметров.

    param_grid -- словарь {имя_параметра: перечень значений}, например::

        {"fast": range(5, 30, 5), "slow": [30, 50, 100]}
    """

    def __init__(
        self,
        objective: Objective,
        param_grid: dict[str, Iterable[Any]],
        maximize: bool = True,
        verbose: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        batch_evaluator: Optional[Any] = None,
        should_stop: Optional[ShouldStop] = None,
    ) -> None:
        super().__init__(
            objective, maximize=maximize, verbose=verbose,
            progress_callback=progress_callback, batch_evaluator=batch_evaluator,
            should_stop=should_stop,
        )
        self.param_grid = {name: list(values) for name, values in param_grid.items()}
        if not self.param_grid:
            raise ValueError("Пустая сетка параметров")

    def optimize(self) -> OptimizationResult:
        names = list(self.param_grid)
        combos = list(product(*(self.param_grid[n] for n in names)))
        param_dicts = [dict(zip(names, combo)) for combo in combos]
        if self.verbose:
            print(f"Grid Search: {len(combos)} комбинаций")

        history: list[tuple[dict[str, Any], float]] = []
        best_score = -math.inf if self.maximize else math.inf
        best_params: dict[str, Any] = {}

        if self.batch_evaluator is not None:
            # параллельно: результаты приходят по мере готовности (по порядку)
            scores = self.batch_evaluator.evaluate_iter(param_dicts)
        else:
            scores = (self._evaluate(p) for p in param_dicts)

        for k, (params, raw) in enumerate(zip(param_dicts, scores), start=1):
            score = self._sanitize(raw)
            history.append((params, score))
            if self._is_better(score, best_score):
                best_score, best_params = score, params
                if self.verbose:
                    print(f"  [{k}/{len(combos)}] новый лучший {score:.4f}: {params}")
            self._report_progress(k, len(combos))
            self._check_stop()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=history,
            method="Grid Search",
        )


# ---------------------------------------------------------------------- #
# Генетический алгоритм
# ---------------------------------------------------------------------- #
class GeneticOptimizer(BaseOptimizer):
    """Легковесный генетический алгоритм.

    param_space -- словарь {имя: IntParam | FloatParam | ChoiceParam}::

        {
            "fast": IntParam(5, 50),
            "sl_pct": FloatParam(0.005, 0.05),
            "mode": ChoiceParam(["close", "hl2"]),
        }
    """

    def __init__(
        self,
        objective: Objective,
        param_space: dict[str, ParamSpec],
        population_size: int = 30,
        generations: int = 20,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.25,
        elite_count: int = 2,
        tournament_size: int = 3,
        maximize: bool = True,
        seed: Optional[int] = None,
        verbose: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        batch_evaluator: Optional[Any] = None,
        should_stop: Optional[ShouldStop] = None,
    ) -> None:
        super().__init__(
            objective, maximize=maximize, verbose=verbose,
            progress_callback=progress_callback, batch_evaluator=batch_evaluator,
            should_stop=should_stop,
        )
        if not param_space:
            raise ValueError("Пустое пространство параметров")
        if elite_count >= population_size:
            raise ValueError("elite_count должен быть меньше population_size")
        self.param_space = dict(param_space)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.rng = random.Random(seed)

    # --- генетические операторы ---------------------------------------- #
    def _random_individual(self) -> dict[str, Any]:
        return {name: spec.sample(self.rng) for name, spec in self.param_space.items()}

    def _crossover(
        self, parent_a: dict[str, Any], parent_b: dict[str, Any]
    ) -> dict[str, Any]:
        """Равномерный кроссовер: каждый ген наследуется от случайного родителя."""
        return {
            name: (parent_a if self.rng.random() < 0.5 else parent_b)[name]
            for name in self.param_space
        }

    def _mutate(self, individual: dict[str, Any]) -> dict[str, Any]:
        mutated = dict(individual)
        for name, spec in self.param_space.items():
            if self.rng.random() < self.mutation_rate:
                mutated[name] = spec.mutate(mutated[name], self.rng)
        return mutated

    def _tournament_select(
        self, scored: list[tuple[dict[str, Any], float]]
    ) -> dict[str, Any]:
        contenders = self.rng.sample(scored, min(self.tournament_size, len(scored)))
        best = max(contenders, key=lambda x: x[1]) if self.maximize else min(
            contenders, key=lambda x: x[1]
        )
        return best[0]

    # --- основной цикл --------------------------------------------------#
    def optimize(self) -> OptimizationResult:
        history: list[tuple[dict[str, Any], float]] = []
        cache: dict[tuple, float] = {}  # не пересчитываем одинаковые геномы

        def evaluate_generation(population: list[dict[str, Any]]) -> None:
            """Считает все новые геномы поколения (пачкой, если есть пул)."""
            fresh: list[tuple[tuple, dict[str, Any]]] = []
            seen: set[tuple] = set()
            for ind in population:
                key = tuple(sorted(ind.items()))
                if key not in cache and key not in seen:
                    seen.add(key)
                    fresh.append((key, dict(ind)))
            if not fresh:
                return
            if self.batch_evaluator is not None:
                scores = self.batch_evaluator.evaluate([ind for _, ind in fresh])
            else:
                scores = [self._evaluate(ind) for _, ind in fresh]
            for (key, ind), raw in zip(fresh, scores):
                cache[key] = self._sanitize(raw)
                history.append((ind, cache[key]))

        population = [self._random_individual() for _ in range(self.population_size)]

        best_params: dict[str, Any] = {}
        best_score = -math.inf if self.maximize else math.inf

        for gen in range(self.generations):
            self._check_stop()
            evaluate_generation(population)
            scored = [
                (ind, cache[tuple(sorted(ind.items()))]) for ind in population
            ]
            scored.sort(key=lambda x: x[1], reverse=self.maximize)

            if self._is_better(scored[0][1], best_score):
                best_params, best_score = dict(scored[0][0]), scored[0][1]
            if self.verbose:
                print(
                    f"Поколение {gen + 1}/{self.generations}: "
                    f"лучший {scored[0][1]:.4f}, глобально {best_score:.4f}"
                )
            self._report_progress(gen + 1, self.generations)

            # Элита переходит без изменений, остальное -- потомки
            next_population = [dict(ind) for ind, _ in scored[: self.elite_count]]
            while len(next_population) < self.population_size:
                parent_a = self._tournament_select(scored)
                if self.rng.random() < self.crossover_rate:
                    parent_b = self._tournament_select(scored)
                    child = self._crossover(parent_a, parent_b)
                else:
                    child = dict(parent_a)
                next_population.append(self._mutate(child))
            population = next_population

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=history,
            method="Genetic Algorithm",
        )


# ---------------------------------------------------------------------- #
# Связка с бэктестером
# ---------------------------------------------------------------------- #
def make_backtest_objective(
    data: pd.DataFrame,
    strategy_cls: Type[BaseStrategy],
    metric: str = "sharpe_ratio",
    fixed_params: Optional[dict[str, Any]] = None,
    engine_kwargs: Optional[dict[str, Any]] = None,
    min_trades: int = 5,
) -> Objective:
    """Создаёт целевую функцию: params -> значение метрики бэктеста.

    Параметры
    ---------
    metric:       имя поля PerformanceMetrics (sharpe_ratio, net_profit, ...).
    fixed_params: параметры стратегии, не участвующие в оптимизации.
    min_trades:   минимум сделок; прогоны с меньшим числом штрафуются
                  (-inf), чтобы оптимизатор не выбирал «пустые» стратегии.
    """
    fixed = dict(fixed_params or {})
    kwargs = dict(engine_kwargs or {})
    needs_trades = metric_needs_trades(metric)

    def objective(params: dict[str, Any]) -> float:
        result = run_backtest(
            data=data,
            strategy_cls=strategy_cls,
            strategy_params={**fixed, **params},
            **kwargs,
        )
        # Фильтр min_trades до расчёта метрик: «пустые» прогоны отсекаются
        # сразу, без сборки trades_df и подсчёта статистики.
        if len(result.trades) < min_trades:
            return -math.inf
        # trades_df строим только для метрик уровня сделок -- для Sharpe и
        # прочих «эквити-метрик» он не нужен вовсе.
        trades_df = result.trades_df() if needs_trades else None
        return compute_optimization_metric(result.equity_curve, trades_df, metric)

    return objective
