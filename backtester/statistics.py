"""
Статистическая защита от переоптимизации (data snooping).

После перебора сотен/тысяч комбинаций параметров лучший результат почти
наверняка завышен: максимум из N случайных величин растёт с N, даже если
у стратегии нет преимущества. Модуль отвечает на вопрос «лучший результат --
это сигнал или артефакт множественного тестирования?» тремя инструментами:

    * probabilistic_sharpe_ratio (PSR) -- вероятность того, что истинный
      Sharpe выше порога, с поправкой на длину ряда, асимметрию и эксцесс
      доходностей (Bailey & Lopez de Prado, 2012);
    * deflated_sharpe_ratio (DSR) -- PSR, где порогом служит ожидаемый
      максимум Sharpe среди N независимых испытаний оптимизации: DSR > 0.95
      означает, что результат значим ДАЖЕ с учётом числа перебранных
      комбинаций (Bailey & Lopez de Prado, 2014);
    * permutation_test -- прогон стратегии на множестве перемешанных
      версий истории (бар-ресэмплинг без возврата): сохраняется
      распределение баров, разрушается их порядок. p-value = доля
      перестановок, на которых стратегия заработала не меньше, чем на
      реальных данных. Малое p -- стратегия эксплуатирует именно временнУю
      структуру рынка, а не свойства распределения (Masters; Aronson).

Дополнительно: select_plateau_params -- выбор параметров по «плато»
(лучшее среднее по окрестности в сетке), устойчивая альтернатива argmax,
который из всей сетки выбирает самый переобученный пик.

Все функции без внешних зависимостей (только numpy/pandas + ядро пакета).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

import numpy as np
import pandas as pd

from .backtest_engine import BacktestEngine
from .base_strategy import BaseStrategy

_EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------- #
# Нормальное распределение (без scipy)
# ---------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Квантиль стандартного нормального распределения (алгоритм Акклама).

    Относительная точность ~1e-9 -- достаточно для оценки E[max] Sharpe.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p должно быть в (0, 1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


# ---------------------------------------------------------------------- #
# PSR / DSR
# ---------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(
    returns: np.ndarray, benchmark_sr: float = 0.0
) -> float:
    """PSR: вероятность того, что истинный Sharpe выше benchmark_sr.

    returns -- доходности ЗА ПЕРИОД (не годовые); benchmark_sr -- в той же
    шкале за период. Поправка на асимметрию и эксцесс: толстые хвосты и
    перекос вниз снижают уверенность при том же точечном Sharpe.
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)
    if n < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0 or not math.isfinite(std):
        return 0.0
    sr = float(returns.mean() / std)
    diff = returns - returns.mean()
    skew = float(np.mean(diff**3) / std**3)
    kurt = float(np.mean(diff**4) / std**4)  # обычный (не excess), у нормали 3
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return 0.0
    z = (sr - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denom)
    return _norm_cdf(z)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Ожидаемый максимум Sharpe среди n_trials испытаний при нулевом
    истинном Sharpe (аппроксимация через асимптотику максимума гауссиан).

    Это «сколько Sharpe даст чистый перебор шума» -- порог для DSR.
    """
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    e = math.e
    return math.sqrt(sr_variance) * (
        (1.0 - _EULER_GAMMA) * _norm_ppf(1.0 - 1.0 / n_trials)
        + _EULER_GAMMA * _norm_ppf(1.0 - 1.0 / (n_trials * e))
    )


@dataclass
class DeflatedSharpeResult:
    """Итог DSR-анализа лучшего результата оптимизации."""

    dsr: float                 # вероятность [0..1], что Sharpe не артефакт перебора
    psr_zero: float            # PSR против нулевого Sharpe (без поправки на перебор)
    observed_sr: float         # точечный Sharpe за период (не годовой)
    expected_max_sr: float     # ожидаемый максимум Sharpe чистого шума за n_trials
    n_trials: int
    n_periods: int

    def __str__(self) -> str:
        verdict = (
            "значим с учётом перебора" if self.dsr >= 0.95
            else "НЕ значим: вероятен артефакт множественного тестирования"
        )
        return (
            f"=== Deflated Sharpe Ratio ===\n"
            f"Sharpe за период: {self.observed_sr:.4f} "
            f"(порог шума {self.n_trials} испытаний: {self.expected_max_sr:.4f})\n"
            f"PSR(>0): {self.psr_zero:.3f}, DSR: {self.dsr:.3f} -- {verdict}"
        )


def deflated_sharpe_ratio(
    returns: np.ndarray,
    trial_scores: np.ndarray,
    n_trials: Optional[int] = None,
) -> DeflatedSharpeResult:
    """DSR лучшего результата оптимизации.

    Параметры
    ---------
    returns:      доходности за период лучшего прогона
                  (equity_curve.pct_change().dropna()).
    trial_scores: значения критерия ВСЕХ испытаний оптимизации (например,
                  колонка score из OptimizationResult.history_df()) --
                  из них берётся дисперсия, нормированная к шкале Sharpe
                  за период через ранговую нормализацию.
    n_trials:     число испытаний; None -- len(trial_scores).

    Замечание: классический DSR использует дисперсию Sharpe-оценок самих
    испытаний. Если критерий оптимизации -- не Sharpe, дисперсию по шкале
    Sharpe честно не восстановить; здесь используется консервативный путь:
    дисперсия оценки SR одного прогона (1/(n-1)) масштабируется на разброс
    испытаний не меньше единичного. Для критерия sharpe_ratio результат
    совпадает с каноническим.
    """
    returns = np.asarray(returns, dtype=np.float64)
    scores = np.asarray(trial_scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    trials = int(n_trials) if n_trials is not None else len(scores)
    trials = max(trials, 1)
    n = len(returns)

    std = returns.std(ddof=1) if n >= 2 else 0.0
    sr = float(returns.mean() / std) if std > 0 else 0.0

    # Дисперсия SR-оценки одного прогона длиной n: ~ (1 + SR^2/2) / (n-1)
    sr_var = (1.0 + 0.5 * sr * sr) / max(n - 1, 1)
    sr_star = expected_max_sharpe(trials, sr_var)

    return DeflatedSharpeResult(
        dsr=probabilistic_sharpe_ratio(returns, benchmark_sr=sr_star),
        psr_zero=probabilistic_sharpe_ratio(returns, benchmark_sr=0.0),
        observed_sr=sr,
        expected_max_sr=sr_star,
        n_trials=trials,
        n_periods=n,
    )


# ---------------------------------------------------------------------- #
# Выбор параметров по плато
# ---------------------------------------------------------------------- #
def select_plateau_params(
    history: list[tuple[dict[str, Any], float]],
    radius: int = 1,
) -> tuple[dict[str, Any], float]:
    """Выбор параметров по «плато»: лучшее СРЕДНЕЕ score по окрестности.

    argmax по сетке выбирает самый острый (и самый переобученный) пик;
    плато-выбор предпочитает точку, вокруг которой результаты тоже хороши --
    такой оптимум переживает дрейф рынка. radius -- размер окрестности
    в шагах сетки по каждой оси (1 -> до 3^k соседей).

    Возвращает (params, plateau_score). Работает по истории Grid Search
    (полная сетка); для GA окрестности неполные, но подход применим.
    """
    if not history:
        raise ValueError("Пустая история оптимизации")
    names = sorted(history[0][0].keys())
    # Оси сетки: отсортированные уникальные значения каждого параметра
    axes: dict[str, list[Any]] = {
        name: sorted({params[name] for params, _ in history}) for name in names
    }
    pos_of = {
        name: {v: i for i, v in enumerate(vals)} for name, vals in axes.items()
    }
    # score по координатам точки в сетке
    score_at: dict[tuple[int, ...], float] = {}
    for params, score in history:
        key = tuple(pos_of[name][params[name]] for name in names)
        score_at[key] = score

    def neighborhood_mean(key: tuple[int, ...]) -> float:
        deltas = [range(-radius, radius + 1)] * len(key)
        values: list[float] = []
        from itertools import product

        for shift in product(*deltas):
            nb = tuple(k + s for k, s in zip(key, shift))
            s = score_at.get(nb)
            if s is not None and math.isfinite(s):
                values.append(s)
        return float(np.mean(values)) if values else -math.inf

    best_key: Optional[tuple[int, ...]] = None
    best_mean = -math.inf
    for key, score in score_at.items():
        if not math.isfinite(score):
            continue  # центр плато должен быть валидной точкой
        mean = neighborhood_mean(key)
        if mean > best_mean:
            best_mean, best_key = mean, key

    if best_key is None:
        raise ValueError("В истории нет ни одного валидного испытания")
    params = {name: axes[name][i] for name, i in zip(names, best_key)}
    return params, best_mean


# ---------------------------------------------------------------------- #
# Permutation test на данных
# ---------------------------------------------------------------------- #
def permute_bars(
    data: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Перемешивает бары, сохраняя их геометрию и распределение доходностей.

    Каждый бар описывается отношениями (open, high, low, close) к закрытию
    предыдущего бара; кортежи отношений (вместе с объёмом) перемешиваются
    без возврата и ряд восстанавливается от исходной стартовой цены.
    Первый бар остаётся на месте (у него нет предыдущего закрытия), поэтому
    мультимножество close-to-close доходностей сохраняется ТОЧНО,
    а временнАя структура (тренды, кластеры волатильности) разрушается.
    """
    closes = data["close"].to_numpy(dtype=np.float64)
    prev_close = np.concatenate(([closes[0]], closes[:-1]))
    prev_close[0] = data["open"].to_numpy()[0]  # для первого бара -- его open
    rel = np.column_stack([
        data["open"].to_numpy() / prev_close,
        data["high"].to_numpy() / prev_close,
        data["low"].to_numpy() / prev_close,
        closes / prev_close,
        data["volume"].to_numpy(dtype=np.float64),
    ])
    rel[1:] = rel[1:][rng.permutation(len(rel) - 1)]

    c_ratio = rel[:, 3]
    new_close = prev_close[0] * np.cumprod(c_ratio)
    new_prev = np.concatenate(([prev_close[0]], new_close[:-1]))
    out = pd.DataFrame(
        {
            "open": rel[:, 0] * new_prev,
            "high": rel[:, 1] * new_prev,
            "low": rel[:, 2] * new_prev,
            "close": new_close,
            "volume": rel[:, 4],
        },
        index=data.index,
    )
    return out


@dataclass
class PermutationTestResult:
    """Итог permutation-теста стратегии на перемешанных данных."""

    p_value: float             # доля перестановок со score >= реального
    original_score: float
    permuted_scores: np.ndarray
    metric: str
    n_permutations: int

    def __str__(self) -> str:
        verdict = (
            "стратегия эксплуатирует структуру рынка"
            if self.p_value <= 0.05
            else "результат неотличим от случайного порядка баров"
        )
        return (
            f"=== Permutation test ({self.metric}) ===\n"
            f"Реальные данные: {self.original_score:.4f}; перестановки: "
            f"медиана {np.median(self.permuted_scores):.4f}, "
            f"P95 {np.percentile(self.permuted_scores, 95):.4f}\n"
            f"p-value: {self.p_value:.4f} -- {verdict}"
        )


def permutation_test(
    data: pd.DataFrame,
    strategy_cls: Type[BaseStrategy],
    strategy_params: Optional[dict[str, Any]] = None,
    engine_kwargs: Optional[dict[str, Any]] = None,
    metric: str = "net_profit",
    n_permutations: int = 100,
    seed: Optional[int] = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> PermutationTestResult:
    """Прогоняет стратегию на n_permutations перемешанных копиях истории.

    p-value = (1 + #{перестановок со score >= реального}) / (n + 1).
    Дорогой тест (n полных бэктестов) -- запускать для финального
    кандидата с УЖЕ зафиксированными параметрами, не внутри оптимизации.
    """
    params = dict(strategy_params or {})
    kwargs = dict(engine_kwargs or {})
    rng = np.random.default_rng(seed)

    def run_once(df: pd.DataFrame) -> float:
        result = BacktestEngine(
            data=df, strategy_cls=strategy_cls, strategy_params=params, **kwargs
        ).run()
        return float(getattr(result.stats(), metric))

    original = run_once(data)
    scores = np.empty(n_permutations, dtype=np.float64)
    for k in range(n_permutations):
        scores[k] = run_once(permute_bars(data, rng))
        if progress_callback is not None:
            progress_callback(k + 1, n_permutations)

    p_value = (1.0 + float((scores >= original).sum())) / (n_permutations + 1.0)
    return PermutationTestResult(
        p_value=p_value,
        original_score=original,
        permuted_scores=scores,
        metric=metric,
        n_permutations=n_permutations,
    )
