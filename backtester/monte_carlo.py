"""
Монте-Карло анализ результатов бэктеста: перемешивание/ресэмплинг сделок.

Идея: исторический бэктест даёт ОДНУ последовательность сделок -- одну
реализацию случайного процесса. Реальное будущее почти наверняка даст
те же сделки в другом порядке (или похожие, но не те же). Монте-Карло
строит тысячи альтернативных кривых эквити из тех же сделок и отвечает
на вопросы:

    * какова РАСПРЕДЕЛЁННАЯ (а не точечная) оценка доходности и просадки;
    * с какой вероятностью стратегия достигнет цели по прибыли;
    * с какой вероятностью просадка превысит допустимый предел / порог
      разорения (risk of ruin);
    * не является ли красивая кривая бэктеста просто удачным порядком
      сделок (защита от переоптимизации: перцентиль исходного результата
      внутри облака симуляций + доверительный интервал матожидания).

Методы генерации последовательностей:
    * permute  -- перестановка без возврата: те же сделки, новый порядок.
      Итоговая прибыль фиксирована, меняется только ПУТЬ -- честная
      оценка распределения просадок и серий убытков.
    * resample -- бутстрэп с возвратом: каждая сделка тянется случайно,
      возможны повторы. Меняется и итог, и путь -- даёт доверительные
      интервалы доходности и матожидания.
    * block    -- блочный бутстрэп (циклический): тянутся блоки подряд
      идущих сделок длиной ~sqrt(N) -- сохраняет автокорреляцию
      (кластеры прибыльных/убыточных периодов).

Режимы счёта эквити:
    * percent -- реинвестирование: доходности сделок в долях от баланса
      перед сделкой, кривая компаундится (как торговля % от счёта).
    * fixed   -- фиксированный размер: PnL сделок в деньгах просто
      суммируется (как торговля постоянным лотом).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # только для аннотаций -- без циклического импорта
    from .backtest_engine import BacktestResult

#: Перцентили конверта веера эквити (нижний хвост, квартиль, медиана...)
ENVELOPE_PERCENTILES: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)

#: Сколько отдельных кривых сохранять для отрисовки «спагетти»
MAX_SAMPLE_CURVES = 200

#: Потолок числа кривых, по которым строится перцентильный конверт
#: (защита памяти: агрегаты считаются по ВСЕМ симуляциям чанками)
MAX_ENVELOPE_CURVES = 4000


@dataclass
class MonteCarloConfig:
    """Настройки Монте-Карло симуляции."""

    n_sims: int = 2000
    method: str = "resample"        # resample | permute | block
    sizing: str = "percent"         # percent (реинвест) | fixed (пост. размер)
    horizon_trades: int = 0         # сделок на симуляцию; 0 = как в бэктесте
    ruin_dd_pct: float = 50.0       # порог разорения: потеря % от старта
    target_return_pct: float = 20.0  # цель по прибыли, %
    dd_limit_pct: float = 20.0      # допустимая просадка, %
    seed: Optional[int] = 42

    def validate(self) -> None:
        if self.n_sims < 10:
            raise ValueError("Слишком мало симуляций (минимум 10)")
        if self.method not in ("resample", "permute", "block"):
            raise ValueError(f"Неизвестный метод: {self.method}")
        if self.sizing not in ("percent", "fixed"):
            raise ValueError(f"Неизвестный режим сайзинга: {self.sizing}")
        if not 0 < self.ruin_dd_pct <= 100:
            raise ValueError("Порог разорения должен быть в (0, 100]%")


@dataclass
class MonteCarloResult:
    """Итог Монте-Карло: распределения по симуляциям + сводные вероятности.

    Массивы длиной n_sims:
        final_equity, total_return_pct, max_dd_pct, max_dd_abs,
        max_losing_streak, mean_trade_ret_pct.
    Кривые (для графиков):
        envelope         -- перцентильные конверты, shape (len(q), steps+1);
        sample_curves    -- подвыборка кривых, shape (k, steps+1);
        original_curve   -- кривая исходного порядка сделок, shape (steps+1,).
    """

    config: MonteCarloConfig
    n_trades: int                       # сделок в исходном бэктесте
    horizon: int                        # сделок на симуляцию
    initial_equity: float

    final_equity: np.ndarray
    total_return_pct: np.ndarray
    max_dd_pct: np.ndarray
    max_dd_abs: np.ndarray
    max_losing_streak: np.ndarray
    mean_trade_ret_pct: np.ndarray      # средний % на сделку в симуляции
    reached_target: np.ndarray          # bool: пик эквити достиг цели
    dd_exceeded: np.ndarray             # bool: просадка превысила лимит
    ruined: np.ndarray                  # bool: эквити падала ниже порога

    envelope_q: tuple[float, ...]
    envelope: np.ndarray
    sample_curves: np.ndarray

    original_curve: np.ndarray
    original_return_pct: float
    original_max_dd_pct: float
    original_max_streak: int

    # ------------------------------------------------------------------ #
    # Сводные вероятности и интервалы
    # ------------------------------------------------------------------ #
    @property
    def n_sims(self) -> int:
        return len(self.final_equity)

    @property
    def prob_profit_pct(self) -> float:
        """Вероятность закончить горизонт в плюсе, %."""
        return float((self.total_return_pct > 0).mean() * 100.0)

    @property
    def prob_target_pct(self) -> float:
        """Вероятность достижения цели по прибыли (хотя бы раз за путь), %."""
        return float(self.reached_target.mean() * 100.0)

    @property
    def prob_dd_exceeded_pct(self) -> float:
        """Вероятность превысить допустимую просадку, %."""
        return float(self.dd_exceeded.mean() * 100.0)

    @property
    def risk_of_ruin_pct(self) -> float:
        """Вероятность разорения (потеря ruin_dd_pct от старта), %."""
        return float(self.ruined.mean() * 100.0)

    def return_percentile(self, q: float) -> float:
        return float(np.percentile(self.total_return_pct, q))

    def dd_percentile(self, q: float) -> float:
        return float(np.percentile(self.max_dd_pct, q))

    def streak_percentile(self, q: float) -> float:
        return float(np.percentile(self.max_losing_streak, q))

    @property
    def var95_pct(self) -> float:
        """VaR 95%: доходность, хуже которой лишь 5% сценариев."""
        return self.return_percentile(5.0)

    @property
    def cvar95_pct(self) -> float:
        """CVaR 95%: средняя доходность худших 5% сценариев."""
        tail = self.total_return_pct[
            self.total_return_pct <= np.percentile(self.total_return_pct, 5.0)
        ]
        return float(tail.mean()) if len(tail) else math.nan

    @property
    def expectancy_ci_pct(self) -> tuple[float, float]:
        """90% доверительный интервал среднего % на сделку (по бутстрэпу)."""
        return (
            float(np.percentile(self.mean_trade_ret_pct, 5.0)),
            float(np.percentile(self.mean_trade_ret_pct, 95.0)),
        )

    # --- защита от переоптимизации ------------------------------------ #
    @property
    def original_return_rank_pct(self) -> float:
        """Перцентиль доходности бэктеста в облаке симуляций.

        > 95% -- исходный порядок сделок аномально удачен: итоговая кривая,
        скорее всего, обязана везению/подгонке, а не преимуществу стратегии.
        """
        return float((self.total_return_pct <= self.original_return_pct).mean() * 100.0)

    @property
    def original_dd_rank_pct(self) -> float:
        """Доля симуляций с просадкой НЕ БОЛЬШЕ исходной, %.

        Маленькое значение (< 5%) -- в бэктесте просадка нетипично мала:
        реальная торговля почти наверняка покажет более глубокую.
        """
        return float((self.max_dd_pct <= self.original_max_dd_pct).mean() * 100.0)

    @property
    def p_loss_pct(self) -> float:
        """«p-value прибыльности»: доля сценариев с убытком, %.

        Малое значение -- статистическое преимущество устойчиво
        к пересэмплированию сделок; большое -- результат хрупок.
        """
        return float((self.total_return_pct <= 0).mean() * 100.0)


# ---------------------------------------------------------------------- #
# Генерация индексов сделок по методам
# ---------------------------------------------------------------------- #
def _sample_indices(
    rng: np.random.Generator, method: str, n_trades: int, n_sims: int, horizon: int
) -> np.ndarray:
    """Матрица индексов сделок shape (n_sims, horizon) для одного чанка."""
    if method == "permute":
        # независимая перестановка в каждой строке
        return np.argsort(rng.random((n_sims, n_trades)), axis=1)
    if method == "resample":
        return rng.integers(0, n_trades, size=(n_sims, horizon))
    # block: циклический блочный бутстрэп, длина блока ~ sqrt(N)
    block = max(1, int(round(math.sqrt(n_trades))))
    n_blocks = -(-horizon // block)  # ceil
    starts = rng.integers(0, n_trades, size=(n_sims, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n_trades
    return idx.reshape(n_sims, -1)[:, :horizon]


def _build_curves(
    idx: np.ndarray,
    trade_returns: np.ndarray,
    trade_pnl: np.ndarray,
    sizing: str,
    initial: float,
) -> np.ndarray:
    """Кривые эквити по сделкам, shape (n_sims, horizon+1), старт = initial."""
    n_sims = idx.shape[0]
    if sizing == "percent":
        # фактор <= 0 означает потерю всего счёта -- дальше кривая лежит в нуле
        factors = np.maximum(1.0 + trade_returns[idx], 0.0)
        curves = initial * np.cumprod(factors, axis=1)
    else:
        curves = initial + np.cumsum(trade_pnl[idx], axis=1)
    return np.hstack([np.full((n_sims, 1), initial), curves])


def _max_losing_streaks(losses: np.ndarray) -> np.ndarray:
    """Максимальная серия подряд идущих убытков в каждой строке (векторно)."""
    cs = np.cumsum(losses, axis=1)
    # в точках без убытка запоминаем счётчик; running max сбрасывает серию
    resets = np.where(~losses, cs, 0)
    streaks = cs - np.maximum.accumulate(resets, axis=1)
    return streaks.max(axis=1).astype(np.int64)


def _curve_drawdowns(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(макс. просадка в деньгах, в %) по каждой строке-кривой."""
    peaks = np.maximum.accumulate(curves, axis=1)
    dd = peaks - curves
    dd_abs = dd.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = np.where(peaks > 0, dd / peaks, 0.0).max(axis=1) * 100.0
    return dd_abs, dd_pct


# ---------------------------------------------------------------------- #
# Основной запуск
# ---------------------------------------------------------------------- #
def run_monte_carlo(
    result: "BacktestResult",
    config: Optional[MonteCarloConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> MonteCarloResult:
    """Монте-Карло анализ по сделкам завершённого бэктеста.

    Параметры
    ---------
    result:            BacktestResult с непустым списком сделок.
    config:            настройки (методы, пороги, число симуляций).
    progress_callback: callable(done, total) -- прогресс по симуляциям.
    """
    config = config or MonteCarloConfig()
    config.validate()

    trades_df = result.trades_df()
    pnl = trades_df["pnl"].to_numpy(dtype=np.float64)
    n_trades = len(pnl)
    if n_trades < 5:
        raise ValueError(
            f"Сделок всего {n_trades} -- для Монте-Карло нужно минимум 5 "
            f"(а для надёжных выводов лучше 30+)"
        )

    initial = float(result.initial_cash)
    # Доходность сделки в долях от БАЛАНСА перед ней (без плавающей прибыли):
    # так перестановка сделок корректно компаундится в режиме percent.
    balance_before = initial + np.concatenate([[0.0], np.cumsum(pnl)[:-1]])
    trade_returns = np.where(balance_before > 1e-9, pnl / balance_before, 0.0)

    horizon = (
        n_trades
        if config.method == "permute" or config.horizon_trades <= 0
        else int(config.horizon_trades)
    )

    target_equity = initial * (1.0 + config.target_return_pct / 100.0)
    ruin_equity = initial * (1.0 - config.ruin_dd_pct / 100.0)

    rng = np.random.default_rng(config.seed)
    n_sims = int(config.n_sims)
    chunk_size = max(64, min(n_sims, int(2e7 // max(horizon, 1)) or 1))

    finals, dd_abs_all, dd_pct_all = [], [], []
    streaks_all, mean_ret_all = [], []
    reached_all, exceeded_all, ruined_all = [], [], []
    stored_curves: list[np.ndarray] = []
    stored = 0

    done = 0
    while done < n_sims:
        size = min(chunk_size, n_sims - done)
        idx = _sample_indices(rng, config.method, n_trades, size, horizon)
        curves = _build_curves(idx, trade_returns, pnl, config.sizing, initial)

        finals.append(curves[:, -1])
        d_abs, d_pct = _curve_drawdowns(curves)
        dd_abs_all.append(d_abs)
        dd_pct_all.append(d_pct)
        streaks_all.append(_max_losing_streaks(pnl[idx] < 0))
        mean_ret_all.append(trade_returns[idx].mean(axis=1) * 100.0)
        reached_all.append(curves.max(axis=1) >= target_equity)
        exceeded_all.append(d_pct >= config.dd_limit_pct)
        ruined_all.append(curves.min(axis=1) <= ruin_equity)

        if stored < MAX_ENVELOPE_CURVES:
            take = min(size, MAX_ENVELOPE_CURVES - stored)
            stored_curves.append(curves[:take])
            stored += take

        done += size
        if progress_callback is not None:
            progress_callback(done, n_sims)

    envelope_pool = np.vstack(stored_curves)
    envelope = np.percentile(envelope_pool, ENVELOPE_PERCENTILES, axis=0)
    k = min(MAX_SAMPLE_CURVES, len(envelope_pool))
    sample_rows = rng.choice(len(envelope_pool), size=k, replace=False)
    sample_curves = envelope_pool[sample_rows]

    # Кривая исходного порядка сделок -- в тех же осях (номер сделки)
    original_idx = np.arange(n_trades)[None, :]
    original_curve = _build_curves(
        original_idx, trade_returns, pnl, config.sizing, initial
    )[0]
    orig_dd_abs, orig_dd_pct = _curve_drawdowns(original_curve[None, :])
    orig_streak = int(_max_losing_streaks((pnl < 0)[None, :])[0])

    final_equity = np.concatenate(finals)
    return MonteCarloResult(
        config=config,
        n_trades=n_trades,
        horizon=horizon,
        initial_equity=initial,
        final_equity=final_equity,
        total_return_pct=(final_equity / initial - 1.0) * 100.0,
        max_dd_pct=np.concatenate(dd_pct_all),
        max_dd_abs=np.concatenate(dd_abs_all),
        max_losing_streak=np.concatenate(streaks_all),
        mean_trade_ret_pct=np.concatenate(mean_ret_all),
        reached_target=np.concatenate(reached_all),
        dd_exceeded=np.concatenate(exceeded_all),
        ruined=np.concatenate(ruined_all),
        envelope_q=ENVELOPE_PERCENTILES,
        envelope=envelope,
        sample_curves=sample_curves,
        original_curve=original_curve,
        original_return_pct=float(original_curve[-1] / initial - 1.0) * 100.0,
        original_max_dd_pct=float(orig_dd_pct[0]),
        original_max_streak=orig_streak,
    )
