"""Технические индикаторы на чистом numpy/pandas (без внешних TA-библиотек).

Каждая функция принимает ``pd.Series``/``pd.DataFrame`` и возвращает Series/DataFrame
с тем же индексом, чтобы результаты легко присоединялись к OHLCV-кадру.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False).mean()


def returns(close: pd.Series, periods: int = 1) -> pd.Series:
    return close.pct_change(periods)


def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(close / close.shift(periods))


def rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Скользящее стандартное отклонение доходностей (прокси волатильности)."""
    return close.pct_change().rolling(window).std()


def momentum(close: pd.Series, window: int = 10) -> pd.Series:
    return close - close.shift(window)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (метод Уайлдера)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100.0)  # при нулевых потерях RSI -> 100


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD: линия, сигнальная линия и гистограмма."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Полосы Боллинджера + позиция цены внутри полос ``bb_pct`` в [0, 1]."""
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower).replace(0.0, np.nan)
    pct = (close - lower) / width
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_pct": pct}
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (волатильность диапазона)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def stochastic_k(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Стохастический осциллятор %K в [0, 100]."""
    lowest = low.rolling(window).min()
    highest = high.rolling(window).max()
    rng = (highest - lowest).replace(0.0, np.nan)
    return 100 * (close - lowest) / rng
