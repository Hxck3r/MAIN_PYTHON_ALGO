"""Общие фикстуры тестов."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import generate_synthetic_data


@pytest.fixture(scope="session")
def synth_1h() -> pd.DataFrame:
    """Синтетический часовой ряд, достаточно длинный для SMA-кроссовера."""
    return generate_synthetic_data(n_bars=1500, freq="1h", seed=7)


def make_bars(prices: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Ручные бары (open, high, low, close) -> канонический DataFrame."""
    arr = np.array(prices, dtype=np.float64)
    index = pd.date_range("2024-01-01", periods=len(arr), freq="1h", name="datetime")
    return pd.DataFrame(
        {
            "open": arr[:, 0],
            "high": arr[:, 1],
            "low": arr[:, 2],
            "close": arr[:, 3],
            "volume": np.full(len(arr), 1000.0),
        },
        index=index,
    )
