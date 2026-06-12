"""Тесты загрузчика данных: формат MT5, очистка, спред, ресэмплинг."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from backtester import DataLoader

MT5_CSV = (
    "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
    "2024.01.02\t10:00:00\t100.0\t101.0\t99.5\t100.5\t1500\t0\t12\n"
    "2024.01.02\t11:00:00\t100.5\t102.0\t100.0\t101.5\t1600\t0\t11\n"
    "2024.01.02\t11:00:00\t100.5\t102.0\t100.0\t101.5\t1600\t0\t11\n"  # дубликат
    "2024.01.02\t12:00:00\t101.5\t101.0\t100.0\t100.5\t1700\t0\t10\n"  # high < open
    "2024.01.02\t13:00:00\t100.5\t101.5\t100.0\t101.0\t1800\t0\t13\n"
)


def test_mt5_format_parsed_with_spread():
    loader = DataLoader()
    df = loader.load_csv(io.StringIO(MT5_CSV))
    report = loader.last_report

    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns[:5]) == ["open", "high", "low", "close", "volume"]
    assert "spread" in df.columns          # колонка MT5 сохранена
    assert df.dtypes.eq(np.float64).all()
    assert report.duplicates_removed == 1
    assert report.invalid_ohlc_dropped == 1
    assert len(df) == 3


def test_missing_datetime_raises():
    csv = "open,high,low,close,volume\n1,2,0.5,1.5,10\n"
    with pytest.raises(ValueError, match="датой/временем"):
        DataLoader().load_csv(io.StringIO(csv))


def test_resample_ohlcv():
    index = pd.date_range("2024-01-01", periods=4, freq="1h", name="datetime")
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [2.0, 3.0, 4.0, 5.0],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.5, 2.5, 3.5, 4.5],
            "volume": [10.0, 20.0, 30.0, 40.0],
        },
        index=index,
    )
    out = DataLoader.resample(df, "2h")
    assert len(out) == 2
    assert out.iloc[0]["open"] == 1.0
    assert out.iloc[0]["high"] == 3.0
    assert out.iloc[0]["low"] == 0.5
    assert out.iloc[0]["close"] == 2.5
    assert out.iloc[0]["volume"] == 30.0


def test_gap_detection():
    rows = ["datetime,open,high,low,close,volume"]
    times = pd.date_range("2024-01-01", periods=10, freq="1h").tolist()
    times[6:] = [t + pd.Timedelta(hours=10) for t in times[6:]]  # разрыв
    for t in times:
        rows.append(f"{t},100,101,99,100,10")
    loader = DataLoader()
    loader.load_csv(io.StringIO("\n".join(rows)))
    assert loader.last_report.gaps_detected == 1
