"""
Загрузка, очистка и валидация исторических данных.

Канонический формат данных во всём фреймворке:
    pd.DataFrame с DatetimeIndex (по возрастанию, без дубликатов)
    и колонками: open, high, low, close, volume (float64).

Поддерживается автоопределение формата экспорта MetaTrader 5
(колонки вида <DATE>, <TIME>, <OPEN>, ..., разделитель табуляция).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


@dataclass
class DataValidationReport:
    """Отчёт об очистке и валидации загруженных данных."""

    rows_loaded: int = 0
    rows_after_cleaning: int = 0
    duplicates_removed: int = 0
    nan_rows_dropped: int = 0
    invalid_ohlc_dropped: int = 0
    gaps_detected: int = 0
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    issues: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "=== Отчёт о загрузке данных ===",
            f"Загружено строк:            {self.rows_loaded}",
            f"После очистки:              {self.rows_after_cleaning}",
            f"Удалено дубликатов:         {self.duplicates_removed}",
            f"Удалено строк с NaN:        {self.nan_rows_dropped}",
            f"Удалено некорректных OHLC:  {self.invalid_ohlc_dropped}",
            f"Обнаружено разрывов:        {self.gaps_detected}",
            f"Период: {self.start} -- {self.end}",
        ]
        if self.issues:
            lines.append("Замечания:")
            lines.extend(f"  - {issue}" for issue in self.issues)
        return "\n".join(lines)


class DataLoader:
    """Загрузчик исторических данных из CSV-файлов.

    Параметры
    ---------
    datetime_col:
        Имя колонки с датой/временем (если дата и время в одной колонке).
    column_map:
        Отображение имён колонок исходного файла на канонические
        ("open", "high", "low", "close", "volume").
        Пример: {"Цена открытия": "open", ...}
    tz:
        Часовой пояс для локализации индекса (например, "UTC").
    """

    def __init__(
        self,
        datetime_col: str = "datetime",
        column_map: Optional[Mapping[str, str]] = None,
        tz: Optional[str] = None,
    ) -> None:
        self.datetime_col = datetime_col
        self.column_map = dict(column_map) if column_map else {}
        self.tz = tz
        self.last_report: Optional[DataValidationReport] = None

    # ------------------------------------------------------------------ #
    # Публичный API
    # ------------------------------------------------------------------ #
    def load_csv(
        self,
        path: "str | Path | Any",
        sep: Optional[str] = None,
        dropna: bool = True,
        validate_ohlc: bool = True,
    ) -> pd.DataFrame:
        """Загружает CSV, приводит к каноническому формату, чистит и валидирует.

        Параметры
        ---------
        path:          путь к .csv файлу либо файловый объект
                       (например, загруженный через GUI буфер).
        sep:           разделитель; None -- автоопределение.
        dropna:        удалять ли строки с пропусками (иначе forward-fill).
        validate_ohlc: удалять ли строки с нарушением high >= low и т.п.
        """
        if hasattr(path, "read"):  # файловый объект (буфер из GUI)
            raw = pd.read_csv(path, sep=sep, engine="python")
        else:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Файл данных не найден: {path}")
            raw = pd.read_csv(path, sep=sep, engine="python")
        report = DataValidationReport(rows_loaded=len(raw))

        df = self._normalize_columns(raw)
        df = self._build_datetime_index(df)
        df = self._clean(df, report, dropna=dropna, validate_ohlc=validate_ohlc)
        self._detect_gaps(df, report)

        report.rows_after_cleaning = len(df)
        if len(df):
            report.start = df.index[0]
            report.end = df.index[-1]
        self.last_report = report

        if df.empty:
            raise ValueError(f"После очистки данных не осталось строк: {path}")
        return df

    @staticmethod
    def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Пересэмплирование OHLCV на старший таймфрейм (например, '4h', '1D')."""
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        out = df.resample(timeframe).agg(agg).dropna(subset=["open", "close"])
        return out

    # ------------------------------------------------------------------ #
    # Внутренняя кухня
    # ------------------------------------------------------------------ #
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Приводит имена колонок к каноническим (включая формат MT5)."""
        df = df.copy()
        # Формат MetaTrader 5: <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
        df.columns = [str(c).strip().strip("<>").strip().lower() for c in df.columns]

        rename: dict[str, str] = {}
        for src, dst in self.column_map.items():
            rename[src.strip().strip("<>").lower()] = dst.lower()

        # Стандартные синонимы
        synonyms = {
            "o": "open", "h": "high", "l": "low", "c": "close",
            "vol": "volume", "tickvol": "volume", "tick_volume": "volume",
            "real_volume": "volume",
            "time": "datetime" if "date" not in df.columns else "time",
            "date": "date", "timestamp": "datetime",
        }
        for src, dst in synonyms.items():
            if src in df.columns and dst not in rename.values():
                rename.setdefault(src, dst)
        df = df.rename(columns=rename)

        # Если объёма нет вовсе -- заполняем нулями (например, форекс без volume)
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df

    def _build_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Собирает DatetimeIndex из колонок date/time или datetime."""
        dt_col = self.datetime_col.lower()
        if "date" in df.columns and "time" in df.columns:
            stamp = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                format="mixed",
            )
            df = df.drop(columns=["date", "time"])
        elif dt_col in df.columns:
            stamp = pd.to_datetime(df[dt_col], format="mixed")
            df = df.drop(columns=[dt_col])
        elif "date" in df.columns:
            stamp = pd.to_datetime(df["date"], format="mixed")
            df = df.drop(columns=["date"])
        else:
            raise ValueError(
                "Не найдена колонка с датой/временем. "
                f"Ожидалась '{self.datetime_col}', 'date'+'time' или 'date'."
            )

        # as_unit("ns"): pandas >= 2.0 может распарсить даты с разрешением
        # s/us -- приводим к каноническим наносекундам, чтобы весь код
        # (графики, сшивка WFA) работал с единым представлением времени
        df.index = pd.DatetimeIndex(stamp, name="datetime").as_unit("ns")
        if self.tz:
            df.index = df.index.tz_localize(self.tz)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"В данных отсутствуют обязательные колонки: {missing}")
        return df[list(REQUIRED_COLUMNS)].astype(np.float64)

    @staticmethod
    def _clean(
        df: pd.DataFrame,
        report: DataValidationReport,
        dropna: bool,
        validate_ohlc: bool,
    ) -> pd.DataFrame:
        df = df.sort_index()

        dup_mask = df.index.duplicated(keep="first")
        report.duplicates_removed = int(dup_mask.sum())
        if report.duplicates_removed:
            df = df[~dup_mask]
            report.issues.append("обнаружены дубликаты по времени (оставлен первый бар)")

        nan_mask = df[list(REQUIRED_COLUMNS[:4])].isna().any(axis=1)
        report.nan_rows_dropped = int(nan_mask.sum())
        if report.nan_rows_dropped:
            if dropna:
                df = df[~nan_mask]
            else:
                df = df.ffill()
                report.issues.append("пропуски заполнены методом forward-fill")
        df["volume"] = df["volume"].fillna(0.0)

        if validate_ohlc:
            invalid = (
                (df["high"] < df["low"])
                | (df["high"] < df[["open", "close"]].max(axis=1))
                | (df["low"] > df[["open", "close"]].min(axis=1))
                | (df[list(REQUIRED_COLUMNS[:4])] <= 0).any(axis=1)
            )
            report.invalid_ohlc_dropped = int(invalid.sum())
            if report.invalid_ohlc_dropped:
                df = df[~invalid]
                report.issues.append("удалены бары с нарушенной целостностью OHLC")
        return df

    @staticmethod
    def _detect_gaps(df: pd.DataFrame, report: DataValidationReport) -> None:
        """Ищет разрывы во времени (пропущенные бары) относительно медианного шага."""
        if len(df) < 3:
            return
        deltas = df.index.to_series().diff().dropna()
        median_step = deltas.median()
        if median_step <= pd.Timedelta(0):
            return
        # Разрывом считаем шаг, превышающий медианный более чем втрое
        # (выходные на форексе при таком пороге обычно тоже попадут в отчёт).
        gaps = deltas[deltas > median_step * 3]
        report.gaps_detected = int(len(gaps))
        if len(gaps):
            report.issues.append(
                f"медианный шаг {median_step}; крупнейший разрыв {gaps.max()}"
            )


def generate_synthetic_data(
    n_bars: int = 2000,
    start: str = "2022-01-01",
    freq: str = "1h",
    start_price: float = 100.0,
    drift: float = 0.00002,
    volatility: float = 0.004,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Генерирует синтетический OHLCV-ряд (геом. случайное блуждание).

    Полезно для тестов и демонстрации работы фреймворка без реальных данных.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=drift, scale=volatility, size=n_bars)
    close = start_price * np.exp(np.cumsum(log_returns))
    open_ = np.empty_like(close)
    open_[0] = start_price
    open_[1:] = close[:-1]

    spread = np.abs(rng.normal(scale=volatility, size=n_bars)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(100, 10_000, size=n_bars).astype(np.float64)

    index = pd.date_range(start=start, periods=n_bars, freq=freq, name="datetime")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
