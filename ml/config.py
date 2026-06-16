"""Константы ML-пакета: пути, имена столбцов, значения по умолчанию.

Самодостаточный модуль без зависимости от Qt и от остального приложения —
импортируется любым слоем ML (features / models / trainer / experiment).
"""
from __future__ import annotations

from pathlib import Path

# Версия артефактов (пишется в RunBundle для воспроизводимости).
APP_VERSION = "0.1.0"

# --- Пути ----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent       # MAIN_PYTHON_ALGO/
MODELS_DIR = ROOT_DIR / "models"                         # сохранённые RunBundle
CACHE_DIR = ROOT_DIR / ".cache" / "ml_features"          # кеш матриц признаков

# --- Канонические имена столбцов OHLCV (совпадают с backtester) ----------
COL_DATE = "date"
COL_OPEN = "open"
COL_HIGH = "high"
COL_LOW = "low"
COL_CLOSE = "close"
COL_VOLUME = "volume"
OHLCV = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]

# --- Значения по умолчанию для пайплайна ---------------------------------
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_FORWARD_HORIZON = 1          # на сколько баров вперёд предсказываем
DEFAULT_RETURN_THRESHOLD = 0.0       # порог доходности для метки "вверх"
DEFAULT_PROB_THRESHOLD = 0.5         # порог вероятности для входа в позицию
