"""Инженерия признаков и формирование целевой переменной.

``build_feature_set`` принимает OHLCV-кадр и :class:`FeatureConfig`, возвращает
:class:`ml.domain.FeatureSet`: кадр с признаками + target, список столбцов-признаков
и метаданные для отрисовки (overlay/RSI/MACD).

Важно для трейдинга:
* признаки делаем по возможности стационарными (отношения/доходности, а не уровни цены);
* целевая переменная смотрит в БУДУЩЕЕ (``close.shift(-h)``), поэтому последние ``h``
  баров отбрасываются — это исключает заглядывание вперёд (look-ahead bias).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ml import config
from ml.domain import FeatureSet
from ml.features import indicators as ind
from ml.features import labeling as lab

log = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Какие признаки считать и как строить target."""

    return_periods: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    sma_windows: list[int] = field(default_factory=lambda: [10, 20, 50])
    ema_windows: list[int] = field(default_factory=lambda: [12, 26])
    use_rsi: bool = True
    rsi_window: int = 14
    use_macd: bool = True
    use_bollinger: bool = True
    bb_window: int = 20
    use_atr: bool = True
    atr_window: int = 14
    use_volatility: bool = True
    vol_window: int = 20
    use_momentum: bool = True
    mom_window: int = 10
    use_stochastic: bool = True
    use_volume: bool = True
    # Календарные признаки (час/день недели) — дешёвый внутридневной край.
    use_calendar: bool = True
    # Режимные признаки (режим волатильности + долгий моментум).
    use_regime: bool = True
    # Отбор признаков: убрать near-zero-variance и один из сильно коррелированных
    # (|corr| > порога). Меньше признаков → быстрее и устойчивее на слабом ПК.
    feature_select: bool = False
    feature_select_corr: float = 0.95
    # Хранить матрицу признаков в float32 (вдвое меньше памяти, быстрее деревья).
    use_float32: bool = True

    # Фрак-диф признак: стационарный, но с памятью (López de Prado, гл. 5).
    use_fracdiff: bool = False
    fracdiff_d: float = 0.4
    # Порог обрезки весов: 1e-3 даёт окно ~55 баров при d=0.4 (умеренный разогрев,
    # дёшево на слабом ПК); меньше — длиннее память, но больше потери на разогреве.
    fracdiff_threshold: float = 1e-3

    # Целевая переменная.
    target_mode: str = "direction"       # "direction" | "triple_barrier" | "meta"
    target_horizon: int = config.DEFAULT_FORWARD_HORIZON
    target_threshold: float = config.DEFAULT_RETURN_THRESHOLD
    # Параметры тройного барьера (множители волатильности).
    tb_pt: float = 1.5
    tb_sl: float = 1.5
    tb_vol_span: int = 20
    # Первичный сигнал для мета-разметки (target_mode="meta", AFML гл. 3).
    primary_kind: str = "ma_cross"       # "ma_cross" | "momentum"
    primary_fast: int = 20
    primary_slow: int = 50

    TARGET_COL: str = "target"


def select_features(
    frame: pd.DataFrame, columns: list[str], max_corr: float = 0.95
) -> list[str]:
    """Отобрать признаки: убрать near-zero-variance и один из коррелированных пар.

    Жадно идёт по столбцам и выкидывает тот, что слишком похож (|corr| > ``max_corr``) на
    уже принятый. Дёшево (одна корреляционная матрица), считается раз на чистых данных.
    """
    if len(columns) <= 1:
        return list(columns)
    sub = frame[columns]
    # near-zero variance — бесполезный признак (константа на этом отрезке).
    keep = [c for c in columns if float(sub[c].std()) > 1e-12]
    if len(keep) <= 1:
        return keep
    corr = sub[keep].corr().abs().to_numpy()
    selected: list[int] = []
    for i in range(len(keep)):
        if all(corr[i, j] <= max_corr for j in selected):
            selected.append(i)
    return [keep[i] for i in selected]


def build_feature_set(df: pd.DataFrame, cfg: FeatureConfig) -> FeatureSet:
    """Построить :class:`FeatureSet` из каноничного OHLCV-кадра."""
    close = df[config.COL_CLOSE]
    high = df[config.COL_HIGH]
    low = df[config.COL_LOW]
    volume = df[config.COL_VOLUME]

    out = df.copy()
    feature_cols: list[str] = []
    overlay_cols: list[str] = []
    rsi_col: str | None = None
    macd_cols: tuple[str, str] | None = None
    side_col: str | None = None

    # --- Доходности за разные лаги (стационарные признаки) ---------------
    for p in cfg.return_periods:
        name = f"ret_{p}"
        out[name] = ind.returns(close, p)
        feature_cols.append(name)

    # --- SMA: рисуем уровень (overlay) + признак-отношение ---------------
    for w in cfg.sma_windows:
        raw = f"sma_{w}"
        out[raw] = ind.sma(close, w)
        overlay_cols.append(raw)
        feat = f"close_sma_{w}"
        out[feat] = close / out[raw] - 1.0
        feature_cols.append(feat)

    # --- EMA: аналогично ------------------------------------------------
    for w in cfg.ema_windows:
        raw = f"ema_{w}"
        out[raw] = ind.ema(close, w)
        overlay_cols.append(raw)
        feat = f"close_ema_{w}"
        out[feat] = close / out[raw] - 1.0
        feature_cols.append(feat)

    # --- RSI ------------------------------------------------------------
    if cfg.use_rsi:
        out["rsi"] = ind.rsi(close, cfg.rsi_window)
        feature_cols.append("rsi")
        rsi_col = "rsi"

    # --- MACD -----------------------------------------------------------
    if cfg.use_macd:
        m = ind.macd(close)
        out["macd"] = m["macd"]
        out["macd_signal"] = m["macd_signal"]
        out["macd_hist"] = m["macd_hist"]
        feature_cols += ["macd", "macd_hist"]
        macd_cols = ("macd", "macd_signal")

    # --- Полосы Боллинджера --------------------------------------------
    if cfg.use_bollinger:
        bb = ind.bollinger_bands(close, cfg.bb_window)
        for col in ("bb_mid", "bb_upper", "bb_lower"):
            out[col] = bb[col]
            overlay_cols.append(col)
        out["bb_pct"] = bb["bb_pct"]
        feature_cols.append("bb_pct")

    # --- ATR (нормируем на цену для стационарности) ---------------------
    if cfg.use_atr:
        out["atr"] = ind.atr(high, low, close, cfg.atr_window)
        out["atr_pct"] = out["atr"] / close
        feature_cols.append("atr_pct")

    # --- Волатильность доходностей --------------------------------------
    if cfg.use_volatility:
        out["volatility"] = ind.rolling_volatility(close, cfg.vol_window)
        feature_cols.append("volatility")

    # --- Моментум (нормируем на цену) -----------------------------------
    if cfg.use_momentum:
        out["momentum"] = ind.momentum(close, cfg.mom_window) / close
        feature_cols.append("momentum")

    # --- Стохастик ------------------------------------------------------
    if cfg.use_stochastic:
        out["stoch_k"] = ind.stochastic_k(high, low, close)
        feature_cols.append("stoch_k")

    # --- Фрак-диф лог-цены (стационарность + память) --------------------
    if cfg.use_fracdiff:
        from ml.features.fracdiff import frac_diff_ffd

        log_close = np.log(close.clip(lower=1e-12))
        out["fracdiff"] = frac_diff_ffd(log_close, cfg.fracdiff_d, cfg.fracdiff_threshold)
        feature_cols.append("fracdiff")

    # --- Объём ----------------------------------------------------------
    if cfg.use_volume and (volume.abs().sum() > 0):
        out["vol_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
        vol_sma = volume.rolling(20).mean().replace(0.0, np.nan)
        out["vol_ratio"] = volume / vol_sma
        feature_cols += ["vol_change", "vol_ratio"]

    # --- Календарные признаки (циклическое кодирование) -----------------
    # Добавляются только если в данных есть соответствующая вариация: для дневного
    # ряда час константен (пропускаем), для внутридневного — несёт сессионный эффект.
    if cfg.use_calendar:
        idx = out.index
        if pd.Index(idx.hour).nunique() > 1:
            ang = 2.0 * np.pi * idx.hour.to_numpy() / 24.0
            out["cal_hour_sin"] = np.sin(ang)
            out["cal_hour_cos"] = np.cos(ang)
            feature_cols += ["cal_hour_sin", "cal_hour_cos"]
        if pd.Index(idx.dayofweek).nunique() > 1:
            out["cal_dow"] = idx.dayofweek.to_numpy() / 6.0
            feature_cols.append("cal_dow")

    # --- Режимные признаки ----------------------------------------------
    if cfg.use_regime:
        vol = ind.rolling_volatility(close, cfg.vol_window)
        med = vol.rolling(max(50, cfg.vol_window * 3)).median().replace(0.0, np.nan)
        out["vol_regime"] = vol / med                  # >1 — повышенная волатильность
        out["mom_50"] = (close - close.shift(50)) / close
        feature_cols += ["vol_regime", "mom_50"]

    # --- Целевая переменная ---------------------------------------------
    # ``_t1_off`` — длина «жизни» метки в барах (для весов уникальности, AFML гл. 4):
    # для тройного барьера = бар касания, для направления = горизонт.
    T1_COL = "_t1_off"
    h = max(1, cfg.target_horizon)
    if cfg.target_mode == "meta":
        from ml.features.primary import primary_signal

        side = primary_signal(close, cfg.primary_kind, cfg.primary_fast, cfg.primary_slow)
        out["prim_side"] = side.to_numpy()
        feature_cols.append("prim_side")          # сторона — признак мета-модели
        side_col = "prim_side"
        events = lab.meta_labels(
            high, low, close, side, h, cfg.tb_pt, cfg.tb_sl, cfg.tb_vol_span
        )
        out[cfg.TARGET_COL] = events["label"].to_numpy()
        out[T1_COL] = events["t1"].to_numpy()
        log.info("Таргет: мета-разметка (первичный=%s, h=%d)", cfg.primary_kind, h)
    elif cfg.target_mode == "triple_barrier":
        events = lab.triple_barrier_events(
            high, low, close, h, cfg.tb_pt, cfg.tb_sl, cfg.tb_vol_span
        )
        out[cfg.TARGET_COL] = events["label"].to_numpy()
        out[T1_COL] = events["t1"].to_numpy()
        log.info("Таргет: тройной барьер (pt=%.2f, sl=%.2f, h=%d)", cfg.tb_pt, cfg.tb_sl, h)
    else:
        out[cfg.TARGET_COL] = lab.direction_labels(close, h, cfg.target_threshold)
        out[T1_COL] = float(h)
        log.info("Таргет: направление (порог=%.4f, h=%d)", cfg.target_threshold, h)

    # --- Очистка: убрать строки разогрева/без метки ---------------------
    before = len(out)
    needed = feature_cols + [cfg.TARGET_COL]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    out[cfg.TARGET_COL] = out[cfg.TARGET_COL].astype(int)

    # Позиционный конец метки (для весов уникальности); ряды после dropna считаем
    # квазинепрерывными по барам (как и purge в валидации).
    n_out = len(out)
    offsets = np.nan_to_num(out.pop(T1_COL).to_numpy(dtype=float), nan=float(h))
    event_end = np.minimum(np.arange(n_out) + offsets.astype(np.int64), max(n_out - 1, 0))
    log.info(
        "Признаки готовы: %d столбцов, %d строк (отброшено %d на разогрев/метку).",
        len(feature_cols), len(out), before - len(out),
    )

    if out.empty:
        raise ValueError("После расчёта признаков не осталось строк — мало данных?")

    # --- Отбор признаков (опционально) и float32 для скорости/памяти -----
    if cfg.feature_select:
        kept = select_features(out, feature_cols, cfg.feature_select_corr)
        if kept and len(kept) < len(feature_cols):
            log.info("Отбор признаков: %d → %d (убрана корреляция/константы).",
                     len(feature_cols), len(kept))
            feature_cols = kept
    if cfg.use_float32:
        out[feature_cols] = out[feature_cols].astype(np.float32)

    pos = int(out[cfg.TARGET_COL].sum())
    log.info("Баланс классов: вверх=%d (%.1f%%), вниз=%d",
             pos, 100 * pos / len(out), len(out) - pos)

    return FeatureSet(
        frame=out,
        feature_columns=feature_cols,
        target_column=cfg.TARGET_COL,
        horizon=h,
        overlay_indicators=[c for c in overlay_cols if c in out.columns],
        rsi_column=rsi_col,
        macd_columns=macd_cols,
        config_meta=asdict(cfg),
        event_end=event_end,
        side_column=side_col,
    )
