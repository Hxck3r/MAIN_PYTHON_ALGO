"""Мост: ML-модель как торговая стратегия движка MAIN.

``MLStrategy`` «проигрывает» out-of-fold вероятности обученной модели
(:class:`ml.experiment.RunBundle`) через штатный движок ``backtester`` — поэтому
к результату применяются без изменений метрики, Монте-Карло, PBO, Walk-Forward и
Deflated Sharpe основного приложения.

Почему именно так (защита от переобучения)
------------------------------------------
В бэктесте используются ТОЛЬКО out-of-fold предсказания: вероятность на баре ``i``
получена моделью, которая этот бар при обучении не видела (purged walk-forward).
Поэтому стратегия не «подсматривает» будущее и не торгует на in-sample участке —
бары без OOS-предсказания просто игнорируются. Это честный торговый слой поверх
честной валидации модели.

Логика входа/выхода (понятная трактовка классификатора P(рост)):
* P ≥ ``prob_threshold``           → быть в лонге;
* P ≤ ``exit_threshold``           → выйти в кэш (или развернуться в шорт, если разрешено);
* между порогами                   → удерживать текущую позицию (гистерезис от «дёрганья»).

``min_hold`` задаёт минимальное удержание в барах (защитные SL/TP у брокера срабатывают
всё равно). Стратегия наследует :class:`~backtester.BaseStrategy` и исполняется
событийным движком, поэтому умеет возвращаться в кэш (быстрый numba-движок этого не
позволяет — он всегда в рынке).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar

from backtester.base_strategy import Bar, BaseStrategy

# Кеш загруженных артефактов: путь -> (mtime, RunBundle). Оптимизатор/WFA создают
# множество экземпляров стратегии — без кеша каждый бы перечитывал .joblib с диска.
_BUNDLE_CACHE: dict[str, tuple[float, Any]] = {}


def _load_bundle_cached(path: str):
    """Загрузить RunBundle с кешированием по пути и времени модификации файла."""
    from ml.experiment import load_bundle

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл модели не найден: {path}")
    mtime = p.stat().st_mtime
    cached = _BUNDLE_CACHE.get(str(p))
    if cached is not None and cached[0] == mtime:
        return cached[1]
    bundle = load_bundle(p)
    _BUNDLE_CACHE[str(p)] = (mtime, bundle)
    return bundle


class MLStrategy(BaseStrategy):
    """Торговля по предсказаниям ML-классификатора (через RunBundle)."""

    params: ClassVar[dict[str, Any]] = {
        # Путь к артефакту .joblib (обязателен). Строковый параметр — не оптимизируется.
        "bundle_path": "",
        # Порог входа в лонг по вероятности. 0 → берётся порог из модели (bundle.threshold).
        "prob_threshold": 0.0,
        # Порог выхода/разворота (гистерезис). 0 → равен prob_threshold (без нейтральной зоны).
        "exit_threshold": 0.0,
        # Разрешить шорты (иначе только лонг/кэш).
        "allow_short": False,
        # Минимальное удержание позиции в барах (борьба с переторговлей).
        "min_hold": 0,
        # Защитные уровни в долях цены входа (0 → не выставлять).
        "sl_pct": 0.0,
        "tp_pct": 0.0,
        # Доля эквити на сделку (0 → автосайзинг движка default_size_pct). При сайзинге
        # по уверенности — это ВЕРХНЯЯ доля (множится на размер ставки из вероятности).
        "size_pct": 0.0,
        # Сайзинг: "fixed" — постоянная доля; "confidence"/"linear" — bet sizing из
        # калиброванной вероятности (AFML гл. 10): больше риска на уверенных сигналах.
        "size_mode": "fixed",
    }

    # ------------------------------------------------------------------ #
    def init(self) -> None:
        path = str(self.p["bundle_path"]).strip()
        if not path:
            raise ValueError(
                "MLStrategy: не задан bundle_path — обучите модель во вкладке "
                "«Машинное обучение» и нажмите «→ В стратегию»."
            )
        bundle = _load_bundle_cached(path)

        thr = float(self.p["prob_threshold"])
        self._thr_hi = thr if thr > 0.0 else float(bundle.threshold)
        exit_thr = float(self.p["exit_threshold"])
        self._thr_lo = exit_thr if exit_thr > 0.0 else self._thr_hi
        # На всякий случай: порог выхода не выше порога входа (иначе нет гистерезиса).
        self._thr_lo = min(self._thr_lo, self._thr_hi)

        self._allow_short = bool(self.p["allow_short"])
        self._min_hold = max(0, int(self.p["min_hold"]))
        self._sl_pct = max(0.0, float(self.p["sl_pct"]))
        self._tp_pct = max(0.0, float(self.p["tp_pct"]))
        self._size_pct = max(0.0, float(self.p["size_pct"]))
        self._size_mode = str(self.p.get("size_mode", "fixed")).strip() or "fixed"

        # Выравниваем OOS-вероятности по барам данных. Бары без предсказания → NaN
        # (модель их не оценивала) → на них не торгуем. Это и есть граница in/out-of-sample.
        proba = bundle.oos_proba.reindex(self._data.index)
        self._proba = proba.to_numpy(dtype=float)

    # ------------------------------------------------------------------ #
    def next(self, bar: Bar) -> None:
        i = bar.index
        p = self._proba[i]
        if math.isnan(p):                 # нет OOS-предсказания на этом баре — пропуск
            return

        pos = self.position
        want_long = p >= self._thr_hi
        want_short = self._allow_short and p <= self._thr_lo

        if pos is None:                   # в кэше — ищем вход
            if want_long:
                self._enter(+1, bar)
            elif want_short:
                self._enter(-1, bar)
            return

        # В позиции — уважаем минимальное удержание (SL/TP у брокера сработают и так).
        if i - pos.entry_bar < self._min_hold:
            return

        if pos.is_long:
            if p <= self._thr_lo:         # уверенность в росте упала
                if want_short:
                    self._enter(-1, bar)  # разворот: неттинг закроет лонг и откроет шорт
                else:
                    self.close_position(tag="ml_exit")
        else:                             # короткая позиция
            if p >= self._thr_hi:
                if want_long:
                    self._enter(+1, bar)
                else:
                    self.close_position(tag="ml_exit")

    # ------------------------------------------------------------------ #
    def _size_fraction(self, bar: Bar) -> float:
        """Доля эквити на сделку: фикс или bet sizing из вероятности (AFML гл. 10)."""
        if self._size_mode in ("confidence", "linear"):
            from ml.bet_sizing import bet_size_from_proba

            p = self._proba[bar.index]
            if math.isnan(p):
                return 0.0
            cap = self._size_pct if self._size_pct > 0.0 else 0.95
            return cap * bet_size_from_proba(float(p), self._size_mode)
        return self._size_pct                       # fixed (0 → автосайзинг движка)

    def _enter(self, side: int, bar: Bar) -> None:
        """Открыть позицию заданного направления с опциональными SL/TP и размером."""
        price = bar.close
        kw: dict[str, Any] = {}
        frac = self._size_fraction(bar)
        if frac > 0.0:
            kw["size"] = self.equity * frac / price
        if side == 1:
            if self._sl_pct > 0.0:
                kw["stop_loss"] = price * (1.0 - self._sl_pct)
            if self._tp_pct > 0.0:
                kw["take_profit"] = price * (1.0 + self._tp_pct)
            self.buy(**kw)
        else:
            if self._sl_pct > 0.0:
                kw["stop_loss"] = price * (1.0 + self._sl_pct)
            if self._tp_pct > 0.0:
                kw["take_profit"] = price * (1.0 - self._tp_pct)
            self.sell(**kw)
