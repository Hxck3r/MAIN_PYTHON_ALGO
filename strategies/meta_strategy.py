"""Мета-лейблинг как торговая стратегия (López de Prado, AFML гл. 3).

В отличие от :class:`MLStrategy` (где направление выводится из вероятности роста), здесь
**сторону** сделки задаёт первичный сигнал (он сохранён в артефакте как ``oos_side``), а
обученная мета-модель решает, **стоит ли** эту сделку брать: ``P(первичный прав) ≥ порога``.
Размер позиции — опционально по уверенности мета-модели (bet sizing, AFML гл. 10).

Как и :class:`MLStrategy`, играет ТОЛЬКО out-of-fold предсказания (purged walk-forward):
бары без OOS-оценки или без стороны игнорируются — честный торговый слой поверх честной
валидации. Исполняется штатным событийным движком, поэтому к результату применимы
Монте-Карло / PBO / Walk-Forward основного приложения.
"""
from __future__ import annotations

import math
from typing import Any, ClassVar

from backtester.base_strategy import Bar

from .ml_strategy import MLStrategy, _load_bundle_cached


class MetaLabelStrategy(MLStrategy):
    """Торговля по первичному сигналу с фильтром/сайзингом мета-модели."""

    params: ClassVar[dict[str, Any]] = {
        **MLStrategy.params,
        # Для меты осмыслен сайзинг по уверенности — включён по умолчанию.
        "size_mode": "confidence",
    }

    def init(self) -> None:
        super().init()                              # грузит модель, пороги, sizing, _proba
        bundle = _load_bundle_cached(str(self.p["bundle_path"]).strip())
        side = getattr(bundle, "oos_side", None)
        if side is None or len(side) == 0:
            raise ValueError(
                "MetaLabelStrategy: в артефакте нет стороны первичного сигнала — "
                "обучите модель в режиме «Мета-разметка»."
            )
        # Сторона на барах данных; вне OOS — NaN (там не торгуем), как и вероятности.
        self._side = side.reindex(self._data.index).to_numpy(dtype=float)

    def next(self, bar: Bar) -> None:
        i = bar.index
        p = self._proba[i]
        s = self._side[i]
        if math.isnan(p) or math.isnan(s) or s == 0:    # нет OOS-оценки/стороны — пропуск
            return

        take = p >= self._thr_hi                        # мета-модель одобряет сделку
        desired = int(math.copysign(1, s)) if take else 0
        pos = self.position

        if pos is None:
            if desired != 0:
                self._enter(desired, bar)
            return

        if i - pos.entry_bar < self._min_hold:          # уважаем минимальное удержание
            return

        current = 1 if pos.is_long else -1
        if desired == 0:                                # мета-модель «не берём» → в кэш
            self.close_position(tag="meta_exit")
        elif desired != current:                        # сменилась сторона → разворот
            self._enter(desired, bar)
