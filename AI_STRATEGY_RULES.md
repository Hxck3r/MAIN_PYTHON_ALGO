# Инструкция для ИИ: как писать стратегии для этого проекта

Этот документ — **контракт и чек-лист**, которому обязана следовать языковая
модель (ИИ), создающая торговые стратегии на Python для данного фреймворка.
Цель: сгенерированная стратегия должна **без правок** подключаться к проекту,
запускаться в бэктесте/оптимизации/WFA/GUI и **не падать с ошибками**.

> Если нужно понять проект «по-человечески» (как всё устроено, как запускать
> GUI и демо) — читайте [GUIDE.md](GUIDE.md). Здесь же — строгие правила.

---

## 0. TL;DR — минимальный валидный результат

ИИ обязан выдать **один файл** `strategies/<имя>.py` с классом-наследником
`BaseStrategy` **или** `SignalStrategy`, и **одну строку регистрации** в
[strategies/__init__.py](strategies/__init__.py). Всё. Никаких изменений в
`backtester/`, `gui/`, `main.py` вносить не нужно и не следует.

Два пути на выбор:

| Что выбрать | Когда | Базовый класс | Метод(ы) для реализации |
|---|---|---|---|
| **Событийная** | сложная логика: трейлинг, частичные закрытия, состояние между барами, нестандартные ордера | `backtester.BaseStrategy` | `init()`, `next(bar)` |
| **Сигнальная** (рекомендуется) | одна неттинг-позиция, рыночный вход на след. баре, защитные SL/TP | `backtester.fast_engine.SignalStrategy` | `compute_signals()` |

Сигнальная стратегия исполняется и быстрым numba-ядром (десятки крат быстрее на
оптимизации/WFA), и обычным движком — результаты идентичны. **Если логика
укладывается в сигнальную модель — всегда выбирайте её.**

> 🧠 **Третий путь — НЕ для ручного кода.** В реестре уже есть готовые
> ML-стратегии (`MLStrategy`, `MetaLabelStrategy` из `strategies/ml_strategy.py`
> и `meta_strategy.py`) — они «проигрывают» предсказания обученной модели. Их
> **не пишут руками**: модель обучается во вкладке «🧠 Машинное обучение» и
> подставляется кнопкой «→ В стратегию». Если пользователь просит «стратегию на
> машинном обучении / нейросети / классификаторе» — **не генерируйте свой
> класс**, а направьте его в эту вкладку и к Части II
> [МЕТОДОЛОГИЯ_ТЕСТИРОВАНИЯ.md](МЕТОДОЛОГИЯ_ТЕСТИРОВАНИЯ.md). Правила ниже — про
> стратегии на правилах, которые вы пишете сами.

---

## 1. Жёсткие правила (нарушение = ошибка при запуске или нечестный результат)

### 1.1 Структура и регистрация
1. Класс наследуется **строго** от `BaseStrategy` (из `backtester`) или
   `SignalStrategy` (из `backtester.fast_engine`). Не от чего-либо иного.
2. Файл кладётся в каталог `strategies/`.
3. Стратегия должна быть зарегистрирована в `strategies/__init__.py` в словаре
   `STRATEGIES` (см. §6), иначе она **не появится в GUI**.
4. Импорт ядра — только из публичного API: `from backtester import Bar,
   BaseStrategy` и `from backtester.fast_engine import SignalStrategy`. Не
   импортируйте приватные модули глубже, чем нужно.

### 1.2 Параметры
5. Все настраиваемые числа — **только** через словарь класса `params`. Никаких
   «магических» констант в теле логики (их нельзя оптимизировать и менять в GUI).
6. Значения в `params` — простые типы: `int`, `float`, `bool`, `str`. Никаких
   списков/словарей/объектов как значений по умолчанию.
7. Доступ к параметрам — **только** через `self.p["имя"]` (не `self.params`).
   `self.p` — это слитый словарь «дефолты + переопределения конструктора».
8. Передача неизвестного параметра в конструктор вызывает `ValueError`. Поэтому
   любой параметр, который вы используете, обязан присутствовать в `params`.
9. **Числовые** параметры (`int`/`float`, кроме `bool`) автоматически попадают в
   таблицу GUI и доступны оптимизатору. `bool`/`str` параметры GUI показывает,
   но оптимизатор их не перебирает по диапазону — закладывайте это в дизайн.

### 1.3 Запрет заглядывания в будущее (lookahead bias) — критично
10. Решение/индикатор на баре `i` имеет право зависеть **только** от данных до
    бара `i` включительно. Это закон. Его нарушение делает все метрики
    бессмысленными.
11. Используйте `rolling()`, `ewm()`, сдвиги назад `shift(k)` с `k > 0`.
    **Запрещено:** `center=True` в rolling, `shift(-k)`, `bfill()`,
    `.iloc[i+1:]`, нормировки/масштабирование по всему ряду (min/max/mean/std
    всего массива), любое чтение «правого хвоста» данных.
12. В событийной `next(bar)` берите цены через срезы `self.close`, `self.high`,
    … — они автоматически обрезаны «по текущий бар включительно». Полные массивы
    `self.full_close` и т.п. — **только** в `init()` для предрасчёта индикаторов.
13. Для сигнальной стратегии прогоняйте проверку причинности
    `assert_no_lookahead(StrategyCls, data)` (см. §5.1) — она ловит протечку,
    которую не поймает ничто другое.

### 1.4 Ордера и сайзинг
14. Тип ордера задаётся взаимоисключающими аргументами:
    - без `limit_price`/`stop_price` → **рыночный** (исполнение на открытии
      следующего бара);
    - `limit_price` → лимитный; `stop_price` → стоп-вход.
    - `limit_price` и `stop_price` **вместе нельзя** → `ValueError`.
15. Сайзинг — тоже взаимоисключающий:
    - `size` — явный размер в единицах актива (должен быть `> 0`);
    - `risk_pct` — доля эквити под риском; **требует `stop_loss`** и `size` не
      задаётся одновременно (иначе `ValueError`);
    - ничего — автосайзинг `default_size_pct` от эквити.
16. `risk_pct` без `stop_loss` → `ValueError`. `size` вместе с `risk_pct` →
    `ValueError`. `size <= 0` → `ValueError`. Эти проверки работают **и** в
    быстром ядре (через `_validate_signal_arrays`), молчаливого пропуска ордера
    не будет.
17. SL/TP должны лежать по правильную сторону от цены входа. Если `risk_pct`
    задан, а стоп оказался по неверную сторону (нулевая/отрицательная дистанция),
    размер выйдет `0` и ордер будет **отклонён** (REJECTED) — это не ошибка
    запуска, но сделки не будет; проверяйте знаки.

### 1.5 Корректность вычислений
18. Индикаторы на этапе прогрева дают `NaN`. Перед использованием значения в
    `next()` проверяйте `np.isnan(...)` и выходите (`return`), пока не прогрелось.
    В векторной форме `NaN` в сравнении даёт `False` — это безопасно, но размеры
    массивов должны совпадать.
19. Длина любого индикаторного массива в сигнальной стратегии = `len(self.full_close)`.
    Массивы `self._sig/_sl/_tp/_size/_risk` уже созданы нужной длины — **только
    заполняйте их по индексам**, не пересоздавайте с другой длиной.
20. Не делайте сетевых запросов, не читайте внешние файлы, не используйте текущее
    время/`random` без seed внутри стратегии — прогон должен быть детерминирован.

---

## 2. Контракт событийной стратегии (`BaseStrategy`)

```python
"""Краткое описание стратегии: идея, входы/выходы, защита позиции."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester import Bar, BaseStrategy


class MyStrategy(BaseStrategy):
    """Одна строка о сути."""

    params = {
        "fast_period": 10,
        "slow_period": 30,
        "sl_pct": 0.02,
        "tp_pct": 0.04,
    }

    def init(self) -> None:
        # Одноразовый предрасчёт по ПОЛНЫМ массивам. rolling/ewm причинны.
        fast = int(self.p["fast_period"])
        slow = int(self.p["slow_period"])
        if fast >= slow:
            raise ValueError("fast_period должен быть меньше slow_period")
        close = pd.Series(self.full_close)
        self.fast_sma = close.rolling(fast).mean().to_numpy()
        self.slow_sma = close.rolling(slow).mean().to_numpy()

    def next(self, bar: Bar) -> None:
        # Вызывается на КАЖДОМ баре по порядку. bar.index — текущий бар i.
        i = bar.index
        if i < 1 or np.isnan(self.slow_sma[i - 1]):
            return  # ещё не прогрелось

        crossed_up = (self.fast_sma[i - 1] <= self.slow_sma[i - 1]
                      and self.fast_sma[i] > self.slow_sma[i])
        price = bar.close
        if crossed_up and (self.position is None or not self.position.is_long):
            self.buy(
                stop_loss=price * (1.0 - self.p["sl_pct"]),
                take_profit=price * (1.0 + self.p["tp_pct"]),
                tag="entry_long",
            )
```

Что доступно внутри:
- `bar.index/open/high/low/close/volume/datetime` — текущий бар.
- `self.position` — открытая позиция или `None`; `self.position.is_long`,
  `.size`, `.entry_price`. **Перед входом проверяйте позицию**, чтобы не плодить
  лишние развороты.
- `self.equity`, `self.cash` — состояние счёта.
- `self.buy(...)`, `self.sell(...)`, `self.close_position()`,
  `self.cancel_all_orders()` — торговый API (см. §1.4).
- Срезы `self.close/high/low/open/volume` — безопасны от lookahead в `next()`.

Образец: [strategies/sma_cross.py](strategies/sma_cross.py).

---

## 3. Контракт сигнальной стратегии (`SignalStrategy`) — рекомендуется

Реализуется **только** метод `compute_signals()`. Базовый `init()` уже создаёт
массивы длины `n` и вызывает `compute_signals()`; универсальный `next()` уже
написан — переопределять его не нужно.

Заполняемые массивы (длины `n = len(self.full_close)`):

| Массив | Тип | Смысл |
|---|---|---|
| `self._sig[i]` | `int` | решение на закрытии бара `i`: `+1` лонг, `-1` шорт, `0` ничего (исполнится рыночно на открытии `i+1`) |
| `self._sl[i]` | `float` | абсолютный уровень Stop-Loss (`NaN` = нет) |
| `self._tp[i]` | `float` | абсолютный уровень Take-Profit (`NaN` = нет) |
| `self._size[i]` | `float` | явный размер (`NaN` = авто) |
| `self._risk[i]` | `float` | сайзинг от риска, доля эквити под SL (`NaN` = не используется) |

```python
"""Сигнальная стратегия — исполняется быстрым numba-ядром."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.fast_engine import SignalStrategy


class MySignal(SignalStrategy):
    """Одна строка о сути."""

    params = {
        "fast_period": 10,
        "slow_period": 30,
        "sl_pct": 0.02,
        "tp_pct": 0.04,
    }

    def compute_signals(self) -> None:
        fast = int(self.p["fast_period"])
        slow = int(self.p["slow_period"])
        if fast >= slow:
            raise ValueError("fast_period должен быть меньше slow_period")

        sl_pct = float(self.p["sl_pct"])
        tp_pct = float(self.p["tp_pct"])
        prices = self.full_close
        close = pd.Series(prices)
        fast_sma = close.rolling(fast).mean().to_numpy()
        slow_sma = close.rolling(slow).mean().to_numpy()

        # Векторное причинное пересечение через сдвиг назад.
        prev_f, prev_s = fast_sma[:-1], slow_sma[:-1]
        cur_f, cur_s = fast_sma[1:], slow_sma[1:]
        valid = ~np.isnan(prev_s)
        crossed_up = valid & (prev_f <= prev_s) & (cur_f > cur_s)
        crossed_down = valid & (prev_f >= prev_s) & (cur_f < cur_s)

        up = np.flatnonzero(crossed_up) + 1   # +1: сигнал на текущем баре i
        down = np.flatnonzero(crossed_down) + 1
        self._sig[up] = 1
        self._sl[up] = prices[up] * (1.0 - sl_pct)
        self._tp[up] = prices[up] * (1.0 + tp_pct)
        self._sig[down] = -1
        self._sl[down] = prices[down] * (1.0 + sl_pct)
        self._tp[down] = prices[down] * (1.0 - tp_pct)
```

Правила сигнальной формы:
- Заполняйте массивы **по индексам** (`self._sig[idx] = ...`), не присваивайте
  новый массив другой длины.
- Не задавайте одновременно `_size[i]` и `_risk[i]` для одного `i`.
- `_risk[i]` без `_sl[i]` — запрещено (валидатор бросит `ValueError`).
- Неттинг и развороты ядро обрабатывает само: достаточно ставить `+1`/`-1`,
  встречный сигнал сначала закроет позицию, остаток откроет противоположную.

Образец: [strategies/sma_cross_signal.py](strategies/sma_cross_signal.py).

---

## 4. Модель исполнения (что делает движок — знать обязательно)

- Ордер из `next(bar i)` исполняется **не раньше** бара `i+1`. Рыночный — по
  открытию `i+1` с проскальзыванием/спредом. Заглянуть в свой бар нельзя.
- SL/TP проверяются внутри бара по `high`/`low`. Если за бар задеты **оба** —
  консервативно считается **Stop-Loss**.
- Гэп через уровень: исполнение по цене открытия (хуже уровня) — реалистично.
- Учёт позиции — **неттинг** (одна агрегированная позиция, как в MT5).
- Спред: данные считаются bid-ценами; любая **покупка** (вход в лонг, выход из
  шорта, SL/TP шорта) исполняется по ask = `price * (1 + spread_pct)`.
- Проскальзывание всегда против трейдера.
- `max_leverage` урезает размер до `equity * max_leverage / price`; если маржи
  нет вовсе — ордер отклоняется (REJECTED).
- В конце данных открытая позиция принудительно закрывается по `close` с
  причиной `END_OF_DATA`.

Эти же правила **побитово** реализованы в numba-ядре, поэтому оба движка дают
идентичный результат.

---

## 5. Обязательная самопроверка перед сдачей

ИИ обязан мысленно (а лучше — реально) прогнать стратегию и убедиться, что:

### 5.1 Скрипт быстрой проверки
```python
from backtester import BacktestEngine, generate_synthetic_data
from backtester.backtest_engine import run_backtest
from strategies import MyStrategy          # или MySignal

data = generate_synthetic_data(n_bars=2000, freq="1h", seed=7)

# 1) запускается без исключений и даёт результат
res = run_backtest(data, MyStrategy)       # авто-выбор движка
print(res.stats())

# 2) для СИГНАЛЬНОЙ — проверка причинности (анти-lookahead)
from backtester.fast_engine import assert_no_lookahead
assert_no_lookahead(MySignal, data)        # бросит AssertionError при протечке

# 3) для СИГНАЛЬНОЙ — эквивалентность движков (необязательно, но желательно)
from backtester.fast_engine import run_fast_backtest
slow = BacktestEngine(data=data, strategy_cls=MySignal).run()
fast = run_fast_backtest(data, MySignal)
assert abs(slow.final_equity - fast.final_equity) < 1e-6
```

### 5.2 Чек-лист
- [ ] Класс наследует `BaseStrategy` или `SignalStrategy`.
- [ ] Все используемые параметры объявлены в `params`, значения — простые типы.
- [ ] Параметры читаются через `self.p[...]`.
- [ ] В логике нет заглядывания в будущее (нет `center=True`, `shift(-k)`,
      `bfill`, нормировки по всему ряду, чтения будущих баров).
- [ ] `NaN` на прогреве обрабатываются (`return` или векторный `valid`).
- [ ] Ордера и сайзинг не нарушают взаимоисключающих правил §1.4.
- [ ] SL/TP по правильную сторону от входа; при `risk_pct` есть `stop_loss`.
- [ ] Стратегия зарегистрирована в `STRATEGIES`.
- [ ] (Сигнальная) `assert_no_lookahead` проходит.

---

## 6. Регистрация в GUI

В [strategies/__init__.py](strategies/__init__.py) добавить импорт и запись в
`STRATEGIES` (ключ — отображаемое имя, значение — класс):

```python
from .my_strategy import MyStrategy

STRATEGIES = {
    "SMA Crossover (пересечение скользящих)": SmaCrossStrategy,
    "Моя стратегия (человекочитаемое имя)": MyStrategy,
}
```

Для **сигнальной** стратегии регистрируйте под `try/except ImportError` (чтобы
проект работал и без numba), по образцу существующего блока `SmaCrossSignal`:

```python
try:
    from .my_signal import MySignal
except ImportError:
    pass
else:
    STRATEGIES["Моя стратегия (быстрый, numba)"] = MySignal
```

После этого стратегия появится в выпадающем списке GUI, её числовые параметры —
в таблице параметров, и она будет доступна оптимизатору/WFA/Монте-Карло.

---

## 7. Метрики для оптимизации (имена полей)

Оптимизатор максимизирует/минимизирует поле `PerformanceMetrics` по имени.
Допустимые имена метрик (наиболее ходовые):
`sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `profit_factor`,
`net_profit`, `total_return_pct`, `cagr_pct`, `win_rate_pct`,
`max_drawdown_pct`, `expectancy`, `recovery_factor`, `complex_criterion`.

Это поля результата, а не настройки стратегии — стратегия их не считает, но
выбирайте параметры так, чтобы целевая метрика имела смысл (например, для
`sharpe_ratio` нужна осмысленная частота сделок).

---

## 8. Частые ошибки, которых ИИ обязан избегать

| Ошибка | Последствие | Правильно |
|---|---|---|
| Константа в логике вместо `params` | нельзя оптимизировать/менять в GUI | вынести в `params`, читать через `self.p` |
| `self.params["x"]` для чтения | читает дефолт класса, игнорит переопределение | `self.p["x"]` |
| `close.rolling(n, center=True)` | заглядывание в будущее | `close.rolling(n)` |
| `series.shift(-1)` | заглядывание в будущее | `series.shift(1)` |
| Нормировка `(x - x.mean())/x.std()` по всему ряду | заглядывание | rolling-нормировка причинно |
| `risk_pct` без `stop_loss` | `ValueError` при запуске | задать `stop_loss` |
| `size` и `risk_pct` вместе | `ValueError` | выбрать одно |
| Использование значения индикатора на прогреве | `NaN` ломает сравнения/сделки | проверять `np.isnan` |
| Пересоздание `self._sig = np.array(...)` другой длины | рассинхрон с данными | заполнять по индексам |
| Стратегия не в `STRATEGIES` | не видна в GUI | зарегистрировать |
| Наследование не от `BaseStrategy`/`SignalStrategy` | не запустится | исправить базовый класс |

---

## 9. Итоговый порядок действий для ИИ

1. Понять идею стратегии; решить — сигнальная или событийная (по таблице §0).
2. Объявить `params` (только простые типы, осмысленные дефолты).
3. Реализовать `compute_signals()` (сигнальная) **или** `init()` + `next()`
   (событийная), строго соблюдая §1.3 (нет lookahead) и §1.4 (ордера).
4. Положить файл в `strategies/`, зарегистрировать в `STRATEGIES` (§6).
5. Прогнать самопроверку §5 (запуск + `assert_no_lookahead` для сигнальной).
6. Сдать: один новый файл стратегии + правка `strategies/__init__.py`. Больше
   ничего менять не нужно.
