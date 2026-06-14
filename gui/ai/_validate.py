"""Изолированная проверка файла стратегии (запускается в подпроцессе).

Вызывается инструментом агента ``validate_strategy``::

    python -m gui.ai._validate strategies/my_strategy.py [n_bars] [seed]

Подпроцесс (а не прямой вызов) выбран намеренно: чужой код стратегии может
зациклиться или упасть — это не должно ронять GUI. Скрипт загружает файл,
находит классы-стратегии, прогоняет быстрый бэктест на синтетике и, для
сигнальных стратегий, проверку причинности ``assert_no_lookahead``.
Печатает человекочитаемый отчёт в stdout; код возврата 0 — проверка прошла,
1 — найдена проблема.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_ai_check_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"не удалось загрузить {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_strategies(module):
    from backtester import BaseStrategy
    from backtester.fast_engine import SignalStrategy

    found = []
    for name in dir(module):
        obj = getattr(module, name)
        if not isinstance(obj, type):
            continue
        if obj.__module__ != module.__name__:
            continue  # импортированные базовые классы пропускаем
        if issubclass(obj, BaseStrategy) and obj not in (BaseStrategy, SignalStrategy):
            found.append(obj)
    return found


def main() -> int:
    if len(sys.argv) < 2:
        print("Использование: _validate.py <путь к файлу стратегии>")
        return 2
    rel = sys.argv[1]
    n_bars = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7

    target = (PROJECT_ROOT / rel).resolve()
    if not target.exists():
        print(f"Файл не найден: {rel}")
        return 1

    try:
        from backtester import generate_synthetic_data
        from backtester.backtest_engine import run_backtest
        from backtester.fast_engine import SignalStrategy, assert_no_lookahead
    except Exception as exc:  # noqa: BLE001
        print(f"Не удалось импортировать backtester: {exc}")
        return 1

    try:
        module = _load_module(target)
    except Exception:  # noqa: BLE001
        print("ОШИБКА ИМПОРТА файла стратегии:")
        print(traceback.format_exc(limit=4))
        return 1

    strategies = _find_strategies(module)
    if not strategies:
        print(
            "В файле не найдено классов-наследников BaseStrategy/SignalStrategy. "
            "Проверьте, что класс объявлен в этом файле и наследует базовый класс."
        )
        return 1

    data = generate_synthetic_data(n_bars=n_bars, freq="1h", seed=seed)
    ok = True
    lines: list[str] = []
    for cls in strategies:
        lines.append(f"### Стратегия: {cls.__name__}")
        is_signal = issubclass(cls, SignalStrategy)
        lines.append(
            f"Тип: {'сигнальная (SignalStrategy)' if is_signal else 'событийная (BaseStrategy)'}"
        )

        # 1) Проверка причинности (только сигнальные).
        if is_signal:
            try:
                assert_no_lookahead(cls, data)
                lines.append("Анти-lookahead: ПРОЙДЕНА (сигналы причинны).")
            except AssertionError as exc:
                ok = False
                lines.append(f"Анти-lookahead: ПРОВАЛЕНА — {exc}")
            except Exception as exc:  # noqa: BLE001
                ok = False
                lines.append(f"Анти-lookahead: ошибка проверки — {exc}")

        # 2) Прогон бэктеста.
        try:
            result = run_backtest(data, cls)
            stats = result.stats()
            lines.append(
                "Бэктест на синтетике: OK — "
                f"сделок {stats.total_trades}, "
                f"доходность {stats.total_return_pct:.2f}%, "
                f"Sharpe {stats.sharpe_ratio:.2f}, "
                f"MaxDD {stats.max_drawdown_pct:.2f}%, "
                f"PF {stats.profit_factor:.2f}, "
                f"WinRate {stats.win_rate_pct:.0f}%"
            )
            if stats.total_trades == 0:
                lines.append(
                    "  ⚠ 0 сделок — логика входов могла не сработать на синтетике "
                    "(проверьте пороги/прогрев индикаторов)."
                )
        except Exception:  # noqa: BLE001
            ok = False
            lines.append("Бэктест: ОШИБКА ВЫПОЛНЕНИЯ:")
            lines.append(traceback.format_exc(limit=4))
        lines.append("")

    print("\n".join(lines).rstrip())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
