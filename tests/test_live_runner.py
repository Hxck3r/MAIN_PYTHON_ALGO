"""Тесты headless-раннера live.runner (без сети и без Qt).

Проверяют сборку EngineConfig из JSON, разбор стратегии, источники ключей
(env vs хранилище) и предохранитель подтверждения реальной торговли.
"""
from __future__ import annotations

import json

import pytest

from live import credentials, runner


def _write_cfg(tmp_path, **over):
    cfg = {
        "exchange": "bingx",
        "mode": "demo",
        "symbol": "BTC/USDT:USDT",
        "timeframe": "4h",
        "strategy": "SmaCrossStrategy",
        "strategy_params": {},
        "leverage": 7,
        "dry_run": False,
        "sizing": {"mode": "balance_pct", "balance_pct": 3.0},
        "risk": {"max_daily_loss_pct": 5.0},
    }
    cfg.update(over)
    p = tmp_path / "trade.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
def test_build_engine_config_maps_all_sections(tmp_path):
    cfg = runner.load_run_config(str(_write_cfg(tmp_path)))
    ec = runner.build_engine_config(cfg)
    assert ec.symbol == "BTC/USDT:USDT"
    assert ec.timeframe == "4h"
    assert ec.leverage == 7
    assert ec.strategy_cls.__name__ == "SmaCrossStrategy"
    assert ec.sizing.mode == "balance_pct"
    assert ec.sizing.balance_pct == pytest.approx(3.0)
    assert ec.risk.max_daily_loss_pct == pytest.approx(5.0)
    # require_protective по умолчанию включён, даже если секции execution нет
    assert ec.execution.require_protective is True


def test_execution_section_overrides_defaults(tmp_path):
    cfg = runner.load_run_config(
        str(_write_cfg(tmp_path, execution={"require_protective": False,
                                            "max_slippage_pct": 0.5}))
    )
    ec = runner.build_engine_config(cfg)
    assert ec.execution.require_protective is False
    assert ec.execution.max_slippage_pct == pytest.approx(0.5)


def test_resolve_strategy_by_registry_name():
    cls = runner.resolve_strategy("SMA Crossover (пересечение скользящих)")
    assert cls.__name__ == "SmaCrossStrategy"


def test_resolve_strategy_unknown_raises():
    with pytest.raises(runner.ConfigError):
        runner.resolve_strategy("НетТакой")


def test_load_run_config_missing_file():
    with pytest.raises(runner.ConfigError):
        runner.load_run_config("/no/such/config.json")


def test_load_run_config_bad_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ не json", encoding="utf-8")
    with pytest.raises(runner.ConfigError):
        runner.load_run_config(str(p))


def test_build_engine_config_requires_symbol():
    with pytest.raises(runner.ConfigError):
        runner.build_engine_config({"strategy": "SmaCrossStrategy"})


# --------------------------------------------------------------------------- #
# Источники ключей
# --------------------------------------------------------------------------- #
def test_resolve_credentials_prefers_env(monkeypatch):
    monkeypatch.setattr(
        credentials, "load", lambda e, m: credentials.Credentials(values={})
    )
    monkeypatch.setenv("ALGO_CRED_API_KEY", "envkey")
    monkeypatch.setenv("ALGO_CRED_SECRET", "envsecret")
    creds = runner.resolve_credentials("bingx", "demo")
    assert creds == {"api_key": "envkey", "secret": "envsecret"}


def test_resolve_credentials_falls_back_to_store(monkeypatch):
    monkeypatch.delenv("ALGO_CRED_API_KEY", raising=False)
    monkeypatch.delenv("ALGO_CRED_SECRET", raising=False)
    monkeypatch.setattr(
        credentials, "load",
        lambda e, m: credentials.Credentials(values={"api_key": "k", "secret": "s"}),
    )
    creds = runner.resolve_credentials("bingx", "demo")
    assert creds == {"api_key": "k", "secret": "s"}


def test_resolve_credentials_missing_required_raises(monkeypatch):
    monkeypatch.delenv("ALGO_CRED_API_KEY", raising=False)
    monkeypatch.delenv("ALGO_CRED_SECRET", raising=False)
    monkeypatch.setattr(
        credentials, "load", lambda e, m: credentials.Credentials(values={})
    )
    with pytest.raises(runner.ConfigError):
        runner.resolve_credentials("bingx", "demo")


# --------------------------------------------------------------------------- #
# Предохранитель подтверждения реала
# --------------------------------------------------------------------------- #
def test_confirm_real_allows_demo_and_dry():
    assert runner._confirm_real("demo", False, False) is True
    assert runner._confirm_real("real", True, False) is True   # dry_run -- безопасно


def test_confirm_real_blocks_noninteractive_without_yes(monkeypatch):
    class _Stdin:
        def isatty(self):
            return False

    monkeypatch.setattr(runner.sys, "stdin", _Stdin())
    assert runner._confirm_real("real", False, False) is False  # нет --yes -> отказ
    assert runner._confirm_real("real", False, True) is True     # --yes -> ок
