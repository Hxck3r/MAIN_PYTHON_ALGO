"""Тесты матричного WFO: сетка длин окон, прогон, плато, кэш IS-оценок."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtester import (
    MATRIX_FIELDS,
    WfaMatrixConfig,
    WfaMatrixResult,
    run_wfa_matrix,
    span_values,
)
from backtester.wfa_matrix import WfaMatrixCell
from strategies import SmaCrossStrategy

GRID = {"fast_period": [5, 10], "slow_period": [20]}


def _matrix_config(**kwargs) -> WfaMatrixConfig:
    start = pd.Timestamp("2022-01-01")
    return WfaMatrixConfig(
        start=start,
        end=start + pd.Timedelta(days=120),
        is_days_list=kwargs.pop("is_days_list", (20, 30)),
        oos_days_list=kwargs.pop("oos_days_list", (10, 15)),
        **kwargs,
    )


# --- span_values ----------------------------------------------------------- #
def test_span_values_geometric_bounds_and_dedup():
    vals = span_values(60, 240, 4)
    assert vals[0] == 60 and vals[-1] == 240
    assert list(vals) == sorted(vals)          # отсортировано
    assert len(set(vals)) == len(vals)         # без дублей
    # геометрия: шаги растут, а не равны
    diffs = np.diff(vals)
    assert diffs[-1] > diffs[0]


def test_span_values_linear_and_edge_cases():
    assert span_values(10, 10, 5) == (10,)     # вырожденный диапазон
    lin = span_values(10, 50, 5, geometric=False)
    assert lin[0] == 10 and lin[-1] == 50
    with pytest.raises(ValueError):
        span_values(0, 10, 3)


def test_from_ranges_builds_lists():
    start = pd.Timestamp("2022-01-01")
    cfg = WfaMatrixConfig.from_ranges(
        start, start + pd.Timedelta(days=365),
        is_range=(60, 180, 3), oos_range=(20, 60, 3),
    )
    assert len(cfg.is_days_list) >= 2
    assert len(cfg.oos_days_list) >= 2
    assert len(cfg.cells()) == len(cfg.is_days_list) * len(cfg.oos_days_list)


# --- прогон матрицы -------------------------------------------------------- #
def test_run_matrix_basic(synth_1h):
    cfg = _matrix_config()
    result = run_wfa_matrix(
        data=synth_1h, strategy_cls=SmaCrossStrategy,
        matrix_config=cfg, grid=GRID, metric="net_profit", min_trades=0,
    )
    assert isinstance(result, WfaMatrixResult)
    # все ячейки сетки присутствуют (прогнанные + пропущенные)
    assert len(result.cells) == len(cfg.cells())
    assert result.ran_cells, "ни одна ячейка не прогналась"
    # матрица имеет форму len(IS) × len(OOS)
    mat = result.matrix()
    assert mat.shape == (len(result.is_axis), len(result.oos_axis))
    # прогнанные ячейки заполнены, пропущенные -- NaN
    for c in result.ran_cells:
        assert c.n_windows >= 2


def test_matrix_skips_oversized_cells():
    # IS=200д + OOS=50д не помещается в 120-дневный период -> ячейка пропущена
    start = pd.Timestamp("2022-01-01")
    cfg = WfaMatrixConfig(
        start=start, end=start + pd.Timedelta(days=120),
        is_days_list=(20, 200), oos_days_list=(10,),
    )
    data = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=pd.date_range(start, periods=120 * 24, freq="1h"),
    )
    result = run_wfa_matrix(
        data=data, strategy_cls=SmaCrossStrategy,
        matrix_config=cfg, grid=GRID, metric="net_profit", min_trades=0,
    )
    big = result.cell(200, 10)
    assert big is not None and not big.ran and big.reason


def test_matrix_all_cells_too_big_raises():
    start = pd.Timestamp("2022-01-01")
    cfg = WfaMatrixConfig(
        start=start, end=start + pd.Timedelta(days=30),
        is_days_list=(200,), oos_days_list=(100,),
    )
    data = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=pd.date_range(start, periods=30 * 24, freq="1h"),
    )
    with pytest.raises(ValueError, match="Ни одна ячейка"):
        run_wfa_matrix(
            data=data, strategy_cls=SmaCrossStrategy,
            matrix_config=cfg, grid=GRID, metric="net_profit", min_trades=0,
        )


def test_matrix_unknown_target_raises(synth_1h):
    with pytest.raises(ValueError, match="целевая метрика"):
        run_wfa_matrix(
            data=synth_1h, strategy_cls=SmaCrossStrategy,
            matrix_config=_matrix_config(), grid=GRID,
            metric="net_profit", target="not_a_field",
        )


# --- кэш IS-оценок: результат не должен зависеть от кэша -------------------- #
def test_cache_does_not_change_results(synth_1h):
    """Последовательный прогон с общим кэшем = независимые прогоны ячеек."""
    cfg = _matrix_config()
    cached = run_wfa_matrix(
        data=synth_1h, strategy_cls=SmaCrossStrategy,
        matrix_config=cfg, grid=GRID, metric="net_profit", min_trades=0,
    )
    # эталон: каждая ячейка как самостоятельный WFA (без общего кэша)
    from backtester import WalkForwardConfig, run_walk_forward

    for c in cached.ran_cells:
        ref = run_walk_forward(
            data=synth_1h, strategy_cls=SmaCrossStrategy,
            config=WalkForwardConfig(
                start=cfg.start, end=cfg.end,
                in_sample_days=c.is_days, out_sample_days=c.oos_days,
            ),
            grid=GRID, metric="net_profit", min_trades=0,
        )
        assert c.oos_return_pct == pytest.approx(
            ref.oos_stats.total_return_pct, rel=1e-9, nan_ok=True
        )
        assert c.n_windows == ref.n_windows


# --- плато / устойчивость -------------------------------------------------- #
def _synthetic_result(target_matrix: dict[tuple[int, int], float]) -> WfaMatrixResult:
    """Строит WfaMatrixResult вручную из словаря (is,oos)->target (wfe_pct)."""
    cells = []
    for (is_d, oos_d), val in target_matrix.items():
        cells.append(WfaMatrixCell(
            is_days=is_d, oos_days=oos_d, ran=True, n_windows=5,
            wfe_pct=val, oos_return_pct=val,  # положит. -> прибыльная ячейка
        ))
    start = pd.Timestamp("2022-01-01")
    cfg = WfaMatrixConfig(
        start=start, end=start + pd.Timedelta(days=365),
        is_days_list=tuple(sorted({k[0] for k in target_matrix})),
        oos_days_list=tuple(sorted({k[1] for k in target_matrix})),
    )
    return WfaMatrixResult(config=cfg, cells=cells, target="wfe_pct")


def test_robust_cell_prefers_plateau_over_spike():
    # сетка 3×3: ровное плато на (20,10) и одинокий пик на (40,30)
    mat = {
        (20, 10): 80.0, (20, 20): 78.0, (20, 30): 10.0,
        (30, 10): 79.0, (30, 20): 77.0, (30, 30): 12.0,
        (40, 10): 11.0, (40, 20): 13.0, (40, 30): 200.0,  # изолированный пик
    }
    res = _synthetic_result(mat)
    best = res.best_cell()
    robust = res.robust_cell()
    assert (best.is_days, best.oos_days) == (40, 30)        # пик -- глобальный max
    # robust выбирает центр плато (высокое среднее по окрестности), не пик
    assert robust.wfe_pct >= 70.0
    assert (robust.is_days, robust.oos_days) != (40, 30)


def test_stability_metrics_ranges():
    mat = {
        (20, 10): 80.0, (20, 20): 78.0,
        (30, 10): 79.0, (30, 20): 77.0,
    }
    res = _synthetic_result(mat)
    st = res.stability()
    assert st.n_ran == 4 and st.n_cells == 4
    assert st.pct_profitable_cells == pytest.approx(100.0)
    assert st.avg_wfe_pct == pytest.approx(78.5, abs=0.1)
    # ровное плато -> низкий CV
    assert math.isfinite(st.target_cv_pct) and st.target_cv_pct < 5.0
    # плато широкое -> высокий процент
    assert st.plateau_pct == pytest.approx(100.0)
    assert "плато" in str(st)


def test_matrix_field_for_lower_is_better():
    # для max_drawdown (меньше -- лучше) robust-ячейка = минимальная просадка
    cells = [
        WfaMatrixCell(20, 10, ran=True, n_windows=5, oos_max_dd_pct=5.0),
        WfaMatrixCell(20, 20, ran=True, n_windows=5, oos_max_dd_pct=6.0),
        WfaMatrixCell(30, 10, ran=True, n_windows=5, oos_max_dd_pct=40.0),
        WfaMatrixCell(30, 20, ran=True, n_windows=5, oos_max_dd_pct=42.0),
    ]
    start = pd.Timestamp("2022-01-01")
    cfg = WfaMatrixConfig(
        start=start, end=start + pd.Timedelta(days=365),
        is_days_list=(20, 30), oos_days_list=(10, 20),
    )
    res = WfaMatrixResult(config=cfg, cells=cells, target="oos_max_dd_pct")
    assert "oos_max_dd_pct" in MATRIX_FIELDS
    best = res.best_cell()
    assert best.oos_max_dd_pct == pytest.approx(5.0)   # минимальная просадка
