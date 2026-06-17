"""Пайплайн обучения и оценки модели.

Поддерживаются два режима валидации:

* **holdout** — один временной сплит train/test с **embargo** на стыке (быстро, для
  предварительной проверки);
* **walk_forward** — честная временная кросс-валидация с purging+embargo
  (:mod:`ml.validation`): метрики усредняются по фолдам, а out-of-fold предсказания
  сшиваются в один непрерывный ряд — его же затем «проигрывает» ML-стратегия в бэктесте.

Дополнительно: калибровка вероятностей и подбор порога решения по выбранной метрике —
это напрямую влияет на качество сигналов в бэктесте.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml import config
from ml import models as model_factory
from ml.domain import FeatureSet, ModelResult
from ml.validation import PurgedWalkForwardSplit

log = logging.getLogger(__name__)

# Прогресс-колбэк: (процент 0..100, сообщение). Возврат False — запрос остановки.
ProgressCb = Optional[Callable[[int, str], Any]]


class TrainingCancelled(Exception):
    """Обучение прервано по запросу (кооперативная отмена из GUI)."""


@dataclass
class TrainConfig:
    model_name: str = field(default_factory=model_factory.default_model_name)
    params: dict[str, Any] | None = None
    train_ratio: float = config.DEFAULT_TRAIN_RATIO
    validation: str = "walk_forward"        # "holdout" | "walk_forward" | "cpcv"
    n_splits: int = 5
    # CPCV (AFML гл. 12): число тест-групп k в комбинации. Групп всего = n_splits.
    # Путей = C(n_splits-1, k-1); обучений = C(n_splits, k). Больше k → больше путей и
    # стоимость. Включается, когда validation == "cpcv".
    cpcv_k: int = 2
    embargo: int = 0                         # доп. зазор в барах сверх горизонта
    balance_classes: bool = True
    calibrate: bool = False                  # калибровать вероятности (Platt/sigmoid)
    optimize_threshold: bool = True
    # Youden's J (max TPR−FPR) — устойчивая рабочая точка для торговли: не вырождается
    # в «всегда лонг» при перекосе классов, в отличие от F1.
    threshold_metric: str = "youden"         # "youden" | "f1"
    # Веса уникальности выборки (AFML гл. 4): компенсируют перекрытие меток при horizon>1.
    use_sample_weights: bool = True
    time_decay_last: float = 1.0             # вес самого старого наблюдения (1.0 — без декея)
    # Честная важность признаков перестановкой на OOS (MDA, AFML гл. 8) — медленнее.
    compute_mda: bool = False
    mda_repeats: int = 3
    # Ранняя остановка LightGBM по валидационному хвосту train (быстрее на слабом ПК,
    # меньше лишних деревьев). Применяется только к некалиброванному LightGBM.
    early_stopping: bool = True
    early_stopping_rounds: int = 30
    # Издержки на оборот (ОДНОсторонние, в базисных пунктах) для честной OOS-оценки
    # сигнала: Sharpe, эквити и дефляция (PSR/DSR) считаются ЗА вычетом издержек ×
    # оборот позиции. 0 — без издержек (оптимистично). Слабый, но «значимый» край на
    # частых данных обычно умирает от costs × turnover — поэтому учитываем сразу.
    cost_bps: float = 0.0
    # Sharpe OOS-сигнала по пробам подбора гиперпараметров — для дефляции (DSR).
    tune_trial_sharpes: list[float] | None = None

    @property
    def cost_frac(self) -> float:
        """Издержки на единицу оборота в долях цены (б.п. → доля)."""
        return max(0.0, float(self.cost_bps)) / 1e4


def _emit(cb: ProgressCb, pct: int, msg: str) -> None:
    """Отправить прогресс; если колбэк вернул False — кооперативно останавливаемся."""
    if cb is not None and cb(pct, msg) is False:
        raise TrainingCancelled()


# --- низкоуровневые помощники -------------------------------------------------
def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Вероятность класса «1» (рост) независимо от порядка классов модели."""
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    col = classes.index(1) if 1 in classes else proba.shape[1] - 1
    return np.asarray(proba[:, col], dtype=float)


def _fit_with_weight(model: Any, X, y, w: np.ndarray | None) -> None:
    """Обучить модель, корректно прокинув ``sample_weight`` сквозь обёртки.

    Маршрут различается: голый estimator и ``CalibratedClassifierCV`` принимают
    ``sample_weight`` напрямую, а ``Pipeline`` — как ``<имя_шага>__sample_weight``.
    Если модель не поддерживает веса — тихий откат на обучение без них.
    """
    if w is None:
        model.fit(X, y)
        return
    from sklearn.pipeline import Pipeline

    try:
        if isinstance(model, Pipeline):
            model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
        else:
            model.fit(X, y, sample_weight=w)
    except (TypeError, ValueError):       # estimator не умеет sample_weight
        model.fit(X, y)


def _fit_lgbm_early_stop(
    cfg: TrainConfig, model: Any, X_tr, y_tr, w_tr: np.ndarray | None, purge_gap: int = 1
) -> bool:
    """Обучить LightGBM с ранней остановкой по валидационному хвосту. True — успех.

    Последние ~15% train (с purge-зазором ``horizon+embargo`` от точки обучения)
    используются как eval_set; число деревьев подбирается автоматически — быстрее и
    устойчивее. При нехватке данных/одном классе в хвосте — откат (False) на обычное
    обучение.
    """
    n = len(X_tr)
    val = max(50, int(n * 0.15))
    gap = max(1, purge_gap)                          # purge-зазор fit↔val (без утечки)
    fit_end = n - val - gap
    if fit_end < 100:
        return False
    X_fit, y_fit = X_tr.iloc[:fit_end], y_tr.iloc[:fit_end]
    X_val, y_val = X_tr.iloc[n - val:], y_tr.iloc[n - val:]
    if y_fit.nunique() < 2 or y_val.nunique() < 2:
        return False
    try:
        import lightgbm as lgb

        kw: dict[str, Any] = {
            "eval_set": [(X_val, y_val)],
            "callbacks": [lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        }
        if w_tr is not None:
            kw["sample_weight"] = w_tr[:fit_end]
        model.fit(X_fit, y_fit, **kw)
        return True
    except Exception:  # noqa: BLE001 — ранняя остановка не критична
        return False


def _fit_calibrated_temporal(
    model: Any, X_tr, y_tr, w_tr: np.ndarray | None, purge_gap: int
) -> Any | None:
    """Калибровка вероятностей (Platt/sigmoid) БЕЗ утечки во времени.

    Обычный ``CalibratedClassifierCV(cv=k)`` перемешивает наблюдения (random KFold) —
    на временном ряду это заглядывание в будущее, а калиброванная вероятность затем
    управляет размером ставки. Здесь честно по времени: базовая модель учится на
    «голове» train, а сигмоидный калибратор подгоняется на её «хвосте» с purge-зазором
    (``horizon+embargo``). Возвращает обёртку или ``None`` (данных мало — откат).
    """
    n = len(X_tr)
    calib = max(50, int(n * 0.3))
    gap = max(1, purge_gap)
    fit_end = n - calib - gap
    if fit_end < 100:
        return None
    y_fit, y_cal = y_tr.iloc[:fit_end], y_tr.iloc[n - calib:]
    if y_fit.nunique() < 2 or y_cal.nunique() < 2:
        return None
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator
    except Exception:  # noqa: BLE001 — старый sklearn без FrozenEstimator
        return None
    w_fit = w_tr[:fit_end] if w_tr is not None else None
    _fit_with_weight(model, X_tr.iloc[:fit_end], y_fit, w_fit)
    try:
        # FrozenEstimator замораживает уже обученную базу — fit лишь подгоняет калибратор
        # на отложенном хвосте (база не переобучается, утечки нет).
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
        cal.fit(X_tr.iloc[n - calib:], y_cal)
    except Exception:  # noqa: BLE001 — калибровка не критична
        return None
    return cal


def _fit_predict(
    cfg: TrainConfig, X_tr, y_tr, X_te, w_tr: np.ndarray | None = None, purge_gap: int = 1
) -> tuple[Any, np.ndarray]:
    """Обучить модель на train-фолде и вернуть (модель, вероятности на test)."""
    model = model_factory.build_model(
        cfg.model_name, cfg.params, balance_classes=cfg.balance_classes
    )
    calibrate = cfg.calibrate and len(X_tr) >= 100 and y_tr.nunique() == 2
    # Ранняя остановка имеет смысл только для голого LightGBM (калибровочная обёртка
    # сама делает CV и несовместима с eval_set).
    if (cfg.early_stopping and not calibrate and cfg.model_name == "lightgbm"
            and _fit_lgbm_early_stop(cfg, model, X_tr, y_tr, w_tr, purge_gap)):
        return model, _predict_proba(model, X_te)
    if calibrate:
        cal = _fit_calibrated_temporal(model, X_tr, y_tr, w_tr, purge_gap)
        if cal is not None:
            return cal, _predict_proba(cal, X_te)
        log.warning("Калибровка пропущена: мало данных для honest temporal split — обучаю без неё.")
    _fit_with_weight(model, X_tr, y_tr, w_tr)
    return model, _predict_proba(model, X_te)


def _sample_weights(fs: FeatureSet, cfg: TrainConfig) -> np.ndarray | None:
    """Веса уникальности по горизонтам меток (None — если отключены/нет данных)."""
    if not cfg.use_sample_weights or fs.event_end is None:
        return None
    from ml.sample_weights import concurrency_weights, time_decay

    w = concurrency_weights(fs.event_end, len(fs.frame))
    if cfg.time_decay_last < 1.0:
        w = time_decay(w, cfg.time_decay_last)
    return w


def _oos_side(fs: FeatureSet, index: pd.Index) -> pd.Series | None:
    """Сторона первичного сигнала на OOS-барах (мета-разметка) или None."""
    if not fs.side_column or fs.side_column not in fs.frame.columns:
        return None
    return fs.frame[fs.side_column].reindex(index).rename("oos_side")


def _signal_returns(
    fwd1: pd.Series, index: pd.Index, proba, thr: float, cost: float = 0.0
) -> pd.Series:
    """OOS-доходность «лонг/флэт» сигнала: позиция × доходность след. бара − издержки.

    ``cost`` — издержки на единицу оборота (доля цены); вычитаются при каждой смене
    позиции (вход/выход), оборот = ``|Δ позиции|``. ``cost=0`` — валовая доходность.
    """
    pos = (np.asarray(proba, dtype=float) >= thr).astype(float)
    r = np.nan_to_num(fwd1.reindex(index).to_numpy(dtype=float), nan=0.0)
    ret = pos * r
    if cost > 0.0:
        turnover = np.abs(np.diff(pos, prepend=0.0))   # вход из кэша/выход = оборот 1
        ret = ret - turnover * cost
    return pd.Series(ret, index=index, name="oos_returns")


def _sharpe(returns) -> float:
    """Безразмерный Sharpe за бар (mean/std); NaN при вырожденном ряде."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float("nan")
    s = arr.std(ddof=1)
    return float(arr.mean() / s) if s > 0 else float("nan")


def _permutation_importance_fold(
    model: Any, X_te, y_te, columns: list[str], n_repeats: int
) -> np.ndarray | None:
    """Падение ROC-AUC при перестановке каждого признака на тест-фолде (MDA)."""
    if len(np.unique(np.asarray(y_te))) < 2:
        return None
    from sklearn.inspection import permutation_importance

    try:
        res = permutation_importance(
            model, X_te, y_te, scoring="roc_auc",
            n_repeats=max(1, n_repeats), random_state=42, n_jobs=1,
        )
    except Exception:  # noqa: BLE001 — важность не критична
        return None
    vals = np.asarray(res.importances_mean, dtype=float)
    return vals if len(vals) == len(columns) else None


def _evaluate(y_true, y_pred, y_proba) -> dict[str, float]:
    """Базовый набор метрик классификации (часть — независимы от порога)."""
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        m["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        m["roc_auc"] = float("nan")
    return m


def _optimize_threshold(y_true, y_proba, metric: str) -> float:
    """Подобрать порог вероятности, максимизирующий выбранную метрику.

    Порог подбирается на out-of-fold предсказаниях — это слегка оптимистично, поэтому
    «честной» метрикой остаётся ROC-AUC (от порога не зависит). Порог нужен скорее как
    разумная стартовая точка входа в позицию для бэктеста.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    if metric == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_proba)
        j = tpr - fpr
        best = thr[int(np.argmax(j))]
        return float(np.clip(best, 0.05, 0.95))
    # f1 по сетке порогов
    grid = np.linspace(0.30, 0.70, 41)
    best_thr, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(t), float(f1)
    return best_thr


def _normalize_importances(columns: list[str], vals: np.ndarray) -> dict[str, float]:
    """Нормировать важности к сумме 1 (клипуя отрицательные) и отсортировать по убыванию."""
    vals = np.clip(np.asarray(vals, dtype=float), 0.0, None)
    if len(vals) != len(columns):
        return {}
    total = vals.sum()
    if total > 0:
        vals = vals / total
    return dict(sorted(zip(columns, vals), key=lambda kv: kv[1], reverse=True))


def _feature_importances(model: Any, columns: list[str]) -> dict[str, float]:
    """Нормированные важности из дерева (feature_importances_) или линейной |coef|."""
    est = model_factory.final_estimator(model)
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        vals = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
    else:
        return {}
    return _normalize_importances(columns, vals)


def _assemble(
    cfg: TrainConfig,
    fs: FeatureSet,
    final_model: Any,
    y_true: pd.Series,
    y_proba: pd.Series,
    threshold: float,
    fold_metrics: list[dict],
    extra: dict[str, float],
    oos_returns: pd.Series | None = None,
    importances_mda: dict[str, float] | None = None,
    oos_fwd: pd.Series | None = None,
    oos_side: pd.Series | None = None,
) -> ModelResult:
    """Собрать ModelResult из сшитых out-of-sample предсказаний."""
    y_pred = (y_proba >= threshold).astype(int)
    metrics = _evaluate(y_true, y_pred, y_proba)
    metrics.update(extra)
    metrics["threshold"] = float(threshold)
    metrics["cost_bps"] = float(cfg.cost_bps)          # издержки, заложенные в OOS-оценку
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    importances = _feature_importances(final_model, fs.feature_columns)
    spec = model_factory.MODEL_SPECS[cfg.model_name]
    log.info(
        "Обучение (%s, %s) завершено: acc=%.3f f1=%.3f auc=%.3f thr=%.2f",
        spec.label, cfg.validation, metrics["accuracy"], metrics["f1"],
        metrics["roc_auc"], threshold,
    )
    return ModelResult(
        model=final_model,
        model_name=cfg.model_name,
        feature_columns=fs.feature_columns,
        metrics=metrics,
        confusion=cm,
        importances=importances,
        test_index=y_true.index,
        y_true=pd.Series(y_true.values, index=y_true.index, name="y_true"),
        y_pred=pd.Series(np.asarray(y_pred), index=y_true.index, name="y_pred"),
        y_proba=pd.Series(np.asarray(y_proba.values), index=y_true.index, name="y_proba"),
        validation=cfg.validation,
        threshold=float(threshold),
        calibrated=cfg.calibrate,
        fold_metrics=fold_metrics,
        importances_mda=importances_mda or {},
        oos_returns=oos_returns if oos_returns is not None else pd.Series(dtype=float),
        oos_fwd=oos_fwd if oos_fwd is not None else pd.Series(dtype=float),
        oos_side=oos_side if oos_side is not None else pd.Series(dtype=float),
    )


# --- режимы валидации ---------------------------------------------------------
def _train_holdout(fs: FeatureSet, cfg: TrainConfig, progress: ProgressCb) -> ModelResult:
    frame = fs.frame
    X = frame[fs.feature_columns]
    y = frame[fs.target_column].astype(int)
    n = len(frame)
    fwd1 = frame["close"].pct_change().shift(-1)     # доходность следующего бара (для OOS-pnl)
    weights = _sample_weights(fs, cfg)

    split = int(n * cfg.train_ratio)
    gap = max(1, fs.horizon) + max(0, cfg.embargo)   # embargo на стыке train/test
    train_end = split - gap
    if train_end < 10 or split > n - 10:
        raise ValueError("Неудачное соотношение train/test — измените долю обучения.")

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[split:], y.iloc[split:]
    w_train = weights[:train_end] if weights is not None else None
    _emit(progress, 15, f"Holdout: train={len(X_train)}, test={len(X_test)} (embargo={gap})")
    if y_train.nunique() < 2:
        raise ValueError("В обучающей выборке один класс — измените порог/горизонт target.")

    _emit(progress, 35, "Обучение модели…")
    model, proba = _fit_predict(cfg, X_train, y_train, X_test, w_train, gap)
    proba_s = pd.Series(proba, index=X_test.index)
    _emit(progress, 80, "Оценка и подбор порога…")
    thr = _optimize_threshold(y_test, proba_s, cfg.threshold_metric) if cfg.optimize_threshold else 0.5

    oos_returns = _signal_returns(fwd1, X_test.index, proba, thr, cfg.cost_frac)
    oos_fwd = fwd1.reindex(X_test.index).rename("oos_fwd")
    mda: dict[str, float] = {}
    if cfg.compute_mda:
        _emit(progress, 92, "Честная важность признаков (MDA)…")
        vals = _permutation_importance_fold(model, X_test, y_test, fs.feature_columns, cfg.mda_repeats)
        if vals is not None:
            mda = _normalize_importances(fs.feature_columns, vals)
    _emit(progress, 100, "Готово.")
    return _assemble(
        cfg, fs, model, y_test, proba_s, thr, fold_metrics=[],
        extra={"train_size": float(len(X_train)), "test_size": float(len(X_test)),
               "oos_sharpe": _sharpe(oos_returns)},
        oos_returns=oos_returns, importances_mda=mda, oos_fwd=oos_fwd,
        oos_side=_oos_side(fs, X_test.index),
    )


def _train_walk_forward(fs: FeatureSet, cfg: TrainConfig, progress: ProgressCb) -> ModelResult:
    frame = fs.frame
    X = frame[fs.feature_columns]
    y = frame[fs.target_column].astype(int)
    fwd1 = frame["close"].pct_change().shift(-1)     # доходность следующего бара (для OOS-pnl)
    weights = _sample_weights(fs, cfg)

    purge_gap = max(1, fs.horizon) + max(0, cfg.embargo)
    splitter = PurgedWalkForwardSplit(
        n_splits=cfg.n_splits, horizon=max(1, fs.horizon), embargo=cfg.embargo
    )
    splits = list(splitter.split(X))
    if not splits:
        raise ValueError("Не удалось сформировать walk-forward фолды — мало данных.")

    proba_parts: list[pd.Series] = []
    true_parts: list[pd.Series] = []
    fold_metrics: list[dict] = []
    mda_acc = np.zeros(len(fs.feature_columns))      # накопитель MDA-важности по фолдам
    mda_weight = 0.0
    for i, (tr, te) in enumerate(splits):
        pct = 10 + int(70 * i / len(splits))
        _emit(progress, pct, f"Фолд {i + 1}/{len(splits)}: train={len(tr)}, test={len(te)}")
        y_tr = y.iloc[tr]
        if y_tr.nunique() < 2:
            log.warning("Фолд %d пропущен: в train один класс.", i + 1)
            continue
        w_tr = weights[tr] if weights is not None else None
        model, proba = _fit_predict(cfg, X.iloc[tr], y_tr, X.iloc[te], w_tr, purge_gap)
        idx = X.iloc[te].index
        proba_s = pd.Series(proba, index=idx)
        y_te = y.iloc[te]
        proba_parts.append(proba_s)
        true_parts.append(y_te)
        fm = _evaluate(y_te, (proba >= 0.5).astype(int), proba)
        fm["fold"] = i + 1
        fm["test_size"] = float(len(te))
        fm["sharpe"] = _sharpe(_signal_returns(fwd1, idx, proba, 0.5, cfg.cost_frac))
        fold_metrics.append(fm)
        if cfg.compute_mda:
            vals = _permutation_importance_fold(model, X.iloc[te], y_te, fs.feature_columns, cfg.mda_repeats)
            if vals is not None:
                mda_acc += vals * len(te)
                mda_weight += len(te)

    if not proba_parts:
        raise ValueError("Все фолды вырождены (один класс в train) — измените target.")

    proba_all = pd.concat(proba_parts).sort_index()
    y_all = pd.concat(true_parts).sort_index()
    thr = _optimize_threshold(y_all, proba_all, cfg.threshold_metric) if cfg.optimize_threshold else 0.5
    oos_returns = _signal_returns(fwd1, proba_all.index, proba_all.to_numpy(), thr, cfg.cost_frac)
    oos_fwd = fwd1.reindex(proba_all.index).rename("oos_fwd")
    mda = _normalize_importances(fs.feature_columns, mda_acc / mda_weight) if mda_weight > 0 else {}

    _emit(progress, 88, "Финальная модель на всех данных…")
    # Для сохранения/важностей дообучаем модель на всей доступной истории.
    final_model, _ = _fit_predict(cfg, X, y, X.iloc[-1:], weights, purge_gap)

    aucs = [fm["roc_auc"] for fm in fold_metrics if fm["roc_auc"] == fm["roc_auc"]]
    extra = {
        "n_folds": float(len(fold_metrics)),
        "cv_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "cv_auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "oos_size": float(len(y_all)),
        "oos_sharpe": _sharpe(oos_returns),
    }
    _emit(progress, 100, "Готово.")
    return _assemble(cfg, fs, final_model, y_all, proba_all, thr, fold_metrics, extra,
                     oos_returns=oos_returns, importances_mda=mda, oos_fwd=oos_fwd,
                     oos_side=_oos_side(fs, proba_all.index))


def _train_cpcv(fs: FeatureSet, cfg: TrainConfig, progress: ProgressCb) -> ModelResult:
    """Комбинаторно-очищенная CV (AFML гл. 12): распределение OOS-Sharpe по путям.

    В отличие от walk-forward (один OOS-путь) строит ``C(N-1, k-1)`` backtest-путей и
    усреднённый по сочетаниям OOS-ряд вероятностей (его и «проигрывает» MLStrategy).
    Распределение Sharpe по путям → честная робастность края.
    """
    from ml.cpcv import CombinatorialPurgedCV, assemble_paths, make_groups

    frame = fs.frame
    X = frame[fs.feature_columns]
    y = frame[fs.target_column].astype(int)
    fwd1 = frame["close"].pct_change().shift(-1)
    weights = _sample_weights(fs, cfg)
    cost = cfg.cost_frac
    n = len(frame)
    purge_gap = max(1, fs.horizon) + max(0, cfg.embargo)

    cv = CombinatorialPurgedCV(
        n_groups=cfg.n_splits, test_groups=cfg.cpcv_k,
        horizon=max(1, fs.horizon), embargo=cfg.embargo,
    )
    groups = make_groups(n, cfg.n_splits)
    N = len(groups)
    combos = list(cv.split(X))
    if not combos:
        raise ValueError("Не удалось сформировать CPCV-сочетания — мало данных/групп.")

    sum_proba = np.zeros(n)
    cnt = np.zeros(n)
    preds_by_group: dict[int, list[np.ndarray]] = {g: [] for g in range(N)}
    fold_metrics: list[dict] = []
    mda_acc = np.zeros(len(fs.feature_columns))
    mda_weight = 0.0

    for i, (tr, te, combo) in enumerate(combos):
        pct = 8 + int(72 * i / len(combos))
        _emit(progress, pct, f"CPCV-сочетание {i + 1}/{len(combos)} (групп в тесте: {len(combo)})")
        y_tr = y.iloc[tr]
        if y_tr.nunique() < 2:
            log.warning("Сочетание %d пропущено: в train один класс.", i + 1)
            continue
        w_tr = weights[tr] if weights is not None else None
        model, proba = _fit_predict(cfg, X.iloc[tr], y_tr, X.iloc[te], w_tr, purge_gap)
        proba_by_pos = dict(zip(te.tolist(), np.asarray(proba, dtype=float)))
        sum_proba[te] += np.asarray(proba, dtype=float)
        cnt[te] += 1.0
        for g in combo:                              # разложить предсказания по группам
            s, e = groups[g]
            preds_by_group[g].append(np.array([proba_by_pos[p] for p in range(s, e)]))
        y_te = y.iloc[te]
        fm = _evaluate(y_te, (np.asarray(proba) >= 0.5).astype(int), proba)
        fm["fold"] = i + 1
        fm["test_size"] = float(len(te))
        fm["sharpe"] = _sharpe(_signal_returns(fwd1, X.iloc[te].index, proba, 0.5, cost))
        fold_metrics.append(fm)
        if cfg.compute_mda:
            vals = _permutation_importance_fold(model, X.iloc[te], y_te, fs.feature_columns, cfg.mda_repeats)
            if vals is not None:
                mda_acc += vals * len(te)
                mda_weight += len(te)

    covered = cnt > 0
    if not covered.any():
        raise ValueError("Все CPCV-сочетания вырождены (один класс в train) — измените target.")

    # Усреднённый по сочетаниям OOS-ряд вероятностей (де-шумлённый) — единый честный OOS.
    avg_proba = sum_proba[covered] / cnt[covered]
    idx = frame.index[covered]
    y_all = pd.Series(y.to_numpy()[covered], index=idx)
    proba_all = pd.Series(avg_proba, index=idx)
    thr = _optimize_threshold(y_all, proba_all, cfg.threshold_metric) if cfg.optimize_threshold else 0.5
    oos_returns = _signal_returns(fwd1, idx, avg_proba, thr, cost)
    oos_fwd = fwd1.reindex(idx).rename("oos_fwd")
    mda = _normalize_importances(fs.feature_columns, mda_acc / mda_weight) if mda_weight > 0 else {}

    # --- Распределение Sharpe по backtest-путям -------------------------------
    paths = assemble_paths(preds_by_group, N)
    path_sharpes: list[float] = []
    for path_proba in paths:
        r = _signal_returns(fwd1, frame.index, path_proba, thr, cost)
        sh = _sharpe(r)
        if np.isfinite(sh):
            path_sharpes.append(float(sh))

    _emit(progress, 88, "Финальная модель на всех данных…")
    final_model, _ = _fit_predict(cfg, X, y, X.iloc[-1:], weights, purge_gap)

    aucs = [fm["roc_auc"] for fm in fold_metrics if fm["roc_auc"] == fm["roc_auc"]]
    extra = {
        "n_folds": float(len(fold_metrics)),
        "cv_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "cv_auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "oos_size": float(len(y_all)),
        "oos_sharpe": _sharpe(oos_returns),
        "cpcv_n_paths": float(len(path_sharpes)),
        "cpcv_n_combos": float(len(combos)),
    }
    if path_sharpes:
        arr = np.asarray(path_sharpes)
        extra.update({
            "cpcv_sharpe_mean": float(arr.mean()),
            "cpcv_sharpe_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "cpcv_sharpe_min": float(arr.min()),
            "cpcv_prob_positive": float(np.mean(arr > 0)),   # доля путей с краем > 0
        })
    _emit(progress, 100, "Готово.")
    mr = _assemble(cfg, fs, final_model, y_all, proba_all, thr, fold_metrics, extra,
                   oos_returns=oos_returns, importances_mda=mda, oos_fwd=oos_fwd,
                   oos_side=_oos_side(fs, idx))
    mr.cpcv_sharpes = path_sharpes
    return mr


def _attach_significance(mr: ModelResult, fs: FeatureSet) -> None:
    """Добавить в метрики значимость AUC (DeLong CI) и IC (корреляция с доходностью).

    Это честный ответ на «есть ли преимущество»: высокий AUC без значимости —
    вероятная случайность на коротком ряду.
    """
    from ml.significance import auc_ci, information_coefficient

    _, lo, hi, p = auc_ci(mr.y_true.to_numpy(), mr.y_proba.to_numpy())
    mr.metrics["auc_ci_low"] = lo
    mr.metrics["auc_ci_high"] = hi
    mr.metrics["auc_pvalue"] = p

    # Будущая доходность на горизонте таргета, выровненная по OOS-индексу.
    close = fs.frame["close"]
    h = max(1, fs.horizon)
    fwd = (close.shift(-h) / close - 1.0).reindex(mr.test_index)
    ic, ic_p = information_coefficient(mr.y_proba, fwd)
    mr.metrics["ic"] = ic
    mr.metrics["ic_pvalue"] = ic_p

    # Диагностика данных/таргета: помогает понять, дело в модели или в данных.
    from ml.diagnostics import data_target_diagnostics

    mr.metrics.update(data_target_diagnostics(fs))


def _attach_deflation(mr: ModelResult, cfg: TrainConfig) -> None:
    """PSR/DSR OOS-сигнала: пережил ли край множественное тестирование при подборе.

    Переиспользует канонический Deflated Sharpe из ``backtester.statistics``. При
    подборе гиперпараметров (``tune_trial_sharpes``) порог шума строится на разбросе
    Sharpe по пробам — честная поправка на «лучший из N» (Bailey & López de Prado).
    """
    rets = mr.oos_returns
    if rets is None or len(rets) < 20:
        return
    arr = rets.to_numpy(dtype=float)
    if not np.isfinite(arr).any() or np.nanstd(arr) == 0:
        return
    from backtester.statistics import deflated_sharpe_ratio, probabilistic_sharpe_ratio

    mr.metrics["psr"] = float(probabilistic_sharpe_ratio(arr, 0.0))
    ts = [s for s in (cfg.tune_trial_sharpes or []) if np.isfinite(s)]
    n_trials = len(ts)
    if n_trials > 1:
        res = deflated_sharpe_ratio(arr, trial_scores=np.asarray(ts),
                                    n_trials=n_trials, trial_sharpes=np.asarray(ts))
        mr.metrics["dsr"] = float(res.dsr)
        mr.metrics["n_trials"] = float(n_trials)


def train(fs: FeatureSet, cfg: TrainConfig, progress: ProgressCb = None) -> ModelResult:
    """Обучить модель на :class:`FeatureSet` и вернуть :class:`ModelResult`."""
    if len(fs.frame) < 80:
        raise ValueError(f"Слишком мало строк для обучения: {len(fs.frame)}.")
    if cfg.validation == "cpcv":
        mr = _train_cpcv(fs, cfg, progress)
    elif cfg.validation == "walk_forward":
        mr = _train_walk_forward(fs, cfg, progress)
    else:
        mr = _train_holdout(fs, cfg, progress)
    _attach_significance(mr, fs)
    _attach_deflation(mr, cfg)
    return mr
