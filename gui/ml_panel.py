"""Вкладка «🧠 Машинное обучение»: обучение классификатора и честная OOS-оценка.

Профессиональный, но понятный путь: выбрать пресет признаков → цель → модель →
нажать «Обучить». Под капотом — purged walk-forward кросс-валидация (защита от
заглядывания в будущее и переобучения). Результат:

* таблица OOS-метрик (AUC по фолдам ± разброс, accuracy/precision/recall/F1, порог);
* столбчатая важность признаков;
* стабильность AUC по фолдам;
* **строка-вердикт** (светофор) — простыми словами предупреждает о переобучении.

Кнопка «→ В стратегию» подставляет обученную модель в ``MLStrategy`` и запускает
обычный бэктест, после чего к ней применимы Монте-Карло / PBO / Walk-Forward.

Панель НЕ импортирует sklearn/lightgbm при создании — тяжёлые зависимости
подтягиваются лениво при первом открытии вкладки (``ensure_models_loaded``), чтобы
старт приложения оставался быстрым на слабом ПК.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtCore
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .chart_widget import BG, make_grid_item

COLOR_GOOD = "#16c784"
COLOR_BAD = "#ff4d4d"
COLOR_WARN = "#ff9f43"
BAR_BRUSH = (79, 156, 247, 200)
CHANCE = "#888888"

PRESETS = [("Лёгкий (быстро)", "light"), ("Стандарт", "standard"),
           ("Расширенный", "extended")]
TARGETS = [("Направление цены", "direction"), ("Тройной барьер", "triple_barrier")]
VALIDATIONS = [("Walk-forward (честно)", "walk_forward"), ("Holdout (быстро)", "holdout")]


def _as_1d_array(series) -> Optional[np.ndarray]:
    """Привести Series/массив к 1-D float ndarray; None, если пусто/не задано."""
    if series is None:
        return None
    arr = np.asarray(getattr(series, "to_numpy", lambda: series)(), dtype=float).ravel()
    return arr if arr.size else None


def _sharpe(returns: np.ndarray) -> float:
    """Безразмерный Sharpe за бар (mean/std); NaN при вырожденном ряде."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    s = arr.std(ddof=1)
    return float(arr.mean() / s) if s > 0 else float("nan")


def verdict(metrics: dict) -> tuple[str, str]:
    """Честный вердикт: значимость преимущества (а не голый AUC) + устойчивость.

    На финансах AUC 0.50–0.55 — норма; решает не величина, а статистическая
    значимость (нижняя граница DeLong-интервала > 0.5 или значимый IC) и
    стабильность по фолдам. Слабый, но значимый и устойчивый край — это «есть
    преимущество»; высокий, но незначимый AUC — случайность.
    """
    auc = metrics.get("cv_auc_mean", metrics.get("roc_auc", float("nan")))
    std = metrics.get("cv_auc_std", 0.0)
    lo = metrics.get("auc_ci_low", float("nan"))
    ic = metrics.get("ic", float("nan"))
    icp = metrics.get("ic_pvalue", float("nan"))
    if not np.isfinite(auc):
        return "Недостаточно данных для оценки качества.", CHANCE

    sig_auc = np.isfinite(lo) and lo > 0.5
    sig_ic = np.isfinite(icp) and icp < 0.05 and np.isfinite(ic) and ic > 0
    if not (sig_auc or sig_ic):
        lo_txt = f", нижняя граница {lo:.3f}" if np.isfinite(lo) else ""
        return (f"⚠ Преимущество над случайностью статистически НЕ значимо "
                f"(AUC {auc:.3f}{lo_txt}). Это не приговор модели: попробуйте "
                "тройной барьер, больший горизонт, календарные/режимные признаки, "
                "подбор гиперпараметров или больше данных.", COLOR_WARN)
    if np.isfinite(std) and std > 0.10:
        return (f"⚠ Край значим, но нестабилен по фолдам (AUC {auc:.3f}±{std:.3f}) — "
                "риск переобучения. Упростите модель/признаки или добавьте данных.",
                COLOR_WARN)
    # Дефляция: при подборе гиперпараметров (n_trials>1) учитываем множественное
    # тестирование — высокий, но не пере­живший дефляцию Sharpe вероятен из-за перебора.
    dsr = metrics.get("dsr", float("nan"))
    n_trials = metrics.get("n_trials", 0.0)
    if n_trials > 1 and np.isfinite(dsr) and dsr < 0.95:
        return (f"⚠ Край не пережил множественное тестирование (DSR {dsr:.2f} при "
                f"{int(n_trials)} пробах подбора). Вероятен артефакт перебора — "
                "сократите число проб, упростите модель или добавьте данных.", COLOR_WARN)
    strength = "уверенное" if auc >= 0.55 else "слабое, но устойчивое и значимое"
    ic_txt = f", IC {ic:+.3f}" if np.isfinite(ic) else ""
    return (f"✅ {strength} преимущество вне обучения (AUC {auc:.3f}±{std:.3f}{ic_txt}). "
            "Подтвердите бэктестом и Монте-Карло/PBO.", COLOR_GOOD)


class MlSummaryTable(QTableWidget):
    """Двухколоночная сводка OOS-метрик модели."""

    def __init__(self, parent=None) -> None:
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Метрика", "Значение"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)

    def show_result(self, mr) -> None:
        m = mr.metrics
        cm = mr.confusion
        auc = m.get("cv_auc_mean", m.get("roc_auc", float("nan")))
        std = m.get("cv_auc_std", 0.0)

        def auc_color(v: float) -> Optional[str]:
            if not np.isfinite(v):
                return None
            return COLOR_GOOD if v >= 0.55 else (COLOR_WARN if v >= 0.52 else COLOR_BAD)

        from ml.diagnostics import vr_label

        lo = m.get("auc_ci_low", float("nan"))
        hi = m.get("auc_ci_high", float("nan"))
        pval = m.get("auc_pvalue", float("nan"))
        ic = m.get("ic", float("nan"))
        icp = m.get("ic_pvalue", float("nan"))
        baseline = m.get("baseline_acc", float("nan"))
        acc = m.get("accuracy", float("nan"))
        vr = m.get("variance_ratio", float("nan"))

        def sig_color(p: float) -> Optional[str]:
            if not np.isfinite(p):
                return None
            return COLOR_GOOD if p < 0.05 else (COLOR_WARN if p < 0.15 else COLOR_BAD)

        rows: list[tuple[str, str, Optional[str]]] = [
            ("OOS ROC-AUC (по фолдам)", f"{auc:.3f} ± {std:.3f}", auc_color(auc)),
            ("AUC 95% дов. интервал",
             f"[{lo:.3f}; {hi:.3f}]" if np.isfinite(lo) else "—",
             COLOR_GOOD if (np.isfinite(lo) and lo > 0.5) else COLOR_WARN),
            ("Значимость AUC (p<0.05 — есть край)",
             f"p={pval:.3f}" if np.isfinite(pval) else "—", sig_color(pval)),
            ("IC (вероятность ↔ доходность)",
             f"{ic:+.3f} (p={icp:.3f})" if np.isfinite(ic) else "—", sig_color(icp)),
            ("— Робастность сигнала —", "", None),
            ("OOS Sharpe сигнала (за бар)",
             f"{m['oos_sharpe']:.3f}" if np.isfinite(m.get("oos_sharpe", float("nan"))) else "—",
             None),
            ("PSR (P[Sharpe>0])",
             f"{m['psr']:.3f}" if np.isfinite(m.get("psr", float("nan"))) else "—",
             sig_color(1.0 - m["psr"]) if np.isfinite(m.get("psr", float("nan"))) else None),
            ("DSR (с поправкой на перебор)",
             f"{m['dsr']:.3f} ({int(m.get('n_trials', 0))} проб)"
             if np.isfinite(m.get("dsr", float("nan"))) else "—",
             (COLOR_GOOD if m.get("dsr", 0.0) >= 0.95 else COLOR_WARN)
             if np.isfinite(m.get("dsr", float("nan"))) else None),
            ("— Порог и валидация —", "", None),
            ("Порог входа", f"{mr.threshold:.3f}", None),
            ("Валидация",
             f"{mr.validation}, фолдов {int(m.get('n_folds', 0)) or '—'}", None),
            ("Калибровка вероятностей", "да" if mr.calibrated else "нет", None),
            ("— Классификация (OOS) —", "", None),
            ("Accuracy vs baseline",
             f"{acc:.3f} vs {baseline:.3f}" if np.isfinite(baseline) else f"{acc:.3f}",
             COLOR_GOOD if (np.isfinite(baseline) and acc > baseline + 0.005) else None),
            ("Precision / Recall",
             f"{m.get('precision', float('nan')):.3f} / {m.get('recall', float('nan')):.3f}", None),
            ("F1", f"{m.get('f1', float('nan')):.3f}", None),
            ("— Диагностика данных —", "", None),
            ("Наблюдений (OOS / всего)",
             f"{int(m.get('oos_size', len(mr.y_true)))} / {int(m.get('n_samples', 0)) or '—'}", None),
            ("Баланс классов (доля роста)",
             f"{m.get('pos_rate', float('nan')) * 100:.1f}%", None),
            ("Режим ряда (VR, q=5)",
             f"{vr:.2f} — {vr_label(vr)}" if np.isfinite(vr) else "—", None),
        ]
        if cm and len(cm) == 2:
            rows.append(("Матрица ошибок [TN FP / FN TP]",
                         f"{cm[0][0]} {cm[0][1]} / {cm[1][0]} {cm[1][1]}", None))

        self.setRowCount(len(rows))
        for row, (name, value, color) in enumerate(rows):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if not value:
                name_item.setForeground(QColor("#888888"))
            value_item = QTableWidgetItem(value)
            value_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if color:
                value_item.setForeground(QColor(color))
            self.setItem(row, 0, name_item)
            self.setItem(row, 1, value_item)
        self.resizeColumnsToContents()


class MlPanel(QWidget):
    """Вкладка обучения ML-модели."""

    #: запрос на обучение со словарём настроек; main_window запускает MlTrainingWorker
    train_requested = Signal(dict)
    save_requested = Signal()
    load_requested = Signal()
    apply_requested = Signal()      # → В стратегию

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._models_loaded = False
        self._has_result = False
        # OOS-массивы для интерактивного пересчёта по порогу (заполняются в show_result).
        self._yt: Optional[np.ndarray] = None     # истинные метки {0,1}
        self._yp: Optional[np.ndarray] = None     # вероятности P(рост)
        self._fwd: Optional[np.ndarray] = None    # доходность след. бара на OOS-баре
        self._roc_marker = None                   # подвижная операционная точка на ROC

        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # --- левая колонка: настройки + сводка ---------------------------- #
        self.combo_preset = QComboBox()
        for label, key in PRESETS:
            self.combo_preset.addItem(label, key)
        self.combo_preset.setCurrentIndex(1)  # Стандарт
        self.combo_preset.setToolTip(
            "Набор признаков. «Лёгкий» — минимум индикаторов: быстрее и устойчивее "
            "на слабом ПК. «Расширенный» — больше признаков (нужно больше данных)."
        )

        feat_box = QGroupBox("Признаки")
        feat_form = QFormLayout(feat_box)
        feat_form.addRow("Пресет:", self.combo_preset)
        self.chk_calendar = QCheckBox("Календарные (час/день недели)")
        self.chk_calendar.setChecked(True)
        self.chk_calendar.setToolTip(
            "Час дня и день недели (циклическое кодирование). Часто несут реальный "
            "внутридневной/сессионный край. На дневных данных час пропускается автоматически."
        )
        self.chk_regime = QCheckBox("Режимные (волатильность/тренд)")
        self.chk_regime.setChecked(True)
        self.chk_regime.setToolTip("Режим волатильности и долгий моментум — контекст рынка.")
        feat_form.addRow(self.chk_calendar)
        feat_form.addRow(self.chk_regime)

        target_box = QGroupBox("Цель (что предсказываем)")
        target_form = QFormLayout(target_box)
        self.combo_target = QComboBox()
        for label, key in TARGETS:
            self.combo_target.addItem(label, key)
        self.combo_target.setCurrentIndex(1)          # по умолчанию — тройной барьер
        self.combo_target.setToolTip(
            "Направление — вырастет ли цена через N баров (трудная цель на h=1). "
            "Тройной барьер (López de Prado) — что сработает первым: тейк или стоп; "
            "экономически осмысленнее и обычно предсказуемее. Рекомендуется."
        )
        self.spin_horizon = QSpinBox(minimum=1, maximum=200, value=10)
        self.spin_horizon.setToolTip("Горизонт прогноза в барах (на сколько вперёд смотрит метка).")
        self.spin_threshold = QDoubleSpinBox(minimum=-5.0, maximum=5.0, value=0.0)
        self.spin_threshold.setDecimals(2)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setSuffix(" %")
        self.spin_threshold.setToolTip(
            "Порог доходности (%), выше которого метка = «рост». Для тройного барьера не используется."
        )
        target_form.addRow("Тип:", self.combo_target)
        target_form.addRow("Горизонт (баров):", self.spin_horizon)
        target_form.addRow("Порог роста:", self.spin_threshold)

        model_box = QGroupBox("Модель и валидация")
        model_form = QFormLayout(model_box)
        self.combo_model = QComboBox()
        self.combo_model.setToolTip("Классическая модель (быстро на CPU). LightGBM — по умолчанию, если установлен.")
        self.combo_validation = QComboBox()
        for label, key in VALIDATIONS:
            self.combo_validation.addItem(label, key)
        self.combo_validation.setToolTip(
            "Walk-forward — честная временная кросс-валидация (рекомендуется). "
            "Holdout — один сплit train/test (быстрее, оптимистичнее)."
        )
        self.spin_splits = QSpinBox(minimum=2, maximum=20, value=5)
        self.spin_splits.setToolTip("Число out-of-sample фолдов walk-forward.")
        self.spin_embargo = QSpinBox(minimum=0, maximum=500, value=0)
        self.spin_embargo.setToolTip(
            "Эмбарго: дополнительный зазор в барах между train и test (сверх горизонта). "
            "Защищает от утечки из-за временной корреляции соседних баров."
        )
        self.chk_balance = QCheckBox("Балансировать классы")
        self.chk_balance.setChecked(True)
        self.chk_calibrate = QCheckBox("Калибровать вероятности")
        self.chk_weights = QCheckBox("Веса уникальности (AFML)")
        self.chk_weights.setChecked(True)
        self.chk_weights.setToolTip(
            "При горизонте > 1 метки соседних баров перекрываются (выборка не-IID). "
            "Веса уникальности (López de Prado, гл. 4) понижают вклад перекрытых "
            "наблюдений — честнее обучение, меньше переобучения."
        )
        self.chk_select = QCheckBox("Отбор признаков (де-корреляция)")
        self.chk_select.setToolTip(
            "Убрать константные и сильно коррелированные признаки. Меньше признаков → "
            "быстрее и устойчивее на слабом ПК."
        )
        self.chk_mda = QCheckBox("Честная важность (MDA, медленнее)")
        self.chk_mda.setToolTip(
            "Важность перестановкой на OOS (López de Prado, гл. 8): честнее импьюрити-"
            "важности, но требует доп. прогонов. Включайте при достатке времени."
        )
        self.chk_tune = QCheckBox("Подбор гиперпараметров")
        self.chk_tune.setToolTip(
            "Случайный поиск лучших гиперпараметров на той же purged walk-forward CV. "
            "Дольше (каждая проба = полный walk-forward), но повышает качество."
        )
        self.spin_tune_trials = QSpinBox(minimum=5, maximum=100, value=15)
        self.spin_tune_trials.setEnabled(False)
        self.chk_tune.toggled.connect(self.spin_tune_trials.setEnabled)
        model_form.addRow("Модель:", self.combo_model)
        model_form.addRow("Валидация:", self.combo_validation)
        model_form.addRow("Фолдов:", self.spin_splits)
        model_form.addRow("Эмбарго:", self.spin_embargo)
        model_form.addRow(self.chk_balance)
        model_form.addRow(self.chk_calibrate)
        model_form.addRow(self.chk_weights)
        model_form.addRow(self.chk_select)
        model_form.addRow(self.chk_mda)
        model_form.addRow(self.chk_tune)
        model_form.addRow("Проб подбора:", self.spin_tune_trials)

        self.btn_train = QPushButton("🧠  Обучить модель")
        self.btn_train.clicked.connect(self._emit_train)
        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("💾 Сохранить")
        self.btn_save.clicked.connect(self.save_requested.emit)
        self.btn_load = QPushButton("📂 Загрузить")
        self.btn_load.clicked.connect(self.load_requested.emit)
        self.btn_apply = QPushButton("→ В стратегию")
        self.btn_apply.clicked.connect(self.apply_requested.emit)
        for b in (self.btn_save, self.btn_apply):
            b.setEnabled(False)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_apply)

        self.lbl_verdict = QLabel("Обучите модель, чтобы увидеть оценку качества.")
        self.lbl_verdict.setWordWrap(True)
        self.lbl_verdict.setStyleSheet("padding: 4px 6px; color: gray;")
        self.summary = MlSummaryTable()

        left_widget = QWidget()
        left_col = QVBoxLayout(left_widget)
        left_col.setContentsMargins(0, 0, 0, 0)
        for w in (feat_box, target_box, model_box, self.btn_train):
            left_col.addWidget(w)
        left_col.addLayout(btn_row)
        left_col.addWidget(self.lbl_verdict)
        left_col.addWidget(self.summary, stretch=1)

        scroll = QScrollArea()
        scroll.setWidget(left_widget)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(380)
        scroll.setFrameShape(QScrollArea.NoFrame)
        root.addWidget(scroll)

        # --- правая часть: графики ---------------------------------------- #
        right = QVBoxLayout()
        self.lbl_hint = QLabel(
            "Машинное обучение оценивается ТОЛЬКО вне обучения (out-of-sample). "
            "Высокий AUC при большом разбросе по фолдам — признак переобучения."
        )
        self.lbl_hint.setStyleSheet("color: gray;")
        right.addWidget(self.lbl_hint)

        # Интерактивный рабочий порог: пересчитывает метрики/эквити без переобучения.
        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Рабочий порог:"))
        self.spin_thr = QDoubleSpinBox(minimum=0.05, maximum=0.95, value=0.50)
        self.spin_thr.setSingleStep(0.01)
        self.spin_thr.setDecimals(2)
        self.spin_thr.setEnabled(False)
        self.spin_thr.setToolTip(
            "Порог вероятности входа в лонг. Двигайте — метрики, операционная точка на "
            "ROC и кривая OOS-эквити пересчитываются мгновенно (без переобучения). "
            "Выбранный порог уходит в стратегию по кнопке «→ В стратегию»."
        )
        self.spin_thr.valueChanged.connect(self._on_threshold_changed)
        thr_row.addWidget(self.spin_thr)
        self.lbl_thr = QLabel("—")
        self.lbl_thr.setStyleSheet("color: #c8c8c8;")
        thr_row.addWidget(self.lbl_thr, stretch=1)
        right.addLayout(thr_row)

        self.chart_tabs = QTabWidget()
        right.addWidget(self.chart_tabs, stretch=1)
        root.addLayout(right, stretch=1)

        # Страница 1: важность признаков + стабильность по фолдам.
        self.imp_plot, self.fold_plot = self._add_chart_page("Качество", "Важность признаков (топ-12)",
                                                             "AUC по фолдам (стабильность вне обучения)")
        # Страница 2: калибровка вероятностей + ROC с подвижной рабочей точкой.
        self.calib_plot, self.roc_plot = self._add_chart_page("Калибровка · ROC",
                                                             "Калибровка вероятностей",
                                                             "ROC-кривая (рабочая точка — по порогу)")
        # Страница 3: распределение вероятностей по классам + OOS-эквити сигнала.
        self.hist_plot, self.equity_plot = self._add_chart_page("Вероятности · Эквити",
                                                               "Распределение P(рост) по классам",
                                                               "OOS-эквити сигнала (лонг/флэт)")

    def _add_chart_page(self, tab_title: str, title_top: str, title_bottom: str):
        """Создать вкладку с двумя графиками (верх/низ) и вернуть (top_plot, bottom_plot)."""
        glw = pg.GraphicsLayoutWidget()
        glw.setBackground(BG)
        top = glw.addPlot(row=0, col=0)
        top.setTitle(title_top)
        bottom = glw.addPlot(row=1, col=0)
        bottom.setTitle(title_bottom)
        for plot in (top, bottom):
            plot.setMenuEnabled(False)
            plot.addItem(make_grid_item())
        self.chart_tabs.addTab(glw, tab_title)
        return top, bottom

    # ------------------------------------------------------------------ #
    def ensure_models_loaded(self) -> bool:
        """Лениво заполнить список моделей. Возвращает True, если ML доступен.

        Импорт sklearn/lightgbm происходит здесь (при первом открытии вкладки),
        а не на старте приложения. Если зависимости не установлены — показываем
        подсказку и блокируем обучение (приложение остаётся работоспособным).
        """
        if self._models_loaded:
            return self.btn_train.isEnabled()
        self._models_loaded = True
        try:
            from ml.models import available_models, default_model_name
        except Exception as exc:  # noqa: BLE001 — деградация без ML-зависимостей
            self.btn_train.setEnabled(False)
            self.lbl_verdict.setText(
                "Машинное обучение недоступно: не установлены зависимости.\n"
                "Установите их командой:  pip install -e .[ml]\n"
                f"({exc})"
            )
            self.lbl_verdict.setStyleSheet("padding: 4px 6px; color: #ff9f43;")
            return False
        default = default_model_name()
        for key, label in available_models():
            self.combo_model.addItem(label, key)
            if key == default:
                self.combo_model.setCurrentIndex(self.combo_model.count() - 1)
        return True

    def make_settings(self) -> dict:
        return {
            "preset": self.combo_preset.currentData(),
            "target_mode": self.combo_target.currentData(),
            "target_horizon": self.spin_horizon.value(),
            "target_threshold": self.spin_threshold.value() / 100.0,
            "use_calendar": self.chk_calendar.isChecked(),
            "use_regime": self.chk_regime.isChecked(),
            "model_name": self.combo_model.currentData() or "random_forest",
            "validation": self.combo_validation.currentData(),
            "n_splits": self.spin_splits.value(),
            "embargo": self.spin_embargo.value(),
            "balance_classes": self.chk_balance.isChecked(),
            "calibrate": self.chk_calibrate.isChecked(),
            "sample_weights": self.chk_weights.isChecked(),
            "feature_select": self.chk_select.isChecked(),
            "compute_mda": self.chk_mda.isChecked(),
            "tune": self.chk_tune.isChecked(),
            "tune_trials": self.spin_tune_trials.value(),
        }

    def _emit_train(self) -> None:
        if not self.ensure_models_loaded():
            return
        self.train_requested.emit(self.make_settings())

    def set_running(self, running: bool) -> None:
        self.btn_train.setEnabled(not running and self._models_loaded)
        self.btn_load.setEnabled(not running)
        self.btn_save.setEnabled(not running and self._has_result)
        self.btn_apply.setEnabled(not running and self._has_result)

    # ------------------------------------------------------------------ #
    def selected_threshold(self) -> float:
        """Выбранный слайдером рабочий порог (0.0 — пока нет результата)."""
        return float(self.spin_thr.value()) if self._has_result else 0.0

    def show_result(self, mr) -> None:
        self._has_result = True
        self.btn_save.setEnabled(True)
        self.btn_apply.setEnabled(True)
        self.summary.show_result(mr)

        text, color = verdict(mr.metrics)
        self.lbl_verdict.setText(text)
        self.lbl_verdict.setStyleSheet(f"padding: 4px 6px; color: {color}; font-weight: bold;")

        # Массивы для интерактивного пересчёта по порогу (могут отсутствовать у старых
        # артефактов v2 — тогда интерактив/часть графиков просто недоступны).
        self._yt = _as_1d_array(getattr(mr, "y_true", None))
        self._yp = _as_1d_array(getattr(mr, "y_proba", None))
        self._fwd = _as_1d_array(getattr(mr, "oos_fwd", None))
        if self._yt is not None and self._yp is not None and self._yt.size != self._yp.size:
            self._yt = None                      # рассинхрон (старый адаптер) — без меток

        for plot in (self.imp_plot, self.fold_plot, self.calib_plot,
                     self.roc_plot, self.hist_plot, self.equity_plot):
            plot.clear()
            plot.addItem(make_grid_item())
        self._roc_marker = None

        self._draw_importances(mr)
        self._draw_folds(mr)
        self._draw_calibration()
        self._draw_roc()
        self._draw_prob_hist()

        # Инициализировать слайдер порогом модели и пересчитать зависимые графики.
        have_interactive = self._yt is not None and self._yp is not None
        self.spin_thr.setEnabled(have_interactive)
        self.spin_thr.blockSignals(True)
        self.spin_thr.setValue(float(np.clip(getattr(mr, "threshold", 0.5), 0.05, 0.95)))
        self.spin_thr.blockSignals(False)
        if have_interactive:
            self._on_threshold_changed()
        else:
            self.lbl_thr.setText("интерактив недоступен для этого артефакта")
            self._draw_equity(self.selected_threshold() or 0.5)

    def _draw_importances(self, mr) -> None:
        # Предпочитаем честную важность перестановкой (MDA), если посчитана.
        mda = getattr(mr, "importances_mda", None) or {}
        src = mda if mda else mr.importances
        self.imp_plot.setTitle(
            "Важность признаков — MDA (топ-12)" if mda else "Важность признаков — импьюрити (топ-12)"
        )
        items = list(src.items())[:12]
        if not items:
            return
        items = items[::-1]                      # самый важный — сверху
        names = [n for n, _ in items]
        vals = np.array([v for _, v in items], dtype=float)
        ypos = np.arange(len(vals))
        self.imp_plot.addItem(pg.BarGraphItem(
            x0=0, x1=vals, y=ypos, height=0.6,
            brush=pg.mkBrush(*BAR_BRUSH), pen=pg.mkPen(BG),
        ))
        axis = self.imp_plot.getAxis("left")
        axis.setTicks([list(zip(ypos.tolist(), names))])
        self.imp_plot.setLabel("bottom", "относительная важность")

    def _draw_folds(self, mr) -> None:
        folds = mr.fold_metrics
        if folds:
            xs = np.array([fm.get("fold", i + 1) for i, fm in enumerate(folds)], dtype=float)
            ys = np.array([fm.get("roc_auc", float("nan")) for fm in folds], dtype=float)
            self.fold_plot.addItem(pg.PlotCurveItem(
                x=xs, y=ys, pen=pg.mkPen("#4f9cf7", width=2)))
            self.fold_plot.addItem(pg.ScatterPlotItem(
                x=xs, y=ys, size=8, brush=pg.mkBrush(79, 156, 247, 220), pen=None))
            mean = float(np.nanmean(ys)) if np.isfinite(ys).any() else float("nan")
            if np.isfinite(mean):
                self.fold_plot.addItem(pg.InfiniteLine(
                    mean, angle=0, pen=pg.mkPen(COLOR_GOOD, style=QtCore.Qt.PenStyle.DashLine),
                    label=f"среднее {mean:.3f}", labelOpts={"color": COLOR_GOOD, "position": 0.05}))
        self.fold_plot.addItem(pg.InfiniteLine(
            0.5, angle=0, pen=pg.mkPen(CHANCE, style=QtCore.Qt.PenStyle.DotLine),
            label="случайно 0.5", labelOpts={"color": CHANCE, "position": 0.95}))
        self.fold_plot.setLabel("bottom", "фолд")
        self.fold_plot.setLabel("left", "ROC-AUC")

    # ------------------------------------------------------------------ #
    # Графики по OOS-вероятностям (калибровка / ROC / распределение / эквити)
    # ------------------------------------------------------------------ #
    def _draw_calibration(self) -> None:
        """Reliability diagram: насколько предсказанная вероятность совпадает с фактической."""
        self.calib_plot.setLabel("bottom", "предсказанная P(рост)")
        self.calib_plot.setLabel("left", "фактическая доля роста")
        diag = np.array([0.0, 1.0])
        self.calib_plot.addItem(pg.PlotCurveItem(
            x=diag, y=diag, pen=pg.mkPen(CHANCE, style=QtCore.Qt.PenStyle.DotLine)))
        if self._yt is None or self._yp is None or len(np.unique(self._yt)) < 2:
            return
        try:
            from sklearn.calibration import calibration_curve

            prob_true, prob_pred = calibration_curve(
                self._yt, self._yp, n_bins=10, strategy="quantile")
        except Exception:  # noqa: BLE001
            return
        self.calib_plot.addItem(pg.PlotCurveItem(
            x=prob_pred, y=prob_true, pen=pg.mkPen("#4f9cf7", width=2)))
        self.calib_plot.addItem(pg.ScatterPlotItem(
            x=prob_pred, y=prob_true, size=7, brush=pg.mkBrush(79, 156, 247, 220), pen=None))

    def _draw_roc(self) -> None:
        """ROC-кривая OOS + точка-маркер текущей рабочей точки (обновляется слайдером)."""
        self.roc_plot.setLabel("bottom", "FPR (ложная тревога)")
        self.roc_plot.setLabel("left", "TPR (попадание)")
        diag = np.array([0.0, 1.0])
        self.roc_plot.addItem(pg.PlotCurveItem(
            x=diag, y=diag, pen=pg.mkPen(CHANCE, style=QtCore.Qt.PenStyle.DotLine)))
        if self._yt is None or self._yp is None or len(np.unique(self._yt)) < 2:
            return
        try:
            from sklearn.metrics import roc_curve

            fpr, tpr, _ = roc_curve(self._yt, self._yp)
        except Exception:  # noqa: BLE001
            return
        self.roc_plot.addItem(pg.PlotCurveItem(x=fpr, y=tpr, pen=pg.mkPen("#4f9cf7", width=2)))
        self._roc_marker = pg.ScatterPlotItem(
            size=12, brush=pg.mkBrush(255, 159, 67, 240), pen=pg.mkPen(BG))
        self.roc_plot.addItem(self._roc_marker)

    def _draw_prob_hist(self) -> None:
        """Гистограммы P(рост) отдельно для фактически выросших и упавших баров."""
        self.hist_plot.setLabel("bottom", "P(рост)")
        self.hist_plot.setLabel("left", "доля баров")
        if self._yt is None or self._yp is None:
            return
        edges = np.linspace(0.0, 1.0, 21)
        for cls, color in ((0, (255, 77, 77, 120)), (1, (22, 199, 132, 120))):
            mask = self._yt == cls
            if not mask.any():
                continue
            hist, _ = np.histogram(self._yp[mask], bins=edges, density=True)
            # stepMode=True: число рёбер (x) на 1 больше числа столбцов (y).
            self.hist_plot.addItem(pg.PlotCurveItem(
                x=edges, y=hist, stepMode=True, fillLevel=0.0,
                brush=pg.mkBrush(*color), pen=pg.mkPen(color[:3] + (220,), width=1)))

    def _draw_equity(self, thr: float) -> None:
        """Кумулятивная OOS-доходность «лонг/флэт» сигнала при заданном пороге."""
        self.equity_plot.clear()
        self.equity_plot.addItem(make_grid_item())
        self.equity_plot.setLabel("bottom", "OOS-бар")
        self.equity_plot.setLabel("left", "кумулятивная доходность")
        self.equity_plot.addItem(pg.InfiniteLine(
            0.0, angle=0, pen=pg.mkPen(CHANCE, style=QtCore.Qt.PenStyle.DotLine)))
        if self._fwd is None or self._yp is None:
            return
        fwd = np.nan_to_num(self._fwd, nan=0.0)
        # Buy & hold по тем же OOS-барам — контекст для оценки сигнала.
        bh = np.cumsum(fwd)
        self.equity_plot.addItem(pg.PlotCurveItem(
            x=np.arange(bh.size), y=bh, pen=pg.mkPen(CHANCE, width=1, style=QtCore.Qt.PenStyle.DashLine)))
        pos = (self._yp >= thr).astype(float)
        equity = np.cumsum(pos * fwd)
        self.equity_plot.addItem(pg.PlotCurveItem(
            x=np.arange(equity.size), y=equity, pen=pg.mkPen(COLOR_GOOD, width=2)))

    def _on_threshold_changed(self) -> None:
        """Пересчитать метрики/маркер/эквити под текущий порог — без переобучения."""
        thr = float(self.spin_thr.value())
        self._draw_equity(thr)
        if self._yt is None or self._yp is None:
            return
        pred = (self._yp >= thr).astype(int)
        yt = self._yt.astype(int)
        tp = int(np.sum((pred == 1) & (yt == 1)))
        fp = int(np.sum((pred == 1) & (yt == 0)))
        fn = int(np.sum((pred == 0) & (yt == 1)))
        tn = int(np.sum((pred == 0) & (yt == 0)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        n_sig = tp + fp
        share = 100.0 * n_sig / len(pred) if len(pred) else 0.0

        parts = [f"P {precision:.2f}", f"R {recall:.2f}", f"F1 {f1:.2f}",
                 f"сигналов {n_sig} ({share:.0f}%)"]
        if self._fwd is not None and n_sig:
            ret = pred.astype(float) * np.nan_to_num(self._fwd, nan=0.0)
            sh = _sharpe(ret)
            exp = float(np.mean(self._fwd[pred == 1])) * 100.0
            parts.append(f"Sharpe {sh:.2f}")
            parts.append(f"E[r] {exp:+.3f}%")
        self.lbl_thr.setText(" · ".join(parts))

        if self._roc_marker is not None:
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            self._roc_marker.setData(x=[fpr], y=[recall])
