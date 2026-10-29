"""Grouped, searchable parameter editor built from FieldSpec lists."""
from __future__ import annotations

import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from ..paramspec import FieldSpec, format_value, parse_value
from ..theme import repolish


def _tooltip(s: FieldSpec) -> str:
    parts = [f"<b>{s.name}</b>" + (f" &nbsp;[{s.unit}]" if s.unit else "")]
    if s.help:
        parts.append(s.help)
    parts.append(f"<span style='color:gray'>default: {format_value(s, s.default)}</span>")
    return "<br>".join(parts)


class _Row:
    def __init__(self, spec: FieldSpec, label: QLabel, editor: QWidget, reset: QToolButton, group: QGroupBox):
        self.spec = spec
        self.label = label
        self.editor = editor
        self.reset = reset
        self.group = group
        self.valid = True

    def widgets(self):
        return (self.label, self.editor, self.reset)


class ParamForm(QWidget):
    changed = Signal()

    def __init__(self, specs: list[FieldSpec], parent=None, show_advanced=False, toolbar=True, scroll=True):
        super().__init__(parent)
        self.specs = list(specs)
        self.rows: dict[str, _Row] = {}
        self._loading = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        if toolbar:
            bar = QHBoxLayout()
            bar.setSpacing(6)
            self.search = QLineEdit()
            self.search.setPlaceholderText("Search parameters (name / description)…")
            self.search.setClearButtonEnabled(True)
            self.search.textChanged.connect(self._apply_filter)
            bar.addWidget(self.search, 1)
            self.adv = QCheckBox("Advanced")
            self.adv.setToolTip("Show grid, numerics and secondary material parameters")
            self.adv.setChecked(show_advanced)
            self.adv.toggled.connect(self._apply_filter)
            has_adv = any(s.advanced for s in self.specs)
            self.adv.setVisible(has_adv)
            bar.addWidget(self.adv)
            lay.addLayout(bar)
            bar2 = QHBoxLayout()
            self.status = QLabel("")
            self.status.setObjectName("muted")
            bar2.addWidget(self.status, 1)
            for text, slot, tip in (("Load…", self.load_dialog, "Load parameters from a JSON preset or a params.json"),
                                    ("Save…", self.save_dialog, "Save the current parameters as a JSON preset"),
                                    ("Reset", self.reset_all, "Reset all parameters to their defaults")):
                b = QPushButton(text)
                b.setObjectName("flat")
                b.setToolTip(tip)
                b.clicked.connect(slot)
                bar2.addWidget(b)
            lay.addLayout(bar2)
        else:
            self.search = None
            self.adv = None
            self.status = None

        inner = QWidget()
        inner.setObjectName("page")
        self._col = QVBoxLayout(inner)
        self._col.setContentsMargins(0, 0, 6, 0)
        self._col.setSpacing(4)
        groups: dict[str, QGroupBox] = {}
        grids: dict[str, QGridLayout] = {}
        for s in self.specs:
            sec = s.section or "Parameters"
            if sec not in groups:
                g = QGroupBox(sec.replace("&", "&&"))      # '&' would become a mnemonic
                grid = QGridLayout(g)
                grid.setContentsMargins(8, 10, 8, 6)
                grid.setHorizontalSpacing(8)
                grid.setVerticalSpacing(5)
                grid.setColumnStretch(1, 1)
                grid.setColumnMinimumWidth(0, 178)
                grid.setColumnMinimumWidth(2, 26)     # reset button column keeps editors aligned across groups
                groups[sec], grids[sec] = g, grid
                self._col.addWidget(g)
            grid = grids[sec]
            r = grid.rowCount()
            lab = QLabel(s.label + (f"  <span style='color:gray'>{s.unit}</span>" if s.unit else ""))
            lab.setTextFormat(Qt.TextFormat.RichText)
            lab.setToolTip(_tooltip(s))
            ed = self._make_editor(s)
            ed.setToolTip(_tooltip(s))
            rb = QToolButton()
            rb.setObjectName("flat")
            rb.setText("↺")
            rb.setFixedWidth(26)
            rb.setToolTip("Reset to default")
            rb.setVisible(False)
            rb.clicked.connect(lambda _=False, n=s.name: self.reset_field(n))
            grid.addWidget(lab, r, 0)
            grid.addWidget(ed, r, 1)
            grid.addWidget(rb, r, 2)
            self.rows[s.name] = _Row(s, lab, ed, rb, groups[sec])
        self._groups = groups
        self._col.addStretch(1)
        if scroll:
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QFrame.Shape.NoFrame)
            area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            area.setWidget(inner)
            lay.addWidget(area, 1)
        else:
            lay.addWidget(inner, 1)
        self._apply_filter()
        self._update_status()

    # ------------------------------------------------------------------ editors
    def _make_editor(self, s: FieldSpec) -> QWidget:
        if s.type == "bool":
            w = QCheckBox()
            w.setChecked(bool(s.default))
            w.toggled.connect(lambda _=False, n=s.name: self._on_edit(n))
        elif s.type == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in s.choices])
            w.setCurrentText(str(s.default))
            w.currentTextChanged.connect(lambda _="", n=s.name: self._on_edit(n))
        else:
            w = QLineEdit(format_value(s, s.default))
            w.setMinimumWidth(110)
            w.textChanged.connect(lambda _="", n=s.name: self._on_edit(n))
            if s.type in ("float", "int"):
                w.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return w

    def _raw(self, row: _Row):
        e = row.editor
        if isinstance(e, QCheckBox):
            return e.isChecked()
        if isinstance(e, QComboBox):
            return e.currentText()
        return parse_value(row.spec, e.text())

    def _set_raw(self, row: _Row, v):
        e = row.editor
        if isinstance(e, QCheckBox):
            e.setChecked(bool(v) if not isinstance(v, str) else v.lower() in ("1", "true"))
        elif isinstance(e, QComboBox):
            e.setCurrentText(str(v))
        else:
            e.setText(format_value(row.spec, v) if not isinstance(v, str) else v)

    @staticmethod
    def _same(a, b):
        if isinstance(a, float) or isinstance(b, float):
            try:
                a, b = float(a), float(b)
            except (TypeError, ValueError):
                return False
            return a == b or abs(a - b) <= 1e-12 * max(abs(a), abs(b))
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return len(a) == len(b) and all(ParamForm._same(x, y) for x, y in zip(a, b))
        return a == b

    def _on_edit(self, name):
        row = self.rows[name]
        try:
            v = self._raw(row)
            if row.spec.type in ("float", "int") and row.spec.minimum is not None and v < row.spec.minimum:
                raise ValueError
            if row.spec.type in ("float", "int") and row.spec.maximum is not None and v > row.spec.maximum:
                raise ValueError
            row.valid = True
            modified = not self._same(v, row.spec.default)
        except (ValueError, TypeError):
            row.valid = False
            modified = True
        e = row.editor
        e.setProperty("invalid", not row.valid)
        e.setProperty("modified", modified and row.valid)
        row.label.setProperty("modified", modified)
        row.reset.setVisible(modified)
        repolish(e)
        repolish(row.label)
        self._update_status()
        if not self._loading:
            self.changed.emit()

    def _update_status(self):
        if self.status is None:
            return
        n_mod = len(self.overrides(strict=False))
        n_bad = sum(not r.valid for r in self.rows.values())
        txt = f"{n_mod} changed" if n_mod else "all defaults"
        if n_bad:
            txt += f" · <span style='color:#d23a3a'>{n_bad} invalid</span>"
        self.status.setText(txt)

    # ------------------------------------------------------------------ filtering
    def _apply_filter(self, *_):
        q = (self.search.text().strip().lower() if self.search else "")
        show_adv = self.adv.isChecked() if self.adv else True
        visible_in = {}
        for row in self.rows.values():
            s = row.spec
            modified = row.label.property("modified") is True
            hay = f"{s.name} {s.label} {s.help} {s.unit}".lower()
            show = (not q or q in hay) and (show_adv or not s.advanced or modified or bool(q))
            for w in row.widgets()[:2]:
                w.setVisible(show)
            row.reset.setVisible(show and modified)
            visible_in[row.group] = visible_in.get(row.group, False) or show
        for g, vis in visible_in.items():
            g.setVisible(vis)

    # ------------------------------------------------------------------ API
    def is_valid(self) -> bool:
        return all(r.valid for r in self.rows.values())

    def invalid_fields(self):
        return [n for n, r in self.rows.items() if not r.valid]

    def values(self) -> dict:
        out = {}
        for n, r in self.rows.items():
            try:
                out[n] = self._raw(r)
            except (ValueError, TypeError):
                out[n] = r.spec.default
        return out

    def overrides(self, strict=True) -> dict:
        """Only the fields that differ from their defaults (invalid fields raise if strict)."""
        out = {}
        for n, r in self.rows.items():
            try:
                v = self._raw(r)
            except (ValueError, TypeError):
                if strict:
                    raise ValueError(f"invalid value for parameter {n}")
                out[n] = None
                continue
            if not self._same(v, r.spec.default):
                out[n] = v
        return out

    def set_values(self, d: dict, reset_others=False):
        self._loading = True
        try:
            for n, r in self.rows.items():
                if n in d:
                    self._set_raw(r, d[n])
                elif reset_others:
                    self._set_raw(r, r.spec.default)
        finally:
            self._loading = False
        self._apply_filter()
        self.changed.emit()

    def set_defaults(self, d: dict):
        """Replace the defaults (e.g. a script's own defaults) and show them."""
        for n, v in d.items():
            if n in self.rows:
                self.rows[n].spec.default = v
        self.set_values(d)

    def reset_field(self, name):
        r = self.rows[name]
        self._set_raw(r, r.spec.default)
        self._apply_filter()

    def reset_all(self):
        self.set_values({}, reset_others=True)

    def set_enabled_fields(self, names, enabled):
        for n in names:
            if n in self.rows:
                self.rows[n].editor.setEnabled(enabled)

    # ------------------------------------------------------------------ presets
    def save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save parameter preset", "preset.json", "JSON (*.json)")
        if not path:
            return
        try:
            data = self.values()
        except ValueError:
            return
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1, default=list)

    def load_dialog(self):
        from .. import paths
        path, _ = QFileDialog.getOpenFileName(self, "Load parameters", str(paths.RESULTS), "JSON (*.json)")
        if not path:
            return
        with open(path) as fh:
            d = json.load(fh)
        if isinstance(d, dict) and "params" in d and isinstance(d["params"], dict):
            d = d["params"]
        self.set_values({k: v for k, v in d.items() if k in self.rows}, reset_others=True)
