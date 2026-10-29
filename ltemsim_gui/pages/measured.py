"""Editor for the measured tip-scan data (data/measured_tip_scan.csv) that the scripts overlay on their plots."""
from __future__ import annotations

import csv
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemDelegate, QAbstractItemView, QApplication, QCheckBox, QFileDialog,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QSizePolicy, QSplitter,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import icons, paths
from ..theme import ThemeManager, repolish
from ..widgets.common import MetricTile, Page, button, card, error_box, label, reveal
from ..widgets.mpl import PlotWidget

COLUMNS = ("bias_V", "X_um", "rms", "rms_err", "note")
HEADERS = COLUMNS
HEADER_TIPS = ("Bias (V)", "X = spot − tip (µm)", "THz signal rms, relative to the +25 V, X = +5 µm reference point",
               "Uncertainty of rms (≥ 0)", "Note: date, number of scans, … (optional)")
NUMERIC = 4                       # the first four columns are numbers
EXTRA_TIP = "Extra column of the file: kept as it is and written back on save"
CONSUMERS = "run_baseline, sweep, compare_scenarios, field_bending, run_pulse_chain, run_tip_positions, fig_slides"

# validated categorical slots (light / dark surface); colour follows the entity (the bias), never the rank
SERIES = {False: ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
          True: ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")}


def series_colors(dark: bool) -> tuple:
    return SERIES[bool(dark)]


def bias_color(bias: float, dark: bool) -> str:
    """Fixed colour per bias: +25 V is slot 1, −25 V slot 2, anything else the following slots."""
    fixed = {25.0: 0, -25.0: 1}
    i = fixed.get(float(bias))
    if i is None:
        i = 2 + int(abs(hash(round(float(bias), 6)))) % 6
    return SERIES[bool(dark)][i]


def read_table(path) -> tuple[list[str], list[list[str]]]:
    """(extra column names, raw text rows [bias_V, X_um, rms, rms_err, note, *extras]).

    Columns other than the five known ones (e.g. added by hand in a spreadsheet) are kept so that saving
    does not drop them. Raises ValueError if a required column is missing.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        r = csv.DictReader(fh)
        fields = [(f or "").strip() for f in (r.fieldnames or [])]
        missing = [c for c in COLUMNS[:NUMERIC] if c not in fields]
        if missing:
            raise ValueError("Missing column(s): " + ", ".join(missing)
                             + " (the header must read bias_V,X_um,rms,rms_err[,note])")
        extras = list(dict.fromkeys(f for f in fields if f and f not in COLUMNS))
        cols = list(COLUMNS) + extras
        rows = []
        for row in r:
            row = {(k or "").strip(): (v or "") for k, v in row.items() if k is not None}
            vals = [str(row.get(c, "") or "").strip() for c in cols]
            if any(vals):
                rows.append(vals)
        return extras, rows


def read_rows(path) -> list[list[str]]:
    """Raw text rows [bias_V, X_um, rms, rms_err, note]; raises ValueError if a required column is missing."""
    return [r[:len(COLUMNS)] for r in read_table(path)[1]]


def cell_error(col: int, text: str) -> str:
    """'' when the cell is valid, otherwise a short lower-case reason."""
    if col >= NUMERIC:
        return ""
    t = text.strip()
    if not t:
        return "empty"
    try:
        v = float(t)
    except ValueError:
        return "not a number"
    if not math.isfinite(v):
        return "not a finite number"
    if col in (2, 3) and v < 0:
        return "must be ≥ 0"
    return ""


def numeric_rows(rows) -> list[tuple[int, float, float, float, float]]:
    """(row index, bias, X, rms, err) for every row whose four numbers are valid."""
    out = []
    for i, r in enumerate(rows):
        if all(not cell_error(c, r[c]) for c in range(NUMERIC)):
            out.append((i, *(float(r[c]) for c in range(NUMERIC))))
    return out


def read_measured(path=None) -> list[tuple[float, float, float, float]]:
    """(bias_V, X_um, rms, rms_err) of the valid rows; [] if the file is missing or unreadable."""
    try:
        rows = read_rows(path or paths.MEASURED_CSV)
    except (OSError, ValueError, csv.Error):
        return []
    return [r[1:] for r in numeric_rows(rows)]


def _norm(rows) -> list[tuple]:
    """Comparable form of rows: numbers by value (so '1.00' == '1.0'), text stripped."""
    out = []
    for r in rows:
        key = []
        for c, v in enumerate(r):
            v = v.strip()
            if c < NUMERIC and not cell_error(c, v):
                key.append(float(v))
            else:
                key.append(v)
        out.append(tuple(key))
    return out


def _rows(n: int) -> str:
    return f"{n} row" + ("" if n == 1 else "s")


def _cells(n: int) -> str:
    return f"{n} cell is" if n == 1 else f"{n} cells are"


class MeasuredPage(Page):
    title = "Measured data"
    subtitle = ("Measured tip scans: THz signal rms versus X = spot − tip at ±25 V bias, normalised to the +25 V, "
                "X = +5 µm reference point. The scripts overlay these points on their plots to compare with the model.")

    def __init__(self, path=None, backup_dir=None):
        super().__init__()
        self.path = Path(path) if path else paths.MEASURED_CSV
        self.backup_dir = Path(backup_dir) if backup_dir else None     # default: <app data>/backups
        self.last_backup: Path | None = None
        self._saved: list[list[str]] = []
        self.extra: list[str] = []            # columns of the file besides COLUMNS, kept as they are
        self._mtime = None
        self._building = False
        self._sort_state = (None, False)
        self._pts = []                        # hover / click targets: (ax, x, y, row)
        self._hover = None                    # hover annotation (made by _draw)
        self._tiles: list[QWidget] = []

        # ---------------- header actions
        self.state = QLabel("")
        self.state.setObjectName("badge")
        self.header.add_action(self.state)
        # lambdas: clicked(bool) would otherwise arrive as the first argument (revert(confirm=False)!)
        self.btn_import = button("Import CSV…", lambda: self.import_csv(),
                                 tooltip="Import rows from another CSV file (replace or append); nothing is written "
                                         "until you save")
        self.btn_revert = button("Revert", lambda: self.revert(),
                                 tooltip="Discard unsaved changes and re-read the file")
        self.btn_save = button("  Save", lambda: self.save(), "primary",
                               tooltip="Save to the CSV file (the previous version is backed up first)  ⌘S")
        for b in (self.btn_import, self.btn_revert, self.btn_save):
            self.header.add_action(b)

        # ---------------- note
        note = card()
        nl = QHBoxLayout(note)
        nl.setContentsMargins(14, 10, 14, 10)
        nl.setSpacing(12)
        self.note_icon = QLabel()
        nl.addWidget(self.note_icon, 0, Qt.AlignmentFlag.AlignTop)
        self.note_text = label("", wrap=True)
        self.note_text.setTextFormat(Qt.TextFormat.RichText)
        nl.addWidget(self.note_text, 1)
        bcol = QVBoxLayout()
        bcol.setSpacing(2)
        self.btn_reveal = button("Show in Finder", lambda: reveal(self.path))
        self.btn_backups = button("Show backups", self._open_backups,
                                  tooltip="Show the folder of automatic backups in Finder")
        for b in (self.btn_reveal, self.btn_backups):
            b.setObjectName("flat")
            bcol.addWidget(b)
        nl.addLayout(bcol)
        self.body.addWidget(note)

        # ---------------- summary tiles
        self.tiles = QHBoxLayout()
        self.tiles.setSpacing(8)
        self.body.addLayout(self.tiles)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        # table and plot take the height that is left: their size hints must not make the page scroll in a
        # small window (the info line under the table, with the invalid-cell messages, would end up below the fold)
        split.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        split.setMinimumHeight(320)
        # ---------------- left: table
        left = card()
        tl = QVBoxLayout(left)
        tl.setContentsMargins(12, 10, 12, 12)
        tl.setSpacing(8)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(label("Data", "cardTitle"))
        bar.addStretch(1)
        self.btn_add = button("Add row", self.add_row, tooltip="Insert a row after the selected one")
        self.btn_dup = button("Duplicate", self.duplicate_rows, tooltip="Duplicate the selected rows")
        self.btn_del = button("Delete", self.delete_rows, tooltip="Delete the selected rows (Delete key)")
        self.btn_sort = button("Sort", self.sort_default,
                               tooltip="Sort by bias (high → low), then X (low → high); click a column header to sort "
                                       "by that column")
        for b in (self.btn_add, self.btn_dup, self.btn_del, self.btn_sort):
            bar.addWidget(b)
        tl.addLayout(bar)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
                                   | QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.table.verticalHeader().setDefaultSectionSize(28)
        hh = self.table.horizontalHeader()
        self._set_columns([])
        hh.setSectionsClickable(True)
        hh.sectionClicked.connect(self._sort_by_column)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection)
        tl.addWidget(self.table, 1)
        self.info = label("", "muted", wrap=True)         # rows / biases / first invalid cell (always current)
        self.status = label("", "muted", wrap=True)       # last action (saved, imported, ...)
        self.status.setVisible(False)
        tl.addWidget(self.info)
        tl.addWidget(self.status)
        left.setMinimumWidth(420)
        split.addWidget(left)

        # ---------------- right: plot
        right = card()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(12, 10, 12, 12)
        rl.setSpacing(6)
        rl.addWidget(label("THz rms versus X", "cardTitle"))
        orow = QHBoxLayout()            # own row: next to the title it clips at 1100 px window width
        orow.setSpacing(14)
        self.opt_mirror = QCheckBox("Mirror −25 V (X → −X)")
        self.opt_mirror.setToolTip("Flip the sign of X for negative-bias data, to compare it with +25 V at the same "
                                   "anode orientation")
        self.opt_norm = QCheckBox("Normalise per bias")
        self.opt_norm.setToolTip("Divide each bias by its own maximum rms")
        self.opt_line = QCheckBox("Lines")
        self.opt_line.setToolTip("Connect the points of each bias")
        self.opt_line.setChecked(True)
        for cb in (self.opt_mirror, self.opt_norm, self.opt_line):
            cb.toggled.connect(self._on_option)
            orow.addWidget(cb)
        orow.addStretch(1)
        rl.addLayout(orow)
        self.plot = PlotWidget(figsize=(6, 4))
        self.plot.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.plot.canvas.mpl_connect("button_press_event", self._on_click)
        rl.addWidget(self.plot, 1)
        rl.addWidget(label("Hover over a point to see its note; click it to select its row in the table.",
                           "muted", wrap=True))
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([560, 700])
        self.body.addWidget(split, 1)

        QShortcut(QKeySequence.StandardKey.Save, self, activated=self.save,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence.StandardKey.Delete, self.table, activated=self.delete_rows,
                  context=Qt.ShortcutContext.WidgetShortcut)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self.table, activated=self.delete_rows,
                  context=Qt.ShortcutContext.WidgetShortcut)

        self.load()
        self._retheme()
        ThemeManager.instance().changed.connect(self._retheme)

    # ------------------------------------------------------------------ file
    def load(self):
        """(Re)read the CSV from disk, discarding edits."""
        extras = []
        try:
            extras, rows = read_table(self.path)
            err = ""
        except FileNotFoundError:
            rows, err = [], f"{paths.rel(self.path)} does not exist yet. Add rows and save to create it."
        except (OSError, ValueError, csv.Error, UnicodeDecodeError) as e:
            rows, err = [], f"Cannot read {paths.rel(self.path)}: {e}"
        self._set_columns(extras)
        self._saved = [list(r) for r in rows]
        self._mtime = self._disk_mtime()
        self._sort_state = (None, False)
        self.table.horizontalHeader().setSortIndicatorShown(False)
        self._set_rows(rows)
        self._update_note()
        self.message(err)

    def message(self, text: str):
        self.status.setText(text)
        self.status.setVisible(bool(text))

    def _disk_mtime(self):
        try:
            return self.path.stat().st_mtime
        except OSError:
            return None

    def save(self) -> bool:
        self._commit_edit()
        rows = self.rows()
        bad = self.invalid_cells()
        if bad:
            r, c, why = bad[0]
            error_box(self, "Cannot save", f"{_cells(len(bad))} invalid; fix {'it' if len(bad) == 1 else 'them'} "
                      f"first.\nRow {r + 1}, {COLUMNS[c]}: {why}")
            return False
        if self._disk_mtime() != self._mtime:
            r = QMessageBox.question(self, "File changed on disk",
                                     f"{paths.rel(self.path)} was changed by another program after it was loaded. "
                                     "Overwrite it with this table?\n\nThe version on disk is backed up first. "
                                     "Choose No and then Revert to load it instead.")
            if r != QMessageBox.StandardButton.Yes:
                return False
        bdir = self.backup_folder()
        try:
            backup = None
            if self.path.exists():
                bdir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                backup = bdir / f"measured_tip_scan-{stamp}.csv"
                i = 2
                while backup.exists():
                    backup = bdir / f"measured_tip_scan-{stamp}_{i}.csv"
                    i += 1
                shutil.copy2(self.path, backup)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(list(COLUMNS) + self.extra)
                for r in rows:
                    w.writerow([v.strip() for v in r])
            os.replace(tmp, self.path)
        except OSError as e:
            error_box(self, "Save failed", f"Cannot write {self.path}: {e}")
            return False
        self.last_backup = backup
        self._saved = [list(r) for r in rows]
        self._mtime = self._disk_mtime()
        self._refresh_state()
        when = time.strftime("%H:%M:%S")
        self.message(f"Saved {_rows(len(rows))} at {when}"
                     + (f"; the previous version was backed up as {backup.name}" if backup else "") + ".")
        return True

    def revert(self, confirm=True):
        if self.is_dirty() and confirm:         # is_dirty() also commits an open cell editor
            r = QMessageBox.question(self, "Revert changes", "Discard all unsaved changes and re-read the file?")
            if r != QMessageBox.StandardButton.Yes:
                return
        self.load()
        self.message("Reverted to the version on disk.")

    def import_csv(self, path=None, mode=None):
        """Load rows from another CSV into the table (mode 'replace' | 'append'); nothing is written until Save."""
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Import measured data", str(paths.DATA),
                                                  "CSV files (*.csv);;All files (*)")
            if not path:
                return
        self._commit_edit()
        try:
            extras, new = read_table(path)
        except (OSError, ValueError, csv.Error, UnicodeDecodeError) as e:
            error_box(self, "Import failed", f"Cannot read {Path(path).name}:\n{e}")
            return
        # keep this file's extra columns; imported columns with the same name fill them
        idx = {c: len(COLUMNS) + i for i, c in enumerate(extras)}
        new = [r[:len(COLUMNS)] + [r[idx[c]] if c in idx else "" for c in self.extra] for r in new]
        if mode is None:
            if self.table.rowCount():
                m = QMessageBox(self)
                m.setWindowTitle("Import CSV")
                m.setText(f"{Path(path).name} contains {_rows(len(new))}. "
                          "Replace the current table, or append to it?")
                b_rep = m.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
                b_app = m.addButton("Append", QMessageBox.ButtonRole.ActionRole)
                m.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                m.exec()
                clicked = m.clickedButton()
                mode = "replace" if clicked is b_rep else "append" if clicked is b_app else None
                if mode is None:
                    return
            else:
                mode = "replace"
        rows = new if mode == "replace" else self.rows() + new
        self._sort_state = (None, False)
        self.table.horizontalHeader().setSortIndicatorShown(False)
        self._set_rows(rows)
        self.message(f"{'Imported' if mode == 'replace' else 'Appended'} {_rows(len(new))} from {Path(path).name} "
                     "(not saved yet).")

    # ------------------------------------------------------------------ table
    def rows(self) -> list[list[str]]:
        """Text of every row: [bias_V, X_um, rms, rms_err, note, *extra columns]."""
        nc = self.table.columnCount()
        out = []
        for r in range(self.table.rowCount()):
            out.append([(self.table.item(r, c).text() if self.table.item(r, c) else "").strip() for c in range(nc)])
        return out

    def _set_columns(self, extras):
        self.extra = list(extras)
        self.table.setColumnCount(len(COLUMNS) + len(self.extra))
        self.table.setHorizontalHeaderLabels(list(HEADERS) + self.extra)
        for c, tip in enumerate(list(HEADER_TIPS) + [EXTRA_TIP] * len(self.extra)):
            self.table.horizontalHeaderItem(c).setToolTip(tip)
        hh = self.table.horizontalHeader()
        for c in range(self.table.columnCount()):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
            self.table.setColumnWidth(c, 68 if c < NUMERIC else 90)
        hh.setSectionResizeMode(NUMERIC, QHeaderView.ResizeMode.Stretch)

    def _commit_edit(self):
        """Write an open cell editor back into the table (⌘S, and on macOS clicked buttons, keep the editor open)."""
        if self.table.state() != QAbstractItemView.State.EditingState:
            return
        ed = QApplication.focusWidget()
        if ed is None or not self.table.viewport().isAncestorOf(ed):
            eds = [w for w in self.table.viewport().findChildren(QLineEdit) if w.isVisible()]
            ed = eds[-1] if eds else None
        if ed is not None:
            self.table.commitData(ed)
            self.table.closeEditor(ed, QAbstractItemDelegate.EndEditHint.NoHint)

    def _set_rows(self, rows, select=None):
        nc = self.table.columnCount()
        self._building = True
        try:
            self.table.setRowCount(0)
            self.table.setRowCount(len(rows))
            for r, vals in enumerate(rows):
                vals = (list(vals) + [""] * nc)[:nc]
                for c, v in enumerate(vals):
                    self.table.setItem(r, c, self._make_item(c, v))
            for r in range(len(rows)):
                for c in range(nc):
                    self._validate(self.table.item(r, c))
        finally:
            self._building = False
        if select is not None and 0 <= select < len(rows):
            self.table.selectRow(select)
        self._changed()

    def _make_item(self, col, text):
        it = QTableWidgetItem(text)
        if col < NUMERIC:
            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return it

    @staticmethod
    def _v(b: float) -> str:
        return f"{b:+g} V".replace("-", "−")

    def _validate(self, it):
        if it is None:
            return
        why = cell_error(it.column(), it.text())
        t = ThemeManager.instance().tokens
        if why:
            c = QColor(t.danger)
            c.setAlpha(60)
            it.setBackground(QBrush(c))
            it.setToolTip(why[0].upper() + why[1:])
        else:
            it.setBackground(QBrush())
            it.setToolTip(it.text() if it.column() >= NUMERIC else "")

    def _on_item_changed(self, it):
        if self._building:
            return
        self._building = True
        try:
            self._validate(it)
        finally:
            self._building = False
        self._changed()

    def invalid_cells(self) -> list[tuple[int, int, str]]:
        out = []
        for r, vals in enumerate(self.rows()):
            for c in range(NUMERIC):
                why = cell_error(c, vals[c])
                if why:
                    out.append((r, c, why))
        return out

    def _selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.table.selectionModel().selectedRows()}) or (
            [self.table.currentRow()] if self.table.currentRow() >= 0 and self.table.selectedItems() else [])

    def add_row(self):
        self._commit_edit()
        rows = self.rows()
        sel = self._selected_rows()
        at = (sel[-1] + 1) if sel else len(rows)
        bias = rows[sel[-1]][0] if sel and not cell_error(0, rows[sel[-1]][0]) else "25"
        rows.insert(at, [bias, "", "", "", ""])
        self._set_rows(rows, select=at)
        self.table.setCurrentCell(at, 1)
        self.table.editItem(self.table.item(at, 1))

    def duplicate_rows(self):
        self._commit_edit()
        sel = self._selected_rows()
        if not sel:
            self.message("Select the rows to duplicate first.")
            return
        rows = self.rows()
        copies = [list(rows[i]) for i in sel]
        at = sel[-1] + 1
        rows[at:at] = copies
        self._set_rows(rows, select=at)

    def delete_rows(self):
        self._commit_edit()
        sel = self._selected_rows()
        if not sel:
            self.message("Select the rows to delete first.")
            return
        rows = [r for i, r in enumerate(self.rows()) if i not in set(sel)]
        self._set_rows(rows, select=min(sel[0], len(rows) - 1) if rows else None)
        self.message(f"Deleted {_rows(len(sel))} (not saved yet).")

    @staticmethod
    def _num(v):
        try:
            x = float(v)
            return (0, x) if math.isfinite(x) else (1, 0.0)
        except ValueError:
            return (1, 0.0)

    def sort_default(self):
        """Bias high → low, then X ascending (the file's original order)."""
        self._commit_edit()
        rows = self.rows()
        rows.sort(key=lambda r: (self._num(r[0])[0], -self._num(r[0])[1], self._num(r[1])))
        self._sort_state = (None, False)
        self.table.horizontalHeader().setSortIndicatorShown(False)
        self._set_rows(rows)

    def _sort_by_column(self, col):
        last, desc = self._sort_state
        desc = (not desc) if last == col else False
        self._sort_state = (col, desc)
        self._commit_edit()
        rows = self.rows()
        if col < NUMERIC:
            valid = [r for r in rows if not self._num(r[col])[0]]
            bad = [r for r in rows if self._num(r[col])[0]]
            valid.sort(key=lambda r: float(r[col]), reverse=desc)
            rows = valid + bad
        else:
            rows.sort(key=lambda r: r[col].lower(), reverse=desc)
        self._set_rows(rows)
        self.table.horizontalHeader().setSortIndicatorShown(True)
        self.table.horizontalHeader().setSortIndicator(
            col, Qt.SortOrder.DescendingOrder if desc else Qt.SortOrder.AscendingOrder)

    # ------------------------------------------------------------------ state
    def is_dirty(self) -> bool:
        """Unsaved edits? (asked by the main window before quitting; an open cell editor counts)"""
        self._commit_edit()
        return self._dirty()

    def _dirty(self) -> bool:
        return _norm(self.rows()) != _norm(self._saved)

    def _changed(self):
        self._refresh_state()
        self._update_tiles()
        self.replot()

    def _refresh_state(self):
        dirty = self._dirty()
        bad = self.invalid_cells()
        if bad:
            self.state.setText(f"● {_cells(len(bad))} invalid")
            self.state.setObjectName("badgeErr")
        elif dirty:
            self.state.setText("● Unsaved changes")
            self.state.setObjectName("badgeWarn")
        else:
            self.state.setText("✓ Matches the file")
            self.state.setObjectName("badgeOk")
        repolish(self.state)
        self.btn_save.setEnabled(dirty and not bad)
        self.btn_revert.setEnabled(dirty)
        n = self.table.rowCount()
        data = numeric_rows(self.rows())
        nb = len({r[1] for r in data})
        txt = f"{_rows(n)} · {nb} {'bias' if nb == 1 else 'biases'} · double-click a cell to edit"
        seen, dup = {}, []
        for i, b, x, *_ in data:
            if (b, x) in seen:
                dup.append(f"{seen[(b, x)] + 1} & {i + 1}")
            seen.setdefault((b, x), i)
        if dup:
            t = ThemeManager.instance().tokens
            txt += (f" · <span style='color:{t.warning}'>⚠ same bias and X in rows {', '.join(dup[:3])}"
                    f"{' …' if len(dup) > 3 else ''}</span>")
        if bad:
            r, c, why = bad[0]
            t = ThemeManager.instance().tokens
            txt = (f"{_rows(n)} · <span style='color:{t.danger}'>row {r + 1}, {COLUMNS[c]}: {why}</span>"
                   f"{f' (+{len(bad) - 1} more)' if len(bad) > 1 else ''}"
                   " · invalid rows are left out of the plot; fix them to save")
        self.info.setText(txt)

    def backup_folder(self) -> Path:
        return self.backup_dir or (paths.app_data_dir() / "backups")

    def _update_note(self):
        t = ThemeManager.instance().tokens
        bdir = str(self.backup_folder())
        home = str(Path.home())
        bdir = "~" + bdir[len(home):] if bdir.startswith(home + os.sep) else bdir
        cols = ", ".join(self.extra)
        extra = (f"<br><span style='color:{t.muted}'>Extra column{'s' if len(self.extra) > 1 else ''} {cols}: "
                 f"shown at the right of the table and written back unchanged.</span>") if self.extra else ""
        self.note_text.setText(
            f"<b>{paths.rel(self.path)}</b> is read by the scripts when they run ({CONSUMERS}). "
            f"Saving affects <b>later</b> runs and figures only; figures already made are not updated."
            f"<br><span style='color:{t.muted}'>Each save first backs up the previous file to {bdir}/</span>{extra}")

    def _update_tiles(self):
        data = numeric_rows(self.rows())
        mirror = self.opt_mirror.isChecked() if hasattr(self, "opt_mirror") else False
        specs = [("Valid points", f"{len(data)}",
                  f"X ∈ [{min(d[2] for d in data):+g}, {max(d[2] for d in data):+g}] µm" if data else "no valid data")]
        for b in sorted({d[1] for d in data}, reverse=True):
            pts = [d for d in data if d[1] == b]
            k = int(np.argmax([d[3] for d in pts]))
            xp, rp, ep = pts[k][2], pts[k][3], pts[k][4]
            hint = f"peak at X = {xp:+g} µm (rms {rp:.2f} ± {ep:.2f})"
            if mirror and b < 0:
                hint += f" · mirrored {-xp:+g}"
            specs.append((self._v(b), f"{len(pts)} point{'' if len(pts) == 1 else 's'}", hint))
        # tiles are reused (hidden when not needed) so nothing stale lingers until a deferred delete
        while len(self._tiles) < len(specs):
            tile = MetricTile("", "—")
            self.tiles.addWidget(tile, 1)
            self._tiles.append(tile)
        for tile, (lab, val, hint) in zip(self._tiles, specs):
            tile.lab.setText(lab)
            tile.set(val, hint.replace("-", "−"))
            tile.show()
        for tile in self._tiles[len(specs):]:
            tile.hide()

    # ------------------------------------------------------------------ plot
    def replot(self, *_):
        self.plot.plot(self._draw)

    def _draw(self, fig):
        t = ThemeManager.instance().tokens
        ax = fig.add_subplot(111)
        self._pts = []
        self._hover = None
        data = numeric_rows(self.rows())
        if not data:
            ax.text(0.5, 0.5, "No valid data points", ha="center", va="center", color=t.muted, transform=ax.transAxes)
            ax.set_axis_off()
            return
        mirror, norm, line = self.opt_mirror.isChecked(), self.opt_norm.isChecked(), self.opt_line.isChecked()
        sel = set(self._selected_rows())
        ax.axvline(0, color=t.border, lw=1, zorder=0)
        ax.axhline(0, color=t.border, lw=1, zorder=0)
        for b in sorted({d[1] for d in data}, reverse=True):
            pts = [d for d in data if d[1] == b]
            idx = np.array([d[0] for d in pts])
            X = np.array([d[2] for d in pts])
            y = np.array([d[3] for d in pts])
            e = np.array([d[4] for d in pts])
            flip = mirror and b < 0
            if flip:
                X = -X
            if norm and np.max(np.abs(y)) > 0:
                s = np.max(np.abs(y))
                y, e = y / s, e / s
            o = np.argsort(X, kind="stable")
            c = bias_color(b, t.dark)
            lab = self._v(b) + ("  (X → −X)" if flip else "")
            ax.errorbar(X[o], y[o], yerr=e[o], fmt="-o" if line else "o", color=c, lw=2, ms=7, capsize=3.5,
                        elinewidth=1.4, mec=t.surface, mew=1.2, label=lab, zorder=3)
            for i, xx, yy in zip(idx, X, y):
                self._pts.append((ax, float(xx), float(yy), int(i)))
                if i in sel:
                    ax.plot([xx], [yy], "o", ms=15, mfc="none", mec=t.text, mew=1.6, zorder=4)
        ax.set_xlabel("X = spot − tip (µm)")
        ax.set_ylabel("THz rms / max per bias" if norm else "THz rms (rel. to ref.)")
        ax.grid(True, axis="y", lw=0.6)
        ax.legend(loc="best")
        self._hover = ax.annotate("", (0, 0), xytext=(12, 12), textcoords="offset points", fontsize=8.5,
                                  bbox=dict(boxstyle="round,pad=0.45", fc=t.surface, ec=t.border), color=t.text,
                                  zorder=10, annotation_clip=False)
        self._hover.set_visible(False)

    def _nearest(self, event, radius=9.0):
        if event.inaxes is None or not self._pts:
            return None
        best, bd = None, radius
        for ax, x, y, row in self._pts:
            if ax is not event.inaxes:
                continue
            px, py = ax.transData.transform((x, y))
            d = math.hypot(px - event.x, py - event.y)
            if d < bd:
                best, bd = (ax, x, y, row), d
        return best

    def _on_hover(self, event):
        if self._hover is None:
            return
        hit = self._nearest(event)
        if hit is None:
            if self._hover.get_visible():
                self._hover.set_visible(False)
                self.plot.canvas.draw_idle()
            return
        ax, x, y, row = hit
        r = self.rows()[row]
        txt = f"row {row + 1}:  {float(r[0]):+g} V,  X = {float(r[1]):+g} µm"
        if x != float(r[1]):
            txt += f"  (mirrored: {x:+g})"
        txt += f"\nrms = {r[2]} ± {r[3]}"
        if r[4]:
            txt += f"\n{r[4]}"
        self._hover.xy = (x, y)
        self._hover.set_text(txt)
        # keep the box inside the axes: flip to the left on the right half
        x0, x1 = ax.get_xlim()
        self._hover.set_position((-12, 12) if x > (x0 + x1) / 2 else (12, 12))
        self._hover.set_ha("right" if x > (x0 + x1) / 2 else "left")
        self._hover.set_visible(True)
        self.plot.canvas.draw_idle()

    def _on_click(self, event):
        hit = self._nearest(event)
        if hit is not None:
            self.table.selectRow(hit[3])
            self.table.scrollToItem(self.table.item(hit[3], 0))

    def _on_option(self, *_):
        self._update_tiles()
        self.replot()

    def _on_selection(self):
        if not self._building:
            self.replot()

    # ------------------------------------------------------------------ misc
    def _open_backups(self):
        d = self.backup_folder()
        d.mkdir(parents=True, exist_ok=True)
        reveal(d)

    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        self.btn_save.setIcon(icons.icon("file", t.on_accent))
        self.note_icon.setPixmap(icons.pixmap("table", t.accent, 20))
        self._building = True
        try:
            for r in range(self.table.rowCount()):
                for c in range(NUMERIC):
                    self._validate(self.table.item(r, c))
        finally:
            self._building = False
        self._update_note()

    def on_show(self):
        m = self._disk_mtime()
        if m != self._mtime:
            if not self.is_dirty():
                self.load()
                self.message("The file was changed on disk and has been re-read.")
            else:
                self.message("⚠ The file was changed on disk. Save overwrites that version (after backing it up); "
                             "Revert loads it.")
