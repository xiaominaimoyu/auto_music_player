"""Editable score dialog used by the SQLite library page."""

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from core.score_model import validate_notes
from gui.widgets import AppDialog


class ScoreEditorDialog(QDialog):
    def __init__(self, score, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑乐谱")
        self.resize(720, 560)
        self._restoring = False
        self._history = []
        self._history_index = -1

        root = QVBoxLayout(self)
        meta = QHBoxLayout()
        meta.addWidget(QLabel("名称"))
        self.name_edit = QLineEdit(str(score.get("name") or ""))
        meta.addWidget(self.name_edit, 1)
        meta.addWidget(QLabel("BPM"))
        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(30, 300)
        self.bpm_spin.setValue(int(score.get("bpm_default") or 100))
        meta.addWidget(self.bpm_spin)
        root.addLayout(meta)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["音符/和弦", "时值(拍)", "半音"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        root.addWidget(self.table, 1)

        actions = QHBoxLayout()
        for text, slot in (
            ("添加音符", self._add_note),
            ("添加休止", self._add_rest),
            ("删除选中", self._delete_selected),
            ("上移", lambda: self._move_selected(-1)),
            ("下移", lambda: self._move_selected(1)),
        ):
            button = QPushButton(text)
            button.setObjectName("BtnSecondary")
            button.clicked.connect(slot)
            actions.addWidget(button)
        actions.addStretch(1)
        self.undo_btn = QPushButton("撤销")
        self.redo_btn = QPushButton("重做")
        self.undo_btn.clicked.connect(self._undo)
        self.redo_btn.clicked.connect(self._redo)
        actions.addWidget(self.undo_btn)
        actions.addWidget(self.redo_btn)
        root.addLayout(actions)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存修改")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._set_rows(score.get("notes") or [])
        self._push_history()
        self.table.itemChanged.connect(self._push_history)

    def _snapshot(self):
        rows = []
        for row in range(self.table.rowCount()):
            rows.append(
                tuple(
                    self.table.item(row, column).text()
                    if self.table.item(row, column) is not None
                    else ""
                    for column in range(3)
                )
            )
        return tuple(rows)

    def _set_snapshot(self, snapshot):
        self._restoring = True
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(0)
            for values in snapshot:
                row = self.table.rowCount()
                self.table.insertRow(row)
                for column, value in enumerate(values):
                    self.table.setItem(row, column, QTableWidgetItem(value))
        finally:
            self.table.blockSignals(False)
            self._restoring = False
        self._update_history_buttons()

    def _set_rows(self, notes):
        snapshot = []
        for item in notes:
            note_text = ",".join(item.get("notes") or []) or "(休止)"
            snapshot.append(
                (note_text, f"{float(item.get('dur', 1.0)):g}", "#" if item.get("semitone") else "")
            )
        self._set_snapshot(snapshot)

    def _push_history(self, *_args):
        if self._restoring:
            return
        snapshot = self._snapshot()
        if self._history_index >= 0 and self._history[self._history_index] == snapshot:
            return
        del self._history[self._history_index + 1 :]
        self._history.append(snapshot)
        self._history_index = len(self._history) - 1
        self._update_history_buttons()

    def _update_history_buttons(self):
        if hasattr(self, "undo_btn"):
            self.undo_btn.setEnabled(self._history_index > 0)
            self.redo_btn.setEnabled(self._history_index + 1 < len(self._history))

    def _append_row(self, values):
        snapshot = list(self._snapshot())
        snapshot.append(tuple(values))
        self._set_snapshot(snapshot)
        self.table.selectRow(len(snapshot) - 1)
        self._push_history()

    def _add_note(self):
        self._append_row(("mid_1", "1", ""))

    def _add_rest(self):
        self._append_row(("(休止)", "1", ""))

    def _delete_selected(self):
        rows = sorted({item.row() for item in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        snapshot = list(self._snapshot())
        for row in rows:
            snapshot.pop(row)
        self._set_snapshot(snapshot)
        self._push_history()

    def _move_selected(self, delta):
        selected = sorted({item.row() for item in self.table.selectedIndexes()})
        if len(selected) != 1:
            return
        row = selected[0]
        target = row + int(delta)
        snapshot = list(self._snapshot())
        if not 0 <= target < len(snapshot):
            return
        snapshot[row], snapshot[target] = snapshot[target], snapshot[row]
        self._set_snapshot(snapshot)
        self.table.selectRow(target)
        self._push_history()

    def _undo(self):
        if self._history_index <= 0:
            return
        self._history_index -= 1
        self._set_snapshot(self._history[self._history_index])

    def _redo(self):
        if self._history_index + 1 >= len(self._history):
            return
        self._history_index += 1
        self._set_snapshot(self._history[self._history_index])

    def notes(self):
        notes = []
        for row in range(self.table.rowCount()):
            cells = [self.table.item(row, column) for column in range(3)]
            if cells[0] is None or cells[1] is None:
                raise ValueError(f"第 {row + 1} 行不完整")
            note_text = cells[0].text().strip()
            try:
                duration = float(cells[1].text().strip())
            except ValueError:
                raise ValueError(f"第 {row + 1} 行时值不是数字") from None
            note_ids = [] if note_text == "(休止)" else [
                value.strip() for value in note_text.split(",") if value.strip()
            ]
            item = {"notes": note_ids, "dur": duration}
            semitone_text = cells[2].text().strip() if cells[2] is not None else ""
            if semitone_text not in ("", "0", "#", "1"):
                raise ValueError(f"第 {row + 1} 行半音只能填写 #、1、0 或留空")
            if semitone_text in ("#", "1"):
                item["semitone"] = 1
            notes.append(item)
        errors = validate_notes(notes, bpm=self.bpm_spin.value())
        if errors:
            detail = "\n".join(str(error) for error in errors[:8])
            raise ValueError(detail)
        if not notes:
            raise ValueError("乐谱不能为空")
        return notes

    def _save(self):
        if not self.name_edit.text().strip():
            AppDialog.show_warning(self, "无法保存", "乐谱名称不能为空")
            return
        try:
            self.notes()
        except ValueError as exc:
            AppDialog.show_error(self, "无法保存", str(exc))
            return
        self.accept()

    def values(self):
        return {
            "name": self.name_edit.text().strip(),
            "bpm_default": self.bpm_spin.value(),
            "notes": self.notes(),
        }
