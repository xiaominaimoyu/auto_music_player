"""乐谱库页：SQLite 列表、编辑及 JSON/MIDI 导入导出。"""

import sqlite3

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.score_io import (
    export_json,
    export_midi,
    import_many,
    import_midi,
    inspect_midi,
)
from core.score_model import ScoreValidationError
from core.transport import MIN_GAME_NOTE_MS, prepare_score
from gui.import_dialog import MidiImportDialog
from gui.score_editor_dialog import ScoreEditorDialog
from gui.theme import BRAND, INK_2, STATE_INFO, STATE_SUCCESS
from gui.widgets import AppDialog


class LibraryTab(QWidget):
    go_play = pyqtSignal(int)
    changed = pyqtSignal()

    def __init__(self, db):
        super().__init__()
        self._db = db
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(16)

        header = QHBoxLayout()
        title = QLabel("已保存乐谱")
        title.setObjectName("PageTitle")
        self.count_label = QLabel("")
        self.count_label.setObjectName("PageSub")
        header.addWidget(title)
        header.addWidget(self.count_label)
        header.addStretch(1)
        root.addLayout(header)

        desc = QLabel("保存过的乐谱无需再次上传,可直接演奏")
        desc.setObjectName("SectionSubtitle")
        root.addWidget(desc)

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("刷新")
        refresh_btn.setObjectName("BtnSecondary")
        refresh_btn.clicked.connect(self.refresh)
        play_btn = QPushButton("去演奏")
        play_btn.setObjectName("BtnPrimary")
        play_btn.clicked.connect(self._go_play_selected)
        edit_btn = QPushButton("编辑选中")
        edit_btn.setObjectName("BtnSecondary")
        edit_btn.clicked.connect(self._edit_selected)
        del_btn = QPushButton("删除选中")
        del_btn.setObjectName("BtnDanger")
        del_btn.clicked.connect(self._delete_selected)
        import_btn = QPushButton("导入乐谱")
        import_btn.setObjectName("BtnSecondary")
        import_btn.setToolTip("从 JSON 或 MIDI 文件导入乐谱(.json / .mid / .midi)")
        import_btn.clicked.connect(self._import_score)
        exp_json_btn = QPushButton("导出 JSON")
        exp_json_btn.setObjectName("BtnSecondary")
        exp_json_btn.setToolTip("导出选中乐谱为 JSON(完整数据结构)")
        exp_json_btn.clicked.connect(lambda: self._export_selected("json"))
        exp_midi_btn = QPushButton("导出 MIDI")
        exp_midi_btn.setObjectName("BtnSecondary")
        exp_midi_btn.setToolTip("导出选中乐谱为标准 MIDI 文件(Type 0)")
        exp_midi_btn.clicked.connect(lambda: self._export_selected("midi"))
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(play_btn)
        btn_row.addWidget(edit_btn)
        btn_row.addWidget(del_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(import_btn)
        btn_row.addWidget(exp_json_btn)
        btn_row.addWidget(exp_midi_btn)
        root.addLayout(btn_row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "名称", "来源", "默认BPM", "保存时间"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.cellDoubleClicked.connect(lambda r, c: self._go_play_row(r))
        root.addWidget(self.table, 1)
        self.refresh()

    def refresh(self):
        scores = self._db.list_scores()
        self.count_label.setText(f"共 {len(scores)} 首")
        self.table.setRowCount(0)
        for s in scores:
            row = self.table.rowCount()
            self.table.insertRow(row)
            id_item = QTableWidgetItem(str(s["id"]))
            id_item.setForeground(QColor(INK_2))
            name_item = QTableWidgetItem(s["name"])
            name_item.setForeground(QColor("#EDEDF2"))
            src = s["source_type"] or "-"
            src_item = QTableWidgetItem(
                "图片" if src == "image"
                else "文档" if src == "document"
                else "导入" if src == "import"
                else "手动" if src == "manual"
                else "-"
            )
            src_item.setForeground(QColor(STATE_INFO if src == "image" else STATE_SUCCESS))
            bpm_item = QTableWidgetItem(str(s["bpm_default"]))
            bpm_item.setForeground(QColor(BRAND))
            date_item = QTableWidgetItem(s["created_at"])
            date_item.setForeground(QColor(INK_2))
            for col, item in enumerate((id_item, name_item, src_item, bpm_item, date_item)):
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

    def _selected_id(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        if not rows:
            AppDialog.show_info(self, "提示", "请先在列表中选择一条乐谱")
            return None
        row = sorted(rows)[0]
        return int(self.table.item(row, 0).text())

    def _go_play_row(self, row):
        self.go_play.emit(int(self.table.item(row, 0).text()))

    def _go_play_selected(self):
        score_id = self._selected_id()
        if score_id is not None:
            self.go_play.emit(score_id)

    def _delete_selected(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        if not rows:
            AppDialog.show_info(self, "提示", "请先在列表中选择一条乐谱")
            return
        row = sorted(rows)[0]
        score_id = int(self.table.item(row, 0).text())
        name = self.table.item(row, 1).text()
        if not AppDialog.confirm(self, "删除乐谱", f"确定删除《{name}》吗?此操作不可恢复。"):
            return
        try:
            self._db.delete_score(score_id)
        except sqlite3.Error as exc:
            AppDialog.show_error(self, "删除失败", str(exc))
            return
        self.refresh()
        self.changed.emit()

    def _edit_selected(self):
        score_id = self._selected_id()
        if score_id is None:
            return
        score = self._db.get_score(score_id)
        if score is None:
            return
        dialog = ScoreEditorDialog(score, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        values = dialog.values()
        try:
            self._db.update_score(
                score_id,
                values["name"],
                values["notes"],
                raw_text=score.get("raw_text") or "",
                bpm_default=values["bpm_default"],
            )
        except (ValueError, sqlite3.Error) as exc:
            AppDialog.show_error(self, "保存失败", str(exc))
            return
        self.refresh()
        self.changed.emit()
        AppDialog.show_success(self, "保存成功", f"《{values['name']}》已更新")

    # ---------- 导入 / 导出 ----------

    def _import_score(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入乐谱", "", "乐谱文件 (*.json *.mid *.midi)")
        if not path:
            return
        try:
            if path.lower().endswith((".mid", ".midi")):
                parsed = inspect_midi(path)
                if not parsed.song.notes:
                    detail = "\n".join(parsed.warnings) or "没有找到可演奏的旋律音符"
                    AppDialog.show_warning(self, "导入失败", detail)
                    return
                dialog = MidiImportDialog(parsed.song, self)
                if dialog.exec() != dialog.DialogCode.Accepted:
                    return
                options = dialog.options()
                result = import_midi(
                    path,
                    track=options.track,
                    style=options.style,
                    transpose=options.transpose,
                    fold_octaves=options.fold_octaves,
                    bpm=int(options.bpm) if options.bpm is not None else None,
                )
                prepared = prepare_score(
                    result.notes,
                    bpm=result.bpm,
                    min_playable_ms=MIN_GAME_NOTE_MS,
                )
                timing_cleanup_count = sum(
                    "游戏可演奏下限" in item.reason
                    for item in prepared.degradations
                )
                result.notes = prepared.notes
                result.degradations.extend(prepared.degradations)
                if timing_cleanup_count:
                    result.warnings.append(
                        f"已将 {timing_cleanup_count} 个短于 {MIN_GAME_NOTE_MS:g}ms、"
                        "游戏难以稳定采样的碎片并入相邻旋律，歌曲总时长不变。"
                    )
                results = [result]
            else:
                results = import_many(path)
        except ScoreValidationError as e:
            AppDialog.show_error(self, "导入失败", f"数据未通过校验:\n{e}")
            return
        except (ValueError, OSError) as e:
            AppDialog.show_error(self, "导入失败", str(e))
            return
        if not results or any(not result.notes for result in results):
            AppDialog.show_warning(self, "导入失败", "至少一首乐谱未能解析出任何音符")
            return

        if len(results) == 1:
            result = results[0]
            info = f"《{result.name}》 · BPM {result.bpm} · {len(result.notes)} 个时间元素"
        else:
            names = "、".join(result.name for result in results[:5])
            if len(results) > 5:
                names += f"等 {len(results)} 首"
            info = f"识别到 {len(results)} 首乐谱：{names}"
        warnings = []
        for result in results:
            warnings.extend(f"《{result.name}》{warning}" for warning in result.warnings)
        if warnings:
            info += "\n\n处理报告：\n" + "\n".join(warnings[:8])
            if len(warnings) > 8:
                info += f"\n…另有 {len(warnings) - 8} 条"
        if not AppDialog.confirm(self, "导入确认", info + "\n\n确定加入乐谱库吗?"):
            return
        records = [
            {
                "name": result.name,
                "notes": result.notes,
                "source_file": path,
                "source_type": "import",
                "bpm_default": result.bpm,
            }
            for result in results
        ]
        try:
            self._db.add_scores(records)
        except (ValueError, sqlite3.Error) as exc:
            AppDialog.show_error(self, "导入失败", f"写入本地曲库失败：{exc}")
            return
        self.refresh()
        self.changed.emit()
        AppDialog.show_success(self, "导入成功", f"已将 {len(results)} 首乐谱加入曲库")

    def _export_selected(self, fmt: str):
        score_id = self._selected_id()
        if score_id is None:
            return
        score = self._db.get_score(score_id)
        if score is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, f"导出乐谱({fmt.upper()})", f"{score['name']}.{fmt}", f"{fmt.upper()} (*.{fmt})"
        )
        if not path:
            return
        try:
            if fmt == "json":
                export_json(path, score)
            else:
                export_midi(path, score)
        except (ValueError, OSError) as e:
            AppDialog.show_error(self, "导出失败", str(e))
            return
        AppDialog.show_success(self, "导出成功", f"已导出到:\n{path}")
