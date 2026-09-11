"""乐谱库页：列表、详情编辑、试听、导入导出与演奏入口。"""

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

from core.score_io import export_json, export_midi, import_any
from core.score_model import ScoreValidationError
from gui.theme import BRAND, INK_2, STATE_INFO, STATE_SUCCESS
from gui.widgets import AppDialog


class LibraryTab(QWidget):
    go_play = pyqtSignal(int)

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
        detail_btn = QPushButton("查看详情")
        detail_btn.setObjectName("BtnSecondary")
        detail_btn.clicked.connect(self._open_selected)
        play_btn = QPushButton("去演奏")
        play_btn.setObjectName("BtnPrimary")
        play_btn.clicked.connect(self._go_play_selected)
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
        btn_row.addWidget(detail_btn)
        btn_row.addWidget(play_btn)
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
        self.table.cellDoubleClicked.connect(lambda r, _c: self._open_row(r))
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

    def _open_row(self, row):
        self._open_detail(int(self.table.item(row, 0).text()))

    def _open_selected(self):
        score_id = self._selected_id()
        if score_id is not None:
            self._open_detail(score_id)

    def _open_detail(self, score_id):
        score = self._db.get_score(score_id)
        if score is None:
            return
        from gui.score_detail_dialog import ScoreDetailDialog

        dialog = ScoreDetailDialog(self, self._db, score)
        dialog.exec()
        if dialog.saved:
            self.refresh()
        if dialog.play_requested:
            self.go_play.emit(score_id)

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
        self._db.delete_score(score_id)
        self.refresh()

    # ---------- 导入 / 导出 ----------

    def _import_score(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入乐谱", "", "乐谱文件 (*.json *.mid *.midi)")
        if not path:
            return
        try:
            result = import_any(path)
        except ScoreValidationError as e:
            AppDialog.show_error(self, "导入失败", f"数据未通过校验:\n{e}")
            return
        except (ValueError, OSError) as e:
            AppDialog.show_error(self, "导入失败", str(e))
            return
        if not result.notes:
            AppDialog.show_warning(self, "导入失败", "未能从文件中解析出任何音符")
            return
        info = f"《{result.name}》 · BPM {result.bpm} · {len(result.notes)} 个音符"
        if result.warnings:
            info += "\n" + "\n".join(result.warnings[:5])
        if not AppDialog.confirm(self, "导入确认", info + "\n\n确定加入乐谱库吗?"):
            return
        self._db.add_score(
            name=result.name,
            notes=result.notes,
            source_file=path,
            source_type="import",
            bpm_default=result.bpm,
        )
        self.refresh()
        AppDialog.show_success(self, "导入成功", f"《{result.name}》已加入乐谱库")

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
