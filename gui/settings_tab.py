"""模型设置页(对齐 Web 设计稿):供应商列表 + 配置表单 + 测试连接 + 激活。"""

import threading

import requests
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.theme import BRAND, BRAND_SOFT, INK_2, STATE_ERROR, STATE_SUCCESS
from gui.widgets import AppDialog


class SettingsTab(QWidget):
    providers_saved = pyqtSignal()
    test_done = pyqtSignal(bool, str)

    def __init__(self, settings_store):
        super().__init__()
        self._store = settings_store
        self._current_id = None
        self._testing = False
        self._build_ui()
        self.test_done.connect(self._on_test_done)
        self.refresh_provider_status()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(16)

        header = QHBoxLayout()
        title = QLabel("模型设置")
        title.setObjectName("PageTitle")
        self.provider_status = QLabel("")
        self.provider_status.setObjectName("PageSub")
        header.addWidget(title)
        header.addWidget(self.provider_status)
        header.addStretch(1)
        root.addLayout(header)

        note = QLabel("模型用于识别上传的乐谱图或乐谱文档")
        note.setFixedHeight(40)
        note.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        note.setStyleSheet(
            f"color: {INK_2}; font-size: 13px; padding: 0 14px; "
            f"background: #1E1E28; border-radius: 8px; border: 1px solid #26262F;"
        )
        root.addWidget(note)

        body = QHBoxLayout()
        body.setSpacing(16)

        # 左侧:供应商列表
        left = QFrame()
        left.setObjectName("SectionCard")
        left.setFixedWidth(260)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(10)
        list_header = QHBoxLayout()
        lt = QLabel("供应商列表")
        lt.setObjectName("SectionTitle")
        list_header.addWidget(lt, 1)
        add_btn = QPushButton("添加")
        add_btn.setObjectName("BtnPrimary")
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.clicked.connect(self._add_provider)
        list_header.addWidget(add_btn)
        left_layout.addLayout(list_header)
        self.provider_list = QListWidget()
        self.provider_list.setObjectName("ProviderList")
        self.provider_list.currentRowChanged.connect(self._on_select_provider)
        left_layout.addWidget(self.provider_list, 1)
        body.addWidget(left)

        # 右侧:配置表单
        right = QFrame()
        right.setObjectName("SectionCard")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(24, 20, 24, 20)
        right_layout.setSpacing(14)

        detail_header = QHBoxLayout()
        dt = QLabel("供应商配置")
        dt.setObjectName("SectionTitle")
        self.active_tag = QLabel("")
        self.active_tag.setFixedHeight(24)
        self.active_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.active_tag.setStyleSheet(
            f"font-size: 12px; color: {STATE_SUCCESS}; padding: 0 10px; "
            f"background: rgba(74, 222, 128, 0.1); border-radius: 999px;"
        )
        detail_header.addWidget(dt)
        detail_header.addStretch(1)
        detail_header.addWidget(self.active_tag)
        right_layout.addLayout(detail_header)

        self.name_edit = self._form_row(right_layout, "供应商名称", "如: OpenAI")
        self.url_edit = self._form_row(
            right_layout, "Base URL", "https://api.example.com/v1", mono=True,
            hint="多模态图片识别模型的 API 基础地址,需兼容 OpenAI Chat Completions 格式",
        )
        self.key_edit = self._form_row(
            right_layout, "API Key", "sk-...", mono=True, password=True,
            hint="用于鉴权的 API 密钥,仅本地存储",
        )
        self.model_edit = self._form_row(
            right_layout, "模型名称", "如: qwen-vl-max, glm-4v", mono=True,
            hint="支持图片输入的多模态模型名称",
        )

        actions = QHBoxLayout()
        self.test_btn = QPushButton("测试连接")
        self.test_btn.setObjectName("BtnSecondary")
        self.test_btn.clicked.connect(self._test_connection)
        actions.addWidget(self.test_btn)
        actions.addStretch(1)
        self.delete_btn = QPushButton("删除供应商")
        self.delete_btn.setObjectName("BtnDanger")
        self.delete_btn.clicked.connect(self._delete_provider)
        self.activate_btn = QPushButton("使用此模型")
        self.activate_btn.setObjectName("BtnSecondary")
        self.activate_btn.clicked.connect(self._activate)
        self.save_btn = QPushButton("保存配置")
        self.save_btn.setObjectName("BtnPrimary")
        self.save_btn.clicked.connect(self._save)
        actions.addWidget(self.delete_btn)
        actions.addWidget(self.activate_btn)
        actions.addWidget(self.save_btn)
        right_layout.addLayout(actions)
        right_layout.addStretch(1)
        body.addWidget(right, 1)

        root.addLayout(body, 1)

    def _form_row(self, parent_layout, label_text, placeholder, mono=False, password=False, hint=""):
        col = QVBoxLayout()
        col.setSpacing(6)
        label = QLabel(label_text)
        label.setObjectName("FieldLabel")
        col.addWidget(label)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        if mono:
            edit.setStyleSheet("font-family: Consolas;")
        if password:
            edit.setEchoMode(QLineEdit.EchoMode.Password)
        col.addWidget(edit)
        if hint:
            h = QLabel(hint)
            h.setObjectName("HintText")
            col.addWidget(h)
        parent_layout.addLayout(col)
        return edit

    # ---------- 数据加载 ----------

    def refresh_provider_status(self):
        active = self._store.get_active()
        if active and active.get("api_key"):
            self.provider_status.setText(f"识别模型: {active['name']} · {active['model']}")
        elif active:
            self.provider_status.setText(f"已选 {active['name']} · {active['model']}（API Key 未填写）")
        else:
            self.provider_status.setText("识别模型: 未配置")
        self._reload_list()

    def _reload_list(self):
        selected_id = self._current_id
        self.provider_list.blockSignals(True)
        self.provider_list.clear()
        for p in self._store.get_providers():
            item = QListWidgetItem(f"{p['name']}  {'● 当前使用' if p.get('active') else ''}")
            item.setData(Qt.ItemDataRole.UserRole, p["id"])
            if p.get("active"):
                item.setForeground(Qt.GlobalColor.white)
            self.provider_list.addItem(item)
        self.provider_list.blockSignals(False)
        if self.provider_list.count() > 0:
            target_row = 0
            for row in range(self.provider_list.count()):
                if self.provider_list.item(row).data(Qt.ItemDataRole.UserRole) == selected_id:
                    target_row = row
                    break
            self.provider_list.setCurrentRow(target_row)
        else:
            self._current_id = None
            self.name_edit.clear()
            self.url_edit.clear()
            self.key_edit.clear()
            self.model_edit.clear()
            self.active_tag.setText("未配置")
            self.active_tag.setStyleSheet(
                f"font-size: 12px; color: {INK_2}; padding: 0 10px; "
                f"background: rgba(148, 148, 162, 0.1); border-radius: 999px;"
            )

    def _on_select_provider(self, row):
        if row < 0:
            return
        item = self.provider_list.item(row)
        pid = item.data(Qt.ItemDataRole.UserRole)
        self._load_form(pid)

    def _load_form(self, pid):
        p = self._store.get_provider(pid)
        if p is None:
            return
        self._current_id = pid
        self.name_edit.setText(p.get("name", ""))
        self.url_edit.setText(p.get("base_url", ""))
        self.key_edit.setText(p.get("api_key", ""))
        self.model_edit.setText(p.get("model", ""))
        if p.get("active"):
            self.active_tag.setText("✓ 当前使用")
            self.active_tag.setStyleSheet(
                f"font-size: 12px; color: {STATE_SUCCESS}; padding: 0 10px; "
                f"background: rgba(74, 222, 128, 0.1); border-radius: 999px;"
            )
            self.activate_btn.setText("当前模型")
            self.activate_btn.setEnabled(False)
        else:
            self.active_tag.setText("未使用")
            self.active_tag.setStyleSheet(
                f"font-size: 12px; color: {INK_2}; padding: 0 10px; "
                f"background: rgba(148, 148, 162, 0.1); border-radius: 999px;"
            )
            self.activate_btn.setText("使用此模型")
            self.activate_btn.setEnabled(True)

    # ---------- 操作 ----------

    def _save(self):
        if not self._current_id:
            AppDialog.show_info(self, "提示", "请先添加并选择供应商")
            return
        fields = {
            "name": self.name_edit.text().strip(),
            "base_url": self.url_edit.text().strip(),
            "api_key": self.key_edit.text().strip(),
            "model": self.model_edit.text().strip(),
        }
        if not fields["name"] or not fields["base_url"] or not fields["model"]:
            AppDialog.show_warning(self, "提示", "供应商名称、Base URL、模型名称不能为空")
            return
        self._store.update_provider(self._current_id, fields)
        self.providers_saved.emit()
        self.refresh_provider_status()
        AppDialog.show_success(self, "成功", "配置已保存")

    def _activate(self):
        if not self._current_id:
            AppDialog.show_info(self, "提示", "请先添加并选择供应商")
            return
        self._store.activate(self._current_id)
        self.providers_saved.emit()
        self.refresh_provider_status()

    def _add_provider(self):
        pid = self._store.add_provider()
        self._current_id = pid
        self.providers_saved.emit()
        self.refresh_provider_status()
        for i in range(self.provider_list.count()):
            if self.provider_list.item(i).data(Qt.ItemDataRole.UserRole) == pid:
                self.provider_list.setCurrentRow(i)
                break

    def _delete_provider(self):
        if not self._current_id:
            AppDialog.show_info(self, "提示", "请先添加并选择供应商")
            return
        providers = self._store.get_providers()
        if len(providers) <= 1:
            AppDialog.show_warning(self, "提示", "至少保留一个供应商")
            return
        active = self._store.get_active()
        if active and active["id"] == self._current_id:
            AppDialog.show_warning(self, "提示", "请先激活其他供应商后再删除")
            return
        self._store.delete_provider(self._current_id)
        self.refresh_provider_status()

    def _test_connection(self):
        if self._testing:
            return
        url = self.url_edit.text().strip().rstrip("/")
        key = self.key_edit.text().strip()
        if not url or not key:
            AppDialog.show_warning(self, "提示", "请先填写 Base URL 和 API Key")
            return
        self._testing = True
        self.test_btn.setEnabled(False)
        self.test_btn.setText("测试中...")
        t = threading.Thread(target=self._test_worker, args=(url, key), daemon=True)
        t.start()

    def _test_worker(self, url, key):
        try:
            resp = requests.get(f"{url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=15)
            ok = resp.status_code == 200
            self.test_done.emit(ok, f"HTTP {resp.status_code}")
        except Exception as e:
            self.test_done.emit(False, str(e))

    def _on_test_done(self, ok, msg):
        self._testing = False
        self.test_btn.setEnabled(True)
        self.test_btn.setText("测试连接")
        if ok:
            AppDialog.show_success(self, "测试连接", f"连接成功({msg})")
        else:
            AppDialog.show_error(self, "测试连接", f"连接失败: {msg}")
