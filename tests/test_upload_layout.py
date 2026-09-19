"""上传识别页校对卡片布局回归测试。"""

import os
import tempfile

from PyQt6.QtCore import QObject, QPoint, pyqtSignal
from PyQt6.QtWidgets import QApplication

from core.database import ScoreDB
from gui.upload_tab import UploadTab
from gui.widgets import BottomResizableCard


class _FakePreviewPlayer(QObject):
    finished = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)

    def play(self, *args, **kwargs):
        return True

    def stop(self):
        pass


def _qapp():
    return QApplication.instance() or QApplication([])


def test_proofreading_card_keeps_recording_controls_inside_card():
    app = _qapp()
    with tempfile.TemporaryDirectory() as temp_dir:
        db = ScoreDB(os.path.join(temp_dir, "scores.db"))
        preview = _FakePreviewPlayer()
        tab = UploadTab(db, preview_player=preview)
        try:
            tab.resize(1000, 900)
            tab.show()
            QApplication.processEvents()

            cards = tab.findChildren(BottomResizableCard)
            assert len(cards) >= 2
            card = cards[1]
            assert card.height() >= card.layout().sizeHint().height()

            controls = (
                tab.table,
                tab.preview_btn,
                tab.record_octave_combo,
                tab.record_dur_combo,
                tab.record_semitone_btn,
                tab.live_record_btn,
                tab.live_stop_btn,
                tab.live_record_status,
                tab.preview_status,
            )
            for widget in controls:
                top_left = widget.mapTo(card, QPoint(0, 0))
                assert top_left.y() >= 0
                assert top_left.y() + widget.height() <= card.height()
        finally:
            tab.close()
            tab.deleteLater()
            app.processEvents()
            db.conn.close()
