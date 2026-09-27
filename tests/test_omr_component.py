import json
import tempfile
import unittest
from pathlib import Path

from core.omr_component import find_jianpu_component


class OmrComponentTests(unittest.TestCase):
    def test_source_only_wrapper_is_not_reported_as_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "omr" / "jianpu_omr"
            root.mkdir(parents=True)
            (root / "jianpu_omr.cmd").write_text("@echo off\n", encoding="utf-8")
            info = find_jianpu_component(tmp)
            self.assertFalse(info.installed)

    def test_bundled_node_wrapper_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "omr" / "jianpu_omr"
            root.mkdir(parents=True)
            (root / "jianpu_omr.cmd").write_text("@echo off\n", encoding="utf-8")
            (root / "node.exe").write_bytes(b"test")
            (root / "component.json").write_text(
                json.dumps({"managed_by_app": True}), encoding="utf-8"
            )
            info = find_jianpu_component(tmp)
            self.assertTrue(info.installed)
            self.assertEqual(Path(info.executable).name, "jianpu_omr.cmd")


if __name__ == "__main__":
    unittest.main()
