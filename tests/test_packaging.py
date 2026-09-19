"""打包配置防回退:锁定运行依赖、资源与 Windows ICU 过滤规则。

hiddenimports 让缺失依赖在 Analysis 阶段失败；ICU 过滤避免构建机 Poppler/Conda
目录中的同名 DLL 遮蔽 Windows 系统 ICU，导致 QtCore 启动失败。
"""

import ast
import os
import unittest


SPEC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AutoMusicPlayer.spec"
)


class TestSpecPackaging(unittest.TestCase):
    def setUp(self):
        with open(SPEC_PATH, encoding="utf-8") as f:
            self.spec_text = f.read()

        tree = ast.parse(self.spec_text, filename=SPEC_PATH)
        keep_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_keep"
        )
        namespace = {}
        keep_module = ast.Module(
            body=[ast.Import(names=[ast.alias(name="fnmatch")]), keep_node],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(keep_module), SPEC_PATH, "exec"), namespace)
        self.keep = namespace["_keep"]

    def test_hiddenimports_declares_pynput(self):
        self.assertIn('"pynput"', self.spec_text)

    def test_hiddenimports_declares_mido(self):
        self.assertIn('"mido"', self.spec_text)

    def test_datas_bundles_profiles_and_config(self):
        self.assertIn('("profiles", "profiles")', self.spec_text)
        self.assertIn('("config.yaml", ".")', self.spec_text)
        self.assertIn('("THIRD_PARTY_NOTICES.md", ".")', self.spec_text)

    def test_filters_icu_dll_even_when_archive_name_has_no_qt_path(self):
        self.assertFalse(self.keep("icuuc.dll"))
        self.assertFalse(self.keep(r"poppler\Library\bin\icudt78.dll"))
        self.assertTrue(self.keep(r"PyQt6\Qt6\bin\Qt6Core.dll"))


if __name__ == "__main__":
    unittest.main()
