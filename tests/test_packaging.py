"""打包配置防回退:M7 首次冒烟发现 exe 缺 pynput(TRAE 打包环境漏装且 PyInstaller 静默跳过)。

修复:AutoMusicPlayer.spec 的 hiddenimports 显式声明 pynput——打包环境缺失时
Analysis 会报错终止,而不是静默产出启动即崩的残缺 exe。本测试锁死该声明。
"""

import os
import unittest


SPEC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AutoMusicPlayer.spec"
)


class TestSpecPackaging(unittest.TestCase):
    def setUp(self):
        with open(SPEC_PATH, encoding="utf-8") as f:
            self.spec_text = f.read()

    def test_hiddenimports_declares_pynput(self):
        self.assertIn('hiddenimports=["pynput"]', self.spec_text)

    def test_datas_bundles_profiles_and_config(self):
        self.assertIn('("profiles", "profiles")', self.spec_text)
        self.assertIn('("config.yaml", ".")', self.spec_text)


if __name__ == "__main__":
    unittest.main()