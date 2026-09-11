"""模型供应商默认配置与激活行为。"""

import json
import os
import tempfile
import unittest

from core.settings_store import SettingsStore


class TestSettingsStore(unittest.TestCase):
    def test_first_run_has_active_test_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(os.path.join(directory, "settings.json"))
            active = store.get_active()
            self.assertIsNotNone(active)
            self.assertEqual(active["base_url"], "https://api.774966.xyz/v1")
            self.assertEqual(active["model"], "gpt-6-astra")
            self.assertTrue(active["api_key"])

    def test_existing_provider_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "settings.json")
            custom = [{
                "id": "custom",
                "name": "自定义",
                "base_url": "https://example.test/v1",
                "api_key": "key",
                "model": "vision",
                "active": True,
            }]
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(custom, stream)
            store = SettingsStore(path)
            self.assertEqual(store.get_providers(), custom)

    def test_legacy_embedded_provider_model_is_upgraded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "settings.json")
            legacy = [{
                "id": "test-774966",
                "name": "774966 测试渠道",
                "base_url": "https://api.774966.xyz/v1",
                "api_key": "temporary-key",
                "model": "gpt-5.6-sol",
                "active": True,
            }]
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(legacy, stream)
            store = SettingsStore(path)
            self.assertEqual(store.get_active()["model"], "gpt-6-astra")

    def test_new_provider_becomes_active(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(os.path.join(directory, "settings.json"))
            provider_id = store.add_provider("新模型", "https://new.test/v1", "key", "vision")
            self.assertEqual(store.get_active()["id"], provider_id)
            self.assertEqual(sum(bool(item["active"]) for item in store.get_providers()), 1)


if __name__ == "__main__":
    unittest.main()
