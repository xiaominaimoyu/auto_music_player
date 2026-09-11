"""多供应商模型配置存储(JSON 本地文件,含 API Key,不入库不提交)。"""

import copy
import json
import os
import uuid

# 测试阶段的首次启动配置。仅在本地没有任何供应商时写入，已有配置不会被覆盖。
DEFAULT_PROVIDERS = [
    {
        "id": "test-774966",
        "name": "774966 测试渠道",
        "base_url": "https://api.774966.xyz/v1",
        "api_key": "sk-zAbqTwB7SUsdQ6mFT0lImnWf98Z5QU6255Qr73WxJpBxcFCG",
        "model": "gpt-6-astra",
        "active": True,
    }
]


class SettingsStore:
    """供应商配置存取。api_key 仅存本地 JSON 文件。"""

    def __init__(self, path: str):
        self.path = path
        self._providers = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    for provider in data:
                        if (
                            provider.get("id") == "test-774966"
                            and provider.get("model") == "gpt-5.6-sol"
                        ):
                            provider["model"] = "gpt-6-astra"
                    return data
            except Exception:
                pass
        return copy.deepcopy(DEFAULT_PROVIDERS)

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._providers, f, ensure_ascii=False, indent=2)

    def get_providers(self):
        return self._providers

    def get_active(self):
        for p in self._providers:
            if p.get("active"):
                return p
        return None

    def get_provider(self, pid):
        for p in self._providers:
            if p["id"] == pid:
                return p
        return None

    def update_provider(self, pid, fields: dict):
        p = self.get_provider(pid)
        if p is None:
            return False
        p.update(fields)
        self.save()
        return True

    def activate(self, pid):
        changed = False
        for p in self._providers:
            want = p["id"] == pid
            if p.get("active") != want:
                p["active"] = want
                changed = True
        if changed:
            self.save()
        return changed

    def add_provider(self, name="新供应商", base_url="", api_key="", model=""):
        for item in self._providers:
            item["active"] = False
        p = {
            "id": uuid.uuid4().hex[:8],
            "name": name,
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "active": True,
        }
        self._providers.append(p)
        self.save()
        return p["id"]

    def delete_provider(self, pid):
        p = self.get_provider(pid)
        if p is None:
            return False
        self._providers.remove(p)
        self.save()
        return True
