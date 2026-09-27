import base64
from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest.mock import patch

from core.update_checker import (
    UpdateManifest,
    check_for_update,
    verify_manifest,
)


class UpdateCheckerTests(unittest.TestCase):
    def setUp(self):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        except ImportError:
            self.skipTest("cryptography 未安装")
        self.private = Ed25519PrivateKey.generate()
        public = self.private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.public_b64 = base64.b64encode(public).decode("ascii")

    def _payload(self, *, version="1.4.3", sequence=143, **overrides):
        payload = {
            "schema_version": 1,
            "version": version,
            "sequence": sequence,
            "min_supported": "1.4.2",
            "urls": ["https://example.invalid/AutoMusicPlayer.exe"],
            "size": 10,
            "sha256": "0" * 64,
            "release_notes": "test",
        }
        payload.update(overrides)
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        payload["signature"] = base64.b64encode(self.private.sign(canonical)).decode("ascii")
        return payload

    def test_signed_manifest_is_verified_and_detected(self):
        payload = self._payload()
        manifest = UpdateManifest.from_dict(payload)
        verify_manifest(manifest, self.public_b64)
        with patch("core.update_checker.fetch_manifest", return_value=payload):
            result = check_for_update(
                "1.4.2",
                ["https://example.invalid/manifest.json"],
                self.public_b64,
                current_sequence=142,
            )
        self.assertTrue(result.update_available)
        self.assertEqual(result.manifest.version, "1.4.3")

    def test_newest_valid_mirror_wins_over_first_stale_mirror(self):
        stale = self._payload(version="1.4.2", sequence=142)
        fresh = self._payload(version="1.4.3", sequence=143)
        with patch("core.update_checker.fetch_manifest", side_effect=[stale, fresh]):
            result = check_for_update(
                "1.4.2",
                ["https://a.invalid/manifest.json", "https://b.invalid/manifest.json"],
                self.public_b64,
                current_sequence=142,
            )
        self.assertTrue(result.update_available)
        self.assertEqual(result.manifest.sequence, 143)
        self.assertEqual(result.source_url, "https://b.invalid/manifest.json")

    def test_forged_signature_is_rejected(self):
        payload = self._payload()
        payload["version"] = "9.9.9"
        manifest = UpdateManifest.from_dict(payload)
        with self.assertRaisesRegex(Exception, "签名校验失败"):
            verify_manifest(manifest, self.public_b64)

    def test_expired_manifest_is_rejected_and_next_mirror_is_used(self):
        expired = self._payload(
            expires_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        )
        fresh = self._payload()
        with patch("core.update_checker.fetch_manifest", side_effect=[expired, fresh]):
            result = check_for_update(
                "1.4.2",
                ["https://a.invalid/manifest.json", "https://b.invalid/manifest.json"],
                self.public_b64,
                current_sequence=142,
            )
        self.assertTrue(result.update_available)
        self.assertEqual(result.source_url, "https://b.invalid/manifest.json")

    def test_invalid_or_rollback_manifest_is_rejected(self):
        with self.assertRaisesRegex(Exception, "HTTPS"):
            UpdateManifest.from_dict(self._payload(urls=["http://example.invalid/app.exe"]))
        with self.assertRaisesRegex(Exception, "sha256"):
            UpdateManifest.from_dict(self._payload(sha256="not-a-hash"))
        with patch(
            "core.update_checker.fetch_manifest",
            return_value=self._payload(version="1.4.1", sequence=141),
        ):
            result = check_for_update(
                "1.4.2",
                ["https://example.invalid/manifest.json"],
                self.public_b64,
                current_sequence=142,
            )
        self.assertFalse(result.update_available)
        self.assertIn("低于当前序列号", result.error)


if __name__ == "__main__":
    unittest.main()
