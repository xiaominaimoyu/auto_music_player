import base64
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from core.update_checker import (
    UpdateManifest,
    UpdateCancelled,
    check_for_update,
    download_update,
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

    def _download_manifest(self, **overrides):
        values = dict(
            urls=["https://a.invalid/app.exe", "https://b.invalid/app.exe"],
            size=7, sha256=hashlib.sha256(b"MZ12345").hexdigest(),
        )
        values.update(overrides)
        return UpdateManifest.from_dict(self._payload(**values))

    def _download(self, manifest, directory, **options):
        with patch("core.update_checker.RELEASE_PUBLIC_KEY", self.public_b64):
            return download_update(
                manifest, directory, current_version="1.4.2", current_sequence=142, **options,
            )

    def test_download_retries_and_publishes_only_verified_content(self):
        for broken in (URLError("offline"), b"123", b"MZ54321", b"MZ12345678"):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as directory:
                first = io.BytesIO(broken) if isinstance(broken, bytes) else broken
                progress = Mock()
                with patch("core.update_checker._open_https", side_effect=[first, io.BytesIO(b"MZ12345")]):
                    path = self._download(self._download_manifest(), directory, progress=progress)
                self.assertEqual(path.read_bytes(), b"MZ12345")
                self.assertEqual(path.name, "AutoMusicPlayer-v1.4.3.exe")
                self.assertEqual(list(Path(directory).iterdir()), [path])
                progress.assert_called_with(100)

    def test_download_cancel_and_corruption_leave_no_files(self):
        with tempfile.TemporaryDirectory() as directory:
            cancelled = False

            def progress(percent):
                nonlocal cancelled
                cancelled = percent > 0

            with patch("core.update_checker._open_https", return_value=io.BytesIO(b"MZ12345")) as opener:
                with self.assertRaises(UpdateCancelled):
                    self._download(self._download_manifest(), directory, progress=progress, cancelled=lambda: cancelled)
                self.assertEqual(opener.call_count, 1)
            with patch("core.update_checker._open_https", side_effect=[io.BytesIO(b""), io.BytesIO(b"bad")]):
                with self.assertRaisesRegex(Exception, "所有更新镜像"):
                    self._download(self._download_manifest(), directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_download_never_overwrites_file_created_during_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory, "AutoMusicPlayer-v1.4.3.exe")

            def progress(percent):
                if percent > 0:
                    target.write_bytes(b"keep me")

            with patch("core.update_checker._open_https", return_value=io.BytesIO(b"MZ12345")):
                with self.assertRaises(FileExistsError):
                    self._download(self._download_manifest(), directory, progress=progress)
            self.assertEqual(target.read_bytes(), b"keep me")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_download_revalidates_signature_expiry_and_version_before_network_io(self):
        tampered = self._download_manifest()
        tampered.raw["urls"] = ["https://forged.invalid/file.exe"]
        with tempfile.TemporaryDirectory() as directory, patch("core.update_checker._open_https") as opener:
            for invalid in (
                tampered, self._download_manifest(expires_at="2000-01-01T00:00:00Z"),
                self._download_manifest(sequence=141), self._download_manifest(version="1.4.1"),
            ):
                with self.subTest(invalid=invalid), self.assertRaises(Exception):
                    self._download(invalid, directory)
            opener.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_incompatible_schema_and_minimum_version_block_before_network_or_files(self):
        with tempfile.TemporaryDirectory() as directory, patch("core.update_checker._open_https") as opener:
            for values, expected in (({"db_schema_version": 99}, "曲库格式"),
                                     ({"min_supported": "1.5.0"}, "低于此更新要求")):
                with self.subTest(values=values), self.assertRaisesRegex(Exception, expected):
                    self._download(self._download_manifest(**values), directory)
            opener.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_compatibility_fields_are_rejected(self):
        for values in ({"db_schema_version": -1}, {"min_supported": "broken"}):
            with self.subTest(values=values), self.assertRaises(Exception):
                UpdateManifest.from_dict(self._payload(**values))


if __name__ == "__main__":
    unittest.main()
