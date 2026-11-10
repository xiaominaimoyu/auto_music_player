"""Release checks stop mismatched metadata and do not replace valid manifests on failure."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools.release_check import check_artifact, check_versions
from tools.update_manifest import generate_key, prepare_manifest


class ReleaseCheckTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "core").mkdir()
        (self.root / "updates").mkdir()
        (self.root / "core" / "version.py").write_text('APP_VERSION="1.5.1"\nUPDATE_SEQUENCE=151\n')
        (self.root / "version.txt").write_text("1.5.1\n")
        self.manifest_path = self.root / "updates" / "stable.json"
        self.manifest_path.write_text("{}")
        self.artifact = self.root / "app.exe"
        self.artifact.write_bytes(b"MZsample executable")
        self.key = self.root / "test-private.pem"
        with redirect_stdout(io.StringIO()) as captured:
            generate_key(SimpleNamespace(private_key=str(self.key), public_key_file=None, force=False))
        self.public = captured.getvalue().strip()
        self.options = SimpleNamespace(artifact=str(self.artifact), private_key=str(self.key),
                                       min_supported="1.4.2", url=["https://example.invalid/app.exe"],
                                       notes=None, output=str(self.manifest_path))
        self.prepare()

    def prepare(self):
        with patch("tools.update_manifest.PROJECT_ROOT", self.root), \
             patch("tools.update_manifest.RELEASE_PUBLIC_KEY", self.public), \
             redirect_stdout(io.StringIO()):
            prepare_manifest(self.options)

    def test_generated_manifest_matches_versions_and_final_artifact(self):
        with patch("tools.release_check.RELEASE_PUBLIC_KEY", self.public):
            self.assertEqual(check_artifact(self.root, self.artifact).version, "1.5.1")
        self.artifact.write_bytes(b"MZchanged executable")
        with patch("tools.release_check.RELEASE_PUBLIC_KEY", self.public), self.assertRaisesRegex(ValueError, "SHA-256"):
            check_artifact(self.root, self.artifact)

    def test_partial_version_changes_are_rejected_before_build(self):
        (self.root / "version.txt").write_text("1.5.2\n")
        with self.assertRaisesRegex(ValueError, "version.txt=1.5.2"):
            check_versions(self.root)
        (self.root / "version.txt").write_text("1.5.1\n")
        value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        value["sequence"] = 150
        self.manifest_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "updates/stable.json=1.5.1/150"):
            check_versions(self.root)

    def test_wrong_signing_key_preserves_previous_manifest(self):
        original = self.manifest_path.read_bytes()
        with patch("tools.update_manifest.PROJECT_ROOT", self.root), self.assertRaises(Exception):
            prepare_manifest(self.options)
        self.assertEqual(self.manifest_path.read_bytes(), original)

    def test_spec_runs_version_gate_before_analysis(self):
        spec = Path(__file__).resolve().parents[1] / "AutoMusicPlayer.spec"
        (self.root / "version.txt").write_text("1.5.2\n")
        with self.assertRaisesRegex(ValueError, "发布版本不一致"):
            exec(compile(spec.read_text(encoding="utf-8"), str(spec), "exec"), {"SPECPATH": str(self.root)})
