import os

import main
import pytest


def test_store_data_dir_is_always_under_local_state(monkeypatch, tmp_path):
    local_state = tmp_path / "Packages" / "AutoMusicPlayer_test" / "LocalState"
    monkeypatch.setenv("AUTOMUSIC_STORE_PACKAGE", "1")
    monkeypatch.setenv("PACKAGE_FAMILY_NAME", "AutoMusicPlayer_test")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(main, "is_store_package", lambda: True)

    resolved = main._resolve_data_dir(
        str(tmp_path / "ignored" / "config.yaml"), "data", "auto"
    )
    assert resolved == str(local_state / "data")


def test_store_absolute_data_setting_cannot_escape_package_state(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOMUSIC_STORE_PACKAGE", "1")
    monkeypatch.setenv("PACKAGE_FAMILY_NAME", "AutoMusicPlayer_test")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(main, "is_store_package", lambda: True)

    resolved = main._resolve_data_dir(
        str(tmp_path / "ignored" / "config.yaml"), os.path.abspath("C:/outside"), "auto"
    )
    assert resolved == str(tmp_path / "Packages" / "AutoMusicPlayer_test" / "LocalState" / "data")


def test_store_relative_data_setting_cannot_traverse_outside(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOMUSIC_STORE_PACKAGE", "1")
    monkeypatch.setenv("PACKAGE_FAMILY_NAME", "AutoMusicPlayer_test")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(main, "is_store_package", lambda: True)

    with pytest.raises(ValueError, match="LocalState"):
        main._resolve_data_dir(
            str(tmp_path / "ignored" / "config.yaml"), "../outside", "auto"
        )


def test_store_frozen_config_is_copied_to_local_state(monkeypatch, tmp_path):
    bundle = tmp_path / "package-bundle"
    bundle.mkdir()
    (bundle / "config.yaml").write_text("config_version: 2\n", encoding="utf-8")
    monkeypatch.setenv("AUTOMUSIC_STORE_PACKAGE", "1")
    monkeypatch.setenv("PACKAGE_FAMILY_NAME", "AutoMusicPlayer_test")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(main.sys, "frozen", True, raising=False)
    monkeypatch.setattr(main.sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(main.sys, "executable", str(bundle / "AutoMusicPlayer.exe"))

    config_path = main.ensure_config()

    expected = tmp_path / "Packages" / "AutoMusicPlayer_test" / "LocalState" / "config.yaml"
    assert config_path == str(expected)
    assert expected.read_text(encoding="utf-8") == "config_version: 2\n"
