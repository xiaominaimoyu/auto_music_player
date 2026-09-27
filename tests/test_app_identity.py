from core.app_identity import (
    channel_name,
    is_store_package,
    stable_app_root,
    store_local_state_dir,
)


def test_store_package_uses_pfn_local_state(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOMUSIC_STORE_PACKAGE", "1")
    monkeypatch.setenv("PACKAGE_FAMILY_NAME", "AutoMusicPlayer_abc123")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert is_store_package() is True
    assert channel_name() == "store"
    assert store_local_state_dir() == str(
        tmp_path / "Packages" / "AutoMusicPlayer_abc123" / "LocalState"
    )


def test_explicit_portable_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOMUSIC_STORE_PACKAGE", "0")
    monkeypatch.setenv("PACKAGE_FAMILY_NAME", "AutoMusicPlayer_abc123")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert is_store_package() is False
    assert channel_name() == "portable"
    assert stable_app_root() == str(tmp_path / "AutoMusicPlayer")


def test_store_path_is_safe_without_package_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOMUSIC_STORE_PACKAGE", "1")
    monkeypatch.delenv("PACKAGE_FAMILY_NAME", raising=False)
    monkeypatch.delenv("APP_PACKAGE_FAMILY_NAME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert store_local_state_dir() == str(
        tmp_path / "AutoMusicPlayer" / "store-local-state"
    )
