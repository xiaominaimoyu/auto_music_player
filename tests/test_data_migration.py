import json

from core.data_migration import migrate_store_data


def test_store_migration_copies_data_and_profiles_without_deleting_source(tmp_path):
    legacy_data = tmp_path / "legacy" / "data"
    legacy_profiles = tmp_path / "legacy" / "profiles"
    target = tmp_path / "Packages" / "PFN" / "LocalState" / "data"
    legacy_data.mkdir(parents=True)
    legacy_profiles.mkdir(parents=True)
    (legacy_data / "scores.db").write_bytes(b"sqlite-placeholder")
    (legacy_data / "sources" / "original.mid").parent.mkdir()
    (legacy_data / "sources" / "original.mid").write_bytes(b"midi")
    (legacy_profiles / "default.yaml").write_text("id: default\n", encoding="utf-8")

    result = migrate_store_data(
        str(target),
        candidate_data_dirs=[str(legacy_data)],
        candidate_profile_dirs=[str(legacy_profiles)],
    )

    assert result.status == "migrated"
    assert result.copied_files == 3
    assert (target / "scores.db").read_bytes() == b"sqlite-placeholder"
    assert (target / "sources" / "original.mid").read_bytes() == b"midi"
    assert (target / "profiles" / "default.yaml").read_text(encoding="utf-8") == "id: default\n"
    assert (legacy_data / "scores.db").exists()
    marker = target.parent / "migration" / "store-v1.json"
    assert marker.is_file()
    assert json.loads(marker.read_text(encoding="utf-8"))["copied_files"] == 3

    repeat = migrate_store_data(
        str(target),
        candidate_data_dirs=[str(legacy_data)],
        candidate_profile_dirs=[str(legacy_profiles)],
    )
    assert repeat.status == "already-complete"


def test_store_migration_preserves_conflicts(tmp_path):
    source = tmp_path / "legacy-data"
    target = tmp_path / "LocalState" / "data"
    source.mkdir()
    target.mkdir(parents=True)
    (source / "scores.db").write_bytes(b"legacy")
    (target / "scores.db").write_bytes(b"newer")

    result = migrate_store_data(
        str(target),
        candidate_data_dirs=[str(source)],
        candidate_profile_dirs=[],
    )

    assert result.status == "migrated"
    assert result.conflict_files == 1
    assert (target / "scores.db").read_bytes() == b"newer"
    assert len(list(target.glob("scores.db.legacy-*"))) == 1
    assert result.backup_path is not None


def test_store_migration_without_source_is_retryable(tmp_path):
    target = tmp_path / "LocalState" / "data"
    result = migrate_store_data(
        str(target), candidate_data_dirs=[], candidate_profile_dirs=[]
    )

    assert result.status == "no-source"
    assert not (target.parent / "migration" / "store-v1.json").exists()
