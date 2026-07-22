"""Path validation for user-supplied file paths (backup/restore, CSV export/import).

Policy: paths inside the HA config directory are implicitly allowed (the
integration's defaults and documented examples live there) EXCEPT the
internal .storage directory; anything else must pass
hass.config.is_allowed_path (allowlist_external_dirs).  Validation runs
before any file access, so traversal cannot be used as a write primitive
or a file-existence oracle.
"""
from unittest.mock import MagicMock, patch

import pytest

from homeassistant.exceptions import HomeAssistantError

from custom_components.heating_analytics.storage import StorageManager


@pytest.fixture
def storage(mock_coordinator, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    mock_coordinator.hass.config.path = MagicMock(return_value=str(config_dir))
    mock_coordinator.hass.config.is_allowed_path = MagicMock(return_value=False)
    with patch("custom_components.heating_analytics.storage.Store"):
        sm = StorageManager(mock_coordinator)
    return sm, config_dir, tmp_path


def test_config_dir_paths_allowed(storage):
    sm, config_dir, _ = storage
    inside = config_dir / "backup.json"
    assert sm._validate_external_path(str(inside)) == str(inside)
    # Traversal that RESOLVES back inside the config dir is fine.
    dodgy_but_inside = config_dir / "sub" / ".." / "backup.json"
    assert sm._validate_external_path(str(dodgy_but_inside)) == str(inside)


def test_traversal_escaping_config_dir_denied(storage):
    sm, config_dir, tmp_path = storage
    outside_via_traversal = config_dir / ".." / "etc" / "passwd"
    with pytest.raises(HomeAssistantError, match="allowlist_external_dirs"):
        sm._validate_external_path(str(outside_via_traversal))
    with pytest.raises(HomeAssistantError):
        sm._validate_external_path("/etc/passwd")


def test_storage_dir_denied_even_inside_config(storage):
    sm, config_dir, _ = storage
    with pytest.raises(HomeAssistantError, match=r"\.storage"):
        sm._validate_external_path(str(config_dir / ".storage" / "auth"))
    with pytest.raises(HomeAssistantError, match=r"\.storage"):
        sm._validate_external_path(str(config_dir / ".storage"))
    # ...but a file merely PREFIXED .storage-ish is fine.
    ok = config_dir / ".storage_notes.json"
    assert sm._validate_external_path(str(ok)) == str(ok)


def test_allowlisted_external_dir_allowed(storage, mock_coordinator, tmp_path):
    sm, _, _ = storage
    external = tmp_path / "media" / "export.csv"
    mock_coordinator.hass.config.is_allowed_path.return_value = True
    assert sm._validate_external_path(str(external)) == str(external)
    mock_coordinator.hass.config.is_allowed_path.assert_called_once_with(str(external))


async def test_service_methods_gate_before_any_file_access(storage, tmp_path):
    """A disallowed path raises HomeAssistantError from all four service
    methods before any filesystem access — no write, no existence oracle."""
    sm, config_dir, _ = storage
    outside = tmp_path / "evil" / "target.json"

    with pytest.raises(HomeAssistantError):
        await sm.async_backup_data(str(outside))
    assert not outside.parent.exists()  # os.makedirs never ran

    # Restore/import on a NONEXISTENT disallowed path must raise
    # HomeAssistantError, not FileNotFoundError — the gate runs before the
    # exists() check, so denied paths leak no existence information.
    with pytest.raises(HomeAssistantError):
        await sm.async_restore_data(str(outside))
    with pytest.raises(HomeAssistantError):
        await sm.import_csv_data(str(outside), {})
    with pytest.raises(HomeAssistantError):
        await sm.export_csv_data(str(outside), "daily")

    # Control: an allowed nonexistent path passes the gate and fails on
    # the ordinary exists() check instead.
    inside_missing = config_dir / "missing.json"
    with pytest.raises(FileNotFoundError):
        await sm.async_restore_data(str(inside_missing))
    with pytest.raises(FileNotFoundError):
        await sm.import_csv_data(str(inside_missing), {})


def test_csv_autolog_skips_and_warns_once_on_disallowed_path(storage, tmp_path, caplog):
    """Configured auto-log paths use skip-and-report (never raise into the
    hourly loop), warning once per path."""
    sm, config_dir, _ = storage
    outside = tmp_path / "evil" / "log.csv"

    row = {"timestamp": "2023-10-27T00:00:00", "kwh": 1.0}
    sm._append_to_csv_with_schema_evolution(str(outside), row)
    sm._append_to_csv_with_schema_evolution(str(outside), row)

    assert not outside.parent.exists()
    assert len(sm._csv_paths_warned) == 1
    assert sum("CSV auto-logging skipped" in r.message for r in caplog.records) == 1

    # Allowed path still writes normally.
    inside = config_dir / "log.csv"
    sm._append_to_csv_with_schema_evolution(str(inside), row)
    assert inside.exists()
    assert "kwh" in inside.read_text()
