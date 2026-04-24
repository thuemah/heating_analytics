"""Regression tests for the HA Store migration wiring.

Guards against re-introducing the bug where ``async_migrate_func`` was
passed to ``Store.__init__`` as a kwarg — HA's ``Store`` does not accept
that kwarg, and the integration crashed with ``TypeError`` at entity
setup.  The supported migration API is one of:

  (a) Subclass ``Store`` and override ``_async_migrate_func``.
  (b) Assign ``_async_migrate_func`` as an instance attribute on a
      concrete ``Store`` instance — HA looks it up via normal attribute
      protocol which respects instance-dict overrides.

We use (b) because ``conftest.py`` mocks ``homeassistant.helpers.storage``
as a ``MagicMock``, which prevents a module-level subclass of ``Store``
from ever being a real class at test time.

Implementation note
-------------------
These tests operate at the AST level against ``storage.py`` source text
rather than importing the runtime class.  Reason: the conftest mock
makes class-level inspection unreliable.  A proper end-to-end test that
loads the integration against a real HA would catch the full class of
bug (signature drift on HA's side); it requires installing
``homeassistant`` in CI and is tracked separately.
"""
import ast
import inspect
from pathlib import Path

from custom_components.heating_analytics.storage import StorageManager

_STORAGE_PATH = Path(__file__).parent.parent / "custom_components" / "heating_analytics" / "storage.py"


def _parse_storage() -> ast.Module:
    return ast.parse(_STORAGE_PATH.read_text())


def _find_class(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Class {name!r} not found in storage.py")


def _find_method(cls: ast.ClassDef, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"Method {name!r} not found on class {cls.name}")


def test_storage_init_does_not_pass_forbidden_kwarg_to_store():
    """HA's ``Store.__init__`` does not accept ``async_migrate_func``.

    ``StorageManager.__init__`` must not pass this kwarg to ``Store``
    anywhere in its body (the form that caused the TypeError crash).
    """
    module = _parse_storage()
    init = _find_method(_find_class(module, "StorageManager"), "__init__")
    for node in ast.walk(init):
        if isinstance(node, ast.keyword) and node.arg == "async_migrate_func":
            raise AssertionError(
                "StorageManager.__init__ must not pass 'async_migrate_func' "
                "kwarg — HA's Store does not accept it.  Assign "
                "_async_migrate_func as an instance attribute instead."
            )


def test_storage_init_wires_migration_hook_on_both_stores():
    """Both ``_store`` and ``_legacy_store`` must get ``_async_migrate_func`` set.

    HA's ``Store`` calls ``self._async_migrate_func(...)`` on version
    mismatch.  Missing the wiring on either store lets a legacy shape
    slip through to ``async_load_data`` without being migrated.
    """
    src = inspect.getsource(StorageManager.__init__)
    assert "self._store._async_migrate_func" in src, (
        "StorageManager.__init__ must assign _async_migrate_func on self._store."
    )
    assert "self._legacy_store._async_migrate_func" in src, (
        "StorageManager.__init__ must assign _async_migrate_func on self._legacy_store."
    )


def test_async_migrate_has_ha_three_arg_signature():
    """``_async_migrate`` must match HA's override contract.

    HA invokes the hook as ``await self._async_migrate_func(
    old_major_version, old_minor_version, old_data)``.  Any other
    signature is rejected at runtime.
    """
    module = _parse_storage()
    method = _find_method(_find_class(module, "StorageManager"), "_async_migrate")
    assert isinstance(method, ast.AsyncFunctionDef), (
        "_async_migrate must be async — HA awaits it."
    )
    params = [arg.arg for arg in method.args.args]
    assert params == ["self", "old_major_version", "old_minor_version", "old_data"], (
        f"_async_migrate signature must match HA's 3-arg contract "
        f"(self, old_major_version, old_minor_version, old_data); got {params!r}"
    )
