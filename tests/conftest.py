"""Shared pytest fixtures.

Isolates every test behind a throwaway toolbox root (via AITOOLBOX_ROOT) so
that path/config singletons never read or write the developer's real project
directories.
"""

import pytest

from ai_toolbox.core import paths as paths_mod
from ai_toolbox.core import config as config_mod


@pytest.fixture(autouse=True)
def isolated_toolbox_root(tmp_path, monkeypatch):
    """Point the toolbox at a per-test temp root and reset the singletons."""
    monkeypatch.setenv("AITOOLBOX_ROOT", str(tmp_path))
    paths_mod.reset_paths()
    # Clear the cached config WITHOUT writing defaults to disk (reset_config
    # would persist a file); tests that need a file call save_config themselves.
    config_mod._config = None
    yield tmp_path
    paths_mod.reset_paths()
    config_mod._config = None
