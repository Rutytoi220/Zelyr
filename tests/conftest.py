"""Pytest configuration and shared fixtures for Zelyr."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

os.environ["QT_QPA_PLATFORM"] = "offscreen"
_GLOBAL_TEST_TEMP_DIR = tempfile.TemporaryDirectory(prefix="zelyr_test_global_")
os.environ.setdefault("ZELYR_MEMORY_DB", os.path.join(_GLOBAL_TEST_TEMP_DIR.name, "default_test_memory.db"))

def pytest_configure(config):
    """Pytest configuration hook."""
    pass

@pytest.fixture(autouse=True)
def mock_home_directory(monkeypatch):
    """Ensure tests never write to actual ~/.local or ~/.config by mocking Path.home()"""
    import logging
    existing_handlers = logging.root.handlers[:]
    with tempfile.TemporaryDirectory() as temp_home:
        # We must mock it where it's used, but setting HOME env var often helps too
        monkeypatch.setenv("HOME", temp_home)
        monkeypatch.setenv("USERPROFILE", temp_home) # Windows
        monkeypatch.setenv("XDG_CONFIG_HOME", os.path.join(temp_home, ".config"))
        monkeypatch.setenv("XDG_DATA_HOME", os.path.join(temp_home, ".local", "share"))
        monkeypatch.setenv("ZELYR_MEMORY_DB", os.path.join(temp_home, "test_isolated_memory.db"))
        
        with patch("pathlib.Path.home", return_value=Path(temp_home)):
            yield temp_home

        # Clean up any logging handlers attached to deleted temporary directories
        for h in logging.root.handlers[:]:
            if h not in existing_handlers:
                try:
                    h.close()
                except Exception:
                    pass
                logging.root.removeHandler(h)
            elif isinstance(h, logging.FileHandler) and h.baseFilename.startswith(temp_home):
                try:
                    h.close()
                except Exception:
                    pass
                logging.root.removeHandler(h)

@pytest.fixture
def mock_orchestrator():
    orchestrator = MagicMock()
    return orchestrator

@pytest.fixture
def mock_gui_bridge():
    bridge = MagicMock()
    return bridge

@pytest.fixture(autouse=True)
def block_external_io(request, monkeypatch):
    """
    Strictly mocks external IO (network, local LLM, disk) unless the test
    is explicitly marked 'integration', 'e2e', or 'requires_ollama'.
    """
    markers = [m.name for m in request.node.iter_markers()]
    if any(m in markers for m in ["integration", "e2e", "requires_ollama"]):
        return

    # 1. Block outbound HTTPX requests
    def mock_sync_httpx_post(*args, **kwargs):
        raise RuntimeError(f"Strict Mock: Network IO is blocked for unit tests (Attempted: {args})")

    def mock_sync_httpx_get(*args, **kwargs):
        raise RuntimeError(f"Strict Mock: Network IO is blocked for unit tests (Attempted: {args})")

    async def mock_async_httpx_post(*args, **kwargs):
        raise RuntimeError(f"Strict Mock: Network IO is blocked for unit tests (Attempted: {args})")

    async def mock_async_httpx_get(*args, **kwargs):
        raise RuntimeError(f"Strict Mock: Network IO is blocked for unit tests (Attempted: {args})")

    try:
        import httpx
        monkeypatch.setattr(httpx, "post", mock_sync_httpx_post)
        monkeypatch.setattr(httpx, "get", mock_sync_httpx_get)
        monkeypatch.setattr(httpx.AsyncClient, "post", mock_async_httpx_post)
        monkeypatch.setattr(httpx.AsyncClient, "get", mock_async_httpx_get)
    except ImportError:
        pass


    # 4. Block Model Config Initialization to prevent CLI stall
    try:
        import zelyr.core.config_service
        if hasattr(zelyr.core.config_service, "initialize_model_config"):
            monkeypatch.setattr(zelyr.core.config_service, "initialize_model_config", lambda *args, **kwargs: None)
    except ImportError:
        pass

    # 3. Block urllib.request
    def mock_urlopen(*args, **kwargs):
        raise RuntimeError("Strict Mock: Network IO is blocked via urllib.request.")
    try:
        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
    except ImportError:
        pass

    # 2. Block LiteLLM Completions
    async def mock_acompletion(*args, **kwargs):
        raise RuntimeError("Strict Mock: LiteLLM IO is blocked. Mock your LLM responses!")
    def mock_completion(*args, **kwargs):
        raise RuntimeError("Strict Mock: LiteLLM IO is blocked. Mock your LLM responses!")
    
    try:
        import litellm
        monkeypatch.setattr(litellm, "acompletion", mock_acompletion)
        monkeypatch.setattr(litellm, "completion", mock_completion)
    except ImportError:
        pass



import sys

try:
    from PySide6.QtWidgets import QApplication
    _HAS_QT = True
except (ImportError, OSError):
    QApplication = None
    _HAS_QT = False


if _HAS_QT:
    @pytest.fixture(autouse=True)
    def qt_lifecycle_teardown(qapp):
        yield
        if QApplication:
            for widget in QApplication.topLevelWidgets():
                widget.close()
                widget.deleteLater()
            QApplication.processEvents()

    @pytest.fixture(scope="session")
    def qapp_session():
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        yield app
        app.processEvents()

    @pytest.fixture
    def qapp(qapp_session):
        yield qapp_session
        qapp_session.processEvents()
else:
    @pytest.fixture
    def qapp():
        pytest.skip("PySide6 / libEGL is not available in this environment")

    @pytest.fixture
    def qapp_session():
        pytest.skip("PySide6 / libEGL is not available in this environment")
