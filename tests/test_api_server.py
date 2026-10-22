"""API regression tests for the FastAPI server surface.

The webhook/watcher worker subsystem referenced by ``axiom/api/server.py`` is
not present in the repository (only stale ``.pyc`` files remain), so no test
asserts webhook behavior.  These tests pin the parts of the server that are
still supported: the app imports, liveness endpoints answer, and invalid
requests are rejected by the router rather than crashing the process.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi.testclient import TestClient


def test_v1_health_returns_online():
    from axiom.api.server import app

    client = TestClient(app)
    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_legacy_health_and_invalid_requests():
    from axiom.api.server import app

    client = TestClient(app)

    legacy = client.get("/health")
    assert legacy.status_code == 200
    assert legacy.json()["status"] == "healthy"

    wrong_method = client.post("/v1/health")
    assert wrong_method.status_code == 405

    missing_route = client.get("/v1/definitely-not-a-route")
    assert missing_route.status_code == 404


def test_optional_worker_reports_missing_dependency(caplog):
    """A missing webhook/watcher module must be logged, not left unretrieved."""
    from axiom.api.server import _supervise_optional_worker

    async def worker_with_missing_import() -> None:
        raise ImportError("No module named 'axiom.core.webhooks'")

    with caplog.at_level(logging.ERROR, logger="axiom.api.server"):
        asyncio.run(
            _supervise_optional_worker("webhook", worker_with_missing_import)
        )

    assert "Optional webhook background worker disabled" in caplog.text
    assert "axiom.core.webhooks" in caplog.text
    assert "rest of the API server remains available" in caplog.text


def test_api_remains_usable_when_optional_workers_are_missing():
    """Startup must not be aborted by the absent webhook/watcher subsystem."""
    from axiom.api.server import app

    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "online"
