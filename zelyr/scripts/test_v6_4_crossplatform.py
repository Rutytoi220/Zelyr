import asyncio
from unittest.mock import patch
from zelyr.services.updater import ZelyrUpdateManager
from zelyr.plugins.sandbox_plugin import SandboxRuntime, SandboxBackend

async def test_cross_platform():
    updater = ZelyrUpdateManager()

    print("[Test] Mocking platform.system() == 'Windows'")
    with patch("platform.system", return_value="Windows"):
        # Sandbox Backend Check
        backend = SandboxRuntime.detect_backend()
        assert backend in [SandboxBackend.DOCKER, SandboxBackend.NONE], f"Sandbox backend should not be BWRAP on Windows. Got: {backend}"
        print("  - Sandbox Backend Degraded Correctly.")

        # Updater OS check
        with patch("zelyr.services.updater.urllib.request.urlopen") as mock_urlopen:
            with patch("zelyr.services.updater.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                await updater.perform_update(mock_mode=False)

    print("[Test] Mocking platform.system() == 'Darwin'")
    with patch("platform.system", return_value="Darwin"):
        backend = SandboxRuntime.detect_backend()
        assert backend in [SandboxBackend.DOCKER, SandboxBackend.NONE], f"Sandbox backend should not be BWRAP on macOS. Got: {backend}"
        print("  - Sandbox Backend Degraded Correctly.")

    print("All Cross-Platform mock tests passed.")

if __name__ == "__main__":
    asyncio.run(test_cross_platform())
