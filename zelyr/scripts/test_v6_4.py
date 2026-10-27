import asyncio
import os
from unittest.mock import patch, mock_open
from zelyr.services.updater import ZelyrUpdateManager
import urllib.error

async def test_os_detection():
    print("[Test] Testing OS Detection...")
    updater = ZelyrUpdateManager()

    # Mock /etc/os-release for Debian
    debian_mock = "NAME=\"Ubuntu\"\nID=ubuntu\nID_LIKE=debian"
    with patch("builtins.open", mock_open(read_data=debian_mock)):
        with patch("zelyr.services.updater.urllib.request.urlopen") as mock_urlopen:
            with patch("zelyr.services.updater.subprocess.run") as mock_run:
                # We expect an error because mock_urlopen will fail, but we just want to see if it uses 'deb'
                mock_run.return_value.returncode = 0
                await updater.perform_update(mock_mode=False)

    print("[Test] OS Detection passed.")

async def test_rate_limit():
    print("[Test] Testing Rate Limit Handling...")
    updater = ZelyrUpdateManager()

    # Mock /etc/os-release for Fedora
    fedora_mock = "NAME=\"Fedora\"\nID=fedora\nID_LIKE=rhel"
    with patch("builtins.open", mock_open(read_data=fedora_mock)):
        with patch("zelyr.services.updater.urllib.request.urlopen") as mock_urlopen:
            # Raise a 403 Forbidden (Rate Limit)
            mock_urlopen.side_effect = urllib.error.HTTPError(
                "http://api.github.com", 403, "rate limit exceeded", {}, None
            )
            success = await updater.perform_update(mock_mode=False)
            assert not success, "Update should have failed on rate limit"

    print("[Test] Rate Limit Handling passed.")

async def main():
    await test_os_detection()
    await test_rate_limit()

if __name__ == "__main__":
    asyncio.run(main())
