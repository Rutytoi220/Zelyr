"""Systemd User Service Management Module for Zelyr."""
from zelyr.compat import state_dir
import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

class SystemdServiceManager:
    """Manages the Zelyr headless daemon as a systemd user service."""
    SERVICE_NAME = 'zelyr.service'

    def __init__(self):
        self.config_dir = Path.home() / '.config' / 'systemd' / 'user'
        self.service_path = self.config_dir / self.SERVICE_NAME
        self.zelyr_dir = state_dir()

    @staticmethod
    def is_supported() -> bool:
        """Check if the current OS supports systemd user services."""
        if sys.platform != 'linux':
            return False
        if not shutil.which('systemctl'):
            return False
        return True

    def _get_zelyr_executable(self) -> str:
        """Resolve the absolute path to the zelyr executable."""
        exe = shutil.which('zelyr')
        if exe:
            return exe
        return sys.executable

    def generate_service_file(self) -> str:
        """Generate the systemd unit file content."""
        exe_path = self._get_zelyr_executable()
        work_dir = str(self.zelyr_dir)
        return f"""[Unit]
Description=Zelyr Local-First AI Operating System Daemon
After=network.target graphical-session.target

[Service]
Type=simple
ExecStart={exe_path} -m zelyr.server.daemon
WorkingDirectory={work_dir}
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""

    def install(self) -> bool:
        """Install and enable the systemd service."""
        if not self.is_supported():
            raise RuntimeError('Systemd user services are not supported on this OS.')
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.zelyr_dir.mkdir(parents=True, exist_ok=True)
            content = self.generate_service_file()
            self.service_path.write_text(content)
            logger.info(f'Generated service file at {self.service_path}')
            subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
            subprocess.run(['systemctl', '--user', 'enable', self.SERVICE_NAME], check=True)
            return True
        except Exception as e:
            logger.error(f'Failed to install service: {e}')
            return False

    def start(self) -> bool:
        """Start the systemd service."""
        return self._run_systemctl('start')

    def stop(self) -> bool:
        """Stop the systemd service."""
        return self._run_systemctl('stop')

    def restart(self) -> bool:
        """Restart the systemd service."""
        return self._run_systemctl('restart')

    def status(self) -> str:
        """Get the status of the systemd service."""
        if not self.is_supported():
            return 'Systemd not supported.'
        try:
            result = subprocess.run(['systemctl', '--user', 'status', self.SERVICE_NAME], capture_output=True, text=True)
            return result.stdout or result.stderr
        except Exception as e:
            return f'Error fetching status: {e}'

    def _run_systemctl(self, command: str) -> bool:
        """Run a systemctl command."""
        if not self.is_supported():
            raise RuntimeError('Systemd user services are not supported on this OS.')
        try:
            subprocess.run(['systemctl', '--user', command, self.SERVICE_NAME], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f'systemctl {command} failed: {e}')
            return False
        except Exception as e:
            logger.error(f'Failed to execute systemctl {command}: {e}')
            return False
