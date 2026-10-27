import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Signal, QObject
from zelyr.services.profile_service import ProfileService, ProfileLevel
from zelyr.gui.main_window import MainWindow

def test_profile_switcher():
    app = QApplication(sys.argv)

    # We must mock bridge since we don't have daemon running
    class MockBridge:
        class Signals(QObject):
            token_received = Signal(str)
            tool_status_changed = Signal(str, str)
            telemetry_updated = Signal(dict)
            response_finished = Signal(str)
            error_occurred = Signal(str)
            request_gui_auth = Signal(str, str, dict)
            swarm_agent_started = Signal(str, str)
            swarm_agent_token = Signal(str, str)
            swarm_agent_completed = Signal(str, str)
            connection_status_changed = Signal(bool)

        def __init__(self):
            self._s = self.Signals()
            self.token_received = self._s.token_received
            self.tool_status_changed = self._s.tool_status_changed
            self.telemetry_updated = self._s.telemetry_updated
            self.response_finished = self._s.response_finished
            self.error_occurred = self._s.error_occurred
            self.request_gui_auth = self._s.request_gui_auth
            self.swarm_agent_started = self._s.swarm_agent_started
            self.swarm_agent_token = self._s.swarm_agent_token
            self.swarm_agent_completed = self._s.swarm_agent_completed
            self.connection_status_changed = self._s.connection_status_changed

            # Needed for main_window.py
            self.refresh_models = lambda: None
            self.submit_task = lambda t: None

    bridge = MockBridge()
    from unittest.mock import patch
    with patch("zelyr.services.ollama_monitor.OllamaHealthMonitor.start"):
        win = MainWindow(bridge)

        ps = ProfileService.instance()

        # Test 1: Standard Mode Defaults
        print("[Test] Verifying Standard Mode Defaults")
        ps.set_profile(ProfileLevel.STANDARD)
        app.processEvents()

        assert win.cloud_btn.isHidden(), "Cloud btn should be hidden in Standard"
        assert win.hardware_btn.isHidden(), "Hardware btn should be hidden in Standard"
        assert win._thermal_label.isHidden(), "Thermal label should be hidden in Standard"
        print("  ✓ Standard Mode verified.")

        # Test 2: Live Reactive Toggle (Advanced)
        print("[Test] Verifying Advanced Mode Toggle")
        ps.set_profile(ProfileLevel.ADVANCED)
        app.processEvents()
        assert not win._thermal_label.isHidden(), "Thermal label should be visible in Advanced"
        assert win.cloud_btn.isHidden(), "Cloud btn should still be hidden in Advanced"
        print("  ✓ Advanced Mode verified.")

        # Test 3: Live Reactive Toggle (Developer)
        print("[Test] Verifying Developer Mode Toggle")
        ps.set_profile(ProfileLevel.DEVELOPER)
        app.processEvents()
        assert not win.cloud_btn.isHidden(), "Cloud btn should be visible in Developer"
        assert not win.hardware_btn.isHidden(), "Hardware btn should be visible in Developer"
        print("  ✓ Developer Mode verified.")

        # Clean up
        win.deleteLater()
        app.quit()

if __name__ == "__main__":
    test_profile_switcher()
