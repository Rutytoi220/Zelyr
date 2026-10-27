import sys
import asyncio
from PySide6.QtWidgets import QApplication
from unittest.mock import patch

def test_onboarding():
    app = QApplication(sys.argv)

    # Mock bridge
    class MockBridge:
        def __init__(self):
            self.foo = "bar"

    bridge = MockBridge()

    from zelyr.gui.onboarding_window import OnboardingWindow

    def on_complete():
        print("OOBE Completed. Handoff to MainWindow triggered.")
        app.quit()

    win = OnboardingWindow(bridge, on_complete)
    win.show()

    print("[Test] OOBE Window created and shown.")
    assert win.logo_effect.opacity() == 0.0, "Logo should start at 0 opacity"

    # We can't easily wait for the 5-second animation sequence in a blocking synchronous test
    # without a real event loop spinning for 5 seconds.
    # Let's just manually fast-forward the state to test the final button click.
    print("[Test] Fast-forwarding to completion click...")
    win._on_initialize()

    # Check config
    from zelyr.config import get_config
    c = get_config()
    assert c.first_launch == False, "Config first_launch should be set to False"
    print("  ✓ Config updated properly")

    # Clean up (usually fade_out handles this after 1.5s, we force it here)
    app.processEvents()
    win._handoff()

if __name__ == "__main__":
    test_onboarding()
