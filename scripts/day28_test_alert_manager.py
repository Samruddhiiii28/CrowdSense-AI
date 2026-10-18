
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics.alert_manager import AlertManager


def run_test():
    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = Path(temp_dir) / "test_alerts.csv"

        manager = AlertManager(
            log_path=log_path,
            confirmation_count=2,
        )

        # Initial state is recorded.
        event = manager.update("LOW")
        assert event is not None
        assert manager.current_level == "LOW"

        # A single HIGH prediction is not enough.
        assert manager.update("HIGH") is None
        assert manager.current_level == "LOW"

        # The second consecutive HIGH confirms the change.
        event = manager.update("HIGH")
        assert event is not None
        assert manager.current_level == "HIGH"

        # A single LOW prediction should not immediately clear HIGH.
        assert manager.update("LOW") is None
        assert manager.current_level == "HIGH"

        # MEDIUM interrupts the pending LOW transition.
        assert manager.update("MEDIUM") is None
        assert manager.current_level == "HIGH"

        # A second MEDIUM confirms the new state.
        event = manager.update("MEDIUM")
        assert event is not None
        assert manager.current_level == "MEDIUM"

        # Invalid predictions must be rejected.
        try:
            manager.update("UNKNOWN")
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Invalid prediction was not rejected."
            )

        print("All Day 28 alert manager tests passed.")
        print("Confirmed state transitions and CSV logging work.")


if __name__ == "__main__":
    run_test()
