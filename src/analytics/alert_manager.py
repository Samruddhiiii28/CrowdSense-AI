
from datetime import datetime
from pathlib import Path
import csv


class AlertManager:
    """
    Converts risk predictions into alert states.

    This module uses model predictions; it does not independently
    verify real-world crowd safety.
    """

    VALID_LEVELS = {"LOW", "MEDIUM", "HIGH"}

    def __init__(
        self,
        log_path=None,
        confirmation_count=2,
    ):
        if confirmation_count < 1:
            raise ValueError(
                "confirmation_count must be at least 1"
            )

        self.confirmation_count = confirmation_count
        self.current_level = None
        self.pending_level = None
        self.pending_count = 0

        self.log_path = (
            Path(log_path)
            if log_path
            else Path(__file__).resolve().parents[2]
            / "data"
            / "processed"
            / "risk_alerts.csv"
        )

        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._ensure_log_exists()

    def _ensure_log_exists(self):
        if not self.log_path.exists():
            with self.log_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp",
                    "previous_level",
                    "new_level",
                    "event",
                ])

    def _log_event(
        self,
        previous_level,
        new_level,
        event,
    ):
        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        with self.log_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                previous_level or "",
                new_level,
                event,
            ])

        return {
            "timestamp": timestamp,
            "previous_level": previous_level,
            "new_level": new_level,
            "event": event,
        }

    def update(self, prediction):
        """
        Process one model prediction.

        Returns an event dictionary when the confirmed state changes;
        otherwise returns None.
        """
        if prediction is None:
            return None

        level = str(prediction).strip().upper()

        if level not in self.VALID_LEVELS:
            raise ValueError(
                f"Invalid risk prediction: {prediction!r}. "
                f"Expected one of {sorted(self.VALID_LEVELS)}."
            )

        if self.current_level is None:
            self.current_level = level

            return self._log_event(
                previous_level=None,
                new_level=level,
                event="INITIAL_STATE",
            )

        if level == self.current_level:
            self.pending_level = None
            self.pending_count = 0
            return None

        if level == self.pending_level:
            self.pending_count += 1
        else:
            self.pending_level = level
            self.pending_count = 1

        if self.pending_count < self.confirmation_count:
            return None

        previous = self.current_level
        self.current_level = level

        self.pending_level = None
        self.pending_count = 0

        return self._log_event(
            previous_level=previous,
            new_level=level,
            event="RISK_LEVEL_CHANGED",
        )
