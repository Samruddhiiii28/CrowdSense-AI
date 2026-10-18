from pathlib import Path
from datetime import datetime
import csv
import time

import cv2
import numpy as np
from ultralytics import YOLO

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.realtime_risk import RealtimeRiskEngine


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "yolo11n.pt"

CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.40

SAMPLE_INTERVAL = 1.0

WINDOW_NAME = "CrowdSense AI - Day 16"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

LOG_FILE = OUTPUT_DIR / "live_crowd_data.csv"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CSV INITIALIZATION
# ============================================================

CSV_COLUMNS = [
    "timestamp",
    "total_people",
    "zone_a",
    "zone_b",
    "zone_c",
    "zone_d",
    "avg_movement",
    "entries",
    "exits",
    "risk_label",
    "score_low",
    "score_medium",
    "score_high",
]


if not LOG_FILE.exists():

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        writer.writeheader()


# ============================================================
# HELPER: DETERMINE ZONE
# ============================================================

def get_zone(cx, cy, width, height):

    middle_x = width // 2
    middle_y = height // 2

    if cx < middle_x and cy < middle_y:
        return "A"

    elif cx >= middle_x and cy < middle_y:
        return "B"

    elif cx < middle_x and cy >= middle_y:
        return "C"

    else:
        return "D"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CrowdSense AI - Day 16")
    print("Live Crowd Analytics + Entry/Exit + Data Logging")
    print("=" * 70)

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print("\nLoading YOLO11 model...")

    model = YOLO(MODEL_NAME)

    print("YOLO11 loaded successfully.")

    # --------------------------------------------------------
    # Load risk model
    # --------------------------------------------------------

    print("Loading CrowdSense risk model...")

    risk_engine = RealtimeRiskEngine(
        window_seconds=5
    )

    print("Risk model loaded successfully.")

    # --------------------------------------------------------
    # Open webcam
    # --------------------------------------------------------

    print("\nOpening webcam...")

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    print("Camera opened successfully.")
    print("\nPress Q to stop.")
    print(f"Logging data to:\n{LOG_FILE}")

    # --------------------------------------------------------
    # Tracking state
    # --------------------------------------------------------

    previous_positions = {}

    previous_y_positions = {}

    counted_entry_ids = set()

    counted_exit_ids = set()

    total_entries = 0
    total_exits = 0

    last_sample_time = 0

    latest_result = None

    frame_count = 0

    # --------------------------------------------------------
    # Camera loop
    # --------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:

            print("Failed to read frame.")

            break

        frame_count += 1

        height, width = frame.shape[:2]

        # ----------------------------------------------------
        # YOLO11 + ByteTrack
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        result = results[0]

        current_positions = {}

        zone_counts = {
            "A": 0,
            "B": 0,
            "C": 0,
            "D": 0
        }

        movement_values = []

        # ----------------------------------------------------
        # Process detections
        # ----------------------------------------------------

        if result.boxes is not None:

            boxes = result.boxes

            if boxes.xyxy is not None:

                xyxy = boxes.xyxy.cpu().numpy()

                ids = None

                if boxes.id is not None:
                    ids = boxes.id.cpu().numpy().astype(int)

                for index, box in enumerate(xyxy):

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    cx = int(
                        (x1 + x2) / 2
                    )

                    cy = int(
                        (y1 + y2) / 2
                    )

                    # ----------------------------------------
                    # Person ID
                    # ----------------------------------------
                    track_id = None

                    if ids is not None and index < len(ids):
                        track_id = int(ids[index])
                        current_positions[track_id] = (cx, cy)
                    if track_id in previous_positions:
                         old_x, old_y = previous_positions[track_id]
                         distance = np.sqrt(
                         (cx - old_x) ** 2
                         + (cy - old_y) ** 2
                            )
                         movement_values.append(float(distance))


                    # ----------------------------------------
                    # Zone
                    # ----------------------------------------

                    zone = get_zone(
                        cx,
                        cy,
                        width,
                        height
                    )

                    zone_counts[zone] += 1

                    # ----------------------------------------
                    # Movement
                    # ----------------------------------------

                    if track_id in previous_positions:

                        old_x, old_y = (
                            previous_positions[
                                track_id
                            ]
                        )

                        distance = np.sqrt(
                            (cx - old_x) ** 2
                            +
                            (cy - old_y) ** 2
                        )

                        movement_values.append(
                            float(distance)
                        )

                    # ----------------------------------------
                    # Entry / Exit line
                    # ----------------------------------------

                    entry_line_y = height // 2

                    if track_id in previous_y_positions:

                        previous_y = (
                            previous_y_positions[
                                track_id
                            ]
                        )

                        # Moving from upper half
                        # to lower half = ENTRY

                        if (
                            previous_y < entry_line_y
                            and cy >= entry_line_y
                            and track_id not in counted_entry_ids
                        ):

                            total_entries += 1

                            counted_entry_ids.add(
                                track_id
                            )

                        # Moving from lower half
                        # to upper half = EXIT

                        elif (
                            previous_y >= entry_line_y
                            and cy < entry_line_y
                            and track_id not in counted_exit_ids
                        ):

                            total_exits += 1

                            counted_exit_ids.add(
                                track_id
                            )

                    previous_y_positions[
                        track_id
                    ] = cy

                    # ----------------------------------------
                    # Draw bounding box
                    # ----------------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        2
                    )

                    # ----------------------------------------
                    # Draw tracking ID
                    # ----------------------------------------

                    cv2.putText(
                        frame,
                        f"ID {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    # ----------------------------------------
                    # Draw center
                    # ----------------------------------------

                    cv2.circle(
                        frame,
                        (cx, cy),
                        4,
                        (255, 255, 255),
                        -1
                    )

        # ----------------------------------------------------
        # Update previous positions
        # ----------------------------------------------------

        previous_positions = current_positions

        # ----------------------------------------------------
        # Current crowd statistics
        # ----------------------------------------------------

        total_people = len(current_positions)

        if movement_values:

            avg_movement = float(
                np.mean(movement_values)
            )

        else:

            avg_movement = 0.0

        # ----------------------------------------------------
        # Draw zones
        # ----------------------------------------------------

        middle_x = width // 2
        middle_y = height // 2

        cv2.line(
            frame,
            (middle_x, 0),
            (middle_x, height),
            (255, 255, 255),
            2
        )

        cv2.line(
            frame,
            (0, middle_y),
            (width, middle_y),
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Draw entry/exit line
        # ----------------------------------------------------

        entry_line_y = height // 2

        cv2.line(
            frame,
            (0, entry_line_y),
            (width, entry_line_y),
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            "ENTRY / EXIT LINE",
            (20, entry_line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Sample once per second
        # ----------------------------------------------------

        current_time = time.time()

        if (
            current_time - last_sample_time
            >= SAMPLE_INTERVAL
        ):

            last_sample_time = current_time

            timestamp = datetime.now()

            # -----------------------------------------------
            # Send observation to risk engine
            # -----------------------------------------------

            latest_result = (
                risk_engine.add_observation(
                    timestamp=timestamp,
                    total_people=total_people,
                    zone_a=zone_counts["A"],
                    zone_b=zone_counts["B"],
                    zone_c=zone_counts["C"],
                    zone_d=zone_counts["D"],
                    avg_movement=avg_movement
                )
            )

            # -----------------------------------------------
            # Risk values
            # -----------------------------------------------

            risk_label = latest_result.get(
                "risk_label",
                "WAITING"
            )

            score_low = latest_result.get(
                "score_low",
                0.0
            )

            score_medium = latest_result.get(
                "score_medium",
                0.0
            )

            score_high = latest_result.get(
                "score_high",
                0.0
            )

            # -----------------------------------------------
            # Log observation
            # -----------------------------------------------

            with open(
                LOG_FILE,
                "a",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=CSV_COLUMNS
                )

                writer.writerow({

                    "timestamp":
                        timestamp.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "total_people":
                        total_people,

                    "zone_a":
                        zone_counts["A"],

                    "zone_b":
                        zone_counts["B"],

                    "zone_c":
                        zone_counts["C"],

                    "zone_d":
                        zone_counts["D"],

                    "avg_movement":
                        round(
                            avg_movement,
                            3
                        ),

                    "entries":
                        total_entries,

                    "exits":
                        total_exits,

                    "risk_label":
                        risk_label,

                    "score_low":
                        round(
                            float(score_low),
                            3
                        ),

                    "score_medium":
                        round(
                            float(score_medium),
                            3
                        ),

                    "score_high":
                        round(
                            float(score_high),
                            3
                        )
                })

        # ----------------------------------------------------
        # Display information
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"People: {total_people}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Zone A: {zone_counts['A']}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Zone B: {zone_counts['B']}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Zone C: {zone_counts['C']}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Zone D: {zone_counts['D']}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Movement: {avg_movement:.2f}",
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Entries: {total_entries}",
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Exits: {total_exits}",
            (20, 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Risk display
        # ----------------------------------------------------

        risk_label = "WAITING"

        if latest_result is not None:

            risk_label = latest_result.get(
                "risk_label",
                "WAITING"
            )

        cv2.putText(
            frame,
            f"RISK: {risk_label}",
            (width - 250, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            3
        )

        # ----------------------------------------------------
        # Show frame
        # ----------------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("CrowdSense AI - Day 16 stopped")
    print("=" * 70)

    print(f"Frames processed: {frame_count}")

    print(
        f"Total entries detected: {total_entries}"
    )

    print(
        f"Total exits detected: {total_exits}"
    )

    print(
        f"Live data saved to:\n{LOG_FILE}"
    )


if __name__ == "__main__":
    main()