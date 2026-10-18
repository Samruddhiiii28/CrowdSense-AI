import time
from pathlib import Path
import sys
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.analytics.realtime_risk import RealtimeRiskEngine


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "yolo11n.pt"

CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.40

WINDOW_NAME = "CrowdSense AI - Live Crowd Intelligence"


# ============================================================
# COLORS
# ============================================================

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
BLUE = (255, 0, 0)


# ============================================================
# RISK COLOR
# ============================================================

def get_risk_color(risk):

    if risk == "LOW":
        return GREEN

    if risk == "MEDIUM":
        return YELLOW

    if risk == "HIGH":
        return RED

    return WHITE


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 65)
    print("CrowdSense AI - Day 15")
    print("YOLO11 + ByteTrack + Crowd Analytics + ML Risk")
    print("=" * 65)

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print("\nLoading YOLO11 model...")

    model = YOLO(MODEL_NAME)

    print("YOLO11 loaded successfully.")

    # --------------------------------------------------------
    # Load Risk Engine
    # --------------------------------------------------------

    print("Loading CrowdSense risk model...")

    risk_engine = RealtimeRiskEngine(
        window_seconds=5
    )
    last_sample_time = 0
    SAMPLE_INTERVAL = 1.0

    latest_risk = {
    "risk_label": "WAITING"
}

    print("Risk model loaded successfully.")

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    print("\nOpening webcam...")

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open webcam.\n"
            "Try changing CAMERA_INDEX from 0 to 1."
        )

    print("Camera opened successfully.")
    print("\nPress Q to stop.\n")

    # --------------------------------------------------------
    # Previous positions
    # --------------------------------------------------------

    previous_positions = {}

    last_risk = "WAITING"

    last_scores = {}

    frame_count = 0

    # ========================================================
    # LIVE LOOP
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:

            print("Unable to read camera frame.")

            break

        frame_count += 1

        height, width = frame.shape[:2]

        # ====================================================
        # YOLO + BYTE TRACK
        # ====================================================

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

        people_count = 0

        zone_a = 0
        zone_b = 0
        zone_c = 0
        zone_d = 0

        movement_values = []

        # ====================================================
        # PROCESS DETECTIONS
        # ====================================================

        if result.boxes is not None:

            boxes = result.boxes

            if boxes.xyxy is not None:

                xyxy = boxes.xyxy.cpu().numpy()

                # Track IDs may not exist during the very
                # first frames.
                if boxes.id is not None:

                    track_ids = (
                        boxes.id
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                else:

                    track_ids = []

                for index, box in enumerate(xyxy):

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    # ----------------------------------------
                    # Count person
                    # ----------------------------------------

                    people_count += 1

                    # ----------------------------------------
                    # Center point
                    # ----------------------------------------

                    center_x = int(
                        (x1 + x2) / 2
                    )

                    center_y = int(
                        (y1 + y2) / 2
                    )

                    # ----------------------------------------
                    # Zone determination
                    # ----------------------------------------

                    horizontal_middle = width // 2
                    vertical_middle = height // 2

                    if center_x < horizontal_middle:

                        if center_y < vertical_middle:
                            zone_a += 1
                        else:
                            zone_c += 1

                    else:

                        if center_y < vertical_middle:
                            zone_b += 1
                        else:
                            zone_d += 1

                    # ----------------------------------------
                    # Tracking
                    # ----------------------------------------

                    if index < len(track_ids):

                        track_id = int(
                            track_ids[index]
                        )

                        current_positions[
                            track_id
                        ] = (
                            center_x,
                            center_y
                        )

                        # Calculate movement.
                        if track_id in previous_positions:

                            previous_x, previous_y = (
                                previous_positions[
                                    track_id
                                ]
                            )

                            movement = np.sqrt(
                                (center_x - previous_x) ** 2
                                +
                                (center_y - previous_y) ** 2
                            )

                            movement_values.append(
                                movement
                            )

                        # ------------------------------------
                        # Draw tracking ID
                        # ------------------------------------

                        cv2.putText(
                            frame,
                            f"ID {track_id}",
                            (
                                x1,
                                max(y1 - 10, 20)
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            BLUE,
                            2
                        )

                    # ----------------------------------------
                    # Draw bounding box
                    # ----------------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        GREEN,
                        2
                    )

                    # ----------------------------------------
                    # Draw center
                    # ----------------------------------------

                    cv2.circle(
                        frame,
                        (center_x, center_y),
                        4,
                        RED,
                        -1
                    )

        # ====================================================
        # UPDATE TRACK HISTORY
        # ====================================================

        previous_positions = current_positions

        # ====================================================
        # AVERAGE MOVEMENT
        # ====================================================

        if movement_values:

            avg_movement = float(
                np.mean(movement_values)
            )

        else:

            avg_movement = 0.0

        # ====================================================
        # RISK ENGINE
        # ====================================================

        current_time = time.time()

        if current_time - last_sample_time >= SAMPLE_INTERVAL:
            latest_risk = risk_engine.add_observation(
                timestamp=datetime.now(),
                total_people=people_count,
                zone_a=zone_a,
                zone_b=zone_b,
                zone_c=zone_c,
                zone_d=zone_d,
                avg_movement=avg_movement
           )
            last_sample_time = current_time
            last_risk = latest_risk.get("risk_label", "WAITING")
            last_scores = latest_risk

        # ====================================================
        # DRAW ZONE LINES
        # ====================================================

        cv2.line(
            frame,
            (
                width // 2,
                0
            ),
            (
                width // 2,
                height
            ),
            WHITE,
            2
        )

        cv2.line(
            frame,
            (
                0,
                height // 2
            ),
            (
                width,
                height // 2
            ),
            WHITE,
            2
        )

        # ====================================================
        # TOP INFORMATION PANEL
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (420, 175),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            "CrowdSense AI",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            WHITE,
            2
        )

        cv2.putText(
            frame,
            f"People: {people_count}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            WHITE,
            2
        )

        cv2.putText(
            frame,
            f"Zone A: {zone_a}  B: {zone_b}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            2
        )

        cv2.putText(
            frame,
            f"Zone C: {zone_c}  D: {zone_d}",
            (15, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            2
        )

        cv2.putText(
            frame,
            f"Movement: {avg_movement:.2f}",
            (15, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            2
        )

        # ====================================================
        # RISK DISPLAY
        # ====================================================

        risk_color = get_risk_color(
            last_risk
        )

        cv2.putText(
            frame,
            f"RISK: {last_risk}",
            (15, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            risk_color,
            2
        )

        # ====================================================
        # SHOW FRAME
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    print("\nCrowdSense AI stopped.")
    print(f"Frames processed: {frame_count}")


if __name__ == "__main__":
    main()