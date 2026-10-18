import cv2
import math
import csv
import os
from collections import defaultdict, deque
from datetime import datetime

from ultralytics import YOLO


# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = "yolo11n.pt"
CAMERA_INDEX = 0

CONFIDENCE = 0.35

HISTORY_LENGTH = 20

DATASET_PATH = "data/processed/crowd_features.csv"


# =====================================================
# CREATE DATASET DIRECTORY
# =====================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# =====================================================
# CREATE CSV FILE IF IT DOES NOT EXIST
# =====================================================

if not os.path.exists(DATASET_PATH):

    with open(
        DATASET_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "total_people",
            "zone_a",
            "zone_b",
            "zone_c",
            "zone_d",
            "avg_movement",
            "entries",
            "exits",
            "net_change"
        ])


# =====================================================
# LOAD MODEL
# =====================================================

model = YOLO(MODEL_PATH)


# =====================================================
# OPEN CAMERA
# =====================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("Error: Could not open camera")
    exit()


# =====================================================
# TRACK HISTORY
# =====================================================

track_history = defaultdict(
    lambda: deque(maxlen=HISTORY_LENGTH)
)


previous_positions = {}


# =====================================================
# ENTRY / EXIT
# =====================================================

counted_ids = set()

entry_count = 0
exit_count = 0


# =====================================================
# FUNCTIONS
# =====================================================

def calculate_distance(
    previous_point,
    current_point
):

    if previous_point is None:
        return 0

    x1, y1 = previous_point
    x2, y2 = current_point

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def get_density(count):

    if count <= 5:
        return "LOW"

    elif count <= 15:
        return "MEDIUM"

    else:
        return "HIGH"


# =====================================================
# MAIN LOOP
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Error reading camera")
        break


    height, width = frame.shape[:2]


    # =================================================
    # DEFINE ZONES
    # =================================================

    zone_width = width // 2
    zone_height = height // 2


    zone_counts = {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0
    }


    # =================================================
    # COUNTING LINE
    # =================================================

    line_y = height // 2


    # =================================================
    # YOLO + BYTE TRACK
    # =================================================

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=CONFIDENCE,
        tracker="bytetrack.yaml",
        verbose=False
    )


    annotated_frame = results[0].plot()

    boxes = results[0].boxes


    total_movement = 0
    moving_people = 0


    # =================================================
    # PROCESS PEOPLE
    # =================================================

    if boxes.id is not None:

        track_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )

        coordinates = (
            boxes.xyxy
            .cpu()
            .numpy()
        )


        for track_id, box in zip(
            track_ids,
            coordinates
        ):

            x1, y1, x2, y2 = box


            # -----------------------------------------
            # CENTER POINT
            # -----------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            current_point = (
                center_x,
                center_y
            )


            # -----------------------------------------
            # PREVIOUS POSITION
            # -----------------------------------------

            previous_point = (
                track_history[track_id][-1]
                if len(track_history[track_id]) > 0
                else None
            )


            # -----------------------------------------
            # MOVEMENT
            # -----------------------------------------

            distance = calculate_distance(
                previous_point,
                current_point
            )


            if distance > 5:

                total_movement += distance
                moving_people += 1


            # -----------------------------------------
            # SAVE POSITION
            # -----------------------------------------

            track_history[track_id].append(
                current_point
            )


            previous_positions[track_id] = center_y


            # -----------------------------------------
            # ZONE DETECTION
            # -----------------------------------------

            if (
                center_x < zone_width
                and center_y < zone_height
            ):

                zone_counts["A"] += 1


            elif (
                center_x >= zone_width
                and center_y < zone_height
            ):

                zone_counts["B"] += 1


            elif (
                center_x < zone_width
                and center_y >= zone_height
            ):

                zone_counts["C"] += 1


            else:

                zone_counts["D"] += 1


            # -----------------------------------------
            # ENTRY / EXIT
            # -----------------------------------------

            if previous_point is not None:

                previous_y = previous_point[1]

                current_y = center_y


                # Above → Below
                if (
                    previous_y < line_y
                    and current_y >= line_y
                ):

                    if track_id not in counted_ids:

                        entry_count += 1

                        counted_ids.add(track_id)


                # Below → Above
                elif (
                    previous_y > line_y
                    and current_y <= line_y
                ):

                    if track_id not in counted_ids:

                        exit_count += 1

                        counted_ids.add(track_id)


    # =================================================
    # TOTAL PEOPLE
    # =================================================

    total_people = sum(
        zone_counts.values()
    )


    # =================================================
    # AVERAGE MOVEMENT
    # =================================================

    if moving_people > 0:

        average_movement = (
            total_movement /
            moving_people
        )

    else:

        average_movement = 0


    # =================================================
    # NET CHANGE
    # =================================================

    net_change = (
        entry_count -
        exit_count
    )


    # ================================================
    # SAVE DATA
    # =================================================

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    with open(
        DATASET_PATH,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            total_people,
            zone_counts["A"],
            zone_counts["B"],
            zone_counts["C"],
            zone_counts["D"],
            round(average_movement, 2),
            entry_count,
            exit_count,
            net_change
        ])


    # =================================================
    # DRAW ZONES
    # =================================================

    cv2.line(
        annotated_frame,
        (zone_width, 0),
        (zone_width, height),
        (255, 255, 255),
        2
    )

    cv2.line(
        annotated_frame,
        (0, zone_height),
        (width, zone_height),
        (255, 255, 255),
        2
    )


    # =================================================
    # DRAW COUNTING LINE
    # =================================================

    cv2.line(
        annotated_frame,
        (0, line_y),
        (width, line_y),
        (0, 255, 255),
        2
    )


    # =================================================
    # DISPLAY INFORMATION
    # =================================================

    cv2.putText(
        annotated_frame,
        f"People: {total_people}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    cv2.putText(
        annotated_frame,
        f"A: {zone_counts['A']} "
        f"B: {zone_counts['B']} "
        f"C: {zone_counts['C']} "
        f"D: {zone_counts['D']}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    cv2.putText(
        annotated_frame,
        f"Movement: "
        f"{average_movement:.2f} px/frame",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    cv2.putText(
        annotated_frame,
        f"Entered: {entry_count}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


    cv2.putText(
        annotated_frame,
        f"Exited: {exit_count}",
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


    cv2.putText(
        annotated_frame,
        f"Net: {net_change}",
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    # =================================================
    # SHOW FRAME
    # =================================================

    cv2.imshow(
        "CrowdSense AI - Real Time Analytics",
        annotated_frame
    )


    # =================================================
    # QUIT
    # =================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =====================================================
# CLEANUP
# =====================================================

cap.release()

cv2.destroyAllWindows()