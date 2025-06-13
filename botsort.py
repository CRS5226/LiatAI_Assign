import os
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Allow OpenMP workaround
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------- CONFIG --------
VIDEO_PATH = "./liat_ai_data/15sec_input_720p.mp4"
MODEL_PATH = "./model/yolo11l.pt"
OUTPUT_PATH = "./output_botsort.mp4"
TRACKER_CONFIG = "botsort.yaml"  # BoT-SORT tracking config

PERSON_CLASS = 0
BALL_CLASS = 32  # Adjust if needed
DRAW_TRAILS = True
MAX_TRAIL_LENGTH = 30
SHOW_BOUNDING_BOXES = False  # Added bounding box toggle


# -------- Load Model --------
model = YOLO(MODEL_PATH)

# -------- Video Setup --------
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# -------- Track History for Trails --------
track_history = defaultdict(lambda: [])

# -------- Main Processing Loop --------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO tracking on the frame with BoT-SORT
    result = model.track(frame, persist=True, tracker=TRACKER_CONFIG, verbose=False)[0]

    # Annotate balls manually (as YOLO tracking doesn't track ball IDs)
    ball_dets = []
    for box in result.boxes:
        cls_id = int(box.cls)
        if cls_id == BALL_CLASS:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            ball_dets.append((x1, y1, x2, y2))

    # Trackable IDs
    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        xyxy = result.boxes.xyxy.cpu().tolist()  # Get bounding box coordinates

        # Draw player trails and bounding boxes
        for box, track_id, bbox in zip(boxes, track_ids, xyxy):
            x, y, w, h = box
            x1, y1, x2, y2 = map(int, bbox)  # Bounding box coordinates
            cx, cy = float(x), float(y)
            track = track_history[track_id]
            track.append((cx, cy))
            if len(track) > MAX_TRAIL_LENGTH:
                track.pop(0)

            if DRAW_TRAILS and len(track) > 1:
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    frame, [points], isClosed=False, color=(255, 255, 255), thickness=3
                )

            # Draw bounding box if enabled
            if SHOW_BOUNDING_BOXES:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw player ID label
            cv2.putText(
                frame,
                f"Player ID {track_id}",
                (int(cx - w / 2), int(cy - h / 2 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # Draw ball bounding boxes
    for x1, y1, x2, y2 in ball_dets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            "Ball",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # Write to video and display
    out.write(frame)
    cv2.imshow("YOLOv11 + BoT-SORT", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------- Cleanup --------
cap.release()
out.release()
cv2.destroyAllWindows()
