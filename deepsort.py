import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------- CONFIG --------
VIDEO_PATH = "./liat_ai_data/15sec_input_720p.mp4"
MODEL_PATH = "./model/yolo11l.pt"
OUTPUT_VIDEO_PATH = "output_deepsort.mp4"  # Path to save the output video
PERSON_CLASS = 0
BALL_CLASS = 32  # Adjust if needed
show_bounding_boxes = False  # Set to False to turn off bounding boxes


# -------- Init Models --------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30, n_init=3)


# Get video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize VideoWriter to save the output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# -------- Processing Loop --------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, verbose=False)[0]

    detections = []
    ball_dets = []

    for box in results.boxes:
        cls_id = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        if cls_id == PERSON_CLASS:
            detections.append(
                ([x1, y1, x2 - x1, y2 - y1], conf, "player")
            )  # (tlwh, conf, class)
        elif cls_id == BALL_CLASS:
            ball_dets.append((x1, y1, x2, y2))

    # Update Deep SORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracked players if enabled
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltrb()

        # Draw bounding box if enabled
        if show_bounding_boxes:
            cv2.rectangle(
                frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 255, 0), 2
            )

        # Always show player ID
        cv2.putText(
            frame,
            f"Player ID {track_id}",
            (int(l), int(t) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Draw balls if enabled
    if show_bounding_boxes:
        for ball in ball_dets:
            x1, y1, x2, y2 = ball
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

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("YOLOv11 + Deep SORT", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
