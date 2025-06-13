import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import torch
from torchreid import utils
from scipy.spatial.distance import cosine
import easyocr
from datetime import datetime

# -------- ENV PATCH --------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------- CONFIG --------
VIDEO_PATH = "./liat_ai_data/15sec_input_720p.mp4"
MODEL_PATH = "./model/best.pt"
OUTPUT_PATH = "./output_video_with_jersey_ocr.mp4"
LOG_PATH = "./reid_log.txt"
TRACKER_CONFIG = "botsort.yaml"

PLAYER_CLASS = 2
REFEREE_CLASS = 3
BALL_CLASS = 0
DRAW_TRAILS = False
MAX_TRAIL_LENGTH = 30
REID_SIM_THRESHOLD = 0.2
REID_BUFFER_SIZE = 30

# -------- Initialize Models --------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(
    OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
)

extractor = utils.FeatureExtractor(
    model_name="osnet_ain_x1_0",
    model_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

# -------- States --------
track_history = defaultdict(lambda: [])
embedding_store = {}  # reid_id: embedding
jersey_store = {}  # reid_id: jersey number
exit_buffer = deque(maxlen=REID_BUFFER_SIZE)
id_mapping = {}  # tracker_id → consistent reid_id
next_reid_id = 0
frame_id = 0

# -------- Logging --------
log_file = open(LOG_PATH, "w")
log_file.write(f"==== ReID Log Started @ {datetime.now()} ====\n\n")


# -------- Helper Functions --------
def get_patch(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return frame[y1:y2, x1:x2]


def get_embedding(image_patch):
    image_patch = cv2.resize(image_patch, (128, 256))
    return extractor([image_patch])[0]


def match_embedding(new_embed):
    best_match, best_score = None, float("inf")
    new_np = new_embed.cpu().numpy()
    for rid, emb in embedding_store.items():
        sim = cosine(new_np, emb.cpu().numpy())
        if sim < REID_SIM_THRESHOLD and sim < best_score:
            best_score, best_match = sim, rid
    return best_match, best_score


def detect_jersey_number(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    for _, text, conf in results:
        if 1 <= len(text) <= 3 and text.isdigit() and conf > 0.2:
            return text
    return "??"


# -------- Main Loop --------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    result = model.track(frame, persist=True, tracker=TRACKER_CONFIG, verbose=False)[0]
    current_ids = set()

    if result.boxes and result.boxes.is_track:
        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        xyxy = result.boxes.xyxy.cpu().tolist()

        for cls, track_id, bbox in zip(classes, track_ids, xyxy):
            if cls not in [PLAYER_CLASS, REFEREE_CLASS]:
                continue

            current_ids.add(track_id)
            label = "Player" if cls == PLAYER_CLASS else "Referee"
            color = (0, 255, 0) if cls == PLAYER_CLASS else (255, 255, 0)

            if track_id not in id_mapping:
                patch = get_patch(frame, bbox)
                emb = get_embedding(patch)
                match_id, score = match_embedding(emb)

                if match_id is not None:
                    id_mapping[track_id] = match_id
                    log_file.write(
                        f"[Frame {frame_id}] {label} matched to ReID:{match_id} | Score: {score:.4f}\n"
                    )
                else:
                    reid_id = next_reid_id
                    id_mapping[track_id] = reid_id
                    embedding_store[reid_id] = emb
                    jersey_text = detect_jersey_number(patch)
                    jersey_store[reid_id] = jersey_text
                    log_file.write(
                        f"[Frame {frame_id}] {label} assigned new ReID:{reid_id} | Jersey: {jersey_text}\n"
                    )
                    next_reid_id += 1

            reid_id = id_mapping[track_id]
            cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)

            if DRAW_TRAILS:
                track_history[reid_id].append((cx, cy))
                if len(track_history[reid_id]) > MAX_TRAIL_LENGTH:
                    track_history[reid_id].pop(0)
                points = np.array(track_history[reid_id], dtype=np.int32).reshape(
                    (-1, 1, 2)
                )
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

            jersey_number = jersey_store.get(reid_id, "??")
            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2,
            )
            cv2.putText(
                frame,
                f"{label} ID:{reid_id} J:{jersey_number}",
                (int(bbox[0]), int(bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    # Ball fallback
    for box in result.boxes:
        if int(box.cls) == BALL_CLASS:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
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

    disappeared_ids = [tid for tid in id_mapping if tid not in current_ids]
    for tid in disappeared_ids:
        exit_buffer.append(id_mapping[tid])
        del id_mapping[tid]

    out.write(frame)
    # cv2.imshow("YOLO + Tracker + ReID", frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

cap.release()
out.release()
log_file.write("\n==== End of Log ====\n")
log_file.close()
# cv2.destroyAllWindows()
