
# Sports Person & Ball Detection and Tracking

This project performs **object detection and tracking** for players, referees, and the ball in a sports video using a YOLO model and ByteTrack/BoT-SORT tracking. It also supports jersey number OCR for players using EasyOCR. All logic is implemented in `main.py`.

---

## Quick Start

### 1. Create a Python Virtual Environment

```bash
# Create a new environment named 'liat_env'
python -m venv liat_env

# Activate the environment
# On Windows:
liat_env\Scripts\activate
# On Linux/Mac:
# source liat_env/bin/activate
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch with CUDA (Optional, for GPU acceleration)

Check your CUDA version and install the appropriate PyTorch build from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you do not have a compatible GPU, install the CPU version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Download Model and Video Data

All necessary model files and video data are available at the following link:
[Project Data &amp; Model Download](https://docs.google.com/document/d/13a64kk9XT2QvSpQ9BnmlzWUGjGPS9YsGA7RIbBNFOL4/edit?tab=t.0)

---

## File Structure and Description

| File/Folder                          | Description                                                     |
| ------------------------------------ | --------------------------------------------------------------- |
| `main.py`                          | Main script for detection, tracking, and jersey OCR.            |
| `requirements.txt`                 | Python dependencies for the project.                            |
| `README.md`                        | This documentation file.                                        |
| `liat_ai_data/`                    | Folder containing input video files.                            |
| ├──`15sec_input_720p.mp4`       | Sample 15-second input video.                                   |
| ├──`broadcast.mp4`              | Broadcast angle video (optional/extra).                         |
| └──`tacticam.mp4`               | Tacticam angle video (optional/extra).                          |
| `model/`                           | Folder to store YOLO model weights.                             |
| └──`best.pt`                    | YOLO model weights file (download from the provided link).      |
| `output_video_with_jersey_ocr.mp4` | Output video with detection, tracking, and jersey OCR overlays. |
| `reid_log.txt`                     | Log file for ReID and jersey number assignments.                |

---

## Running the Project

After setting up the environment, installing requirements, and downloading the model/video files:

```bash
python main.py
```

---

## Output

- Processed video: `./output_video_with_jersey_ocr.mp4`
- ReID log: `./reid_log.txt`

---

## Notes

- For best performance, use a GPU and install the correct CUDA version for your system.
- Make sure the model and video files are placed in the correct folders as described above.
- All detection, tracking, and OCR logic is contained in `main.py`.

---
