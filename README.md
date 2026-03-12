<h1 align="center">🚗 Real-Time Number Plate Detection</h1>

<p align="center">
  <b>A real-time license plate detection system using Haar Cascade Classifier and OpenCV — detects, highlights, and saves number plates from a live webcam feed.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Computer%20Vision-Haar%20Cascade-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Detection-Real--Time-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" />
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [How It Works — Under the Hood](#-how-it-works--under-the-hood)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Module Breakdown](#-module-breakdown)
- [Algorithm Deep Dive — Haar Cascade Classifier](#-algorithm-deep-dive--haar-cascade-classifier)
- [Detection Parameters Explained](#-detection-parameters-explained)
- [Installation & Setup](#-installation--setup)
- [Running the Project](#-running-the-project)
- [Keyboard Controls](#-keyboard-controls)
- [Output](#-output)
- [Key Design Decisions](#-key-design-decisions)
- [Challenges & Learnings](#-challenges--learnings)
- [Future Improvements](#-future-improvements)

---

## 🧠 Overview

This project implements a **real-time vehicle number plate detection system** using classical computer vision techniques. It uses a **pre-trained Haar Cascade XML classifier** to detect license plates from a live webcam video stream. Once a plate is detected and exceeds a minimum area threshold, it is:

- Highlighted with a **green bounding box**
- Labeled with `"Number Plate"` text overlay
- Cropped and shown as a separate **Region of Interest (ROI)** window
- **Saved** as a `.jpg` image to the `plates/` directory on keypress

This is a clean, minimal, zero-deep-learning approach to plate detection — demonstrating strong fundamentals in computer vision, making it an ideal portfolio project for showcasing knowledge of **OpenCV**, **image processing pipelines**, and **real-time video analysis**.

---

## ⚙️ How It Works — Under the Hood

```
Webcam Frame Capture
        │
        ▼
Convert BGR Frame → Grayscale
        │
        ▼
Apply Haar Cascade detectMultiScale()
        │
        ▼
Filter by Minimum Area (> 500 px²)
        │
        ├── Draw bounding rectangle (green)
        ├── Put "Number Plate" label (magenta text)
        ├── Crop ROI from frame
        └── Show ROI in separate window
        │
        ▼
Display Final Annotated Frame ("Result" window)
        │
        ▼
[Press 'S'] → Save ROI as plates/scaned_img_N.jpg
             → Flash "Plate Saved" confirmation banner
             → Increment save counter
```

The pipeline runs **continuously in a real-time loop** using OpenCV's frame-by-frame video capture from the default system webcam.

---

## 🎥 Demo

| Window Name | Description |
|-------------|-------------|
| `Result`    | Full live webcam feed with bounding box drawn on detected plate |
| `ROI`       | Cropped close-up of the detected number plate region only |

When a plate is saved, a full-width green banner overlaid with `"Plate Saved"` in red text is displayed for 500ms as visual confirmation.

---

## 🛠 Tech Stack

| Tool / Library | Version | Purpose |
|----------------|---------|---------|
| Python | 3.8+ | Core programming language |
| OpenCV (`opencv-python`) | 4.x | Video capture, image processing, cascade classification, display |
| Haar Cascade XML | Pre-trained | Russian/standard license plate detection model |

> No deep learning frameworks required. Pure classical computer vision.

---

## 📁 Project Structure

```
Number_plate_detection/
│
├── model/
│   └── haarcascade_russian_plate_number.xml   # Pre-trained Haar Cascade model for plate detection
│
├── plates/                                     # Auto-created at runtime — stores saved plate images
│   └── scaned_img_0.jpg                        # Example saved plate (name increments on each save)
│
├── main.py                                     # Core application — entire detection pipeline
├── requirements.txt                            # Python dependencies (opencv-python)
├── .gitignore                                  # Ignores virtual envs, pycache, etc.
└── README.md                                   # Project documentation
```

> **Note:** The `plates/` directory is not included in the repo but is automatically used as the save destination at runtime. Create it manually before running if you want to save images: `mkdir plates`

---

## 🔍 Module Breakdown

### `main.py` — Core Detection Pipeline

This is the **single entry point** and the **entire application logic**, structured as a clean real-time processing loop.

---

#### Step 1 — Configuration & Initialization

```python
harcascade = "model/haarcascade_russian_plate_number.xml"
cap = cv2.VideoCapture(0)
cap.set(3, 640)   # Set frame width  → 640px
cap.set(4, 480)   # Set frame height → 480px
min_area = 500    # Minimum bounding box area to be considered a valid detection
count = 0         # Tracks number of saved plate images
```

| Variable | Value | Purpose |
|----------|-------|---------|
| `harcascade` | Path to XML | Points to the pre-trained Haar Cascade model |
| `cap` | `VideoCapture(0)` | Opens the default webcam (index 0) |
| `cap.set(3, 640)` | Width | Fixes frame width at 640 pixels |
| `cap.set(4, 480)` | Height | Fixes frame height at 480 pixels |
| `min_area` | `500` | Area filter — ignores very small/false detections |
| `count` | `0` | Integer counter for unique filenames on save |

---

#### Step 2 — Per-Frame Processing Loop

```python
while True:
    success, img = cap.read()
```

Continuously reads frames from the webcam. `success` is a boolean indicating whether the frame was captured correctly. The loop runs until the process is manually terminated.

---

#### Step 3 — Grayscale Conversion

```python
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

Converts the captured BGR (color) frame to **grayscale**. This is required because the Haar Cascade classifier works on **single-channel intensity images**, not color images. Grayscale conversion also improves processing speed.

---

#### Step 4 — Plate Detection via Haar Cascade

```python
plate_cascade = cv2.CascadeClassifier(harcascade)
plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
```

`detectMultiScale()` scans the image at multiple scales and returns a list of bounding rectangles `(x, y, w, h)` for each detected plate.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `scaleFactor` | `1.1` | Image is reduced by 10% at each scale — smaller value = slower but more accurate |
| `minNeighbors` | `4` | A detected region must have at least 4 neighboring detections to be confirmed — reduces false positives |

---

#### Step 5 — Bounding Box & ROI Rendering

```python
for (x, y, w, h) in plates:
    area = w * h
    if area > min_area:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, "Number Plate", (x, y-5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
        img_roi = img[y: y+h, x: x+w]
        cv2.imshow("ROI", img_roi)
```

For each detected plate that passes the area filter:

| Action | Detail |
|--------|--------|
| **Green rectangle** | Drawn around the plate region: color `(0, 255, 0)`, thickness `2` |
| **Magenta label** | `"Number Plate"` text placed just above the bounding box in `FONT_HERSHEY_COMPLEX_SMALL` |
| **ROI crop** | NumPy array slice `img[y:y+h, x:x+w]` extracts the plate region |
| **ROI window** | Cropped plate shown in a dedicated `"ROI"` OpenCV window |

---

#### Step 6 — Save on Keypress (`S` key)

```python
if cv2.waitKey(1) & 0xFF == ord('s'):
    cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)
    cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, "Plate Saved", (150, 265),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
    cv2.imshow("Results", img)
    cv2.waitKey(500)
    count += 1
```

| Action | Detail |
|--------|--------|
| `cv2.imwrite()` | Saves the current ROI as `plates/scaned_img_N.jpg` |
| Green banner | A filled green rectangle drawn across the frame `(0,200)→(640,300)` as a visual flash |
| `"Plate Saved"` text | Bold red confirmation text rendered on the banner |
| `cv2.waitKey(500)` | Holds the saved-frame display for 500ms |
| `count += 1` | Increments the filename counter to avoid overwriting |

---

### `model/haarcascade_russian_plate_number.xml` — Pre-trained Haar Cascade Model

This is an **OpenCV pre-trained XML cascade classifier** specifically trained to detect **Russian and European-style vehicle license plates**.

- It is a **sliding-window, multi-scale classifier** based on Viola-Jones algorithm
- Internally contains a cascade of **boosted weak classifiers** (each checking Haar-like features)
- Loaded via `cv2.CascadeClassifier()` — no GPU, no training required
- Works on **grayscale images** and is fast enough for real-time video processing
- This specific model generalizes reasonably well to standard rectangular plates globally

---

### `requirements.txt` — Dependencies

```
opencv-python
```

The entire project depends only on **one library** — `opencv-python`, which bundles:
- Core image processing (`cvtColor`, array slicing)
- Video capture (`VideoCapture`)
- Haar Cascade classifier (`CascadeClassifier`, `detectMultiScale`)
- GUI windows (`imshow`, `waitKey`)
- File I/O (`imwrite`)

---

## 🔬 Algorithm Deep Dive — Haar Cascade Classifier

The **Viola-Jones Haar Cascade** algorithm works in the following stages:

1. **Haar Feature Extraction** — Computes rectangular feature differences (edges, lines, diagonals) across the image using an **integral image** for O(1) lookups
2. **AdaBoost Training** — Selects the most discriminative Haar features from thousands of candidates, combining weak classifiers into a strong one
3. **Cascade Structure** — Multiple stages of classifiers run in sequence; early stages quickly reject most negative regions, later stages refine positives — this makes real-time detection possible
4. **Multi-Scale Sliding Window** — The image is repeatedly shrunk by `scaleFactor` and the window slides across each scale to detect plates at any size

This is a **classical computer vision approach** — no neural networks, interpretable, fast on CPU, and requires no GPU.

---

## 🎛 Detection Parameters Explained

| Parameter | Current Value | Effect of Increasing | Effect of Decreasing |
|-----------|--------------|----------------------|----------------------|
| `scaleFactor` | `1.1` | Faster, may miss plates at unusual scales | Slower, more thorough |
| `minNeighbors` | `4` | Fewer false positives, may miss real plates | More detections, more noise |
| `min_area` | `500` | Ignores distant/small plates | Detects more plates including distant ones |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- A working **webcam** connected to your system
- `pip` package manager

### 1. Clone the Repository

```bash
git clone https://github.com/JangraTushar/Number_plate_detection.git
cd Number_plate_detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create the Plates Output Directory

```bash
mkdir plates
```

> This directory is where detected number plate images will be saved when you press the `S` key.

---

## ▶️ Running the Project

```bash
python main.py
```

Two windows will open:
- **`Result`** — Live webcam feed with detected plates highlighted
- **`ROI`** — Cropped close-up of the detected plate region

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `S` | Save the currently detected plate ROI as a `.jpg` image in the `plates/` folder |
| `Ctrl + C` | Stop the application from terminal |

> Close the OpenCV windows by terminating the process — `Ctrl+C` in the terminal or closing the terminal window.

---

## 📤 Output

Each time you press `S`, a new image is saved:

```
plates/
├── scaned_img_0.jpg    ← First saved plate
├── scaned_img_1.jpg    ← Second saved plate
├── scaned_img_2.jpg    ← Third saved plate
└── ...
```

A green banner with `"Plate Saved"` text will flash on the screen for 500ms to confirm the save.

---

## 💡 Key Design Decisions

| Decision | Reasoning |
|----------|-----------|
| **Haar Cascade over deep learning** | Lightweight, no GPU required, runs in real-time on any basic machine, demonstrates CV fundamentals |
| **Grayscale conversion before detection** | Haar features work on intensity gradients — color is irrelevant and slows processing |
| **`min_area = 500` filter** | Prevents spurious small detections from noise or background objects triggering false saves |
| **`scaleFactor = 1.1`
