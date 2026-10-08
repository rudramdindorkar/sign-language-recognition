# Sign Language Recognition (ASL Alphabet)

Real-time American Sign Language (ASL) alphabet recognition from a webcam, built on **MediaPipe hand landmarks** and a lightweight **PyTorch MLP classifier**. Hold a sign steady and the letter is typed into an on-screen text buffer.

## Screenshots

<p align="center">
  <img src="screenshots/Screenshot%202026-10-08%20at%207.02.17%E2%80%AFPM.png" alt="Screenshot 1" width="600">
</p>
<p align="center"><em>Live recognition: hand skeleton and predicted letter</em></p>

<p align="center">
  <img src="screenshots/Screenshot%202026-10-08%20at%207.02.29%E2%80%AFPM.png" alt="Screenshot 2" width="600">
</p>
<p align="center"><em>Prediction with confidence score</em></p>

<p align="center">
  <img src="screenshots/Screenshot%202026-10-08%20at%207.02.43%E2%80%AFPM.png" alt="Screenshot 3" width="600">
</p>
<p align="center"><em>Typed text output</em></p>

## Overview

Instead of classifying raw pixels with a heavy CNN, this project extracts 21 3D hand landmarks per image with MediaPipe and classifies the normalized landmark geometry. This makes the model:

- **Fast**: a small MLP runs in real time, even on CPU.
- **Robust**: invariant to background, lighting, skin tone and hand position/scale.
- **Handedness-agnostic**: mirror augmentation lets it work with either hand.

## How It Works

```
Webcam / Image -> MediaPipe HandLandmarker -> 21 x (x, y, z) landmarks
              -> Normalize (wrist at origin, scale to unit length)
              -> MLP classifier (63 -> 256 -> 128 -> classes)
              -> Majority-vote smoothing -> Hold-to-type text output
```

| Stage | Description |
|-------|-------------|
| Landmark extraction | MediaPipe Tasks `HandLandmarker` (1 hand, confidence 0.3 for training, 0.5 for live use) |
| Normalization | Translate wrist to origin; scale so the farthest landmark has unit length |
| Augmentation | Random rotation (±15°), horizontal mirroring, scaling (0.9–1.1x), Gaussian noise |
| Model | `Linear(63,256)` -> BN -> ReLU -> Dropout -> `Linear(256,128)` -> BN -> ReLU -> Dropout -> `Linear(128,N)` |
| Training | AdamW, OneCycleLR, label smoothing 0.05, 60 epochs, stratified 85/15 train/val split |
| Inference | Confidence threshold + majority-vote smoothing + hold-to-confirm typing |

## Project Structure

```
.
├── screenshots/            # Demo screenshots used in this README
├── main.ipynb              # Training + live inference notebook
├── sign_landmark.pt        # Trained landmark MLP checkpoint (used by the app)
├── sign_cnn.pt             # CNN model checkpoint
├── Dataset C.zip           # Zipped dataset archive
├── requiremnets.txt        # Python dependencies
├── .gitignore
├── License.md
└── readme.md
```

Generated locally (not tracked in git): `landmarks_cache.npz`, `hand_landmarker.task`, and the `archive/` dataset folder.

## Getting Started

### Prerequisites

- Python 3.10+ (developed on 3.12)
- A webcam (for live inference)

### Requirements

All Python dependencies are listed in [`requiremnets.txt`](requiremnets.txt):

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >=1.24 | Array operations and landmark caching |
| `opencv-python` | >=4.8 | Image loading, webcam capture, and display |
| `torch` | >=2.0 | MLP model, training, and inference |
| `mediapipe` | >=0.10.14 | Hand landmark detection (Tasks API) |
| `ipykernel` | >=6.0 | Jupyter kernel for the notebook |
| `notebook` | >=7.0 | Running `main.ipynb` |

### Installation

```bash
git clone https://github.com/rudramdindorkar/sign-language-recognition.git
cd sign-language-recognition

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requiremnets.txt
```

> **Note:** For GPU training, install the CUDA build of PyTorch first by following the selector at [pytorch.org](https://pytorch.org/get-started/locally/), then run `pip install -r requiremnets.txt`.

### Download required assets

1. **MediaPipe hand landmarker model**

```bash
   curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

2. **ASL Alphabet dataset** from [Kaggle (grassknoted/asl-alphabet)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). Extract it so the paths are:

```
   archive/asl_alphabet_train/asl_alphabet_train/<class folders>
   archive/asl_alphabet_test/asl_alphabet_test/
```

> A pre-trained `sign_landmark.pt` is included, so you can skip training and go straight to live recognition (only `hand_landmarker.task` is needed).

## Usage

Open `main.ipynb` in Jupyter or VS Code.

### 1. Train

Run the first cell. It will:

1. Extract and cache hand landmarks (`landmarks_cache.npz`) for up to `MAX_PER_CLASS` images per class
2. Train the MLP and save the best checkpoint to `sign_landmark.pt`
3. Report validation accuracy and, if available, accuracy on the test folder

### 2. Run live recognition

Run the second cell. A webcam window opens showing the hand skeleton, the predicted letter with confidence, and the typed text.

| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Clear text |
| `b` | Backspace |
| `Space` | Insert a space |

Gestures `space` and `del` from the dataset are also mapped to their text actions.

## Configuration

Key settings at the top of each notebook cell:

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_PER_CLASS` | 800 | Images sampled per class for training |
| `EPOCHS` | 60 | Training epochs |
| `BATCH` | 128 | Batch size |
| `LR` | 3e-3 | Peak learning rate (OneCycle) |
| `VAL_SPLIT` | 0.15 | Validation fraction |
| `CONF` | 0.50 | Minimum confidence to accept a prediction |
| `HOLD` | 20 | Frames a sign must be held to type it |
| `SMOOTH` | 10 | Frames in the majority-vote window |
| `CAMERA` | 0 | Camera index (try `1` if the camera fails to open) |

## Results

| Metric | Score |
|--------|-------|
| Validation accuracy | _fill in after training_ |
| Test accuracy | _fill in after training_ |

## Limitations

- Recognizes **static** alphabet signs only; dynamic letters (e.g. J, Z) are approximated by a single pose.
- Visually similar signs (e.g. M/N/S/T) can be confused.
- Only one hand is tracked at a time.
- Requires reasonable lighting and a clearly visible hand.

## Roadmap

- [ ] Support dynamic gestures (J, Z) with a temporal model
- [ ] Word suggestion / autocorrect
- [ ] Export to ONNX / TFLite for mobile and web
- [ ] Standalone `train.py` and `run.py` scripts

## Tech Stack

Python · PyTorch · MediaPipe · OpenCV · NumPy

## Acknowledgements

- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [ASL Alphabet dataset by grassknoted](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## Author

**Rudram Dindorkar**
GitHub: [@rudramdindorkar](https://github.com/rudramdindorkar)

## License

Distributed under the MIT License. See [`License.md`](License.md) for details.
