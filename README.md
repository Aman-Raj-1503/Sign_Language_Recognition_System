# Sign Language Recognition System

A real-time, command-line Sign Language Recognition System built with **OpenCV**, **MediaPipe Hands**, and classical **scikit-learn** classifiers. The system detects a hand in a webcam feed, extracts 21 3D hand landmarks, normalizes them into a translation- and scale-invariant feature vector, and classifies the resulting vector into a gesture label (e.g. ASL alphabet letters, "HELLO", "THANKS", "YES", "NO").

Course: **Computer Vision** — VITyarthi "Build Your Own Project" submission.

---

## 1. Features

- **Data collection module** — capture your own labelled gesture dataset from a webcam.
- **Modular preprocessing** — translation/scale-invariant landmark normalization (works regardless of hand distance from camera or position in frame).
- **Configurable classifier** — Random Forest (default), SVM, or MLP, selectable via CLI flag.
- **Evaluation module** — confusion matrix + per-class precision/recall/F1 report.
- **Real-time recognition** — live webcam inference with temporal smoothing (majority vote over recent frames) and a confidence threshold to avoid displaying low-confidence guesses.
- **Single unified CLI** (`python -m src.main ...`) — no GUI setup required, runs entirely from the terminal.
- **Logging** — rotating log file under `logs/app.log` for troubleshooting.

## 2. Technologies / Tools Used

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Hand detection & landmarks | MediaPipe Hands |
| Video capture / display | OpenCV |
| Classifier | scikit-learn (Random Forest / SVM / MLP) |
| Data storage | CSV (pandas) |
| Model persistence | joblib |
| Visualization (confusion matrix) | matplotlib |
| Testing | pytest |

## 3. Project Structure

```
sign-language-recognition/
├── src/
│   ├── config.py              # central configuration
│   ├── utils.py                # logging, FPS counter, camera helpers
│   ├── landmark_extractor.py   # MediaPipe Hands wrapper
│   ├── preprocessing.py        # landmark normalization
│   ├── dataset.py              # CSV dataset read/write
│   ├── model.py                # classifier factory + save/load
│   ├── collect_data.py         # data collection CLI
│   ├── train.py                # training pipeline
│   ├── evaluate.py             # evaluation + confusion matrix
│   ├── recognize.py            # real-time recognition
│   └── main.py                 # unified CLI entry point
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
├── data/
│   ├── raw/                    # (empty; reserved for optional raw frame captures)
│   └── processed/              # landmarks_dataset.csv is generated here
├── models/                     # trained model + evaluation artifacts are generated here
├── docs/
│   ├── diagrams/                # architecture, workflow, UML, ER diagrams
│   └── sample_run/               # sample metrics/confusion matrix from a demo run
├── statement.md
├── requirements.txt
└── README.md
```

## 4. Environment Setup

### Prerequisites
- Python 3.10 or newer
- A working webcam
- Git

### Step-by-step installation

```bash
# 1. Clone the repository
git clone https://github.com/<github-username>/sign-language-recognition.git
cd sign-language-recognition

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 5. How to Run

All functionality is exposed through a single CLI entry point: `python -m src.main <command>`.

### 5.1 Collect training data

Capture samples for one gesture label at a time. Repeat for every gesture you want to recognize (aim for at least 100–150 samples per class, varying hand angle/position slightly for robustness):

```bash
python -m src.main collect --label A --samples 150
python -m src.main collect --label B --samples 150
python -m src.main collect --label HELLO --samples 150
```

While the capture window is focused:
- Press **SPACE** to capture the current frame as a labelled sample.
- Press **Q** to stop collecting for that label.

### 5.2 Train the classifier

```bash
python -m src.main train --model-type random_forest
```

This loads `data/processed/landmarks_dataset.csv`, splits it into train/test sets, fits the classifier, and saves:
- `models/sign_classifier.joblib`
- `models/label_encoder.joblib`
- `models/metrics_report.json`

### 5.3 Evaluate the model

```bash
python -m src.main evaluate
```

Produces `models/confusion_matrix.png` and `models/evaluation_report.json`.

### 5.4 Run real-time recognition

```bash
python -m src.main recognize
```

Shows a live webcam window with the hand skeleton overlay and the predicted gesture + confidence. Press **Q** to quit.

### 5.5 Run the automated tests

```bash
pip install pytest   # already included in requirements.txt
pytest tests/ -v
```

The tests validate the landmark-normalization math and the model save/load/predict roundtrip using synthetic data, so they run without needing a webcam.

## 6. Configuration

All tunable parameters (camera index, confidence threshold, default gesture classes, model type, etc.) live in `src/config.py` — no code changes are needed elsewhere to adjust them.

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `RuntimeError: Could not open camera` | Another application is using the webcam, or the camera index is wrong — try `--camera 1`. |
| `FileNotFoundError: No dataset found` | Run `collect` for at least two gesture labels before `train`. |
| Low accuracy / frequent "Uncertain" | Collect more samples per class, vary hand angle/lighting, or lower `CONFIDENCE_THRESHOLD` in `config.py`. |
| Import error for `mediapipe` / `cv2` | Re-run `pip install -r requirements.txt` inside the activated virtual environment. |

Check `logs/app.log` for detailed error traces from any run.

## 8. Screenshots

See `docs/sample_run/sample_confusion_matrix.png` for a sample evaluation output from a demonstration run.

## 9. Future Enhancements

See Section 14 of the project report (`docs/Project_Report.pdf`) for planned improvements, including dynamic (motion-based) gesture support, a larger vocabulary, and text-to-speech output.

## 10. Author

Aman Raj (24BAI10769) 

