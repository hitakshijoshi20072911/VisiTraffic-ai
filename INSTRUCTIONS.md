# 📋 VisiTraffic AI – Instructions to Run

This document explains how to set up, run, and (optionally) train the VisiTraffic AI system.  
Read it together with `README.md` for the full project context.

---

## 1. Project Structure (what you get)

```
visi-traffic-ai/
├── models/                   # Core AI modules
│   ├── detector.py           # YOLOv8‑m with decoded output
│   ├── classifier.py         # MobileNetV3 helmet/seatbelt classifiers
│   ├── enhancer.py           # Zero‑DCE low‑light enhancement
│   └── alpr.py               # License plate recognition (placeholder)
├── rules/                    # 7 violation rule engines
│   ├── base_rule.py
│   ├── helmet_rule.py
│   ├── seatbelt_rule.py
│   ├── triple_riding.py
│   ├── stopline_redlight.py
│   ├── wrong_side.py
│   └── illegal_parking.py
├── pipeline.py               # End‑to‑end edge orchestrator
├── api/                      # FastAPI backend (future)
├── frontend/                 # Dashboard mockup (HTML/CSS/JS)
│   └── index.html
├── notebooks/                # Training notebooks
│   ├── train_detector.ipynb
│   └── train_classifier.ipynb
├── scripts/
│   ├── prepare_training_data.py
│   └── download_datasets.py
├── docker/                   # Dockerfiles for Jetson & cloud
├── configs/                  # YAML configs
├── requirements.txt
├── .gitignore
├── README.md
└── INSTRUCTIONS.md           ← this file
```

---

## 2. Prerequisites

- **Python 3.11** (strongly recommended – 3.10 may work but is untested)
- **Git**
- **Windows / Linux / macOS** – the code works cross‑platform; all commands below are given for PowerShell (Windows) and Bash (Linux/Mac)
- (Optional) **NVIDIA GPU** – not required for the quick test; the pipeline runs on CPU

---

## 3. Quick Start – Run the Edge Pipeline

This is the fastest way to see the system in action. It processes a randomly generated dummy image and prints any violations found.

### 3.1 Clone the repository

```bash
git clone https://github.com/hitakshijoshi20072911/VisiTraffic-ai.git
cd VisiTraffic-ai
```

### 3.2 Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv311
.\venv311\Scripts\activate
```

**Linux / Mac (Bash):**
```bash
python3 -m venv venv311
source venv311/bin/activate
```

### 3.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 3.4 Run the pipeline

```bash
python pipeline.py
```

**What happens:**
- The first run downloads the YOLOv8‑m backbone weights (~100 MB) from the internet.
- It creates a random 640×640 image (no real camera needed).
- The pipeline enhances the image, runs the detector, applies all seven rule engines, tries to read number plates, and prints the results.

**Expected output:**
```
Initialising VisiTraffic pipeline (this may load weights)...
Processing dummy frame...
Done. Found X violation(s).
  - triple_riding (conf 0.80)
  - …
Pipeline test completed successfully.
```
*The exact number of violations may vary because the dummy image is random, but you should see at least one violation (typically “triple_riding” or “stop_line_violation”).*

**To test with a real image,** open `pipeline.py` and find the `if __name__ == "__main__":` block. Replace the dummy image creation with:

```python
dummy_img = cv2.imread("path/to/your/image.jpg")
```

Then run again.

---

## 4. Frontend Dashboard (Live Demo)

A fully interactive, static mockup of the final user interface is already deployed:

🔗 **https://visitraffic-ai-frontend.vercel.app/**

You can also open `frontend/index.html` directly in any browser.

The dashboard contains:
- **Command Center** – key metrics, heatmap, top‑5 risk list
- **Live Feed** – simulated annotated video with bounding boxes
- **Violation Queue** – risk‑ranked cards with evidence and recommended actions
- **Analytics** – violation trends and charts
- **How It Works** – six‑step flow, architecture diagram, interactive risk formula

> ⚠️ The frontend is a **static prototype** and is not connected to the live backend. The backend pipeline (above) is functional but still under development.

---

## 5. Training the Models (Optional)

The repository contains scripts and notebooks to train the AI models on real traffic data.  
*Note: Training requires the actual datasets to be downloaded and placed in `data/raw/` (see `README.md` for the list of datasets).*

### 5.1 Prepare a Training Subset

If you have the IDD and BDD100K datasets in `data/raw/`, run:

```bash
python scripts/prepare_training_data.py
```

This copies 300 IDD images and 200 BDD100K images (with their annotations if available) into `data/train_subset/`. You can edit the script to change the number of images.

### 5.2 Fine‑tune the YOLOv8 Detector

Open the Jupyter notebook:

```bash
jupyter notebook notebooks/train_detector.ipynb
```

Or open it in VS Code with the Python extension. The notebook will:
- Load images and XML annotations from `data/train_subset/`
- Build the YOLOv8‑m model with the COCO‑pretrained backbone
- Freeze the backbone for the first epochs, then unfreeze and fine‑tune
- Save the trained model as `models/detector_trained.h5`

**Predicted outcome (on a small subset):**
- mAP@0.5 ≈ 55‑60% (improves with more data)
- Inference latency unchanged (model architecture is the same)

After training, modify `models/detector.py` to load your trained weights instead of the default ones (instructions in the notebook).

### 5.3 Train the Helmet / Seatbelt Classifiers

Open the notebook:

```bash
jupyter notebook notebooks/train_classifier.ipynb
```

The notebook will:
- Load helmet / no‑helmet crops from your Kaggle dataset
- Fine‑tune a MobileNetV3‑Small binary classifier
- Save the model as `models/helmet_classifier.h5` (repeat for seatbelt)

**Predicted accuracy:**
- Helmet classifier ≈ 96% (on Kaggle test set)
- Seatbelt classifier ≈ 94% (on synthetic + real samples)

After training, update `models/classifier.py` to load the saved models.

---

## 6. Docker Deployment (Advanced)

If you have Docker and an NVIDIA GPU, you can build the edge container:

```bash
docker build -f docker/Dockerfile.jetson -t visitraffic-edge .
docker run --gpus all -it visitraffic-edge
```

The container includes all dependencies and starts the pipeline automatically.  
For cloud deployment, a similar backend container is defined in `docker/Dockerfile.backend`.

---

## 7. Current Status & Notes for Reviewers

- **This is a prototype submission** for Flipkart Gridlock 2.0 Theme 3 (“idea submission”).
- The core inference pipeline (`pipeline.py`) is **functional** and can process images, detect violations, and apply the rule engine.
- Large‑scale model training, backend‑frontend integration, and real‑time video processing are under active development.
- Performance metrics (e.g., latency, FPS) are **targets** based on literature and early prototyping; they will be validated on Jetson hardware in the next phase.
- The frontend dashboard is a **static mockup** demonstrating the intended user experience.

---

## 8. Support

If you encounter any issues, please open an issue on the [GitHub repository](https://github.com/hitakshijoshi20072911/VisiTraffic-ai) or contact the author.

---

**Thank you for reviewing VisiTraffic AI!**

---
