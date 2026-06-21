

# 🚦 VisiTraffic AI  
**Explainable Multi‑Violation Traffic Intelligence for Bengaluru**

![Python](https://img.shields.io/badge/Python-3.11-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange) ![KerasCV](https://img.shields.io/badge/KerasCV-0.6.4-red) ![YOLOv8](https://img.shields.io/badge/YOLOv8--m--backbone-00D4AA) ![Docker](https://img.shields.io/badge/Docker-ready-2496ED) ![Status](https://img.shields.io/badge/Phase-Prototype-brightgreen)  
[![Live Demo](https://img.shields.io/badge/Demo-Visit%20Dashboard-brightgreen)](https://visitraffic-ai-frontend.vercel.app/)  
[![Source Code](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/hitakshijoshi20072911/VisiTraffic-ai)

> **Flipkart Gridlock 2.0 – Theme 3**  
> *Automated Photo Identification and Classification of Traffic Violations Using Computer Vision*

---

## 📌 Problem Statement

Bengaluru’s 14 million citizens lose billions of hours to traffic gridlock every year.  
Thousands of CCTV cameras watch the roads, but traffic enforcement still relies on **manual, slow, and inconsistent** review.  
Existing automated systems suffer from two critical gaps:

1. **Vision Gap** – Cloud‑dependent AI fails in Bengaluru’s monsoon, low‑light, and motion‑blur conditions.  
2. **Priority Gap** – Violations are reported as a flat, unsorted list; a missing helmet on an empty lane gets the same urgency as a red‑light jump at a crowded school junction.  

Without risk prioritisation, patrol resources are wasted, and truly dangerous violations go unenforced.

---

## 💡 Our Solution: VisiTraffic AI

VisiTraffic AI is a **decision‑support platform** that turns raw CCTV footage into a **risk‑ranked, explainable action queue** for traffic officers.  
It combines two breakthrough innovations:

- **Edge‑native, real‑time detection** – 7 violation types detected in < 30 ms on a Jetson Orin (target), even in rain, darkness, and glare.  
- **Adaptive Violation Intelligence Engine (AVIE)** – an agentic AI layer that queries a knowledge graph of historical hazards, traffic density, and temporal patterns to assign every violation a **risk score (0‑100) and a plain‑English explanation**.

The system transmits only violation metadata and cropped evidence – never raw video – protecting privacy and cutting cloud bandwidth by over 90% (projected).

---

## 🧠 How It Works (Six Simple Steps)

| Step | What Happens | Technology |
|------|--------------|------------|
| 1. **Brighten** | Enhance dark, rainy, or shadowy scenes | Zero‑DCE (unsupervised CNN) |
| 2. **Detect** | Find all vehicles, riders, pedestrians in one pass | YOLOv8‑m (KerasCV backbone) |
| 3. **Verify** | Check helmet, seatbelt, triple‑riding, red‑light, etc. | MobileNetV3 (quantisation‑ready) + deterministic rules |
| 4. **Read Plate** | Extract Indian number‑plate text at any angle | LPRNet (ONNX) |
| 5. **Contextualise** | Query a live knowledge graph for accident history, density, and recency | TigerGraph + LangGraph agent |
| 6. **Score & Explain** | Compute a weighted risk score and generate a human‑readable reason | Weighted fusion formula (configurable weights) |

All inference steps 1‑4 run **on the edge device** in under 30 ms (target); steps 5‑6 run in the cloud to give officers a prioritised, explainable queue.

---

## ✨ Features & USPs

- **Edge‑native, real‑time** – Designed for 25+ FPS on Jetson Orin; works offline.  
- **7 violations, 1 pipeline** – A shared backbone with lightweight conditional heads avoids running seven separate models.  
- **Weather‑robust** – Proprietary Zero‑DCE enhancement adapts to Bengaluru’s monsoons without paired training data.  
- **Explainable priorities** – Every violation comes with a risk score and a natural‑language explanation.  
- **Privacy‑preserving** – Only violation crops and metadata leave the edge; full video never transmitted.  
- **Modular & extensible** – New violation types can be added by writing a single rule file and updating AVIE logic.  
- **Production‑ready** – Docker containers, FastAPI endpoints, and one‑command deployment (planned).  
- **Synthetic data augmentation** – Rare violations (triple riding, wrong‑side) generated via Stable Diffusion + ControlNet.

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| **Deep Learning Framework** | TensorFlow 2.15 + KerasCV 0.6.4 |
| **Object Detection** | YOLOv8‑m (CSPDarknet backbone, COCO pretrained) |
| **Lightweight Classifiers** | MobileNetV3‑Small (quantisation‑ready) |
| **Low‑Light Enhancement** | Zero‑DCE (custom `tf.keras` layer) |
| **License Plate Recognition** | LPRNet (PyTorch → ONNX) |
| **Edge Inference Runtime** | TensorRT / ONNX Runtime |
| **Graph Database (AVIE)** | TigerGraph (OpenCypher) |
| **Agentic Orchestration** | LangGraph (future integration) |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend Dashboard** | HTML/CSS/JS static mockup (Vercel), Streamlit (planned) |
| **Containerisation** | Docker + NVIDIA Container Toolkit |

---

## 🏗️ System Architecture (Simplified)

```
┌───────────────────────────────┐
│  EDGE (Jetson Orin)           │
│  CCTV → Zero‑DCE → YOLOv8‑m   │
│       ├── Helmet/Seatbelt     │
│       ├── Rule Engine         │
│       └── ALPR                │
│       → Evidence JSON         │
└────────┬──────────────────────┘
         │ (metadata only)
         ▼
┌───────────────────────────────┐
│  CLOUD / COMMAND CENTER       │
│  FastAPI → AVIE (LangGraph)   │
│       └── TigerGraph          │
│       → Risk Score + Reason   │
│       → Streamlit Dashboard   │
└───────────────────────────────┘
```

---

## 📂 Project File Structure

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
└── README.md
```
 Frontend provided in the source code folder (.zip)
---

## 📊 Datasets Used

The system is trained and evaluated on the following publicly available datasets:

| Dataset | Content |
|--------|---------|
| **Indian Driving Dataset (IDD)** | 10,000+ images of Indian traffic scenes – vehicles, riders, pedestrians, auto‑rickshaws |
| **BDD100K** | 100,000 images covering diverse weather, time of day, and urban/rural scenes |
| **UA‑DETRAC** | Multi‑weather vehicle detection and tracking, 140k frames |
| **Helmet Detection (Kaggle)** | Dedicated helmet vs. no‑helmet crops |
| **Indian Rider & Number Plates (Kaggle)** | Riders with/without helmets, plus Indian number plates |

> ROD2021 was originally planned but excluded due to access constraints; its role (small‑object detection) is covered by multi‑scale training and the existing helmet/plate datasets.

---

## 📈 Current Prototype Status

### Completed
- System architecture design & documentation
- Frontend dashboard prototype (Vercel‑deployed)
- Violation Intelligence Engine (AVIE) design
- Edge inference pipeline prototype (`pipeline.py`)
- Rule engine for 7 violation types
- Zero‑DCE enhancement layer
- Training scripts & notebooks
- Video walkthrough & pitch deck

### In Progress
- Large‑scale model training on full datasets
- Backend API ↔ frontend dashboard integration
- End‑to‑end performance benchmarking on real hardware
- Live camera feed integration

### Future Work
- Production deployment on Jetson Orin
- Integration with Bengaluru Traffic Police CCTV network & e‑challan system
- Real‑time graph analytics for predictive patrolling
- White‑label product for other Indian cities

---

## 🚀 How to Run the Project

### 1. Prerequisites
- **Python 3.11** (3.10 may also work)
- Git
- (Optional) Docker & NVIDIA Container Toolkit for Jetson deployment

### 2. Clone & Set Up Environment
```bash
git clone https://github.com/hitakshijoshi20072911/VisiTraffic-ai.git
cd VisiTraffic-ai

# Create virtual environment
python -m venv venv311
source venv311/bin/activate      # Linux / Mac
venv311\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Edge Pipeline (Quick Test)
```bash
python pipeline.py
```
**What happens:**  
- First run downloads YOLOv8‑m backbone weights (~100 MB).  
- It processes a random 640×640 dummy image.  
- The pipeline prints the number and type of violations found.  

**Expected output:**  
```
Initialising VisiTraffic pipeline (this may load weights)...
Processing dummy frame...
Done. Found X violation(s).
  - triple_riding (conf 0.80)
  - …
Pipeline test completed successfully.
```

To test with a real image, edit the `if __name__ == "__main__":` block in `pipeline.py` and replace the dummy generation with `cv2.imread("your_image.jpg")`.

---

## 🖥️ Frontend Dashboard

A static, interactive mockup of the final user interface is deployed at:  
🔗 **[https://visitraffic-ai-frontend.vercel.app/](https://visitraffic-ai-frontend.vercel.app/)**

The dashboard shows:  
- **Command Center** – KPI cards, heatmap, top‑5 risk list.  
- **Live Feed** – simulated annotated video with bounding boxes.  
- **Violation Queue** – risk‑ranked cards with evidence and recommended actions.  
- **Analytics** – violation trends and charts.  
- **How It Works** – six‑step flow, architecture diagram, interactive risk formula.

> ⚠️ The frontend is a **static prototype** and is not yet connected to the live backend. The backend AI models are under development; the pipeline prototype demonstrates the core inference logic.

---

## 🏋️‍♀️ Training Your Own Models

### A. Prepare a Training Subset
```bash
python scripts/prepare_training_data.py
```
This copies 300 IDD and 200 BDD100K images (configurable) into `data/train_subset/`.

### B. Fine‑tune the YOLOv8 Detector
Open `notebooks/train_detector.ipynb` in Jupyter / VS Code and run all cells.  
The notebook will:
- Load images and annotations from `data/train_subset/`
- Freeze the COCO‑pretrained backbone, train the head, then unfreeze
- Save the trained model to `models/detector_trained.h5`

**Predicted training outcome (on a small subset):**  
- mAP@0.5: ~55‑60% (limited by data size; full dataset training would exceed 60%)  
- Inference latency: unchanged (~28 ms FP16, ~12 ms INT8) once deployed on Jetson hardware

### C. Train the Helmet / Seatbelt Classifiers
Open `notebooks/train_classifier.ipynb` and run all cells.  
The notebook will:
- Load helmet / no‑helmet crops from your Kaggle dataset
- Fine‑tune MobileNetV3‑Small
- Save to `models/helmet_classifier.h5`

**Predicted accuracy:**  
- Helmet classifier: ~96% (on Kaggle test set)  
- Seatbelt classifier: ~94% (on synthetic + real samples)

After training, update `models/classifier.py` to load your saved `.h5` files instead of using random weights.

---

## 🐳 Docker Deployment

Build and run the edge container (requires NVIDIA GPU):
```bash
docker build -f docker/Dockerfile.jetson -t visitraffic-edge .
docker run --gpus all -it visitraffic-edge
```

The cloud backend container can be built similarly with `docker/Dockerfile.backend`.

---

## 📊 Target Performance Metrics

| Metric | Target |
|--------|--------|
| Inference latency | < 30 ms per frame (Jetson Orin) |
| Throughput | 25+ FPS |
| Bandwidth reduction | > 90% (metadata‑only transmission vs. raw video) |
| Helmet classifier accuracy | > 95% |
| Seatbelt classifier accuracy | > 90% |
| ALPR character accuracy | > 90% |

*These metrics are based on literature and early prototyping; full hardware benchmarking is in progress.*

---

## 📚 Key References

- YOLOv8 – Ultralytics (2023)  
- Zero‑DCE – Guo et al., CVPR 2020  
- MobileNetV3 – Howard et al., ICCV 2019  
- LPRNet – Zherzdev & Gruzdev (2018)  
- Focal Loss – Lin et al., ICCV 2017  
- CIoU Loss – Zheng et al., AAAI 2020  
- LangGraph – LangChain Inc. (2024)

---

