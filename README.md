# VigilantAI — Real-Time Anomalous Activity Detection for CCTV Surveillance

[![IEEE Xplore](https://img.shields.io/badge/IEEE-ICIT%202024-blue)](https://ieeexplore.ieee.org/abstract/document/10859907)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-00FFFF)](https://github.com/ultralytics/yolov5)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey)](https://universe.roboflow.com/detectai/vigilant_ai/dataset/8)

**VigilantAI** is an open-source deep learning pipeline for **real-time detection of robberies, weapons, and threatening behavior** in CCTV and surveillance video streams. It implements the approach described in our **IEEE ICIT 2024** conference paper, using a custom-labeled dataset and **YOLOv5** object detection to help security teams spot anomalous activity faster than manual monitoring alone.

> **Research paper:** [VigilantAI: Real-time detection of anomalous activity from a video stream using deep learning](https://ieeexplore.ieee.org/abstract/document/10859907) — *2024 International Conference on IT and Industrial Technologies (ICIT)*, IEEE Xplore.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Detection Classes](#detection-classes)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Pre-trained Models](#pre-trained-models)
- [Results](#results)
- [Use Cases](#use-cases)
- [Limitations & Responsible Use](#limitations--responsible-use)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Public spaces rely heavily on CCTV, but **human operators cannot watch every feed at once**. Missed robberies, delayed weapon identification, and slow incident response remain critical safety challenges.

**VigilantAI** addresses this with a **computer vision system** that:

1. Processes video frames from surveillance cameras in near real time.
2. Detects **normal persons**, **robbers**, and **weapons** using a fine-tuned **YOLOv5s** model.
3. Supports rapid alerting workflows so security personnel can respond sooner.

The research introduces a **custom dataset** built from diverse CCTV footage and movie scenes, with **data augmentation** and bounding-box annotations—because high-quality labeled data for illegal-activity detection is scarce. According to the published IEEE paper, the system achieves approximately **85% accuracy** on the evaluation setup described in that work.

This repository contains the **official implementation artifacts**: dataset download scripts, YOLOv5 training notebooks, model weights, and deployment-ready **TFLite** exports for edge and IoT targets.

---

## Key Features

| Capability | Description |
|------------|-------------|
| **Multi-class threat detection** | Simultaneous detection of `normal person`, `robber`, and `weapon` |
| **YOLOv5s backbone** | Fast, single-stage object detection optimized for real-time CCTV |
| **Custom VigilantAI dataset** | Curated and augmented imagery from CCTV and cinematic sources |
| **Roboflow integration** | Versioned dataset hosted on Roboflow Universe (`detectai/vigilant_ai`) |
| **Colab-ready training** | Jupyter notebooks for GPU training on Google Colab |
| **Edge deployment** | Exported `.pt` (PyTorch) and `.tflite` (TensorFlow Lite) model files |
| **Peer-reviewed research** | Published at **ICIT 2024** (IEEE Xplore) |

---

## How It Works

```text
CCTV / Video Stream
        │
        ▼
  Frame Extraction
        │
        ▼
  YOLOv5s (VigilantAI weights)
        │
        ├── normal person  → baseline / no alert
        ├── robber         → anomalous activity
        └── weapon         → threat indicator
        │
        ▼
  Alerting & Security Response (application layer)
```

1. **Phase 1 — Data collection & labeling:** Footage is converted to images, labeled with bounding boxes, and augmented.
2. **Phase 2 — Model training:** YOLOv5s is trained on the VigilantAI dataset with standard hyperparameters (`640×640`, batch training, multi-epoch schedule).
3. **Deployment:** Trained weights run inference on new frames; TFLite exports target resource-constrained devices.

---

## Repository Structure

```text
VigilantAI/
├── Phase_1_Data_Collection_and_Labeling/
│   └── download_dataset.py          # Download VigilantAI dataset from Roboflow
├── Phase_2_Model_Training/
│   ├── data.yaml                    # YOLO class names and dataset paths
│   ├── Training.ipynb               # End-to-end YOLOv5 training (25 epochs)
│   ├── Training_VigilantAI_model.ipynb
│   ├── vigilantai_beta_v002 (1).pt  # PyTorch weights
│   └── vigilantai_beta_v002 (2).tflite
└── public/                          # Static assets (fonts, UI resources)
```

---

## Detection Classes

The model is trained on **three object classes** defined in `Phase_2_Model_Training/data.yaml`:

| Class ID | Label | Role |
|----------|-------|------|
| 0 | `normal person` | Non-threatening pedestrian activity |
| 1 | `robber` | Robbery / anomalous criminal behavior |
| 2 | `weapon` | Firearms, knives, and similar threats |

These classes align with the paper’s focus on **robberies** and **gunpoint threats** in surveillance footage.

---

## Dataset

- **Source:** Custom VigilantAI dataset (CCTV clips, movies, augmented frames).
- **Roboflow project:** [detectai/vigilant_ai — Dataset v8](https://universe.roboflow.com/detectai/vigilant_ai/dataset/8)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Splits:** `train`, `valid`, and `test` (YOLO format with images + label files).

### Download the dataset

**Option A — Python (Phase 1 script):**

```bash
cd Phase_1_Data_Collection_and_Labeling
python download_dataset.py
```

**Option B — Inside a Colab / training notebook:**

```python
import torch
torch.hub.download_url_to_file(
    "https://app.roboflow.com/ds/<your-export-id>?key=<your-key>",
    "tmp.zip",
)
# unzip into yolov5/dataset/
```

> Roboflow download URLs are project-specific. Use your Roboflow export link or the notebook in `Phase_2_Model_Training/` for the bundled dataset version used during training.

---

## Requirements

- **Python** 3.10+
- **CUDA-capable GPU** (recommended for training; Google Colab works out of the box)
- **[Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)** (cloned automatically in notebooks)
- **PyTorch** with CUDA (for `.pt` inference)
- **TensorFlow Lite runtime** (optional, for `.tflite` on edge devices)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/danishjavedcodes/VigilantAI.git
cd VigilantAI
```

### 2. Train or run inference with YOLOv5

Open either notebook in Google Colab (GPU runtime recommended):

- [`Phase_2_Model_Training/Training.ipynb`](Phase_2_Model_Training/Training.ipynb)
- [`Phase_2_Model_Training/Training_VigilantAI_model.ipynb`](Phase_2_Model_Training/Training_VigilantAI_model.ipynb)

Each notebook will:

1. Clone [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
2. Download and unzip the VigilantAI dataset
3. Fine-tune `yolov5s.pt` on the three-class task

---

## Training

Default training configuration used in this repository:

| Parameter | Value |
|-----------|-------|
| Base weights | `yolov5s.pt` |
| Image size | `640` |
| Batch size | `16` |
| Epochs | `20`–`25` |
| Data config | `dataset/data.yaml` |
| Caching | RAM cache enabled (`--cache`) |

**Example command** (after cloning YOLOv5 and preparing `dataset/`):

```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 25 \
  --data /path/to/dataset/data.yaml \
  --weights yolov5s.pt \
  --cache
```

Best checkpoints are saved under `runs/train/exp/weights/best.pt`.

---

## Inference

Run detection on an image or video source:

```bash
python detect.py \
  --weights runs/train/exp/weights/best.pt \
  --source /path/to/image_or_video.jpg \
  --img 640 \
  --conf-thres 0.25
```

Or use the pre-exported weights in `Phase_2_Model_Training/`:

- `vigilantai_beta_v002 (1).pt`
- `vigilantai_beta_v002 (2).tflite`

---

## Pre-trained Models

| File | Format | Use case |
|------|--------|----------|
| `vigilantai_beta_v002 (1).pt` | PyTorch | Desktop / server inference with YOLOv5 |
| `vigilantai_beta_v002 (2).tflite` | TensorFlow Lite | Mobile, embedded, and IoT deployment |

Place weights in your YOLOv5 `weights/` directory or pass the full path to `detect.py`.

---

## Results

### Published research (IEEE ICIT 2024)

The peer-reviewed paper reports approximately **85% accuracy** on the VigilantAI evaluation protocol, with real-time threat detection and alerting for security personnel. See the full methodology and metrics in the paper:

**[https://ieeexplore.ieee.org/abstract/document/10859907](https://ieeexplore.ieee.org/abstract/document/10859907)**

### Validation metrics (this repository’s notebooks)

Representative YOLO validation metrics from training runs in the included notebooks:

| Notebook | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|----------|-----------|--------|---------|--------------|
| `Training.ipynb` | 0.515 | 0.489 | 0.431 | 0.184 |
| `Training_VigilantAI_model.ipynb` | 0.735 | 0.501 | 0.501 | 0.215 |

> Metrics depend on dataset version, epoch count, and hardware. Re-train on the latest Roboflow export for production deployments.

---

## Use Cases

- **Retail & banking** — Early robbery detection on indoor CCTV
- **Public venues** — Stadiums, transit hubs, and campuses
- **Smart city surveillance** — Automated triage of multiple camera feeds
- **Law enforcement support** — AI-assisted review of incident footage
- **Research & education** — Benchmark for anomalous-activity object detection

---

## Limitations & Responsible Use

- **Not a standalone security product.** VigilantAI is a research codebase; production systems need calibration, human oversight, privacy compliance, and legal review.
- **False positives/negatives** can occur due to occlusion, lighting, camera angle, and dataset bias (including synthetic movie scenes).
- **Ethical deployment** requires consent, transparent policies, and adherence to local surveillance and biometric regulations.
- **Do not use** for discriminatory profiling or covert surveillance without authorization.

---

## Citation

If you use VigilantAI in academic or commercial work, please cite the IEEE conference paper:

```bibtex
@inproceedings{vigilantai2024,
  title={VigilantAI: Real-time detection of anomalous activity from a video stream using deep learning},
  booktitle={2024 International Conference on IT and Industrial Technologies (ICIT)},
  year={2024},
  organization={IEEE},
  address={Chiniot, Pakistan},
  url={https://ieeexplore.ieee.org/abstract/document/10859907}
}
```

**Paper link:** [https://ieeexplore.ieee.org/abstract/document/10859907](https://ieeexplore.ieee.org/abstract/document/10859907)

---

## License

- **Code in this repository:** Check repository license settings on GitHub (add a `LICENSE` file if not yet specified).
- **VigilantAI dataset (Roboflow):** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — attribution required when using the dataset.
- **YOLOv5:** [Ultralytics YOLOv5 license](https://github.com/ultralytics/yolov5/blob/master/LICENSE)

---

## Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the object detection framework
- [Roboflow](https://roboflow.com/) for dataset versioning and export tooling
- **GIK Institute of Engineering Sciences & Technology** and co-author institutions affiliated with the ICIT 2024 publication
- Contributors and reviewers of the **2024 International Conference on IT and Industrial Technologies (ICIT)**

---

<p align="center">
  <strong>Keywords:</strong>
  VigilantAI · YOLOv5 · weapon detection · robbery detection · CCTV surveillance ·
  real-time object detection · anomalous activity · deep learning · computer vision ·
  smart security · IEEE ICIT 2024 · TensorFlow Lite · public safety AI
</p>
