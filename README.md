# Text-to-Sketch

> **Hand Simulation Pipeline** — converts anime images into Sketchformer-ready
> stroke-5 vector sequences, complete with realistic kinematics and a K-means
> Tok-Dict motion vocabulary.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Stages](#pipeline-stages)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [1. Download Dataset](#1-download-dataset)
  - [2. Extract Sketches](#2-extract-sketches)
  - [3. Run Full Pipeline](#3-run-full-pipeline)
  - [4. Evaluate Ordering](#4-evaluate-ordering)
- [Output Artefacts](#output-artefacts)
- [Environment Variables](#environment-variables)
- [CLI Reference](#cli-reference)

---

## Overview

**Text-to-Sketch** is a data preprocessing pipeline that transforms raw anime
frames into stroke-5 vector sequences suitable for training
[Sketchformer](https://github.com/leosampaio/sketchformer).  It implements the
complete **Hand Simulation Pipeline** (function *H*):

```
Anime image
  └─ 1: Lineart extraction    (ControlNet LineartAnimeDetector)
  └─ 2: Vectorization + RDP   (OpenCV contours → Ramer-Douglas-Peucker)
  └─ 3: Stroke ordering       (Directional / Greedy / TSP)
  └─ 4: Sigma-Lognormal kinematics
  └─ 5: stroke-5 formatting   → Sketchformer input
       + Tok-Dict codebook     → K-means discrete motion vocabulary
```

---

## Pipeline Stages

| Stage | Module | Description |
|---|---|---|
| **1** | `pipeline/steps/sketch_ext  ractor.py` | ControlNet anime lineart extraction |
| **2** | `pipeline/steps/vectorizer.py` | RDP-simplified vector stroke extraction |
| **3** | `pipeline/steps/ordering_algorithms.py` | Directional bias / Greedy NN / TSP ordering |
| **4** | `pipeline/steps/kinematics.py` | Sigma-Lognormal velocity model |
| **5** | `pipeline/steps/stroke5_formatter.py` | stroke-5 `[Δx, Δy, p1, p2, p3]` formatter |
| **Tok-Dict** | `pipeline/tokdict/` | K-means codebook builder + encoder |

---

## Project Structure

```
text-to-sketch/
│
├── pipeline/                          # Core logic — importable Python package
│   ├── __init__.py
│   ├── steps/                         # One module per pipeline stage
│   │   ├── __init__.py
│   │   ├── sketch_extractor.py          # Stage 1 — Lineart extraction
│   │   ├── vectorizer.py                # Stage 2 — Vectorization + RDP
│   │   ├── ordering_algorithms.py       # Stage 3 — Stroke ordering
│   │   ├── kinematics.py                # Stage 4 — Kinematics
│   │   └── stroke5_formatter.py         # Stage 5 — stroke-5 formatting
│   ├── tokdict/                       # Tok-Dict module
│   │   ├── __init__.py
│   │   ├── builder.py                 # K-means codebook builder
│   │   └── encoder.py                 # stroke-5 → token-index encoder/decoder
│   └── utils/
│       ├── __init__.py
│       └── io.py                      # save/load helpers for .npz and .npy
│
├── scripts/                           # Runnable CLI entry-points
│   ├── download_data.py               # Kaggle dataset download
│   ├── extract_sketches.py            # Stage 1 batch lineart extraction
│   ├── run_pipeline.py                # Unified interactive pipeline runner (2–5)
│   └── evaluate_ordering.py           # Ordering visualisation & evaluation
│
├── data/                              # All data (git-ignored)
│   ├── raw/                           # Raw downloaded datasets
│   └── processed/
│       ├── sketches/                  # Stage 1 output — binary line-art .png
│       ├── stroke5/                   # Stage 5 output — stroke-5 .npz files
│       ├── tokdict/                   # Tok-Dict output — codebook.npy + metadata.json
│       └── evaluations/               # Ordering evaluation plots
│
├── .env                               # Local environment variables (git-ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Requirements

- Python **3.10+**
- A valid **Kaggle API** account (for dataset download only)
- GPU recommended for Stage A (CPU supported)

**Core dependencies:**

```
controlnet-aux     # Stage 1 – LineartAnimeDetector
opencv-python      # Stage 2 – contour extraction
python-tsp         # Stage 3 – TSP ordering
scipy              # Stage 4 – Sigma-Lognormal (lognorm.ppf)
scikit-learn       # Tok-Dict – MiniBatchKMeans
numpy
matplotlib
Pillow
tqdm
python-dotenv
kagglehub
```

---

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:naolselemon/text-to-sketch.git
cd text-to-sketch
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle Credentials

**Option A — Kaggle JSON (recommended):**
```bash
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
```

**Option B — Environment variables:**
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

---

## Configuration

Create a `.env` file in the project root:

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
KAGGLE_DATASET=diraizel/anime-images-dataset

DATA_RAW_DIR=data/raw
INPUT_DIR=data/raw/data/anime_images
OUTPUT_DIR=data/processed/sketches

DETECT_RES=512
IMAGE_RES=512
MAX_PER_FOLDER=15
```

---

## Usage

### 1. Download Dataset

```bash
python scripts/download_data.py
```

Downloads the configured Kaggle dataset into `data/raw/`.

---

### 2. Extract Sketches

```bash
python scripts/extract_sketches.py
```

With custom options:

```bash
python scripts/extract_sketches.py \
  --input-dir data/raw/data/anime_images \
  --output-dir data/processed/sketches \
  --detect-resolution 512 \
  --image-resolution 512 \
  --max-per-folder 15
```

Produces binary line-art `.png` files in `data/processed/sketches/`.

---

### 3. Run Full Pipeline

**Interactive mode — asks how many sketches to process and which ordering:**

```bash
python scripts/run_pipeline.py
```

```
╔════════════════════════════════════════════════════════════╗
║          Hand Simulation Pipeline — Text-to-Sketch         ║
║ Stages: Vectorize → Order → Kinematics → Stroke5 → Tok-Dict║
╚════════════════════════════════════════════════════════════╝

  Available sketches : 10300

How many sketches to process? [default: 50]:
Stroke-ordering method:
  1) Directional bias [default]  — top-left → bottom-right
  2) Greedy nearest-neighbor     — minimise pen travel locally
  3) TSP approximation           — globally minimise pen travel
Choose [1/2/3, default: 1]:
```

Produces:
- `data/processed/stroke5/<name>.npz` — stroke-5 arrays, shape `(N+1, 5)`
- `data/processed/tokdict/codebook.npy` — K-means centroids, shape `(K, 2)`
- `data/processed/tokdict/metadata.json` — K, n_samples, timestamp

---

### 4. Evaluate Ordering

```bash
python scripts/evaluate_ordering.py
python scripts/evaluate_ordering.py --samples 20
```

Saves side-by-side evaluation plots to `data/processed/evaluations/`.

---

## Output Artefacts

### stroke-5 format (`data/processed/stroke5/*.npz`)

Each `.npz` contains a single array `stroke5` of shape `(N+1, 5)`:

| Column | Meaning |
|---|---|
| `Δx` | Relative X displacement to previous point |
| `Δy` | Relative Y displacement to previous point |
| `p1` | `1` = pen is drawing (mid-stroke) |
| `p2` | `1` = last point of stroke (pen lifts next) |
| `p3` | `1` = end-of-sketch sentinel (final row only) |

Load with:
```python
from pipeline.utils.io import load_stroke5
s5 = load_stroke5("data/processed/stroke5/my_sketch.npz")
```

### Tok-Dict Codebook (`data/processed/tokdict/`)

| File | Contents |
|---|---|
| `codebook.npy` | `(K, 2)` float32 array of (Δx, Δy) cluster centroids |
| `metadata.json` | `K`, `n_samples`, `codebook_shape`, `timestamp` |

Encode a sketch to token indices:
```python
from pipeline.utils.io import load_codebook
from pipeline.tokdict.encoder import encode_stroke5

codebook = load_codebook("data/processed/tokdict/codebook.npy")
tokens   = encode_stroke5(s5, codebook)   # shape (N+1,), dtype int32
# tokens[i] ∈ [0, K-1]  → motion token
# tokens[i] == K         → pen-lift token
# tokens[i] == K+1       → end-of-sketch token
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KAGGLE_USERNAME` | _(none)_ | Kaggle account username |
| `KAGGLE_KEY` | _(none)_ | Kaggle API key |
| `KAGGLE_DATASET` | `diraizel/anime-images-dataset` | Kaggle dataset slug |
| `DATA_RAW_DIR` | `data/raw` | Destination for downloaded dataset |
| `INPUT_DIR` | `data/raw/data/anime_images` | Source for lineart extraction |
| `OUTPUT_DIR` | `data/processed/sketches` | Output for extracted sketches |
| `DETECT_RES` | `512` | Detector input resolution |
| `IMAGE_RES` | `512` | Output image resolution |
| `MAX_PER_FOLDER` | `15` | Max images processed per subdirectory |

---

## CLI Reference

| Script | Purpose |
|---|---|
| `python scripts/download_data.py` | Download Kaggle dataset |
| `python scripts/extract_sketches.py [--input-dir] [--output-dir] [--detect-resolution] [--image-resolution] [--max-per-folder]` | Run Stage A lineart extraction |
| `python scripts/run_pipeline.py` | Interactive Stages B–E + Tok-Dict |
| `python scripts/evaluate_ordering.py [--samples N]` | Visualise ordering quality |
