# Sketch Generation from Text
This repository prepares anime portrait sketches as vector stroke sequences and
experiments with scaling the 2020 Sketchformer idea to more complex anime-style
line art.

The project currently has two parallel tracks:

1. **Original Sketchformer integration**: prepare anime stroke data and launch a
   local checkout of the original TensorFlow Sketchformer codebase for
   checkpoint-compatible experiments.
2. **Native PyTorch rebuild**: a clean in-repo Sketchformer-style model,
   dataloader, losses, checkpointing, and training loop designed for modern
   CUDA training on a single RTX 3090-class GPU.

The local development machine can be CPU-only. Full native training is expected
to run on the GPU server.

## Current State

| Area | Status |
|---|---|
| Anime image preprocessing | Implemented. Downloads portraits, extracts line art, filters sketches, vectorizes centerlines, orders strokes, and writes stroke-5 files. |
| Sketchformer-ready data | Implemented. Builds a sketch token dictionary and converts stroke-5 sketches into chunked tok-dict `.npz` files with train/valid/test splits. Continuous stroke3 prep remains available for legacy experiments. |
| Original Sketchformer handoff | Implemented as a Docker command launcher. The bundled image is CPU-oriented; GPU use requires a custom compatible image. |
| Native PyTorch Sketchformer | Implemented for tok-dict reconstruction with SDPA attention, gradient checkpointing, length-bucketed loading, masked token cross entropy, token accuracy/perplexity metrics, codebook-decoded plots, evaluation, and export. |
| Native fine-tuning from converted TF weights | Implemented for the released tok-dict checkpoint, including the 200-position checkpoint base and zero-initialized 4096-position residual. |
| Text prompt conditioning | Not implemented in the current codebase. The present focus is anime sketch sequence modeling. |

## Pipeline Overview

```text
Danbooru2019 portraits
  -> anime line-art sketches
  -> filtered sketch images
  -> topology-preserving centerlines
  -> ordered drawing paths
  -> stroke-5 arrays
  -> sketch token dictionary
  -> Sketchformer-style tok-dict chunks
  -> original Sketchformer or native PyTorch training
```

## Repository Layout

```text
.
├── configs/
│   ├── data/                    Dataset paths, tok-dict format, batching
│   ├── experiment/              Smoke test and anime fine-tuning presets
│   ├── model/                   Native Sketchformer architecture config
│   ├── optimizer/               Optimizer, scheduler, and loss weights
│   └── trainer/                 Precision, checkpointing, runtime settings
│
├── prep_data/                   Download, extraction, filtering, tok-dict prep
├── pipeline/                    Vectorization, ordering, timing, stroke-5 export
├── metrics/                     Preprocessing and reconstruction evaluation
├── utils/                       Shared IO, paths, and tokenization helpers
│
├── models/sketchformer/         Native PyTorch Sketchformer-style model
├── dataloaders/                 Token/stroke datasets, masks, collation, loaders
├── core/                        Losses, metrics, checkpointing, train helpers
├── builders/                    Model, optimizer, scheduler, loss factories
├── scripts/sketchformer/        Native train, evaluate, export, inspect CLIs
│
├── integrations/
│   └── original_sketchformer/   Launcher and Docker files for legacy TF code
├── scripts/integrations/        CLI wrappers for integration workflows
│
├── data/                        Local generated data, git-ignored
├── weights/                     Local pretrained and fine-tuned weights
├── dependencies/                Optional local third-party checkouts
├── sketchformer/                Optional original Sketchformer checkout
└── tests/                       Unit and smoke tests
```

| Area | Main Paths | Purpose |
|---|---|---|
| Data preparation | `prep_data/`, `pipeline/`, `scripts/prepare_data/` | Build clean anime sketch data from images and export stroke-5, token dictionary, and tok-dict files. |
| Native training | `models/sketchformer/`, `dataloaders/`, `core/`, `builders/`, `scripts/sketchformer/` | Rebuilt PyTorch Sketchformer-style training path for long anime stroke sequences. |
| Configuration | `configs/` | Compose reusable data, model, optimizer, trainer, and experiment settings. |
| Legacy integration | `integrations/original_sketchformer/`, `scripts/integrations/`, `sketchformer/` | Run the original TensorFlow Sketchformer checkout for compatibility experiments. |
| Outputs | `data/`, `weights/`, `logs/`, `runs/` | Local datasets, checkpoints, logs, and training artifacts. These are not meant for source control. |

## Environment

Use Python 3.10 or 3.11 for the training environment. The repo may be inspected
on CPU, but native training needs PyTorch with CUDA on the server.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For the RTX 3090 server, install the PyTorch CUDA wheel that matches the server
driver before installing the remaining requirements. Example shape:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

The data downloader also requires the system `rsync` binary.

```bash
sudo apt install rsync
```

## Data Preparation

Download portraits:

```bash
tts-download-data --num-images 5000
```

Extract line-art sketches with the default ControlNet anime line-art detector:

```bash
tts-extract-sketches --extractor lineart-anime --max-images 5000
```

Optional Anime2Sketch extraction is supported through a separate local checkout:

```bash
tts-extract-sketches \
  --extractor anime2sketch \
  --anime2sketch-dir dependencies/Anime2Sketch \
  --anime2sketch-python dependencies/Anime2Sketch/.venv/bin/python \
  --anime2sketch-model improved \
  --anime2sketch-gpu-ids "" \
  --max-images 5000
```

Filter noisy sketches:

```bash
tts-filter-sketches --max-points 10000
```

Vectorize, order, time, and export stroke-5 sketches:

```bash
tts-run-pipeline
```

For checkpoint-compatible tok-dict fine-tuning, export the released
Sketchformer dictionary into the native codebook location:

```bash
tts-create-sketch-token-dict \
  --source-token-dict-pkl sketchformer/prep_data/sketch_token/token_dict.pkl \
  --output-dir data/processed/sketch_token
```

For from-scratch native training only, a new anime-specific dictionary can be
built from stroke-5 deltas:

```bash
tts-create-sketch-token-dict \
  --source-dir data/processed/stroke5 \
  --output-dir data/processed/sketch_token \
  --K 1000
```

Convert stroke-5 files into Sketchformer-style tok-dict chunks:

```bash
tts-prepare-sketchformer-tokens \
  --source-dir data/processed/stroke5 \
  --token-dict-dir data/processed/sketch_token \
  --target-dir data/processed/sketchformer-ready-data/tok-dict \
  --n-chunks 10
```

## 4096-token Centerline Workflow

The default centerline workflow preserves grayscale confidence, removes
contour-boundary duplication and synthetic kinematic resampling, uses the
released dictionary with error feedback, and refuses to truncate complete
sketches. Existing extracted binary PNGs cannot recover grayscale confidence,
so start extraction from the raw images.

```bash
tts-extract-sketches \
  --input-dir data/raw/portraits \
  --output-dir data/processed/sketches \
  --extractor lineart-anime \
  --max-images 0

tts-extract-sketches \
  --input-dir data/raw/portraits \
  --output-dir data/processed/sketches \
  --extractor anime2sketch \
  --anime2sketch-dir dependencies/Anime2Sketch \
  --anime2sketch-python dependencies/Anime2Sketch/.venv/bin/python \
  --anime2sketch-model improved \
  --max-images 0

tts-create-sketch-token-dict \
  --source-token-dict-pkl sketchformer/prep_data/sketch_token/token_dict.pkl \
  --output-dir data/processed/sketch_token

tts-evaluate-centerline-preprocessing \
  --extractor-dir lineart-anime=data/processed/sketches/lineart_anime \
  --extractor-dir anime2sketch=data/processed/sketches/anime2sketch \
  --token-dict-dir data/processed/sketch_token \
  --samples 100 \
  --max-token-length 4096 \
  --review-dir data/processed/evaluations/centerline_manual_review \
  --output data/processed/evaluations/centerline_preprocessing.json \
  --enforce
```

Use the report's `selected_extractor` and `selected_profile` for the full run.
The commands below show `lineart-anime` and `hysteresis`; replace those two
values if the benchmark selects another combination. The final manifest must
contain at least 10,000 accepted sketches; 25,000 is the target dataset size.

```bash
tts-filter-sketches \
  --input-dir data/processed/sketches/lineart_anime \
  --output-dir data/processed/sketches-filtered \
  --max-points 20000

tts-run-pipeline \
  --sketches-dir data/processed/sketches-filtered \
  --stroke5-dir data/processed/stroke5-4096 \
  --token-dict-dir data/processed/sketch_token \
  --extractor-name lineart-anime \
  --n-sketches 10000 \
  --vectorizer centerline \
  --threshold-profile hysteresis \
  --ordering continuity \
  --rdp-epsilon 0.5 \
  --max-geometry-error 2.0 \
  --max-token-length 4096 \
  --manifest data/processed/preprocessing_manifest.jsonl \
  --num-workers 8 \
  --fail-on-overlength

tts-prepare-sketchformer-tokens \
  --tokens-dir data/processed/tokens \
  --token-dict-dir data/processed/sketch_token \
  --target-dir data/processed/sketchformer-ready-data/tok-dict-4096 \
  --max-length 4096 \
  --overlength-policy error \
  --n-chunks 10
```

Centerline preprocessing defaults to one worker for compatibility. On a
12-core/24-thread server, start with `--num-workers 8`; each worker loads the
released codebook and builds its quantizer once, while the parent process alone
writes the manifest. Worker outputs use atomic sibling-file replacement, so an
interrupted archive write cannot replace an existing valid `.npz`. Use fresh
stroke, token, and manifest output directories for each run because the pipeline
does not remove artifacts left by an older run.

Prevent OpenCV and numerical libraries from creating nested thread pools before
starting a multi-worker run:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
```

For the 44,000-sketch server run with RDP fixed at exactly 2 pixels, keep
`--rdp-epsilon` and `--max-geometry-error` equal and use a fresh run directory:

```bash
tts-run-pipeline \
  --sketches-dir data/processed/sketches-filtered \
  --stroke5-dir data/processed/rdp2/stroke5 \
  --token-dict-dir data/processed/sketch_token \
  --extractor-name lineart-anime \
  --n-sketches 44000 \
  --vectorizer centerline \
  --threshold-profile hysteresis \
  --ordering continuity \
  --rdp-epsilon 2.0 \
  --max-geometry-error 2.0 \
  --max-token-length 4096 \
  --manifest data/processed/rdp2/preprocessing_manifest.jsonl \
  --num-workers 8 \
  --seed 42
```

This writes token archives under `data/processed/rdp2/tokens`.

Convert a separate long-sequence checkpoint. TensorFlow is needed only for this command:

```bash
tts-convert-sketchformer-checkpoint \
  --experiment anime_tok_dict_long \
  --source weights/pretrained/sketch-transformer-tf2-cvpr_tform_tok_dict/weights/ckpt-12 \
  --output weights/pretrained/sketchformer_tok_dict_4096_init.safetensors
```

Expected conversion output includes `missing_target_keys=0` and
`initialized_long_sequence_keys=2`. The two initialized tensors are the
zero-valued long-position residual; the original 200-position expander remains
checkpoint exact.

Run the server gates and curriculum training:

```bash
tts-train-sketchformer \
  --experiment anime_tok_dict_long \
  --device cuda \
  --precision 16-mixed \
  --dry-run

tts-check-sketchformer-parity \
  --experiment anime_tok_dict_long \
  --checkpoint weights/pretrained/sketchformer_tok_dict_4096_init.safetensors \
  --device cpu

tts-check-sketchformer-memory \
  --experiment anime_tok_dict_long \
  --sequence-length 4096 \
  --batch-size 1 \
  --max-memory-gb 22

tts-train-sketchformer \
  --experiment anime_tok_dict_long \
  --device cuda \
  --precision 16-mixed
```

Evaluate the best checkpoint without teacher forcing:

```bash
tts-evaluate-sketchformer \
  --experiment anime_tok_dict_long \
  --checkpoint weights/finetuned/sketchformer-tok-dict-anime-4096/best.pt \
  --split valid \
  --device cuda \
  --precision 16-mixed \
  --decode-mode free-running \
  --enforce-long-sequence-gates \
  --preprocessing-manifest data/processed/preprocessing_manifest.jsonl \
  --metrics-output data/processed/evaluations/long_sequence_free_running.json \
  --plots-output-dir data/processed/evaluations/long_sequence_free_running_plots
```

Reconstruction plots use the manifest transform to show the original source,
the codebook-decoded target, and the model prediction on the same source
canvas. If source images moved, pass `--source-images-root` with their new
root. Without usable manifest metadata, plotting falls back to a shared
normalized two-panel raster canvas.

The final report must contain
`valid/free_running/geometry_f1_2px_median_length_2049_4096 >= 0.90`.
Complete `manual_review_checklist.json` for the same 100 preprocessing
samples and require at least 95 manual passes.

Continuous stroke3 chunks are still supported for legacy experiments:

```bash
tts-prepare-sketchformer \
  --source-dir data/processed/stroke5 \
  --target-dir data/processed/sketchformer-ready-data/stroke3 \
  --n-chunks 10
```

Expected tok-dict output:

```text
data/processed/sketchformer-ready-data/tok-dict/
├── train_000.npz
├── train_001.npz
├── ...
├── valid.npz
├── test.npz
└── meta.npz
```

## Native PyTorch Training

The native path is the preferred direction for RTX 3090 training. It uses:

- tok-dict variable-length batches with SDPA-compatible masks
- length-bucketed sampling to reduce padding
- gradient checkpointing for long sequences
- CUDA mixed precision with `16-mixed` by default
- TF32 matmul enabled by default on CUDA
- broadcast SDPA padding masks that avoid allocating batch-sized square masks
- cached autoregressive decoding for free-running reconstruction
- token-budget batching and staged 512/1024/2048/4096 fine-tuning
- masked token cross entropy over the sketch token dictionary
- token accuracy and token perplexity validation metrics

CPU dry run:

```bash
tts-train-sketchformer --experiment smoke_test --dry-run
```

RTX 3090 training:

```bash
tts-train-sketchformer \
  --experiment anime_tok_dict_finetune \
  --device cuda \
  --precision 16-mixed
```

Resume a native checkpoint:

```bash
tts-train-sketchformer \
  --experiment anime_tok_dict_finetune \
  --device cuda \
  --resume weights/finetuned/sketchformer-tok-dict-anime/last.pt
```

Evaluate:

```bash
tts-evaluate-sketchformer \
  --experiment anime_tok_dict_finetune \
  --checkpoint weights/finetuned/sketchformer-tok-dict-anime/best.pt \
  --split valid \
  --device cuda \
  --metrics-output data/processed/evaluations/native_valid_metrics.json \
  --plots-output-dir data/processed/evaluations/native_reconstructions
```

Export weights:

```bash
tts-export-sketchformer \
  --experiment anime_tok_dict_finetune \
  --checkpoint weights/finetuned/sketchformer-tok-dict-anime/best.pt \
  --output weights/finetuned/sketchformer-tok-dict-anime/model.safetensors
```

## Original Sketchformer Integration

The legacy path is useful for validating data compatibility against the 2020
codebase and its TensorFlow checkpoints.

Build the bundled CPU image:

```bash
tts-sketchformer-codebase-finetune --sudo build-image
```

Build the RTX 3090-oriented GPU image:

```bash
tts-sketchformer-codebase-finetune --sudo build-gpu-image
```

Prepare legacy-compatible data:

```bash
tts-sketchformer-codebase-finetune --sudo prepare-data \
  --source-dir data/processed/stroke5 \
  --target-dir data/processed/sketchformer-ready-data/stroke3 \
  --n-chunks 10 \
  --n-classes 345
```

Fine-tune with the original checkout:

```bash
tts-sketchformer-codebase-finetune --sudo finetune-continuous \
  --dataset data/processed/sketchformer-ready-data/stroke3 \
  --output-dir weights/finetuned \
  --run-id anime-continuous-finetune \
  --resume weights/pretrained/sketch-transformer-tf2-cvpr_tform_cont/weights/ckpt-12
```

For GPU experiments, provide a custom image compatible with the server CUDA
stack, or use the bundled GPU image, and expose Docker GPUs explicitly:

```bash
tts-sketchformer-codebase-finetune \
  --sudo \
  --image sketchformer-tf2-gpu \
  --gpus all \
  --dry-run \
  finetune-continuous
```

The released continuous TensorFlow checkpoint is sequence-length dependent and
was trained with `max_seq_len=200`. Keep that value when resuming the original
checkpoint. Larger values in the legacy path are for from-scratch compatibility
experiments and will use the old TensorFlow attention implementation, not the
optimized native PyTorch path.

## Configuration

Important configs:

| File | Purpose |
|---|---|
| `configs/train.yaml` | Root composed training config. |
| `configs/model/sketchformer_tok_dict.yaml` | Native tok-dict model architecture and token reconstruction head. |
| `configs/data/anime_tok_dict.yaml` | Tok-dict dataset, token dictionary IDs, sequence length, and batching. |
| `configs/trainer/single_gpu.yaml` | Single-GPU runtime, precision, checkpointing, and logging settings. |
| `configs/experiment/smoke_test.yaml` | Tiny CPU-friendly dry-run/smoke settings. |
| `configs/experiment/anime_tok_dict_finetune.yaml` | RTX 3090-oriented native tok-dict training experiment. |
| `configs/experiment/anime_tok_dict_long.yaml` | 4096-token curriculum with no truncation. |
| `configs/experiment/anime_continuous_finetune.yaml` | Legacy continuous stroke3 experiment. |

The default root config trains the native tok-dict model. To use a custom token
dictionary size, update `data.format.token_dictionary` or provide an experiment
override; config composition copies those IDs into `model.input.token_dictionary`.

## Formats

Stroke-5:

```text
[dx, dy, p1, p2, p3]
```

Tok-dict:

```text
0      = padding token
1..K   = codebook motion tokens
K + 1  = stroke separator token
K + 2  = start-of-sketch token
K + 3  = end-of-sketch token
```

The native tok-dict checkpoint path uses the same token ID layout as the
released TensorFlow Sketchformer dictionary checkpoint. Continuous stroke3
remains available through `anime_stroke3` and `sketchformer_continuous` for
compatibility checks.

Convert the released tok-dict TensorFlow checkpoint after extracting the
archive:

```bash
unzip weights/pretrained/sketch-transformer-tf2-cvpr_tform_tok_dict.zip -d weights/pretrained

tts-convert-sketchformer-checkpoint \
  --experiment anime_tok_dict_finetune \
  --source weights/pretrained/sketch-transformer-tf2-cvpr_tform_tok_dict/weights/ckpt-12 \
  --output weights/pretrained/sketchformer_tok_dict_init.safetensors
```

## Verification

Run the tests with either command:

```bash
python -m unittest discover -s tests -v
pytest -q
```

On a CPU-only development machine, use `--dry-run` and the smoke experiment to
validate config composition without launching full training.

## Known Gaps

- TensorFlow must be installed in the conversion environment to read original
  checkpoint shards. The local PyTorch training environment does not need
  TensorFlow after conversion.
- The native model is tok-dict reconstruction-first; text prompt conditioning is
  not wired into the architecture yet.
- The legacy Docker image is CPU-oriented and intentionally conservative.
