# Sketch Generation from Text
This repository prepares anime portrait sketches as vector stroke sequences and
experiments with scaling the 2020 Sketchformer idea to more complex anime-style
line art.

The project currently has three parallel tracks:

1. **Original Sketchformer integration**: prepare anime stroke data and launch a
   local checkout of the original TensorFlow Sketchformer codebase for
   checkpoint-compatible experiments.
2. **Native PyTorch rebuild**: a clean in-repo Sketchformer-style model,
   dataloader, losses, checkpointing, and training loop designed for modern
   CUDA training on a single RTX 3090-class GPU.
3. **Anchored V3 reconstruction**: a fixed-canvas token grammar, direct
   encoder-memory decoder, geometry-aware objective, strict artifact contracts,
   and free-running model selection for target-faithful anime reconstruction.

The local development machine can be CPU-only. Full native training is expected
to run on the GPU server.

## Current State

| Area | Status |
|---|---|
| Anime image preprocessing | Implemented. Downloads portraits, extracts line art, filters sketches, vectorizes centerlines, orders strokes, and writes stroke-5 files. |
| Sketchformer-ready data | Implemented. Builds a sketch token dictionary and converts stroke-5 sketches into chunked tok-dict `.npz` files with train/valid/test splits. Continuous stroke3 prep remains available for legacy experiments. |
| Original Sketchformer handoff | Implemented as a Docker command launcher. The bundled image is CPU-oriented; GPU use requires a custom compatible image. |
| Native PyTorch Sketchformer | Implemented for tok-dict reconstruction with SDPA attention, gradient checkpointing, length-bucketed loading, masked token cross entropy, token accuracy/perplexity metrics, codebook-decoded plots, evaluation, and export. |
| Native fine-tuning from converted TF weights | Implemented for the released tok-dict checkpoint, including the 200-position checkpoint base and zero-initialized 4096-position V2 residual. |
| Target-faithful anchored V3 | Implemented. Builds content-addressed datasets, anchors every stroke on a 256-pixel canvas, decodes from encoder memory, enforces an overfit gate, and selects checkpoints by free-running geometry. |
| Text prompt conditioning | Not implemented in the current codebase. The present focus is anime sketch sequence modeling. |

## Pipeline Overview

```text
Danbooru2019 portraits
  -> anime line-art sketches
  -> filtered sketch images
  -> topology-preserving centerlines
  -> ordered drawing paths
  -> stroke-5 arrays
  -> legacy V2 tok-dict chunks OR anchored V3 artifacts
  -> original Sketchformer, native V2, or target-faithful V3 training
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
├── services/anchored_sketch_data/ Anchored V3 grammar, cleaning, split, shards
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
  --allow-legacy-checkpoint \
  --enforce-v2-gates \
  --preprocessing-manifest data/processed/preprocessing_manifest.jsonl \
  --metrics-output data/processed/evaluations/v2_free_running.json \
  --plots-output-dir data/processed/evaluations/v2_free_running_plots
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

## Target-Faithful Anchored V3 Workflow

V3 is a new data and checkpoint contract. It does not reuse V2 token shards or
resume V2 training checkpoints. Each stroke begins with absolute X/Y tokens on
a fixed 256×256 canvas and then uses within-stroke motion tokens, so an error in
one stroke cannot move every later stroke.

### 1. Prepare and pin the dataset

Choose a positive cleaned-source minimum for the run. There is no hardcoded
dataset-size floor; `1` accepts any nonempty validated artifact, while a larger
value makes preparation fail if cleaning, deduplication, or encoding leaves too
few originals. For a 7,942-image collection, choose a value no larger than the
number expected to survive those steps. Preparation performs the
deterministic cleaning, deduplication, grouped 80/10/10 split, train-only
augmentation, RDP gate sweep, 2,048-center training-only codebook fit, token
encoding, and atomic artifact publication. The RDP sweep measures the simplified
vector directly against the source-image centerline, not against its own
unsimplified vector.

```bash
tts-prepare-sketchformer-v3 build \
  --source-dir data/processed/sketches-filtered-v3 \
  --output-root data/processed/sketchformer-ready-data/anchored-v3 \
  --seed 42 \
  --calibration-size 256 \
  --augmentation-copies 1 \
  --minimum-accepted-source-sketches 1 \
  --shard-size 1024 \
  --workers 0
```

`--workers 0` automatically uses up to eight CPUs available to the process for
the independent image vectorization and cleaning loop. This conservative cap
avoids memory and scheduler thrashing on large servers. Set an explicit value
such as `--workers 12` after measuring the host if more parallelism helps. Each
worker disables OpenCV's nested thread pool, so worker processes do not multiply
into hundreds of competing threads. The command reports per-stage elapsed time,
image rate, accepted/rejected counts, epsilon calibration metrics, and tqdm
progress bars.
Use `--no-progress` only for quiet automation. This preprocessing stage is CPU
bound; the RTX 3090 is used later during model training, not during this build.

The command publishes an immutable directory named
`anchored_v3-<content-hash>` and atomically updates the relative `current`
symlink. Validate the printed immutable path before training.

```bash
tts-prepare-sketchformer-v3 validate \
  data/processed/sketchformer-ready-data/anchored-v3/anchored_v3-0123456789abcdef
```

Training and evaluation resolve `current` once to the immutable content-hashed
directory. Retargeting the symlink cannot switch shards during a running job.
A V3 checkpoint records and verifies the resolved config plus manifest and
codebook SHA-256 digests. Training, overfit checking, and evaluation accept
`--minimum-source-sketches N`; use the same positive value for each command in
a run. The selected value is part of strict checkpoint compatibility, so resume
or evaluation with a different value is rejected.

### 2. Initialize compatible transformer blocks

Convert the released TensorFlow tok-dict checkpoint, then initialize only
exact-shape encoder and decoder transformer blocks. V3 embeddings, output bias,
and all grammar-specific parameters remain newly initialized.

```bash
tts-convert-sketchformer-checkpoint \
  --experiment anime_tok_dict_long_v2 \
  --source weights/pretrained/sketch-transformer-tf2-cvpr_tform_tok_dict/weights/ckpt-12 \
  --output weights/pretrained/sketchformer_tok_dict_4096_init.safetensors

tts-initialize-sketchformer-v3 \
  --experiment anime_anchored_v3_direct \
  --source weights/pretrained/sketchformer_tok_dict_4096_init.safetensors \
  --output weights/pretrained/sketchformer_anchored_v3_transformer_init.pt \
  --report data/processed/evaluations/anchored_v3_initialization.json
```

Inspect the initialization report before training. It lists every reused,
shape-skipped, ignored, and reinitialized tensor.

### 3. Pass the mandatory 32-sketch overfit gate

The overfit experiment selects a deterministic, length-stratified 32-source
subset from the same pinned V3 training manifest and excludes its precomputed
augmented copies.
Training and validation both run on those 32 examples in FP32 with exposure
corruption disabled.

```bash
tts-train-sketchformer \
  --experiment anime_anchored_v3_overfit \
  --minimum-source-sketches 1 \
  --pretrained weights/pretrained/sketchformer_anchored_v3_transformer_init.pt \
  --device cuda \
  --precision 32-true

tts-check-sketchformer-v3-overfit \
  --experiment anime_anchored_v3_overfit \
  --minimum-source-sketches 1 \
  --checkpoint weights/finetuned/sketchformer-anchored-v3-overfit/best.pt \
  --device cuda \
  --expected-samples 32 \
  --output data/processed/evaluations/anchored_v3_overfit_gate.json
```

The checker exits nonzero unless teacher-forced token accuracy is at least
99.5%, median free-running F1@2px is at least 0.99, and cached and uncached
generation produce identical tokens.

### 4. Train the gated curriculum

The direct-memory experiment runs the 512/1024/2048/4096 curriculum, validates
free-running output every epoch on one fixed 256-sketch, four-bucket subset,
and selects `best.pt` using its four-bucket macro median F1@2px. Full-validation
metrics use a separate `val_full/` namespace at stage boundaries and on every
final-stage epoch, so final early stopping never compares different validation
populations.

```bash
tts-train-sketchformer \
  --experiment anime_anchored_v3_direct \
  --minimum-source-sketches 1 \
  --pretrained weights/pretrained/sketchformer_anchored_v3_transformer_init.pt \
  --overfit-gate-report data/processed/evaluations/anchored_v3_overfit_gate.json \
  --device cuda \
  --precision bf16-mixed
```

Resume only against the same immutable config, manifest, codebook, and token
layout. `last.pt` also carries the final-stage full-validation best score and
non-improving count, so the five-validation early-stop history survives a
restart:

```bash
tts-train-sketchformer \
  --experiment anime_anchored_v3_direct \
  --minimum-source-sketches 1 \
  --resume weights/finetuned/sketchformer-anchored-v3-direct/last.pt \
  --overfit-gate-report data/processed/evaluations/anchored_v3_overfit_gate.json \
  --device cuda \
  --precision bf16-mixed
```

### 5. Evaluate the held-out test split

```bash
tts-evaluate-sketchformer \
  --experiment anime_anchored_v3_direct \
  --minimum-source-sketches 1 \
  --checkpoint weights/finetuned/sketchformer-anchored-v3-direct/best.pt \
  --split test \
  --device cuda \
  --precision bf16-mixed \
  --decode-mode free-running \
  --enforce-v3-gates \
  --num-plots 100 \
  --human-review-template data/processed/evaluations/anchored_v3_human_review.json \
  --metrics-output data/processed/evaluations/anchored_v3_test.json \
  --plots-output-dir data/processed/evaluations/anchored_v3_test_plots
```

The automated gate requires median F1@2px ≥ 0.95, long-sequence median
F1@2px ≥ 0.90, p10 F1@2px ≥ 0.85, p95 symmetric Chamfer ≤ 3 pixels, and both
premature-EOS and maximum-length-hit rates ≤ 2%. Fill every boolean in the
generated review template while inspecting the fixed plots, then enforce the
95/100 face-shape, eyes, hair, and major-accessories gate:

```bash
tts-check-sketchformer-v3-human-review \
  --evaluation-report data/processed/evaluations/anchored_v3_test.json \
  --reviews data/processed/evaluations/anchored_v3_human_review.json \
  --output data/processed/evaluations/anchored_v3_human_review_result.json
```

The review template records the SHA-256 of the evaluation report and every
plot. The validator rejects changed plots, a different checkpoint report,
partial/non-test evaluation, a legacy checkpoint, or adjustable sample/pass
thresholds.

### 6. Run a fixed-membership scaling study

Choose any increasing positive source limits that fit the accepted training
split, then add the full split as the final point. Use the same pinned dataset
root for every run. `--train-source-limit` chooses source IDs by a stable seeded
hash, so the chosen limited sets are nested without changing validation/test
membership, the codebook, or manifest. For example, a dataset with 7,942 total
sources can use 1,400, 5,000, and full when at least 5,000 originals remain in
its cleaned training split.

```bash
tts-train-sketchformer --experiment anime_anchored_v3_direct \
  --minimum-source-sketches 1 \
  --train-source-limit 1400 \
  --output-dir weights/finetuned/sketchformer-anchored-v3-scale-1400 \
  --overfit-gate-report data/processed/evaluations/anchored_v3_overfit_gate.json \
  --device cuda --precision bf16-mixed

tts-train-sketchformer --experiment anime_anchored_v3_direct \
  --minimum-source-sketches 1 \
  --train-source-limit 5000 \
  --output-dir weights/finetuned/sketchformer-anchored-v3-scale-5000 \
  --overfit-gate-report data/processed/evaluations/anchored_v3_overfit_gate.json \
  --device cuda --precision bf16-mixed

tts-train-sketchformer --experiment anime_anchored_v3_direct \
  --minimum-source-sketches 1 \
  --output-dir weights/finetuned/sketchformer-anchored-v3-scale-full \
  --overfit-gate-report data/processed/evaluations/anchored_v3_overfit_gate.json \
  --device cuda --precision bf16-mixed
```

Evaluate each limited checkpoint with the matching `--train-source-limit`.
Output and resume paths are provenance only and do not affect compatibility.
Example:

```bash
tts-evaluate-sketchformer --experiment anime_anchored_v3_direct \
  --minimum-source-sketches 1 \
  --train-source-limit 1400 \
  --checkpoint weights/finetuned/sketchformer-anchored-v3-scale-1400/best.pt \
  --split test --device cuda --precision bf16-mixed \
  --decode-mode free-running \
  --metrics-output data/processed/evaluations/anchored_v3_scale_1400_test.json
```

Repeat the evaluation for every chosen limited point. Omit the limit for the
full run. Change the checkpoint and report name together, then validate and
aggregate the same ordered test records. Pass each numeric limit with its
matching evaluation path and include exactly one `full` report:

```bash
tts-build-sketchformer-v3-scaling-curve \
  --report 1400 data/processed/evaluations/anchored_v3_scale_1400_test.json \
  --report 5000 data/processed/evaluations/anchored_v3_scale_5000_test.json \
  --report full data/processed/evaluations/anchored_v3_scale_full_test.json \
  --output data/processed/evaluations/anchored_v3_scaling_curve.json
```

The aggregator recomputes metrics from every per-sketch record and rejects a
different manifest, codebook, test order, subset limit, partial evaluation, or
edited aggregate. It requires at least one limited point plus the full point.
The legacy `--report-1400`, `--report-5000`, `--report-10000`, and
`--report-full` interface remains available when all four are supplied.

If the narrow model passes the 32-sketch gate but both full-scale training and
validation median F1@2px are below `0.90`, rerun the largest limited point and
the full study with `--experiment anime_anchored_v3_wide`. That conditional
config changes only V3 capacity to 256 dimensions, six encoder/decoder layers,
and a 1024-unit feed-forward block. Measure training F1 with the same evaluator
using `--split train`; keep the checkpoint's matching `--train-source-limit`
value.
Create its complete initialization artifact first:

```bash
tts-initialize-sketchformer-v3 \
  --experiment anime_anchored_v3_wide \
  --source weights/pretrained/sketchformer_tok_dict_4096_init.safetensors \
  --output weights/pretrained/sketchformer_anchored_v3_wide_transformer_init.pt \
  --report data/processed/evaluations/anchored_v3_wide_initialization.json
```

### Artifact and checkpoint compatibility

- V2 datasets and checkpoints remain readable only through their legacy
  configs. They are never converted, overwritten, or resumed as V3.
- V3 checkpoints embed schema version, composed config, token-layout version,
  manifest hash, codebook hash, git commit, and monitored free-running metrics.
- Strict comparison uses the embedded compatibility projection. It excludes
  only output, resume, pretrained-source, and gate-report paths; model, data,
  optimizer, scheduler, precision, curriculum, and subset settings must match.
- V3 resume and evaluation use strict tensor loading and abort on a contract
  mismatch. Partial transfer is allowed only through
  `tts-initialize-sketchformer-v3`, which writes a complete tensor report.
- Release evaluation and human review reject reports unless both the runtime
  and embedded checkpoint use the exact public V3 token layout, tied token
  weights, grammar-constrained generation, and direct encoder memory.
- `--allow-legacy-checkpoint` is an evaluation-only escape hatch for trusted
  pre-contract V2 checkpoints. It never permits missing or unexpected tensor
  keys.

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
  --allow-legacy-checkpoint \
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
| `configs/experiment/anime_tok_dict_long_v2.yaml` | Faithful 4096-token curriculum with no truncation. |
| `configs/data/anime_anchored_v3.yaml` | Fixed-canvas V3 artifact paths and 2,566-token grammar. |
| `configs/model/sketchformer_anchored_v3.yaml` | Direct encoder-memory model with tied token weights. |
| `configs/experiment/anime_anchored_v3_overfit.yaml` | Mandatory 32-sketch FP32 overfit experiment. |
| `configs/experiment/anime_anchored_v3_direct.yaml` | Gated four-stage target-faithful V3 curriculum. |
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

Anchored V3:

```text
0           = padding
1..2048     = within-stroke motion tokens
2049..2304  = absolute X coordinates 0..255
2305..2560  = absolute Y coordinates 0..255
2561        = stroke start
2562        = stroke end
2563        = sketch start
2564        = sketch end
2565        = masked decoder input

SOS (STROKE_START X Y MOTION+ STROKE_END)+ EOS
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
- The four-size V3 scaling study uses one pinned artifact plus deterministic
  nested source limits. GPU training remains an explicit operator-run workflow.
- The legacy Docker image is CPU-oriented and intentionally conservative.
