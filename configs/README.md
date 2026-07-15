# Training Configs

This folder contains the configuration layer for in-repo Sketchformer
fine-tuning.

The configs are intentionally split by responsibility:

- `data/` controls dataset paths, sequence length, batching, and dataloader
  behavior.
- `model/` controls the native PyTorch Sketchformer-style architecture.
- `optimizer/` controls optimizer and learning-rate scheduler settings.
- `trainer/` controls device, precision, logging, checkpointing, and runtime
  behavior.
- `experiment/` selects concrete training runs such as smoke tests and anime
  fine-tuning.

The default entrypoint config is `train.yaml`.

## Research Objective

The in-repo fine-tuning path targets more sophisticated sketches than the
short, simple QuickDraw-style drawings used by the original Sketchformer setup.
For that reason, the base data/model configs are long-sequence capable:

- the default model consumes tok-dict token sequences built from the sketch
  token dictionary using the released TensorFlow token ID layout;
- the model supports sequences up to `2048` tok-dict tokens;
- the anime tok-dict data config trains at `2048` by default for server-side
  runs;
- the smoke-test experiment overrides sequence length down to `256`;
- attention is configured for PyTorch scaled dot-product attention with Flash
  Attention preferred when the hardware supports it;
- SDPA padding masks use broadcast key masks instead of batch-sized square
  tensors, allowing long runs to remain on memory-efficient attention paths;
- sparse attention is represented in config but disabled until the dense
  Flash/SDPA baseline is correct.

For checkpoint-compatible runs, `data/processed/sketch_token/codebook.npy`
must be exported from `sketchformer/prep_data/sketch_token/token_dict.pkl`.
The token IDs are `PAD=0`, motion tokens `1..1000`, `SEP=1001`,
`SOS=1002`, and `EOS=1003`.

`configs/train.yaml` defaults to `anime_tok_dict` plus
`sketchformer_tok_dict`. Continuous stroke3 configs remain in this folder for
legacy compatibility, but they are not the native fine-tuning objective.

The default trainer config uses CUDA `16-mixed` precision and TF32-friendly
runtime settings for an RTX 3090-class server. CPU development should use the
`smoke_test` experiment or explicit CLI overrides.

## 4096-token Long-Sequence Workflow

`experiment/anime_tok_dict_long.yaml` keeps the existing 2048-token run
unchanged and defines the full long-sequence contract: separately converted weights,
complete sequences with truncation disabled, token-budget batches, and the
512/1024/2048/4096 length curriculum. Its dataset and checkpoint paths are
separate from legacy artifacts.

The long-sequence fields are:

- `data.batching.max_tokens_per_batch: 4096`
- `trainer.training.target_tokens_per_step: 32768`
- `model.architecture.latent_expander_base_length: 200`
- `data.sequence.truncate_long_sequences: false`

Curriculum loaders filter by complete sequence length. They never shorten an
example to make it enter an earlier stage.

## Anchored V3

`data/anime_anchored_v3.yaml`, `model/sketchformer_anchored_v3.yaml`, and
`experiment/anime_anchored_v3_direct.yaml` define the target-faithful path.
The stable contract is:

- format `anchored_v3`, version `3`, fixed 256×256 canvas;
- vocabulary size `2566`, with motion `1..2048`, X `2049..2304`, Y
  `2305..2560`, `STROKE_START=2561`, `STROKE_END=2562`, `SOS=2563`,
  `EOS=2564`, and `MASK=2565`;
- direct, mask-aware encoder memory instead of the single-vector latent
  expander;
- complete-stroke windows for early curriculum stages, never token truncation;
- free-running macro median F1@2px as the best-checkpoint metric.

`experiment/anime_anchored_v3_overfit.yaml` disables augmentation, exposure
corruption, and the full-run gate for the mandatory 32-sample FP32 check. The
full experiment refuses to start until its overfit report passes all three
thresholds.

The configured `data.dataset.root`, `manifest.jsonl`, and `codebook.npy` are a
single immutable unit. Checkpoints store their hashes plus the full composed
config, a strict compatibility projection, and token-layout version. The
runtime resolves the `current` symlink once to its content-hashed directory.
`data.dataset.minimum_source_sketches` is a positive configurable cleaned-source
floor, defaults to `1`, and can be overridden consistently with
`--minimum-source-sketches`. It remains in checkpoint compatibility so resume
and evaluation cannot silently change the selected requirement. Explicit
manifest paths are rejected unless they resolve to that same pinned artifact.
The
projection excludes only operational output/resume/initialization/report paths.
Resume and evaluation therefore require the same model, data, training, and
pinned artifact settings. Do not retarget the `current` dataset symlink during
a run. V2 checkpoints remain under V2 configs and cannot resume V3.

For a scaling study, keep one immutable root and choose any increasing positive
`--train-source-limit` values that do not exceed the number of accepted original
training sketches. The manifest reader uses one seeded ordering, so the subsets
are nested while validation/test membership and the codebook stay fixed. Omit
the option for the full training split and pass the same limit to strict
evaluation. Use repeatable `tts-build-sketchformer-v3-scaling-curve --report
LIMIT PATH` arguments, with one final `--report full PATH`, to verify identical
held-out sample order and aggregate the reports.

If both full-scale train and validation median F1@2px remain more than 0.05
below the 0.95 target after the narrow model passes the overfit gate, use
`anime_anchored_v3_wide`. It changes only V3 capacity to `d_model=256`, six
encoder/decoder layers, and a 1024-wide feed-forward block. Rerun only the
largest limited point and full stages; do not widen V2.
