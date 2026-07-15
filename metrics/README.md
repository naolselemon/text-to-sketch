# Metrics

This folder contains evaluation utilities for two stages of the project.

## Preprocessing Metrics

These modules evaluate the sketch extraction and vectorization pipeline:

- `preprocessing/compare_rdp_epsilon.py` ranks dense sketches and compares RDP
  simplification.
- `preprocessing/evaluate_encoder.py` checks sketch-token encode/decode
  reconstruction error.
- `preprocessing/evaluate_ordering.py` visualizes stroke ordering strategies.
- `preprocessing/evaluate_centerline_preprocessing.py` compares extractor/threshold
  profiles on common source images, reports token-decoded fidelity and drift,
  enforces the centerline gates, and writes manual-review pairs.
- `preprocessing/visualisation.py` contains plotting helpers for
  original-vs-simplified sketches.

## Native Sketchformer Metrics

These modules support the in-repo Sketchformer fine-tuning path:

- `sketchformer/reconstruction.py` converts continuous outputs or codebook-
  decoded tok-dict outputs into stroke3 predictions, collects target/prediction
  examples, and writes JSON metric reports.
- `sketchformer/visualisation.py` saves target-vs-prediction reconstruction
  plots on shared fixed 256×256 axes. Mode-qualified filenames prevent
  teacher-forced and free-running runs from overwriting each other.
- `sketchformer/free_running.py` writes per-sample target/generated lengths,
  EOS position, premature-EOS and max-length flags, stroke and structure-count
  errors, unique-motion ratio, longest repeated-token run, first divergence,
  F1@1px/F1@2px, and symmetric Chamfer distance.

Use the native evaluation script to produce artifacts:

```bash
python scripts/sketchformer/evaluate.py \
  --experiment anime_tok_dict_finetune \
  --checkpoint weights/finetuned/sketchformer-tok-dict-anime/last.pt \
  --allow-legacy-checkpoint \
  --metrics-output weights/finetuned/sketchformer-tok-dict-anime/eval_metrics.json \
  --plots-output-dir weights/finetuned/sketchformer-tok-dict-anime/reconstruction_plots \
  --num-plots 8
```

Final long-sequence evaluation defaults to free-running cached decoding and reports the
geometry F1 separately for 1-512, 513-1024, 1025-2048, and 2049-4096 token
groups, including per-bucket medians. During early fine-tuning, prioritize:

- token loss
- token accuracy
- token perplexity
- codebook-decoded target-vs-prediction plots

Anchored V3 evaluation is strict by default:

```bash
tts-evaluate-sketchformer \
  --experiment anime_anchored_v3_direct \
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

`--enforce-v3-gates` checks median F1@2px ≥ 0.95, 2049–4096-token median
F1@2px ≥ 0.90, p10 F1@2px ≥ 0.85, p95 symmetric Chamfer ≤ 3 pixels, and
premature-EOS/max-length-hit rates ≤ 2%. The report contains aggregate metrics
and all per-sample records. Fill the generated template and run
`tts-check-sketchformer-v3-human-review`; the validator ties all 100 reviews to
the exact report and plot SHA-256 values and requires exactly 100 reviews with
at least 95 all-criterion passes.

Use `tts-build-sketchformer-v3-scaling-curve` after the 1.4k, 5k, 10k, and full
test evaluations. It recomputes the curve from per-sample records and rejects
artifact hashes, subset limits, or ordered test identities that differ.

Checkpoint contract validation precedes tensor loading. A V3 evaluation aborts
if the composed compatibility projection, token layout, manifest hash, or
codebook hash differs. Release and human-review reports additionally prove the
canonical token IDs and direct encoder-memory decoder. The configured
cleaned-source minimum is also part of strict compatibility. The complete
config remains embedded for provenance;
only operational output/resume/initialization/report paths are excluded from
comparison. `--allow-legacy-checkpoint` exists only for a trusted V2
pre-contract checkpoint and still performs strict tensor-key loading.
