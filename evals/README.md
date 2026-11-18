# Benchmark evaluation and failure analysis

These tools prepare the Phase 1 study. They do not convert TensorFlow checkpoints,
train a model, or supply real baseline measurements. Run the study on the GPU
server with the actual development dataset, codebook, and compatible weights.
The held-out test split is evaluated only when explicitly selected.

## Before running on the server

Install the repository dependencies and editable package in your server's Python
3.10/3.11 environment. Install the PyTorch CUDA wheels appropriate for its NVIDIA
driver. Check `torch.cuda.is_available()` before requesting `--device cuda`.
Run the CPU regression suite first:

```bash
python -B -m unittest discover -s tests -v
```

Use a checkpoint converted for the selected architecture/sequence-length experiment.
For example, `anime_tok_dict_long` needs the corresponding 4096-token initialization
or fine-tuned checkpoint. A 2048-token checkpoint is not interchangeable with it.
Weight loading is strict by default. `--allow-partial-checkpoint` is an explicit
experimental override; missing/unexpected keys are recorded in the report. A file
with no matching weights is rejected. `--allow-random-init` is for synthetic smoke
tests, not the pretrained baseline.

Benchmarking disables truncation regardless of the training configuration and
checks the entire requested split before inference. An overlong sample causes an
error identifying the sample; choose a compatible longer-sequence experiment and
checkpoint. Targets are never shortened to fit the default 2048-token experiment.
`--max-generation-length` limits generation only; missing output tokens still count
as errors against the complete target. Leave it unset for the full baseline.

## 1. Annotate the development split

```bash
python evals/benchmark_manifest.py \
  --dataset-root data/processed/sketchformer-ready-data/tok-dict-4096 \
  --splits valid \
  --output evals/benchmark_manifest.jsonl
```

Sample IDs match the prepared `sample_ids` array, or fall back to `shard_stem:index`,
as in the model dataloader. IDs must be unique. `sequence_length` and bucket assignment
count non-padding model input tokens **including SOS and EOS**. The legacy
`token_length` field describes content tokens excluding SOS, EOS and padding;
`motion_token_count` additionally excludes separators. Numeric legacy bucket names
(e.g. `1_512`) are normalized to `short`, `medium`, `long`, and `extended` by diagnostics.

## 2. Freeze a real baseline

Replace the artifact locations with those on your server. `--codebook-path` changes
the file location; it does not change the checkpoint's token layout or vocabulary.

```bash
python evals/benchmark_eval.py \
  --experiment anime_tok_dict_long \
  --checkpoint weights/pretrained/sketchformer_tok_dict_4096_init.safetensors \
  --data-root data/processed/sketchformer-ready-data/tok-dict-4096 \
  --codebook-path data/processed/sketch_token/codebook.npy \
  --split valid --device cuda --precision 16-mixed \
  --seed 42 --num-plots 50 --save-predictions --save-latents \
  --output-dir evals/results --run-name baseline_valid
```

The default `--modes teacher-forced free-running` evaluates both modes. Either mode
can be selected independently. Teacher-forced evaluation scores the model's shifted
loss targets and mask. Comparable TF/FR per-sample accuracy excludes the supplied
SOS and padding, includes EOS and separators, and counts missing generated tokens
as incorrect. Global TF loss/accuracy are token-weighted; the per-sample accuracy
means used by gap analysis give every sketch equal weight. Perplexity is the
exponential of pooled token loss, not a mean of batch perplexities.

Reports include all TF loss/accuracy/perplexity and FR accuracy/exact-match/EOS/F1
metrics globally and per populated bucket. Empty buckets are absent, not silently
reported as zero quality. Wall-clock timings include loading/metric computation and
are **end-to-end evaluation times**, not isolated GPU kernel latency.

The report records the checkpoint/codebook hashes, split-file hashes and dataset
snapshot hash, resolved config hash, effective seed/precision, device, code revision,
and peak CUDA memory when available. Only the requested split is loaded; no train
split is needed to benchmark validation data. Seed zero is supported.

Run artifacts are never overwritten: use a new `--run-name` for each experiment.
An interrupted run may leave partial artifacts; preserve/inspect them and retry
with a new run name. Archive the baseline artifacts and the exact code revision.

### Frozen qualitative comparisons

The comparison set is selected deterministically across populated length buckets,
with spare capacity redistributed when a bucket has few samples. When fewer than
50 samples exist, all available samples are selected. For subsequent models reuse
exactly the saved IDs:

```bash
python evals/benchmark_eval.py \
  --experiment anime_tok_dict_long \
  --checkpoint weights/finetuned/sketchformer-tok-dict-anime-4096/best.pt \
  --data-root data/processed/sketchformer-ready-data/tok-dict-4096 \
  --codebook-path data/processed/sketch_token/codebook.npy \
  --split valid --device cuda --precision 16-mixed --seed 42 \
  --save-predictions --save-latents --num-plots 50 \
  --comparison-manifest evals/results/baseline_valid_comparison_manifest.json \
  --output-dir evals/results --run-name candidate_valid
```

The dataset snapshot must match; missing comparison IDs cause an error. Loader
ordering and length bucketing cannot change the selected IDs or plot order.

## 3. Diagnose the baseline

```bash
python evals/failure_analysis.py all \
  --dataset-root data/processed/sketchformer-ready-data/tok-dict-4096 \
  --codebook-path data/processed/sketch_token/codebook.npy --split valid \
  --tf-report evals/results/baseline_valid_report.json \
  --fr-per-sample evals/results/baseline_valid_per_sample.jsonl \
  --predictions evals/results/baseline_valid_per_sample.jsonl \
  --latent-vectors evals/results/baseline_valid_latents.npz \
  --plots --output-dir evals/results/baseline_analysis
```

Individual commands are `gap`, `position-curves`, `separator-confusion`,
`separators`, `codebook`, `quantization`, `latent-capacity`, `conversion`, and `plot`.
Use `<command> --help` for required inputs. Position/cascade analysis needs the
prediction arrays exported by `--save-predictions`; absent arrays cause an error.

Gap analysis joins TF and FR by sample ID and verifies the target lengths/hashes
and scoring convention. It reports per-sample gaps and ranks the worst 20 by
TF-minus-FR accuracy, not geometry F1. A global TF score is never substituted for a
missing bucket. Legacy reports with explicit matching bucket summaries can produce
bucket-only analysis, clearly marked as lacking per-sample gaps.

Missing continuation tokens count as errors in separator cascade windows. Counts
are reported for both comparison groups. If either group is absent, the penalty
and effect flag are `null`, since no comparison is supported. Position curves
report the number of active target samples at each position.

Latent export contains `embeddings`, `sample_ids`, `lengths`, `stroke_counts` and
`tf_token_accuracy`, all in the same row order. Effective dimensionality is the
covariance participation ratio: for singular values `s`, `(sum(s**2))**2 / sum(s**4)`.

## 4. Verify conversion fidelity

The comparison tool consumes outputs already exported from the original TensorFlow
model and the converted PyTorch model. Generate both with identical inputs, sample
order, token IDs, masks, decoding convention, and inference mode. The actual
checkpoint converter is `scripts/sketchformer/convert_checkpoint.py`.

For padded sequence logits, export each `.npz` with `logits` shaped `(B,T,V)` and a
boolean `valid_mask` shaped `(B,T)`, or integer `lengths` shaped `(B,)`. Include only
positions that both implementations are meant to predict; exclude padding and any
supplied prefix. Both files' valid positions must agree if both supply metadata.
Sequence exports without validity metadata are rejected. Genuinely unpadded
position arrays may be exported as `(N,V)`.

```bash
python evals/conversion_fidelity.py \
  --candidate-logits evals/results/pytorch_logits.npz \
  --reference-logits evals/results/tf_reference_logits.npz \
  --output-dir evals/results/conversion_fidelity
```

Alternatively pass `--valid-mask mask.npy` or `--lengths lengths.npy` explicitly.
A comparison with zero valid positions, inconsistent masks, or nonfinite valid
logits fails. The report records input file and mask hashes and evaluated positions.
Thresholds default to cosine >= 0.999 and argmax agreement >= 0.99. A failed check
returns a nonzero exit status, including through `failure_analysis.py conversion`
and `failure_analysis.py all`.

## Outputs and follow-up study work

A complete `baseline_valid` run writes:

- `baseline_valid_report.json`: provenance and global/per-bucket metrics.
- `baseline_valid_teacher_forced_per_sample.jsonl`: TF scores and target identity.
- `baseline_valid_per_sample.jsonl`: FR scores and optional target/predicted arrays.
- `baseline_valid_comparison_manifest.json`: frozen stratified comparison IDs.
- `baseline_valid_plots/`: numbered plots in the manifest's selection order.
- `baseline_valid_latents.npz`: optional embeddings with aligned sample metadata.

`failure_analysis.py all --plots` produces diagnostic reports and available figures.
Its optional analyses run only when their inputs are supplied; it does not generate
TensorFlow reference outputs or missing latent/prediction files.

The GPU study still needs actual baseline measurements, verified TensorFlow exports,
per-sample quantization/reconstruction correlations, complexity-colored embedding
visualizations, and a consolidated interpretation with Week 3 recommendations.
The exported IDs/embeddings support that work. None of these empirical findings are
claimed by CPU regression tests, and they must not be inferred from synthetic runs.

# Frozen evaluation identities and evidence labels

Benchmark reports retain `schema_version: 2` and their existing output names.
`--evaluation-manifest PATH` creates or reuses a frozen evaluation population;
`--eval-samples N` chooses a deterministic subset across populated length buckets
and restricts inference to those IDs. Without a subset, the entire requested
split is evaluated. The evaluation manifest is separate from the smaller
comparison-image manifest. Stored targets are validated before inference, with
truncation disabled and internal padding or missing SOS/EOS rejected.

The manifest contains ordered IDs, target lengths and token hashes, split-file
hashes, dictionary hash and token layout, seed, code revision, and versioned
completeness/scoring rules. Canonical inference settings are stored alongside
their fingerprint so mismatches identify the differing fields. Configuration
hash, device and batch size remain provenance. Standalone benchmarking resolves
an omitted generation limit to model capacity; direct `run_free_running` callers
retain their existing batch-derived default.

Use `--save-predictions` to retain `target_tokens`, `predicted_tokens`, and full
`validation_tokens`, including initial SOS and unexpected post-EOS output.
Candidate identity, checkpoint hash and global step accompany this evidence.
`evals.selection_common.validate_report` checks population coverage, target and
dictionary identity, inference settings, actual checkpoint hashes, recomputed
drawing scores, aggregates, bucket totals and expected image files. A partial
`--limit-batches` diagnostic cannot validate against a complete manifest.

Reports are `quality` or `not_quality`. Applicable non-quality reasons have the
fixed order `random_init`, `smoke_test`, `sanity_preset`, `partial_load`,
`unknown_training_origin`. `--smoke-test` explicitly labels a diagnostic run;
fixture metadata, the `smoke_test` preset and `memorize10_long` are also recognized.
Strict loading alone does not establish training origin. For older checkpoints,
`--training-origin real_training` fills missing origin only. Embedded metadata
wins conflicts, recorded through `origin_override_ignored`; known fixture,
smoke, sanity, random-init and partial-load markers remain non-quality.

`--image-captions` adds checkpoint, report kind/reasons, seed and generation
settings to the existing sample caption. Numbered image names and comparison
order are preserved. Each run's plot directory contains `images_index.json`,
linked from the report, with image paths, sample IDs, checkpoint hash and settings.
The wrapper naming helper prefixes non-quality runs with `not-quality-`;
standalone benchmarking continues honoring explicit `--run-name` values.

CPU fixture reports test reproducibility, validation and labeling. Their scores
are not model-quality evidence. Real-checkpoint generation, image inspection and
acceptance remain the maintainer's hand-off.

# Free-running checkpoint selection

Root `best.pt` is a stage-dependent validation-loss diagnostic: its tracker resets
at curriculum stage boundaries. `selected/model.*` is the checkpoint selected by
complete free-running drawings and their quality. Next-token loss remains a
training diagnostic. Selection alone does not establish satisfactory model quality.

The versioned rule `free-running-selection-v1` ranks eligible candidates by:

1. Higher mean `drawing_score` across the entire frozen evaluation population.
2. Higher completion rate.
3. Higher minimum bucket-mean `drawing_score` across populated length buckets.
4. Earlier recorded global step.
5. Lexicographic candidate ID.

Empty buckets are excluded from the minimum. Each complete drawing receives
`geometry_f1_2px × (1 − out_of_canvas_fraction)`; incomplete drawings receive zero.
The versioned clipping penalty measures continuous drawn-segment length outside
the target-normalized 256×256 canvas before rasterization, excluding pen-up jumps.
Incorrect extra strokes affect quality without automatically making a drawing
syntactically incomplete.

Float criteria use a finite, nonnegative `--tolerance` (default `1e-6`). At each
criterion, sort descending and form a group containing the best remaining value
and all values within tolerance of that anchor. Refine each group using the next
criterion before continuing to the next anchor. Global step and candidate ID use
exact comparisons. For example, scores `0.90`, `0.84`, `0.78` with tolerance `0.10`
group the first two; the third remains separate even though it is close to `0.84`.
This avoids non-transitive pairwise approximate comparisons and produces the same
full ranking and tie-resolution trace regardless of candidate input order.

Selection checks each report against the supplied frozen `valid` manifest, matching
dictionary, inference settings, completeness/scoring versions, target identities,
saved token evidence, aggregates, image artifacts and actual checkpoint hashes.
Candidates require IDs and recorded nonnegative integer global steps. Non-quality
and test reports are always ineligible, as are partial evaluation reports. Duplicate
candidate IDs invalidate every conflicting report. Rejections are recorded; when no
candidate is eligible, selection writes a report with no winner and exits nonzero
without falling back to loss.

Once candidate benchmarks and a shared validation manifest have been produced:

```bash
.venv/bin/python -B evals/select_checkpoint.py select \
  --evaluation-manifest runs/free-running-selection/validation/evaluation_manifest.json \
  --reports-dir runs/free-running-selection/validation \
  --codebook-path data/processed/sketch_token/codebook.npy \
  --output runs/free-running-selection/selection.json \
  --materialize
```

`selection.json` records the rule, tolerance, frozen identities, full ranking,
tie-resolution trace, rejected reports, artifact hashes and winner. `--materialize`
atomically publishes a verified byte-for-byte copy as `selected/model.pt` or
`selected/model.safetensors`, preserving its actual checkpoint format. Existing
selection outputs or materialization directories are refused.

The CPU selection tests use temporary synthetic report fixtures with known rankings.
These scores exercise selection mechanics and are never model-quality evidence.
Checkpoints marked as fixtures remain non-quality and cannot pass production
selection; the CLI has no fixture bypass. The maintainer performs real-checkpoint
generation, image review and model acceptance. If the rule changes after review,
version it and rerun selection consistently across candidates.

# Retained candidates and sequential validation

`trainer.checkpointing.retain_candidates` defaults to `false` and is enabled by
`anime_tok_dict_long`. Every validation boundary retains an immutable model-only
checkpoint at `candidates/<stage>/epoch-XXXX-step-XXXXXX.pt`, with configuration,
training origin, epoch, global step and stage maximum length. Optimizer and scheduler
states are excluded. After successful atomic saving, `candidates/manifest.jsonl`
registers its identity, path, stage, epoch, global step, stage maximum length, SHA256
and validation loss. Collisions are refused; candidates are never overwritten or
deleted. Resume reports unregistered checkpoints and temporary files left by an
interrupted save. `last.pt`, stage-local loss-monitored `best.pt`, and final-stage
early stopping keep their existing behavior.

Evaluate retained checkpoints using one shared complete validation population:

```bash
.venv/bin/python -B evals/select_checkpoint.py evaluate \
  --candidate-manifest weights/finetuned/sketchformer-tok-dict-anime-4096/candidates/manifest.jsonl \
  --experiment anime_tok_dict_long \
  --data-root data/processed/sketchformer-ready-data/tok-dict-4096 \
  --codebook-path data/processed/sketch_token/codebook.npy \
  --device cuda --precision 16-mixed \
  --max-generation-length 4096 --seed 42 --num-plots 50 \
  --output-dir runs/free-running-selection/validation
```

For older checkpoints, replace `--candidate-manifest` with
`--checkpoints <a.pt> <b.pt> --training-origin real_training`. These candidate
sources are mutually exclusive. IDs come from checkpoint metadata or the filename
and hash. Global steps must come from checkpoint metadata or a candidate-manifest
entry; timestamps and default zero are never used. Relative paths in a supplied
manifest resolve against its directory. Supplied hashes, IDs and steps must agree
with embedded metadata. An origin override only fills missing origin: known
fixture, smoke, sanity, random-init and partial-load reasons remain non-quality.

Before inference, the wrapper freezes `evaluation_manifest.json`, the smaller
`comparison_manifest.json`, `inference_config.json`, and verified `candidates.json`.
By default the evaluation population is the entire `valid` split. `--eval-samples N`
freezes and evaluates a deterministic subset using seed 42 unless overridden.
Every stage uses the same inference architecture and generation limit, with strict
loading, greedy free-running decoding, and saved target, predicted and validation
tokens. Candidate evaluation is sequential.

Each candidate has a directory containing reserved attempt identities, process
logs, reports, numbered comparison images, and an `images_index.json` linked from
its report. Non-quality attempt names start with `not-quality-`. Complete artifacts
are skipped only after verifying their checkpoint hash, frozen settings, population,
token evidence, aggregates, labels, comparison order and images. Interrupted or
invalid attempts remain in place and retries use a new attempt name. Changed
settings or candidate identities require a fresh output directory. Each invocation
writes `evaluation_attempt_XXXX.json` with complete, skipped and failed outcomes;
any requested candidate failure produces a nonzero exit after the others finish.

CPU orchestration tests use tiny checkpoints with fixture metadata. Those reports
remain non-quality and their scores are not model-quality evidence. Real generation,
candidate image inspection, ranking review and model acceptance belong to the
maintainer. Generated weights, data and reports stay out of commits.
