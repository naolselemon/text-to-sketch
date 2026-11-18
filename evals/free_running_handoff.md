# Issue #9: maintainer hand-off

Complete free-running drawings and their quality determine the selected
checkpoint. Next-token loss remains a training diagnostic. Root `best.pt` tracks
stage-local validation loss, and the first validation of a new curriculum stage
can replace it. `selected/model.*` records the free-running selection decision;
that decision alone does not establish satisfactory model quality.

The implementation includes no real-checkpoint quality results. The maintainer
performs real generation, reviews images and decides whether to accept the model.
CPU fixtures establish workflow mechanics only.

## CPU verification

From the repository root, use the project environment:

```bash
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B evals/long_sequence_eval.py
```

Milestone 6 CPU verification passed: the full discovery suite, both new hand-off
regressions in a separate targeted run, and `long_sequence_eval.py`. No real
checkpoint generation or model-quality assessment was performed.

The regression suite covers complete drawing validation and the continuous
clipping penalty, frozen evaluation identities and saved token evidence, evidence
classification, deterministic ranking, immutable candidate retention, sequential
evaluation, materialization and held-out isolation. Existing benchmark and
failure-analysis regressions remain part of this suite. The long-sequence check
uses synthetic preprocessing, quantization, attention-mask and parity inputs.

Two hand-off regressions connect these checks:

- `test_fixture_evaluation_cannot_select_or_materialize_a_model` runs the
  `evaluate` CLI with actual tiny CPU checkpoints marked as fixtures. Their
  complete reports remain `not_quality`, including when an origin override is
  attempted; the production `select --materialize` CLI rejects them and creates
  no selected model.
- `test_synthetic_cli_handoff_ranking_materialization_and_test_isolation` uses
  temporary handwritten token/report fixtures with known rankings. An incomplete
  sequence with perfect diagnostic F1 receives drawing score zero. The `select`
  CLI ranks the complete sequences, resolves an exact score tie by global step
  and materializes the expected checkpoint. The `test` CLI entry point uses
  synthetic candidate reports; production artifact verification, manifest
  freezing and report validation run normally. It evaluates only the
  materialized winner, preserves the selection bytes and rejects held-out reports
  through the production selection CLI.

Those synthetic successful selections are tests of mechanics, never trained-model
evidence. Tiny-model generation stays non-quality. There is no fixture bypass in
the production CLI, and all test data, checkpoints, reports and images live in
temporary directories.

## 1. Evaluate real retained candidates on validation drawings

`anime_tok_dict_long` enables `trainer.checkpointing.retain_candidates`; the
default for other configurations is `false`. At each validation end, retention
writes an immutable model-only checkpoint under `candidates/<stage>/` and then
registers its hash, identity, stage, epoch, global step, stage maximum length and
validation loss in `candidates/manifest.jsonl`. Configuration and training origin
are embedded; optimizer and scheduler state are excluded. Existing `last.pt`,
loss-monitored `best.pt` and final-stage early stopping remain operational.
Candidates are never overwritten or deleted. On resume, inspect any reported
unregistered files left by an interrupted save.

With the real dataset, matching codebook, compatible weights and CUDA environment
available, run:

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

For older checkpoints, replace the candidate-manifest argument with:

```bash
--checkpoints <a.pt> <b.pt> --training-origin real_training
```

Missing global steps still require candidate-manifest entries. The origin override
fills missing origin only; it cannot remove embedded fixture, smoke, sanity,
random-initialization or partial-load reasons. Strict loading alone does not prove
real training origin.

The wrapper freezes the entire `valid` population before inference, plus one
inference specification and a smaller comparison-image selection. It evaluates
every candidate sequentially with strict loading, greedy free-running decoding,
saved target/predicted/validation tokens, generation limit 4096 including SOS/EOS,
seed 42 and up to 50 comparison images. All stages use the same specification.
`--eval-samples N` is an optional deterministic subset; it changes the frozen
population and requires a fresh output directory.

Inspect `evaluation_attempt_XXXX.json` for failed candidates and their process
logs. Any requested failure makes evaluation exit nonzero after the remaining
candidates finish. Retries preserve interrupted attempts and use fresh names.
Complete matching reports are skipped only after validation. Changed settings
require a fresh output directory. Generated data, weights, reports and images
stay out of commits; `runs/` is ignored.

## 2. Select and materialize

```bash
.venv/bin/python -B evals/select_checkpoint.py select \
  --evaluation-manifest runs/free-running-selection/validation/evaluation_manifest.json \
  --reports-dir runs/free-running-selection/validation \
  --codebook-path data/processed/sketch_token/codebook.npy \
  --output runs/free-running-selection/selection.json \
  --materialize
```

Eligibility requires a validated quality report on the frozen `valid` population,
a matching checkpoint hash and dictionary, a candidate ID, a nonnegative recorded
global step, and matching inference/completeness/scoring versions. Partial reports,
non-quality reports and test reports are rejected. There is no loss fallback.

The rule ranks higher mean drawing score, higher completion rate, higher minimum
populated-bucket mean drawing score, earlier global step, then lexicographic
candidate ID. Complete drawings receive
`geometry_f1_2px × (1 − out_of_canvas_fraction)`; incomplete drawings receive zero.
The penalty measures continuous drawn-segment length outside the target-normalized
canvas before clipping, excluding pen-up jumps. Empty buckets are excluded.

Float ties use tolerance `1.0e-6`, with groups anchored to the best remaining value
at each criterion. Global step and ID comparisons are exact. The rule and full
tie-resolution trace are recorded in `selection.json`. The winner is atomically
copied in its actual format to `selected/model.pt` or `selected/model.safetensors`
and its hash is verified. Existing outputs are refused.

## 3. Review the candidate images and decision

Inspect every candidate's numbered comparison images and `images_index.json`,
linked from its report. Use the common sample IDs and comparison order to compare
the same drawings. Captions/indexes record checkpoint hashes, report labels, seed
and inference settings. Inspect aggregate and bucket scores, completion failures,
the full ranking, rejections and tie-resolution trace. The comparison manifest is
only the smaller image population; ranking uses all frozen evaluation samples.

Image inspection and model acceptance belong to the maintainer. If review leads to
a changed rule, version it and rerun selection consistently across candidates.
Do not use test results to choose a different candidate or tune this rule.

## 4. Evaluate only the winner on held-out test drawings

```bash
.venv/bin/python -B evals/select_checkpoint.py test \
  --selection runs/free-running-selection/selection.json \
  --data-root data/processed/sketchformer-ready-data/tok-dict-4096 \
  --codebook-path data/processed/sketch_token/codebook.npy \
  --device cuda \
  --output-dir runs/free-running-selection/test
```

Before generation, test evaluation verifies the selection versions, all ranked
artifact hashes and eligibility, the dictionary, inference settings and winner
checkpoint hash. An optional checkpoint override must have that same hash.
Test uses the frozen validation architecture, precision, generation limit and seed,
and freezes a separate complete `test` manifest. Split-file hashes may differ;
test sample IDs must be disjoint from the validation IDs used for selection.

Only the selected winner is evaluated. The report records the selection hash and
is labeled `held_out`, `selection_eligible: false`, `selection_score: null`.
Incomplete test reports fail validation. Test reports never enter selection or
modify `selection.json`. The maintainer reviews the held-out evidence and decides
whether the model is satisfactory.

| Output | Location |
|---|---|
| Validation reports and images | `runs/free-running-selection/validation` |
| Selection | `runs/free-running-selection/selection.json` |
| Selected model | `runs/free-running-selection/selected` |
| Test report | `runs/free-running-selection/test` |

Separator-structure validation, candidate filtering and retention thinning remain
follow-ups. Keep the validation decision and its referenced artifacts available
for held-out verification and reproducibility.
