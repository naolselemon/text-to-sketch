# Tests

This folder contains standard-library `unittest` tests for the native
Sketchformer fine-tuning path.

The tests follow the original flat test layout:

```text
tests/
  test_config_composition.py
  test_stroke_sequence_dataset.py
  test_collate_masks.py
  test_sketchformer_forward.py
  test_losses.py
  test_prepare_sketchformer_tokens.py
  test_checkpoint_mapping.py
  test_train_smoke.py
  test_centerline_preprocessing.py
  test_long_sequence.py
```

They avoid real datasets, Docker, GPU, and large checkpoints. Temporary toy
fixtures are created inside each test.

Run all tests:

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

Run the deterministic periodic long-sequence eval:

```bash
.venv/bin/python -B evals/long_sequence_eval.py
```

Run the serial/parallel preprocessing equivalence eval:

```bash
python -B evals/parallel_preprocessing_eval.py
```

The `-B` flag avoids writing `__pycache__` files into the repository.

Free-running checkpoint selection tests cover drawing completeness and clipping,
frozen evaluation populations, report evidence, deterministic ranking, candidate
retention, sequential evaluation, materialization and held-out isolation. The
suite includes the existing benchmark and failure-analysis tests.

The CLI hand-off regressions exercise actual tiny CPU fixture generation through
`evaluate` and prove those non-quality reports cannot select or materialize a
model. Temporary synthetic report fixtures exercise successful selection and
materialization with known rankings. The held-out CLI entry point uses synthetic
candidate reports, then verifies the materialized winner, complete disjoint test
population, selection hash and rejection of test-based
reselection. There is no fixture bypass in the production CLI. All of these
results establish mechanics, never model quality.

See the [maintainer hand-off](../evals/free_running_handoff.md) for real-checkpoint
commands and the remaining generation, image-review and acceptance work.
