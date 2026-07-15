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
  test_faithful_v2_preprocessing.py
  test_long_sequence_v2.py
  test_anchored_v3_tokenizer.py
  test_anchored_v3_artifacts.py
  test_anchored_v3_configurable_minimum.py
  test_anchored_v3_dataset_loader.py
  test_sketchformer_encoder_memory.py
  test_anchored_v3_objective.py
  test_checkpoint_contract_v3.py
  test_evaluation_contract_v3.py
  test_anchored_v3_overfit_gate.py
  test_anchored_v3_human_review.py
```

They avoid real datasets, Docker, GPU, and large checkpoints. Temporary toy
fixtures are created inside each test.

Run all tests:

```bash
python -B -m unittest discover -s tests
```

Run the deterministic periodic long-sequence eval:

```bash
python -B evals/long_sequence_eval.py
```

Run the serial/parallel preprocessing equivalence eval:

```bash
python -B evals/parallel_preprocessing_eval.py
```

Run the deterministic anchored V3 contract eval:

```bash
python -B evals/anchored_v3_reconstruction_eval.py
```

The `-B` flag avoids writing `__pycache__` files into the repository.

Anchored V3 gate tests cover grammar and round trips, deterministic cleaning,
atomic artifact hashes and split isolation, complete-stroke windows, direct
encoder-memory masking, cached/uncached equality, padding and batch-companion
invariance, geometry-loss gradients, strict checkpoint contracts, collapse
diagnostics, fixed-canvas plots, review/report hash binding, and deterministic
scaling-curve aggregation. Dataset-size tests verify arbitrary positive
cleaned-source minima across preparation, training, evaluation, and overfit.
They use synthetic fixtures and do not replace the
paid GPU overfit, free-running, full-test, or human-review runs.
