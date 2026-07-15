# In-Repo Sketchformer Fine-Tuning Experiments

This folder records the experiment plan for the native PyTorch Sketchformer
fine-tuning path.

The executable configuration remains under:

```text
configs/experiment/
```

This folder is the experiment runbook layer. It explains what each experiment
is for, which command launches it, what outputs should be produced, and what
criteria must pass before moving to the next stage.

## Experiments

| Experiment | Purpose | Canonical Config |
|---|---|---|
| `smoke_test` | Verify the full in-repo training path on a tiny run. | `configs/experiment/smoke_test.yaml` |
| `anime_tok_dict_finetune` | Fine-tune the long-sequence tok-dict model on anime token data. | `configs/experiment/anime_tok_dict_finetune.yaml` |
| `anime_continuous_finetune` | Legacy continuous stroke3 compatibility experiment. | `configs/experiment/anime_continuous_finetune.yaml` |

## Stage Order

### 1. Smoke Test

Run this first after changes to dataloaders, model, losses, builders, core, or
scripts:

```bash
python scripts/sketchformer/train.py --experiment smoke_test
```

The smoke test is intentionally short. It confirms the pipeline can load data,
run the model forward, compute loss, backpropagate, validate, and save a
checkpoint.

### 2. Anime Tok-Dict Fine-Tuning

Run this after the smoke test passes:

```bash
python scripts/sketchformer/train.py --experiment anime_tok_dict_finetune
```

This experiment targets longer, more detailed anime sketches as tok-dict token
sequences. The config runs at 2048 tokens by default. Real training should run
on a CUDA GPU machine; CPU is only practical for smoke tests.

## Common Commands

Dry-run the training config:

```bash
python scripts/sketchformer/train.py --experiment smoke_test --dry-run
```

Evaluate a checkpoint:

```bash
python scripts/sketchformer/evaluate.py \
  --experiment anime_tok_dict_finetune \
  --checkpoint weights/finetuned/sketchformer-tok-dict-anime/last.pt \
  --allow-legacy-checkpoint
```

Export a checkpoint:

```bash
python scripts/sketchformer/export.py \
  --experiment anime_tok_dict_finetune \
  --checkpoint weights/finetuned/sketchformer-tok-dict-anime/last.pt \
  --output weights/finetuned/sketchformer-tok-dict-anime/export.pt
```

Export the released tok-dict dictionary centers for checkpoint-compatible data:

```bash
python -m prep_data.sketch_token.create_token_dict \
  --source-token-dict-pkl sketchformer/prep_data/sketch_token/token_dict.pkl \
  --output-dir data/processed/sketch_token
```

Convert the released tok-dict TensorFlow checkpoint:

```bash
unzip weights/pretrained/sketch-transformer-tf2-cvpr_tform_tok_dict.zip -d weights/pretrained

python scripts/sketchformer/convert_checkpoint.py \
  --experiment anime_tok_dict_finetune \
  --source weights/pretrained/sketch-transformer-tf2-cvpr_tform_tok_dict/weights/ckpt-12 \
  --output weights/pretrained/sketchformer_tok_dict_init.safetensors
```

## Acceptance Criteria

Before trusting a fine-tuning run:

- The dataloader returns non-empty train and validation batches.
- The model forward pass produces token reconstruction logits with expected shape.
- The loss is finite.
- One optimizer step completes.
- Validation metrics are logged.
- `last.pt` checkpoint is saved.
- Evaluation can load the checkpoint.
- Export can write a standalone PyTorch or safetensors file.
