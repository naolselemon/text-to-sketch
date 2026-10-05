"""Deterministic readiness eval for anchored-V3 reconstruction contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root")


_add_project_to_path()

from builders import build_model
from dataloaders.masks import build_sequence_masks
from metrics.sketchformer.free_running import free_running_reconstruction_records
from services.anchored_sketch_data.contract import CODEBOOK_SIZE, TOKEN_LAYOUT
from services.anchored_sketch_data.grammar import validate_tokens
from services.anchored_sketch_data.preprocessing import geometry_f1
from services.anchored_sketch_data.tokenizer import AnchoredTokenizer
from scripts.sketchformer.config import compose_training_config


def _fixture_codebook() -> np.ndarray:
    codebook = np.column_stack(
        (
            np.arange(CODEBOOK_SIZE, dtype=np.float32) + 1000.0,
            np.arange(CODEBOOK_SIZE, dtype=np.float32) + 2000.0,
        )
    )
    codebook[:4] = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    return codebook


def main() -> int:
    config = compose_training_config(
        "configs/train.yaml",
        experiment="anime_anchored_v3_direct",
    )
    token_config = config["data"]["format"]["token_dictionary"]
    expected = {
        "pad_token_id": 0,
        "motion_token_offset": 1,
        "codebook_size": 2048,
        "x_token_offset": 2049,
        "y_token_offset": 2305,
        "stroke_start_token_id": 2561,
        "stroke_end_token_id": 2562,
        "sos_token_id": 2563,
        "eos_token_id": 2564,
        "mask_token_id": 2565,
        "vocab_size": 2566,
    }
    mismatches = {
        name: (token_config.get(name), value)
        for name, value in expected.items()
        if token_config.get(name) != value
    }
    if mismatches or token_config.get("sep_token_id") is not None:
        raise AssertionError(f"anchored V3 token layout mismatch: {mismatches}")

    strokes = [
        [(8.0, 8.0), (9.0, 8.0), (10.0, 8.0)],
        [(200.0, 200.0), (200.0, 201.0), (199.0, 201.0)],
    ]
    tokenizer = AnchoredTokenizer(_fixture_codebook())
    tokens = tokenizer.encode(strokes)
    summary = validate_tokens(tokens)
    decoded = tokenizer.decode(tokens)
    roundtrip_f1 = geometry_f1(strokes, decoded)
    if roundtrip_f1 != 1.0:
        raise AssertionError(f"anchored round-trip F1 is {roundtrip_f1}, expected 1.0")
    if decoded[1][0] != strokes[1][0]:
        raise AssertionError("second stroke did not reset to its absolute anchor")

    model = build_model(config["model"]).eval()
    shared_weight = model.input_embedding.token_embedding.weight
    head = model.reconstruction_head
    assert head is not None
    with torch.no_grad():
        shared_weight.zero_()
        head.projection.bias.zero_()
        head.projection.bias[TOKEN_LAYOUT.stroke_start_token_id] = 100.0
        head.projection.bias[TOKEN_LAYOUT.eos_token_id] = 110.0
        head.projection.bias[TOKEN_LAYOUT.x_token_start] = 90.0
        head.projection.bias[TOKEN_LAYOUT.y_token_start] = 90.0
        head.projection.bias[TOKEN_LAYOUT.motion_token_start] = 80.0
        head.projection.bias[TOKEN_LAYOUT.stroke_end_token_id] = 90.0

    source = torch.as_tensor(tokens, dtype=torch.long).unsqueeze(0)
    masks = build_sequence_masks([len(tokens)], max_length=len(tokens))
    batch = {"tokens": source, "targets": source.clone(), **masks}
    with torch.inference_mode():
        cached = model.generate(batch, use_cache=True)
        uncached = model.generate(batch, use_cache=False)
    if not torch.equal(cached.tokens, uncached.tokens):
        raise AssertionError("cached and uncached anchored decoding diverged")
    generated = cached.tokens[0, : int(cached.lengths[0])].cpu().tolist()
    if generated != [TOKEN_LAYOUT.sos_token_id, TOKEN_LAYOUT.eos_token_id]:
        raise AssertionError(f"premature EOS was not observable: {generated}")
    records = free_running_reconstruction_records(
        cached.tokens,
        cached.lengths,
        {"targets": source, "lengths": torch.tensor([len(tokens)])},
        tokenizer.codebook,
        eos_token_id=TOKEN_LAYOUT.eos_token_id,
        token_layout={
            **token_config,
            "type": "anchored_v3",
            "version": 3,
        },
    )
    if records[0]["premature_eos"] != 1.0:
        raise AssertionError("immediate EOS was not reported as a failure")

    report = {
        "eval": "anchored_v3_reconstruction",
        "status": "pass",
        "token_length": int(len(tokens)),
        "stroke_count": summary.stroke_count,
        "roundtrip_f1_2px": roundtrip_f1,
        "cached_uncached_exact_match": True,
        "premature_eos_observable": True,
        "decoder_memory_source": config["model"]["decoder"]["memory_source"],
        "latent_expander_present": model.latent_expander is not None,
    }
    if report["decoder_memory_source"] != "encoder" or report["latent_expander_present"]:
        raise AssertionError("V3 reconstruction did not bypass the latent expander")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
