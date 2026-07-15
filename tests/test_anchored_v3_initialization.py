from __future__ import annotations

import unittest

import torch
from torch import nn

from scripts.sketchformer.initialize_v3 import (
    initialize_transformer_blocks,
    load_source_state,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Linear(2, 2)])
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([nn.Linear(2, 2)])
        self.input_embedding = nn.Embedding(4, 2)


class AnchoredV3InitializationTest(unittest.TestCase):
    def test_loads_pytorch_checkpoint_payload(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            torch.save({"model": {"encoder.layers.0.value": torch.ones(2)}}, path)

            state = load_source_state(path)

            torch.testing.assert_close(
                state["encoder.layers.0.value"],
                torch.ones(2),
            )

    def test_loads_only_exact_shape_transformer_blocks(self) -> None:
        model = _TinyModel()
        source = {
            "encoder.layers.0.weight": torch.full((2, 2), 3.0),
            "encoder.layers.0.bias": torch.zeros(3),
            "decoder.layers.0.weight": torch.full((2, 2), 4.0),
            "input_embedding.weight": torch.full((4, 2), 9.0),
        }
        original_embedding = model.input_embedding.weight.detach().clone()
        report = initialize_transformer_blocks(model, source)
        self.assertTrue(torch.equal(model.encoder.layers[0].weight, source["encoder.layers.0.weight"]))
        self.assertTrue(torch.equal(model.decoder.layers[0].weight, source["decoder.layers.0.weight"]))
        self.assertTrue(torch.equal(model.input_embedding.weight, original_embedding))
        self.assertIn("encoder.layers.0.bias", report.skipped_shape)
        self.assertIn("input_embedding.weight", report.ignored_source)


if __name__ == "__main__":
    unittest.main()
