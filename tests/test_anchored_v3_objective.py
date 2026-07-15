from __future__ import annotations

import unittest

import torch

from core.anchored_v3_objective import AnchoredV3Objective, AnchoredV3ObjectiveConfig


class AnchoredV3ObjectiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AnchoredV3ObjectiveConfig(
            motion_start=1,
            motion_end=3,
            x_start=3,
            x_end=7,
            y_start=7,
            y_end=11,
            stroke_start_id=11,
            stroke_end_id=12,
            sos_id=13,
            eos_id=14,
            canvas_scale=3.0,
        )
        self.objective = AnchoredV3Objective(
            self.config,
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )

    def test_perfect_sequence_has_small_differentiable_geometry_loss(self) -> None:
        targets = torch.tensor([[11, 4, 9, 1, 2, 12, 14]])
        logits = torch.full((1, len(targets[0]), 15), -8.0, requires_grad=True)
        with torch.no_grad():
            for position, token in enumerate(targets[0].tolist()):
                logits[0, position, token] = 8.0
        mask = torch.ones_like(targets, dtype=torch.bool)
        token, point, endpoint = self.objective(logits, targets, mask)
        total = token + 2.0 * point + endpoint
        total.backward()
        self.assertLess(point.item(), 1e-4)
        self.assertLess(endpoint.item(), 1e-4)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_structural_mistake_is_weighted(self) -> None:
        targets = torch.tensor([[11, 4]])
        logits = torch.zeros(1, 2, 15)
        mask = torch.ones_like(targets, dtype=torch.bool)
        token, _, _ = self.objective(logits, targets, mask)
        self.assertGreater(token.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
