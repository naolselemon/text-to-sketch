from __future__ import annotations

import unittest

import torch

from scripts.sketchformer.exposure import (
    AnchoredTokenRanges,
    corrupt_decoder_prefixes,
    scheduled_sampling_probability,
)


class AnchoredV3ExposureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ranges = AnchoredTokenRanges(
            motion_start=1,
            motion_end=5,
            x_start=5,
            x_end=7,
            y_start=7,
            y_end=9,
            mask_token_id=13,
        )

    def test_replacements_stay_in_true_token_category(self) -> None:
        targets = torch.tensor([[11, 10, 5, 7, 1, 12, 9]])
        logits = torch.zeros(1, 6, 14)
        logits[..., 4] = 9.0
        logits[..., 6] = 8.0
        logits[..., 8] = 7.0
        corrupted = corrupt_decoder_prefixes(
            targets,
            logits,
            self.ranges,
            scheduled_probability=1.0,
            mask_probability=0.0,
        )
        self.assertEqual(corrupted.tolist(), [[11, 10, 6, 8, 4, 12, 9]])

    def test_masking_only_changes_content_tokens(self) -> None:
        targets = torch.tensor([[11, 10, 5, 7, 1, 12, 9]])
        logits = torch.zeros(1, 6, 14)
        corrupted = corrupt_decoder_prefixes(
            targets,
            logits,
            self.ranges,
            scheduled_probability=0.0,
            mask_probability=1.0,
        )
        self.assertEqual(corrupted.tolist(), [[11, 10, 13, 13, 13, 12, 9]])

    def test_probability_ramp_is_bounded(self) -> None:
        self.assertEqual(scheduled_sampling_probability(-1.0), 0.0)
        self.assertEqual(scheduled_sampling_probability(0.5), 0.125)
        self.assertEqual(scheduled_sampling_probability(2.0), 0.25)


if __name__ == "__main__":
    unittest.main()
