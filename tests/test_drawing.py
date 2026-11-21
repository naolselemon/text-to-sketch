import unittest
import numpy as np
from metrics.sketchformer.drawing import (
    validate_drawing, score_drawing, out_of_canvas_fraction, segment_inside_fraction,
)
from utils.tokenizer import decode_tokens


class DrawingTest(unittest.TestCase):
    def setUp(self):
        self.book = np.array([[0, 0], [1, 0], [0, 1], [100, 100]], dtype=np.float32)
        self.target = [6, 1, 2, 5, 7]

    def validate(self, tokens, limit=8):
        return validate_drawing(tokens, self.book, generation_limit=limit)

    def test_eos_and_padding(self):
        for row, limit, ended in [(self.target + [0, 0], 8, False), (self.target, 5, True)]:
            result = self.validate(row, limit)
            self.assertTrue(result['complete'])
            self.assertEqual(result['ended_at_limit'], ended)
            self.assertEqual(result['invalid_step_count'], 0)

    def test_bad_tokens_distinct_positions(self):
        cases = [([6, 1, 2], 'missing_eos', 0),
                 ([6, 1, 2, 5], 'generation_limit_hit', 0),
                 (self.target + [7], 'tokens_after_eos', 1),
                 (self.target + [2], 'tokens_after_eos', 1),
                 ([6, 1, 0, 6, 2, 7], 'invalid_token', 2),
                 ([6, -1, 99, 7], 'invalid_token', 2)]
        for row, reason, count in cases:
            result = self.validate(row, 4 if reason == 'generation_limit_hit' else 8)
            self.assertFalse(result['complete'])
            self.assertIn(reason, result['failure_reasons'])
            self.assertEqual(result['invalid_step_count'], count)
        self.assertIn('generation_limit_hit', self.validate(self.target, 4)['failure_reasons'])

    def test_empty_and_nonfinite(self):
        result = self.validate([6, 7])
        self.assertIn('empty_output', result['failure_reasons'])
        self.assertEqual(result['invalid_step_count'], 0)
        self.book[1, 0] = np.nan
        result = self.validate(self.target)
        self.assertIn('nonfinite_geometry', result['failure_reasons'])
        self.assertEqual(result['invalid_step_count'], 1)

    def test_incomplete_high_f1_zero_score(self):
        result = score_drawing(self.target + [2], self.target, self.book, generation_limit=8)
        self.assertEqual(result['geometry_f1_2px'], 1)
        self.assertEqual(result['drawing_score'], 0)

    def test_extra_strokes_quality_and_penalty(self):
        inside = score_drawing([6, 1, 2, 3, 5, 7], self.target, self.book, generation_limit=8)
        self.assertTrue(inside['complete'])
        self.assertLess(inside['drawing_score'], 1)
        self.assertEqual(inside['out_of_canvas_fraction'], 0)
        outside = score_drawing([6, 1, 2, 5, 4, 2, 5, 7], self.target, self.book, generation_limit=8)
        self.assertTrue(outside['complete'])
        self.assertEqual(outside['geometry_f1_2px'], 1)
        self.assertLess(outside['drawing_score'], 1)
        self.assertAlmostEqual(outside['out_of_canvas_fraction'], .5)
        jump = decode_tokens([6, 1, 2, 5, 4, 5, 7], self.book)
        self.assertEqual(out_of_canvas_fraction(decode_tokens(self.target, self.book), jump), 0)

    def test_crossing_continuous_length(self):
        self.assertAlmostEqual(segment_inside_fraction(np.array([-10., 10]), np.array([10., 10])), .5)
        self.assertEqual(segment_inside_fraction(np.array([0., 0]), np.array([255., 255.])), 1)
        self.assertEqual(segment_inside_fraction(np.array([300., 0]), np.array([300., 200.])), 0)
