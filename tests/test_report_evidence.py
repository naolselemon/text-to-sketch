import unittest
from evals.report_evidence import classify_report, training_origin_from_config, wrapper_run_name


class EvidenceTest(unittest.TestCase):
    def test_strict_loading_does_not_establish_origin(self):
        self.assertEqual(classify_report({})['not_quality_reasons'], ['unknown_training_origin'])
        self.assertEqual(classify_report({}, training_origin_override='real_training')['report_kind'], 'quality')

    def test_wrapper_name_prefix(self):
        self.assertEqual(wrapper_run_name('fixture', classify_report({})), 'not-quality-fixture')
        self.assertEqual(wrapper_run_name('trained', classify_report({'training_origin': 'real_training'})), 'trained')

    def test_all_reasons_have_fixed_order(self):
        result = classify_report({'fixture': True, 'config': {'experiment': {'name': 'memorize10_long'}}},
            random_init=True, smoke_test=True, partial_load=True)
        self.assertEqual(result['not_quality_reasons'],
            ['random_init', 'smoke_test', 'sanity_preset', 'partial_load', 'unknown_training_origin'])

    def test_embedded_metadata_and_fixture_cannot_be_overridden(self):
        for origin, reason in [('fixture', 'smoke_test'), ('smoke_test', 'smoke_test'),
                               ('random_init', 'random_init'), ('sanity_preset', 'sanity_preset')]:
            result = classify_report({'training_origin': origin}, training_origin_override='real_training')
            self.assertEqual(result['report_kind'], 'not_quality')
            self.assertIn(reason, result['not_quality_reasons'])
            self.assertTrue(result['origin_override_ignored'])
        result = classify_report({'training_origin': 'real_training'}, partial_load=True)
        self.assertEqual(result['not_quality_reasons'], ['partial_load'])

    def test_config_training_origin(self):
        self.assertEqual(training_origin_from_config({'experiment': {'name': 'smoke_test'}}), 'smoke_test')
        self.assertEqual(training_origin_from_config({'experiment': {'name': 'memorize10_long'}}), 'sanity_preset')
        result = classify_report({'training_origin': 'real_training', 'config': {'experiment': {'name': 'smoke_test'}}})
        self.assertEqual(result['not_quality_reasons'], ['smoke_test'])
