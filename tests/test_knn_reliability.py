import math
import unittest

import torch

from knn_reliability import KNNReliabilityBank, combine_rescue_masks


def points(angles):
    return torch.tensor([[math.cos(a), math.sin(a)] for a in angles], dtype=torch.float32)


def ready_bank(k=2, **kwargs):
    bank = KNNReliabilityBank(2, 2, k=k, score_threshold=0.6, **kwargs)
    bank.begin_source_refresh()
    # Independent diagonal Gaussians and local bandwidth for geometry tests.
    bank.class_means.copy_(torch.tensor([[1., 0.], [-1., 0.]]))
    bank.class_variances.fill_(0.02)
    bank.distribution_initialized.fill_(True)
    bank.global_var.fill_(0.04)
    bank.sigma_initialized.fill_(True)
    return bank


def fill(bank, angles, labels=None, conf=None, ids=None):
    n = len(angles)
    bank.set_target_memory(
        points(angles), torch.tensor(labels if labels is not None else [0] * n),
        torch.tensor(conf if conf is not None else [0.99] * n),
        torch.tensor(ids if ids is not None else list(range(n))),
    )


def query(bank, angle=0., label=0, sample_id=999, candidate=True):
    return bank.gate(points([angle]), torch.tensor([label]), torch.tensor([sample_id]),
                     torch.tensor([candidate]), torch.tensor([0.8, 0.8]))


class ReliabilityTests(unittest.TestCase):
    def test_dense_vs_sparse_with_equal_class_support(self):
        dense, sparse = ready_bank(), ready_bank()
        fill(dense, [0.02, -0.02])
        fill(sparse, [0.25, -0.25])
        a, b = query(dense), query(sparse)
        self.assertEqual(a['support_fraction'].item(), 1.)
        self.assertEqual(b['support_fraction'].item(), 1.)
        self.assertGreater(a['score'].item(), b['score'].item())
        self.assertTrue(a['pass_mask'].item())
        self.assertFalse(b['pass_mask'].item())
        self.assertTrue(a['dense_mask'].item())
        self.assertFalse(b['dense_mask'].item())

    def test_close_other_class_cannot_support_and_search_is_not_class_filtered(self):
        bank = ready_bank()
        fill(bank, [0.01, -0.01, 0.1, -0.1], labels=[1, 1, 0, 0])
        result = query(bank)
        self.assertGreater(result['density'].item(), 0.9)
        self.assertEqual(result['score'].item(), 0.)

    def test_low_confidence_neighbors_remain_in_search_but_cannot_support(self):
        bank = ready_bank()
        fill(bank, [0.01, -0.01, 0.1, -0.1], conf=[0.2, 0.2, 0.99, 0.99])
        result = query(bank)
        self.assertGreater(result['density'].item(), 0.9)
        self.assertEqual(result['score'].item(), 0.)

    def test_self_excluded_by_id_not_feature_equality(self):
        bank = ready_bank(k=1)
        fill(bank, [0., 0.02], labels=[0, 1], ids=[42, 43])
        self.assertEqual(query(bank, sample_id=42)['score'].item(), 0.)
        self.assertGreater(query(bank, sample_id=999)['score'].item(), 0.99)
        # Another sample may legitimately have the same feature as the query.
        fill(bank, [0., 0.], ids=[42, 43])
        self.assertGreater(query(bank, sample_id=42)['score'].item(), 0.99)

    def test_query_and_neighbors_must_be_in_assigned_class_distribution(self):
        bank = ready_bank()
        fill(bank, [0.7, 0.71])
        result = query(bank, angle=0.7)
        self.assertGreater(result['density'].item(), 0.9)
        self.assertEqual(result['score'].item(), 0.)
        result = query(bank)
        self.assertEqual(result['support_fraction'].item(), 0.)

    def test_distribution_mass_changes_region_membership(self):
        tight = ready_bank(distribution_mass=0.5)
        wide = ready_bank(distribution_mass=0.99)
        fill(tight, [0.25, -0.25])
        fill(wide, [0.25, -0.25])
        self.assertEqual(query(tight)['score'].item(), 0.)
        self.assertGreater(query(wide)['score'].item(), 0.)

    def test_missing_statistics_and_insufficient_neighbors_fail_closed(self):
        bank = ready_bank()
        self.assertFalse(query(bank)['checked_mask'].item())
        fill(bank, [0., 0.01], ids=[10, 11])
        self.assertFalse(query(bank, sample_id=10)['pass_mask'].item())
        self.assertFalse(query(bank, sample_id=10)['checked_mask'].item())
        bank.sigma_initialized.fill_(False)
        self.assertFalse(query(bank)['pass_mask'].item())
        bank.sigma_initialized.fill_(True)
        bank.distribution_initialized[0] = False
        self.assertFalse(query(bank)['pass_mask'].item())

    def test_nonfinite_memory_and_queries_do_not_produce_nan_scores(self):
        bank = ready_bank(k=1)
        bank.set_target_memory(torch.tensor([[float('nan'), 0.], [1., 0.]]),
                               torch.tensor([0, 0]), torch.tensor([0.99, 0.99]),
                               torch.tensor([0, 1]))
        self.assertEqual(bank.memory_ids.tolist(), [1])
        result = bank.gate(torch.tensor([[float('nan'), 0.], [0., 0.]]),
                           torch.tensor([0, 0]), torch.tensor([10, 11]),
                           torch.tensor([True, True]), torch.tensor([0.8, 0.8]))
        self.assertTrue(torch.isfinite(result['score']).all())
        self.assertFalse(result['pass_mask'].any())

    def test_duplicate_memory_ids_rejected(self):
        with self.assertRaises(ValueError):
            fill(ready_bank(), [0., 0.01], ids=[3, 3])

    def test_class_means_variances_and_moment_ema(self):
        bank = KNNReliabilityBank(2, 2, sigma_momentum=0.9)
        centers = torch.tensor([[1., 0.], [-1., 0.]])
        initialized = torch.tensor([True, True])
        x = points([0., 0.2, -0.1, 2.8, 3., 3.2])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        bank.begin_source_refresh()
        bank.accumulate_source(x[:2], labels[:2])
        bank.accumulate_source(x[2:], labels[2:])
        bank.finalize_source()
        means = torch.stack([x[:3].mean(0), x[3:].mean(0)])
        variances = torch.stack([x[:3].var(0, unbiased=False), x[3:].var(0, unbiased=False)])
        self.assertTrue(torch.allclose(bank.class_means, means, atol=1e-6))
        self.assertTrue(torch.allclose(bank.class_variances, variances.clamp_min(1e-4), atol=1e-6))
        old_variances = bank.class_variances.clone()
        self.assertEqual(bank.class_source_counts.tolist(), [3, 3])
        shifted = points([0.3, 0.5, 0.2, 2.5, 2.7, 2.9])
        new_means = torch.stack([shifted[:3].mean(0), shifted[3:].mean(0)])
        new_variances = torch.stack([shifted[:3].var(0, unbiased=False), shifted[3:].var(0, unbiased=False)])
        bank.begin_source_refresh()
        bank.accumulate_source(shifted, labels)
        bank.finalize_source()
        expected_var = (0.9 * old_variances + 0.1 * new_variances
                        + 0.09 * (means - new_means).square()).clamp_min(1e-4)
        self.assertTrue(torch.allclose(bank.class_means, 0.9 * means + 0.1 * new_means, atol=1e-6))
        self.assertTrue(torch.allclose(bank.class_variances, expected_var, atol=1e-6))
        self.assertAlmostEqual(bank.global_var.item(), expected_var.sum(1).mean().item(), places=6)

    def test_chi_square_quantile_and_anisotropic_membership(self):
        bank = ready_bank()
        # Chi-square(2) quantile has the exact closed form -2 log(1-p).
        self.assertAlmostEqual(bank.mahalanobis_threshold.item(), -2 * math.log(0.05), places=5)
        bank.class_means[0].zero_()
        bank.class_variances[0] = torch.tensor([1., 0.01])
        z = torch.tensor([[1., 0.], [0., 1.]])
        distances = bank.distribution_distance(z, torch.tensor([0, 0]))
        self.assertTrue(torch.allclose(distances, torch.tensor([1., 100.])))
        self.assertEqual(bank.in_distribution(z, torch.tensor([0, 0])).tolist(), [True, False])

    def test_sample_can_pass_without_any_prototype(self):
        bank = KNNReliabilityBank(2, 2, k=1)
        # Source Gaussian alone is sufficient; no prototype input or state exists.
        bank.begin_source_refresh()
        bank.accumulate_source(points([-0.1, 0.1]), torch.tensor([0, 0]))
        bank.finalize_source()
        self.assertTrue(bank.in_distribution(points([0.]), torch.tensor([0])).item())
        self.assertFalse(hasattr(bank, 'prototype_in_distribution'))
        fill(bank, [0.01])
        self.assertGreater(query(bank)['score'].item(), 0.5)
        self.assertTrue(query(bank)['pass_mask'].item())

    def test_identical_features_and_missing_classes_fail_closed(self):
        bank = KNNReliabilityBank(2, 2, k=1)
        bank.begin_source_refresh()
        bank.accumulate_source(points([0., 0., math.pi]), torch.tensor([0, 0, 1]))
        bank.finalize_source()
        self.assertEqual(bank.distribution_initialized.tolist(), [True, False])
        self.assertTrue(torch.isfinite(bank.class_variances).all())
        self.assertTrue((bank.class_variances >= bank.variance_floor).all())
        fill(bank, [0.])
        self.assertTrue(query(bank)['pass_mask'].item())
        self.assertFalse(query(bank, angle=math.pi, label=1)['pass_mask'].item())
        bank.begin_source_refresh()
        bank.finalize_source()
        self.assertFalse(bank.distribution_initialized.any())
        self.assertFalse(bank.sigma_initialized.item())

    def test_distribution_is_checkpointed_but_memory_is_not(self):
        bank = ready_bank()
        fill(bank, [0., 0.01])
        state = bank.state_dict()
        self.assertNotIn('memory_features', state)
        restored = KNNReliabilityBank(2, 2, k=2)
        restored.load_state_dict(state)
        self.assertAlmostEqual(restored.sigma.item(), 0.2, places=6)
        self.assertFalse(query(restored)['pass_mask'].item())
        self.assertTrue(torch.equal(restored.class_means, bank.class_means))
        self.assertTrue(torch.equal(restored.class_variances, bank.class_variances))
        restored.begin_source_refresh()
        self.assertEqual(restored.memory_features.size(0), 0)

    def test_score_matches_direct_reference_and_chunking_does_not_change_results(self):
        bank = ready_bank(query_chunk_size=1)
        fill(bank, [0.02, -0.04])
        z = points([0., 0.01, -0.02]).requires_grad_()
        result = bank.gate(z, torch.zeros(3, dtype=torch.long), torch.tensor([11, 12, 13]),
                           torch.ones(3, dtype=torch.bool), torch.tensor([0.8, 0.8]))
        expected = torch.exp(-((z[:, None, :] - bank.memory_features[None, :, :]) ** 2)
                             .sum(dim=2) / (2 * 0.04)).mean(dim=1)
        self.assertTrue(torch.allclose(result['score'], expected.detach(), atol=1e-5))
        self.assertFalse(result['score'].requires_grad)
        bank.query_chunk_size = 100
        other = bank.gate(z, torch.zeros(3, dtype=torch.long), torch.tensor([11, 12, 13]),
                          torch.ones(3, dtype=torch.bool), torch.tensor([0.8, 0.8]))
        self.assertTrue(torch.allclose(result['score'], other['score']))

    def test_or_rescue_warmup_and_disagreement(self):
        confidence = torch.tensor([True, False, False, False])
        agree = torch.tensor([True, True, False, True])
        support = torch.tensor([False, True, True, False])
        final, rescue = combine_rescue_masks(confidence, agree, support, True)
        self.assertEqual(final.tolist(), [True, True, False, False])
        self.assertEqual(rescue.tolist(), [False, True, False, False])
        final, rescue = combine_rescue_masks(confidence, agree, support, False)
        self.assertEqual(final.tolist(), confidence.tolist())
        self.assertFalse(rescue.any())
        self.assertEqual(set(final.float().tolist()), {0., 1.})


if __name__ == '__main__':
    unittest.main()

