import math
import unittest

import torch

from global_gaussian_knn_reliability import GlobalGaussianKNNReliabilityBank


def points(angles):
    return torch.tensor(
        [[math.cos(a), math.sin(a)] for a in angles],
        dtype=torch.float32,
    )


class GlobalGaussianReliabilityTests(unittest.TestCase):
    def test_source_true_labels_build_class_centers_and_one_shared_variance(self):
        bank = GlobalGaussianKNNReliabilityBank(
            2, 2, k=1, sigma_momentum=0.5,
            global_momentum_end=0.9,
            global_momentum_ramp_refreshes=3,
        )
        x = points([0.0, 0.2, -0.1, 2.8, 3.0, 3.2])
        y = torch.tensor([0, 0, 0, 1, 1, 1])

        bank.begin_source_refresh()
        bank.accumulate_source(x, y)
        bank.finalize_source()

        expected_means = torch.stack([x[:3].mean(0), x[3:].mean(0)])
        class_vars = torch.stack([
            x[:3].var(0, unbiased=False),
            x[3:].var(0, unbiased=False),
        ])
        expected_global = class_vars.mean()

        self.assertTrue(torch.allclose(bank.class_means, expected_means, atol=1e-6))
        self.assertAlmostEqual(bank.global_var.item(), expected_global.item(), places=6)
        self.assertTrue(torch.allclose(
            bank.class_variances[0],
            torch.full((2,), bank.global_var.item()),
        ))
        self.assertTrue(torch.allclose(
            bank.class_variances[1],
            torch.full((2,), bank.global_var.item()),
        ))

    def test_global_variance_uses_increasing_momentum(self):
        bank = GlobalGaussianKNNReliabilityBank(
            1, 2, k=1,
            sigma_momentum=0.5,
            global_momentum_end=0.9,
            global_momentum_ramp_refreshes=3,
        )

        for angles in ([0.0, 0.2, -0.2], [0.0, 0.6, -0.6], [0.0, 0.9, -0.9]):
            bank.begin_source_refresh()
            bank.accumulate_source(points(angles), torch.zeros(3, dtype=torch.long))
            old = bank.global_var.item()
            was_ready = bank.sigma_initialized.item()
            bank.finalize_source()
            observed = bank.current_observed_global_var.item()
            beta = bank.current_global_momentum.item()
            if was_ready:
                expected = beta * old + (1.0 - beta) * observed
                self.assertAlmostEqual(bank.global_var.item(), expected, places=6)

        self.assertAlmostEqual(bank.current_global_momentum.item(), 0.9, places=6)

    def test_candidate_and_neighbors_must_lie_in_assigned_source_gaussian(self):
        bank = GlobalGaussianKNNReliabilityBank(
            2, 2, k=2,
            distribution_mass=0.95,
            score_threshold=0.2,
            density_threshold=0.2,
        )
        bank.class_means.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        bank.global_var.fill_(0.01)
        bank.class_variances.fill_(0.01)
        bank.distribution_initialized.fill_(True)
        bank.sigma_initialized.fill_(True)

        bank.set_target_memory(
            points([0.02, -0.02, 2.9, 3.0]),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0.99, 0.99, 0.99, 0.99]),
            torch.tensor([1, 2, 3, 4]),
        )

        result = bank.gate(
            points([0.0]),
            torch.tensor([0]),
            torch.tensor([99]),
            torch.tensor([True]),
            torch.tensor([0.8, 0.8]),
        )
        self.assertTrue(result["in_distribution"].item())
        self.assertEqual(result["support_fraction"].item(), 1.0)
        self.assertTrue(result["pass_mask"].item())

        outside = bank.gate(
            points([1.2]),
            torch.tensor([0]),
            torch.tensor([98]),
            torch.tensor([True]),
            torch.tensor([0.8, 0.8]),
        )
        self.assertFalse(outside["in_distribution"].item())
        self.assertFalse(outside["pass_mask"].item())

    def test_sparse_neighborhood_receives_score_penalty(self):
        dense = GlobalGaussianKNNReliabilityBank(
            1, 2, k=2,
            score_threshold=0.01,
            density_threshold=0.7,
            sparse_penalty=0.25,
        )
        sparse = GlobalGaussianKNNReliabilityBank(
            1, 2, k=2,
            score_threshold=0.01,
            density_threshold=0.7,
            sparse_penalty=0.25,
        )
        for bank in (dense, sparse):
            bank.class_means[0] = torch.tensor([1.0, 0.0])
            bank.global_var.fill_(0.04)
            bank.class_variances.fill_(0.04)
            bank.distribution_initialized.fill_(True)
            bank.sigma_initialized.fill_(True)

        dense.set_target_memory(
            points([0.02, -0.02]),
            torch.zeros(2, dtype=torch.long),
            torch.ones(2),
            torch.tensor([1, 2]),
        )
        sparse.set_target_memory(
            points([0.35, -0.35]),
            torch.zeros(2, dtype=torch.long),
            torch.ones(2),
            torch.tensor([1, 2]),
        )

        args = (
            points([0.0]),
            torch.tensor([0]),
            torch.tensor([99]),
            torch.tensor([True]),
            torch.tensor([0.8]),
        )
        a = dense.gate(*args)
        b = sparse.gate(*args)

        self.assertTrue(a["dense_mask"].item())
        self.assertTrue(b["sparse_mask"].item())
        self.assertGreater(a["score"].item(), b["score"].item())


if __name__ == "__main__":
    unittest.main()
