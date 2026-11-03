import unittest
import torch
from prototype_radius import PrototypeRadiusBank


class RadiusTests(unittest.TestCase):
    def test_streaming_top_seven_per_class(self):
        bank = PrototypeRadiusBank(torch.eye(2), torch.ones(2, dtype=torch.bool))
        angles = torch.linspace(0, 1.5, 10)
        z = torch.stack((angles.cos(), angles.sin()), dim=1)
        bank.accumulate_source(z[:4], torch.zeros(4, dtype=torch.long))
        bank.accumulate_source(z[4:], torch.zeros(6, dtype=torch.long))
        bank.accumulate_source(torch.eye(2)[1:].repeat(7, 1), torch.ones(7, dtype=torch.long))
        bank.finalize_source()
        expected = (z - torch.tensor([1., 0.])).norm(dim=1).topk(7).values.mean()
        torch.testing.assert_close(bank.radii[0], expected)
        self.assertEqual(bank.radii[1].item(), 0.)
        self.assertTrue(bank.ready.all())

    def test_strict_boundary_invalid_and_missing_class(self):
        bank = PrototypeRadiusBank(torch.eye(2), torch.ones(2, dtype=torch.bool))
        bank.ready[0] = True
        bank.radii[0] = 0.0
        self.assertFalse(bank.accepts(torch.tensor([[1., 0.]]), torch.tensor([0])).item())
        bank.radii[0] = 1.0
        queries = torch.tensor([[1., 0.], [0., 1.], [0., 0.], [float('nan'), 1.], [1., 0.]])
        self.assertEqual(bank.accepts(queries, torch.tensor([0, 1, 0, 0, -1])).tolist(),
                         [True, False, False, False, False])

    def test_fewer_than_seven_fails_closed(self):
        bank = PrototypeRadiusBank(torch.eye(2), torch.ones(2, dtype=torch.bool))
        bank.accumulate_source(torch.tensor([[1., 0.]]).repeat(6, 1), torch.zeros(6, dtype=torch.long))
        bank.finalize_source()
        self.assertFalse(bank.ready.any())


if __name__ == '__main__':
    unittest.main()
