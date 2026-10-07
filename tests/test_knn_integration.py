import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms

from dataset import FER
from knn_reliability import KNNReliabilityBank
from prototype_utils import FeatureHook, PrototypeBank
from train import parse_args, refresh_reliability_bank, save_checkpoint, log_gaussian_means


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.feature.weight.copy_(torch.eye(2))

    def forward(self, x, *args, **kwargs):
        features = self.feature(x)
        logits = torch.stack([5 * features[:, 0], -5 * features[:, 0]], dim=1)
        return logits, features


class IntegrationTests(unittest.TestCase):
    def test_parser_returns_namespace_and_rejects_invalid_options(self):
        args = parse_args([])
        self.assertTrue(args.knn_gate)
        self.assertEqual(args.knn_k, 20)
        self.assertFalse(parse_args(['--no_knn_gate']).knn_gate)
        for argv in (['--knn_k', '0'], ['--knn_refresh_interval', '0'],
                     ['--knn_sigma_momentum', '1'], ['--knn_score_threshold', '0'],
                     ['--knn_bandwidth_multiplier', 'nan'],
                     ['--knn_distribution_mass', '1'], ['--knn_distribution_mass', 'nan'],
                     ['--knn_variance_floor', '0'], ['--knn_interval_lambda', '1.5'], ['--pre_epochs', '0']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)
        args = parse_args(['--pre_epochs', '0', '--checkpoint', 'source.pth'])
        self.assertEqual(args.checkpoint, 'source.pth')

    def test_mu_logging_is_complete_and_flags_large_values(self):
        bank = KNNReliabilityBank(2, 1200)
        bank.distribution_initialized[0] = True
        bank.class_source_counts[0] = 12
        bank.class_means[0].fill_(0.01)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            log_gaussian_means(bank)
        text = out.getvalue()
        self.assertIn('bound_ok=True', text)
        self.assertIn('class=1 count=0 ready=False', text)
        self.assertNotIn('...', text)
        vector = text.split('mu=[', 1)[1].split(']', 1)[0]
        self.assertEqual(len(vector.split(',')), 1200)
        bank.class_means[0, 0] = 2.
        with contextlib.redirect_stdout(out):
            log_gaussian_means(bank)
        self.assertIn('bound_ok=False', out.getvalue())
        self.assertIn('WARNING', out.getvalue())

    def test_dataset_ids_match_across_views_and_default_interface_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            root = Path(root)
            for split in ('train', 'val', 'test'):
                folder = root / split / '0'
                folder.mkdir(parents=True)
                for name in ('z.jpg', 'a.jpg', 'm.jpg'):
                    Image.new('RGB', (8, 8), color=(30, 50, 90)).save(folder / name)
            # Match train.py: OpenCV's BGR->RGB slice is converted through PIL.
            transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
            state = np.random.get_state()
            with contextlib.redirect_stdout(io.StringIO()):
                train = FER(str(root), 'train', transform=transform,
                            weak2_transform=transform, strong_transform=transform,
                            return_index=True)
                memory = FER(str(root), 'train', transform=transform, return_index=True)
                val = FER(str(root), 'val', transform=transform)
                test = FER(str(root), 'test', transform=transform)
            after = np.random.get_state()
            self.assertTrue(np.array_equal(state[1], after[1]))
            self.assertEqual(state[2:], after[2:])
            self.assertEqual(train.file_paths, memory.file_paths)
            for i in range(len(train)):
                self.assertEqual(len(train[i]), 5)
                self.assertEqual(len(memory[i]), 3)
                self.assertEqual(train[i][-1], memory[i][-1])
                self.assertEqual(memory[i][-2], FER.FER_TO_CAST[0])
            self.assertEqual(len(val[0]), 2)
            self.assertEqual(len(test[0]), 2)

    def test_refresh_uses_teacher_predictions_not_target_ground_truth(self):
        teacher = TinyTeacher()
        hook = FeatureHook(teacher.feature)
        self.addCleanup(hook.close)
        prototypes = PrototypeBank(2, 2)
        source_features = torch.tensor([[1., 0.], [0.98, 0.2], [-1., 0.], [-0.98, 0.2]])
        source_labels = torch.tensor([0, 0, 1, 1])
        prototypes.accumulate_source(source_features, source_labels)
        prototypes.finalize_source()
        source_loader = [(source_features[:2], source_labels[:2]),
                         (source_features[2:], source_labels[2:])]
        target = torch.tensor([[1., 0.1], [-1., 0.1], [0.9, 0.2]])
        ids = torch.tensor([71, 52, 99])
        banks = []
        for fake_labels in (torch.tensor([999, -999, 999]), torch.tensor([1, 0, 1])):
            bank = KNNReliabilityBank(2, 2, k=1)
            refresh_reliability_bank(teacher, hook, source_loader, [(target, fake_labels, ids)],
                                     bank)
            banks.append(bank)
        self.assertTrue(torch.equal(banks[0].memory_labels, torch.tensor([0, 1, 0])))
        self.assertTrue(torch.equal(banks[0].memory_labels, banks[1].memory_labels))
        self.assertTrue(torch.equal(banks[0].memory_ids, ids))
        self.assertTrue(banks[0].sigma_initialized.item())
        self.assertEqual(banks[0].source_count.item(), 4)
        self.assertFalse(banks[0].memory_features.requires_grad)

        optimizer = torch.optim.SGD(teacher.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            path = str(Path(root) / 'checkpoint.pth')
            save_checkpoint(path, teacher, optimizer, scheduler, 1, 0.5, parse_args([]),
                            teacher=teacher, prototype_bank=prototypes,
                            reliability_bank=banks[0])
            # This file is created locally by this test, never an external checkpoint.
            saved = torch.load(path, map_location='cpu')
        self.assertIn('reliability_bank', saved)
        self.assertIn('global_var', saved['reliability_bank'])
        for key in ('class_means', 'class_variances', 'distribution_initialized',
                    'mahalanobis_threshold'):
            self.assertIn(key, saved['reliability_bank'])
        self.assertNotIn('memory_features', saved['reliability_bank'])
        self.assertEqual(saved['args']['knn_k'], 20)


if __name__ == '__main__':
    unittest.main()

