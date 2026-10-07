import torch.utils.data as data
import cv2
import pandas as pd
import os
import image_utils as util
import random
import glob
import numpy as np


"0: surprise, 1: fear, 2: disgust, 3: happy 4: sad  5: angry 6: neutral 7:attempt"


class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None, strong_transform=None, basic_aug=False, ratio=1):
        self.phase = phase
        self.transform = transform
        self.strong_transform = strong_transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1

        seed = np.random.seed(2000)
        np.random.shuffle(file_names)
        seed = np.random.seed(2000)
        np.random.shuffle(self.label)

        self.file_paths = []
        for f in file_names:
            f = f.split('.')[0] + '_aligned.jpg'
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [util.flip_image, util.add_gaussian_noise, util.crop, util.rotation]
        distribute = np.array(self.label)
        self.label_dis = [
            np.sum(distribute == 0), np.sum(distribute == 1), np.sum(distribute == 2),
            np.sum(distribute == 3), np.sum(distribute == 4), np.sum(distribute == 5),
            np.sum(distribute == 6)
        ]
        print('The dataset distribute: %d, %d, %d, %d, %d, %d, %d' % tuple(self.label_dis))

    def __len__(self):
        return len(self.file_paths)

    def weight(self):
        return np.ones(shape=len(self.label_dis)) / self.label_dis * 1000

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError('Failed to read image: %s' % path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]

        if self.phase == 'train' and self.basic_aug and random.uniform(0, 1) > 0.5:
            index = random.randint(0, 2)
            image = self.aug_func[index](image)

        img = self.transform(image) if self.transform is not None else image

        if self.strong_transform is not None:
            img_aug = self.strong_transform(image)
            return img, img_aug, label
        return img, label


class FER(data.Dataset):
    """FER2013 loader using the CAST/RAF-DB class order.

    For target training this class can return two independently sampled weak
    views plus one strong view. The extra weak view is the only data-pipeline
    change required by the EMA + Dual View ablation.
    """

    FER_TO_CAST = {
        0: 5,
        1: 2,
        2: 1,
        3: 3,
        4: 4,
        5: 0,
        6: 6,
    }

    def __init__(self, path, phase, transform=None, weak2_transform=None,
                 strong_transform=None, basic_aug=False, return_index=False):
        self.phase = phase
        self.transform = transform
        self.weak2_transform = weak2_transform
        self.return_index = bool(return_index)
        self.strong_transform = strong_transform
        self.basic_aug = basic_aug
        self.aug_func = [util.flip_image, util.add_gaussian_noise, util.crop, util.rotation]
        self.file_paths, self.label = [], []

        if phase not in {'train', 'val', 'test'}:
            raise ValueError("FER phase must be one of: 'train', 'val', 'test'")

        split_candidates = {
            'train': ['train'],
            'val': ['val', 'validation'],
            'test': ['test'],
        }[phase]

        files = []
        used_split = None
        for split_name in split_candidates:
            candidate = glob.glob(os.path.join(path, split_name, '*', '*.jpg'))
            if candidate:
                files = candidate
                used_split = split_name
                break

        if not files:
            expected = ' or '.join(os.path.join(path, name, '*', '*.jpg') for name in split_candidates)
            raise FileNotFoundError(
                'No FER2013 images found for phase %s. Expected %s' % (phase, expected)
            )

        # Independent training/memory views must assign the same ID to a path.
        # Use a local RNG so constructing a dataset does not reset training RNGs.
        files = sorted(files)
        if phase == 'train':
            np.random.RandomState(2000).shuffle(files)

        for file in files:
            self.file_paths.append(file)
            original_label = int(os.path.basename(os.path.dirname(file)))
            if original_label not in self.FER_TO_CAST:
                raise ValueError('Unexpected FER2013 class id %s in %s' % (original_label, file))
            self.label.append(self.FER_TO_CAST[original_label])

        distribute = np.array(self.label)
        self.label_dis = [np.sum(distribute == i) for i in range(7)]
        print('FER %s split (%s), samples: %d' % (phase, used_split, len(self.file_paths)))
        print('The dataset distribute: %d, %d, %d, %d, %d, %d, %d' % tuple(self.label_dis))

    def __len__(self):
        return len(self.file_paths)

    def weight(self):
        return np.ones(shape=len(self.label_dis)) / self.label_dis * 1000

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError('Failed to read image: %s' % path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]

        if self.phase == 'train' and self.basic_aug and random.uniform(0, 1) > 0.5:
            index = random.randint(0, 1)
            image = self.aug_func[index](image)

        img = self.transform(image) if self.transform is not None else image

        if self.weak2_transform is not None:
            img_weak2 = self.weak2_transform(image)
            if self.strong_transform is not None:
                img_strong = self.strong_transform(image)
                result = (img, img_weak2, img_strong, label)
            else:
                result = (img, img_weak2, label)
            return result + (idx,) if self.return_index else result

        if self.strong_transform is not None:
            img_strong = self.strong_transform(image)
            result = (img, img_strong, label)
            return result + (idx,) if self.return_index else result

        result = (img, label)
        return result + (idx,) if self.return_index else result
