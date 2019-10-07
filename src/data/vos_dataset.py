import os
import random
import numpy as np
import cv2
import torch

from .helpers import *
from torch.utils.data import Dataset


class VOSDataset(Dataset):
    """DAVIS dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, seqs_key, frame_id, crop_size, root_dir, transform, meanval,
                 multi_object):
        """Loads image to label pairs.
        root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        raise NotImplementedError

    @property
    def num_seqs(self):
        return len(self.seqs)

    @property
    def seqs_names(self):
        return list(self.seqs.keys())

    def set_random_seq(self):
        rnd_key_idx = random.randint(0, self.num_seqs - 1)
        rnd_seq_name = self.seqs_names[rnd_key_idx]

        self.set_seq(rnd_seq_name)

    def set_random_frame_id(self):
        self.frame_id = torch.randint(len(self.imgs), (1,)).item()

    def set_next_frame_id(self):
        if self.frame_id == 'middle':
            self.frame_id = len(self.imgs) // 2
        elif self.frame_id == 'random':
            self.frame_id = torch.randint(len(self.imgs), (1,)).item()

        if self.frame_id + 1 == len(self.imgs):
            self.frame_id = 0
        else:
            self.frame_id += 1

    def get_seq_id(self):
        return self.seqs_names.index(self.seq_key)

    def set_next_seq(self):
        key_idx = self.get_seq_id() + 1
        if key_idx == len(self.seqs.keys()):
            key_idx = 0

        seq_name = self.seqs_names[key_idx]

        self.set_seq(seq_name)

    def set_seq(self, seq_name):
        imgs = self.seqs[seq_name]['imgs']
        labels = self.seqs[seq_name]['labels']

        self.imgs = imgs
        self.labels = labels
        self.seq_key = seq_name

    def __len__(self):
        if self.frame_id is not None:
            return 1
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.frame_id is not None:
            if self.frame_id == 'middle':
                idx = len(self.imgs) // 2
            elif self.frame_id == 'random':
                idx = torch.randint(len(self.imgs), (1,)).item()
            else:
                idx = self.frame_id

        img, gt = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt,
                  'file_name': os.path.splitext(os.path.basename(self.imgs[idx]))[0]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        raise NotImplementedError

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.root_dir, self.imgs[0]))

        return list(img.shape[:2])

