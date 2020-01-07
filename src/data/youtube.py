import os
import random
import numpy as np
import cv2
import re
from collections import OrderedDict

from davis import cfg as eval_cfg
from .helpers import *
from .vos_dataset import VOSDataset


class YouTube(VOSDataset):
    """YouTube-VOS dataset. https://youtube-vos.org/"""

    meanval = (104.00699, 116.66877, 122.67892)

    def __init__(self, *args, **kwargs):
        super(YouTube, self).__init__(*args, **kwargs)

        seqs = OrderedDict()
        imgs = []
        labels = []

        # seqs_key either loads file with sequences or specific sequence
        seqs_dir = os.path.join(self.root_dir, self.seqs_key)
        if not os.path.exists(seqs_dir):
            raise NotImplementedError
            # seqs_keys = [self.seqs_key]

        # Initialize the per sequence images for online training

        seq_names = listdir_nohidden(os.path.join(seqs_dir, 'JPEGImages'))
        for seq_name in seq_names:

            img_names = np.sort(listdir_nohidden(
                os.path.join(seqs_dir, 'JPEGImages', seq_name)))
            img_paths_without_root = list(map(lambda x: os.path.join(
                self.seqs_key, 'JPEGImages', seq_name, x), img_names))

            label_names = np.sort(listdir_nohidden(
                os.path.join(seqs_dir, 'Annotations', seq_name)))
            label_paths_without_root = list(map(lambda x: os.path.join(
                self.seqs_key, 'Annotations', seq_name, x), label_names))

            if len(label_names) != len(img_names):
                print(f'failure in: {self.seqs_key}/{seq_name}')

            seqs[seq_name] = {}
            seqs[seq_name]['imgs'] = img_paths_without_root
            seqs[seq_name]['labels'] = label_paths_without_root

            imgs.extend(img_paths_without_root)
            labels.extend(label_paths_without_root)

        self.seqs = seqs

        self.imgs = imgs
        self.labels = labels
        self.seq_key = None
