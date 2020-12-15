import os
import random
import numpy as np
import cv2
import re
from collections import OrderedDict

from davis import cfg as eval_cfg

from .helpers import listdir_nohidden
from .vos_dataset import VOSDataset


class DAVIS(VOSDataset):
    """DAVIS 16 and 17 datasets.

        The root_dir naming specifies whether it is 16 or 17.
    """

    mean_val = (104.00699, 116.66877, 122.67892)

    def __init__(self, *args, **kwargs):
        super(DAVIS, self).__init__(*args, **kwargs)
        self.year = int(re.sub("[^0-9]", "", self.root_dir))

        if 'test' in self.seqs_key:
            self.test_mode = True

        seqs = OrderedDict()
        imgs = []
        labels = []

        # seqs_key either loads file with sequences or specific sequence
        seqs_file = os.path.join(self.root_dir, f"{self.seqs_key}.txt")
        if os.path.exists(seqs_file):
            with open(seqs_file) as f:
                seqs_keys = [seq.strip() for seq in f.readlines()]
        else:
            seqs_keys = [self.seqs_key]

        res_folder = '480p'
        if self._full_resolution:
            if self.year == 2016:
                res_folder = '1080p'
            else:
                res_folder = 'Full-Resolution'

        # Initialize the per sequence images for online training
        for k in seqs_keys:
            images = np.sort(listdir_nohidden(os.path.join(
                self.root_dir, 'JPEGImages', res_folder, k)))
            imgs_seq = list(map(lambda x: os.path.join(
                self.root_dir, 'JPEGImages', res_folder, k, x), images))

            lab = np.sort(listdir_nohidden(os.path.join(
                self.root_dir, 'Annotations', res_folder, k)))
            labels_seq = list(map(lambda x: os.path.join(
                self.root_dir, 'Annotations', res_folder, k, x), lab))

            if not self.test_mode:
                assert (len(labels_seq) == len(imgs_seq)), f'failure in: {k}'

            seqs[k] = {}
            seqs[k]['imgs'] = imgs_seq
            seqs[k]['labels'] = labels_seq

            imgs.extend(imgs_seq)
            labels.extend(labels_seq)

        self.seqs = seqs

        if os.path.exists(seqs_file):
            self.imgs = imgs
            self.labels = labels
            self.seq_key = None
        else:
            self.set_seq(self.seqs_key)

        self.setup_davis_eval()

    def setup_davis_eval(self):
        eval_cfg.MULTIOBJECT = bool(self.multi_object)
        if self.year == 2016:
            eval_cfg.MULTIOBJECT = False
        if self._full_resolution:
            if self.year == 2016:
                eval_cfg.RESOLUTION = '1080p'
            else:
                eval_cfg.RESOLUTION = 'Full-Resolution'
        if self.test_mode:
            eval_cfg.PHASE = 'test-dev'
        eval_cfg.YEAR = self.year
        eval_cfg.PATH.ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        eval_cfg.PATH.DATA = os.path.abspath(os.path.join(eval_cfg.PATH.ROOT, self.root_dir))
        eval_cfg.PATH.SEQUENCES = os.path.join(eval_cfg.PATH.DATA, "JPEGImages", eval_cfg.RESOLUTION)
        eval_cfg.PATH.ANNOTATIONS = os.path.join(eval_cfg.PATH.DATA, "Annotations", eval_cfg.RESOLUTION)
        eval_cfg.PATH.PALETTE = os.path.abspath(os.path.join(eval_cfg.PATH.ROOT, 'data/palette.txt'))

    def resetup_davis_eval(self):
        self.setup_davis_eval()
        eval_cfg.SEQUENCES = {n: {'name': n, 'attributes': [], 'set': 'train', 'eval_t': False, 'year': 2017, 'num_frames': len(set(v['labels']))}
                              for n, v in self.seqs.items()}
