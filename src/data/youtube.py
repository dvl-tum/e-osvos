import json
import os
import random
import re
from collections import OrderedDict

import cv2
import numpy as np
from davis import cfg as eval_cfg

from .helpers import *
from .vos_dataset import VOSDataset


class YouTube(VOSDataset):
    """YouTube-VOS dataset. https://youtube-vos.org/"""

    meanval = (104.00699, 116.66877, 122.67892)

    def __init__(self, *args, **kwargs):
        super(YouTube, self).__init__(*args, **kwargs)

        if self._full_resolution:
            raise NotImplementedError

        seqs = OrderedDict()
        imgs = []
        labels = []

        # seqs_key either loads file with sequences or specific sequence
        seqs_file = os.path.join(self.root_dir, f"{self.seqs_key}.txt")
        if os.path.exists(seqs_file):
            with open(seqs_file) as f:
                seqs_keys = [seq.strip() for seq in f.readlines()]
        else:
            raise NotImplementedError

        if 'val' in self.seqs_key or 'test' in self.seqs_key:
            self.test_mode = True

        # if not os.path.exists(seqs_dir):
        #     raise NotImplementedError
        #     # seqs_keys = [self.seqs_key]

        # # Initialize the per sequence images for online training

        self._split = self.seqs_key.split('_')[0]
        seqs_dir = os.path.join(self.root_dir, self._split)

        meta_file_path = os.path.join(seqs_dir, 'meta.json')
        with open(meta_file_path, 'r') as f:
            self._meta_data = json.load(f)

        # # seq_names = listdir_nohidden(os.path.join(seqs_dir, 'JPEGImages'))
        for seq_name in seqs_keys:

            img_names = np.sort(listdir_nohidden(
                os.path.join(seqs_dir, 'JPEGImages', seq_name)))
            img_paths = list(map(lambda x: os.path.join(
                seqs_dir, 'JPEGImages', seq_name, x), img_names))

            label_names = np.sort(listdir_nohidden(
                os.path.join(seqs_dir, 'Annotations', seq_name)))
            label_paths = list(map(lambda x: os.path.join(
                seqs_dir, 'Annotations', seq_name, x), label_names))

            if not self.test_mode and len(label_names) != len(img_names):
                print(f'failure in: {self.seqs_key}/{seq_name}')

            seqs[seq_name] = {}
            seqs[seq_name]['imgs'] = img_paths
            seqs[seq_name]['labels'] = label_paths

            imgs.extend(img_paths)
            labels.extend(label_paths)

        self.seqs = seqs

        self.imgs = imgs
        self.labels = labels
        self.seq_key = None

        self.setup_davis_eval()

    @property
    def num_objects(self):
        """
        Retrieve number of objects from first frame ground truth which always
        contains all objects.
        """
        if self.seq_key is None:
            raise NotImplementedError
        if not self.multi_object:
            return 1

        # from PIL import Image
        # im = Image.open(self.labels[0])
        # label = np.atleast_3d(im)[...,0]
        # # label = cv2.imread(os.path.join(self.root_dir, self.labels[0]), cv2.IMREAD_GRAYSCALE)
        # # label = np.array(label, dtype=np.float32)
        # # label = label / 255.0

        # unique_labels = [l for l in np.unique(label)
        #                 #  if l != 0.0 and l != 1.0]
        #                  if l != 0.0]

        # if len(unique_labels) != len(self._meta_data['videos'][self.seq_key]['objects']):
        #     print(self.seq_key)

        return len(self._meta_data['videos'][self.seq_key]['objects'])

    def set_gt_frame_id(self):
        objects_info = self._meta_data['videos'][self.seq_key]['objects']
        objects_info = [v for k, v in sorted(objects_info.items())]

        if 'test' in self.seqs_key:
            first_gt_image_name = objects_info[self.multi_object_id][0]
        else:
            first_gt_image_name = objects_info[self.multi_object_id]["frames"][0]

        self.frame_id = [path.find(first_gt_image_name) != -1 for path in self.imgs].index(True)
        self._label_id = [path.find(
            first_gt_image_name) != -1 for path in self.labels].index(True)

        self._multi_object_id_to_label = [
            int(k) for k in sorted(self._meta_data['videos'][self.seq_key]['objects'].keys())]

        # print(self.seq_key, self.multi_object_id, self.frame_id, self._label_id)

    def setup_davis_eval(self):
        eval_cfg.MULTIOBJECT = bool(self.multi_object)
        # if self._full_resolution:
        #     eval_cfg.RESOLUTION = '1080p'
        # if self.test_mode:
        #     eval_cfg.PHASE = 'test-dev'
        eval_cfg.YEAR = 2017
        eval_cfg.PATH.ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../..'))
        eval_cfg.PATH.DATA = os.path.abspath(
            os.path.join(eval_cfg.PATH.ROOT, self.root_dir, self._split))
        eval_cfg.PATH.SEQUENCES = os.path.join(
            eval_cfg.PATH.DATA, "JPEGImages")
        eval_cfg.PATH.ANNOTATIONS = os.path.join(
            eval_cfg.PATH.DATA, "Annotations")
        # eval_cfg.PATH.PALETTE = os.path.abspath(
        #     os.path.join(eval_cfg.PATH.ROOT, 'data/palette.txt'))

        eval_cfg.SEQUENCES = {n: {'name': n, 'attributes': [], 'set': 'train', 'eval_t': False, 'year': 2017, 'num_frames': len(v['imgs'])}
                              for n, v in self.seqs.items()}
