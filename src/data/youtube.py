import json
import os
import random
import re
from collections import OrderedDict

import cv2
import numpy as np
import torch

from davis import cfg as eval_cfg

from .helpers import listdir_nohidden
from .vos_dataset import VOSDataset


class YouTube(VOSDataset):
    """YouTube-VOS dataset. https://youtube-vos.org/"""

    mean_val = (104.00699, 116.66877, 122.67892)

    def __init__(self, *args, deepcopy=False, **kwargs):
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

        # # Initialize the per sequence images for online training

        self._split = self.seqs_key.split('_')[0]
        seqs_dir = os.path.join(self.root_dir, self._split)

        if self._split in ['valid', 'test', 'valid-all-frames', 'test-all-frames']:
            self.test_mode = True

        self.all_frames = False
        if 'all-frames' in self._split:
            self.all_frames = True

        self._meta_data = None
        self.seq_key = None
        self.seqs = None
        self.imgs = None
        self.labels = None

        if not deepcopy:
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

                # we never train on all frames
                if self.all_frames:
                    label_paths = label_paths + [label_paths[0]] * (len(img_paths) - len(label_paths))

                if not self.test_mode:
                    assert len(img_paths) == len(
                        label_paths), f"{self._split} {img_names} {label_names}"

                seqs[seq_name] = {}
                seqs[seq_name]['imgs'] = img_paths
                seqs[seq_name]['labels'] = label_paths

                imgs.extend(img_paths)
                labels.extend(label_paths)

            self.seqs = seqs
            self.imgs = imgs
            self.labels = labels

            self.setup_davis_eval()

    def get_random_frame_id(self):
        if self.random_frame_id_epsilon is not None:
            random_frame_id_epsilon = self.random_frame_id_epsilon
            if 'all-frames' not in self._split:
                assert random_frame_id_epsilon % 5 == 0, "random_frame_id_epsilon={random_frame_id_epsilon} must be a multiple of 5."

                random_frame_id_epsilon //= 5

            return torch.randint(max(0, self.random_frame_id_anchor_frame - random_frame_id_epsilon),
                                 min(self.random_frame_id_anchor_frame + random_frame_id_epsilon + 1, len(self.imgs)),
                                 (1,)).item()
        else:
            return torch.randint(len(self.imgs), (1,)).item()

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

        return len(self._meta_data['videos'][self.seq_key]['objects'])

    def set_seq(self, seq_name):
        super(YouTube, self).set_seq(seq_name)
        self._multi_object_id_to_label = [
            int(k) for k in sorted(self._meta_data['videos'][self.seq_key]['objects'].keys())]

        eval_cfg.NUM_OBJECTS = self.num_object_groups

    def get_gt_frame_id(self, multi_object_id):
        objects_info = self._meta_data['videos'][self.seq_key]['objects']
        objects_info = [v for k, v in sorted(objects_info.items())]

        if 'test' in self.seqs_key:
            first_gt_image_name = objects_info[multi_object_id][0]
        else:
            first_gt_image_name = objects_info[multi_object_id]["frames"][0]

        frame_id = [path.find(first_gt_image_name) != -1 for path in self.imgs].index(True)
        _label_id = [path.find(first_gt_image_name) != -1 for path in self.labels].index(True)

        return frame_id, _label_id

    def get_gt_object_frames(self):
        return [self.get_gt_frame_id(i) for i in range(self.num_objects)]

    def get_gt_object_steps(self):
        frame_ids = self.get_gt_object_frames()
        steps = []
        for i in range(len(frame_ids) - 1):
            steps.append(frame_ids[i + 1][0] - frame_ids[i][0])
        return steps

    def has_later_objects(self):
        return [f for f, l in self.get_gt_object_frames()].count(0) != self.num_objects

    @property
    def num_object_groups(self):
        if self.multi_object == 'all':
            return len(torch.unique(torch.tensor([f for f, l in self.get_gt_object_frames()])))
        return self.num_objects

    @property
    def object_ids_in_group(self):
        obj_frames = self.get_gt_object_frames()

        frame_id = torch.unique(torch.tensor([f for f, l in obj_frames]))[
            self.multi_object_id].item()
        object_ids = [i for i, (f, _) in enumerate(obj_frames) if f == frame_id]

        if self.sub_group_ids is not None:
            object_ids = [object_ids[i] for i in self.sub_group_ids]

        return object_ids

    def set_gt_frame_id(self):
        if self.multi_object == 'all':
            obj_frames = self.get_gt_object_frames()
            frame_id = torch.unique(torch.tensor([f for f, l in obj_frames]))[
                self.multi_object_id].item()
            obj_frame = obj_frames[[f for f, l in obj_frames].index(frame_id)]
            self.frame_id, self._label_id = obj_frame
        else:
            self.frame_id, self._label_id = self.get_gt_frame_id(self.multi_object_id)

    def setup_davis_eval(self):
        eval_cfg.MULTIOBJECT = bool(self.multi_object)
        eval_cfg.YEAR = 2017
        eval_cfg.PATH.ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../..'))
        eval_cfg.PATH.DATA = os.path.abspath(
            os.path.join(eval_cfg.PATH.ROOT, self.root_dir, self._split))
        eval_cfg.PATH.SEQUENCES = os.path.join(
            eval_cfg.PATH.DATA, "JPEGImages")
        eval_cfg.PATH.ANNOTATIONS = os.path.join(
            eval_cfg.PATH.DATA, "Annotations")

        eval_cfg.SEQUENCES = {n: {'name': n, 'attributes': [], 'set': 'train', 'eval_t': False, 'year': 2017, 'num_frames': len(set(v['labels']))}
                              for n, v in self.seqs.items()}

    def __deepcopy__(self, memo):
        copy_obj = type(self)(self.seqs_key, self.root_dir, deepcopy=True)

        import copy
        for key in self.__dict__:
            copy_obj.__dict__[key] = copy.copy(self.__dict__[key])

        memo[id(self)] = copy_obj

        return copy_obj
