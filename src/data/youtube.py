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

    def __init__(self,
                 seqs_key='train_seqs',  # ['train_seqs', 'test_seqs', 'blackswan', ...]
                 frame_id=None,
                 crop_size=None,
                 root_dir='data/DAVIS-2016',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 multi_object=False,): # [False, 'all', 'single_first', 'single_random']
        """Loads image to label pairs.

        root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.crop_size = crop_size
        self.root_dir = root_dir
        self.transform = transform
        self.meanval = meanval
        self.seqs_key = seqs_key
        self.frame_id = frame_id
        self.multi_object = multi_object

        seqs = OrderedDict()
        imgs = []
        labels = []

        # seqs_key either loads file with sequences or specific sequence
        seqs_file = os.path.join(root_dir, f'{seqs_key}.txt')
        if os.path.exists(seqs_file):
            with open(seqs_file) as f:
                seqs_keys = [seq.strip() for seq in f.readlines()]
        else:
            seqs_keys = [seqs_key]

        # Initialize the per sequence images for online training
        for k in seqs_keys:
            images = np.sort(listdir_nohidden(os.path.join(root_dir, 'JPEGImages/480p/', k)))
            imgs_seq = list(map(lambda x: os.path.join('JPEGImages/480p/', k, x), images))

            lab = np.sort(listdir_nohidden(os.path.join(root_dir, 'Annotations/480p/', k)))
            labels_seq = list(map(lambda x: os.path.join('Annotations/480p/', k, x), lab))

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
            self.set_seq(seqs_key)

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.root_dir, self.imgs[idx]), cv2.IMREAD_COLOR)
        label = cv2.imread(os.path.join(self.root_dir, self.labels[idx]), cv2.IMREAD_GRAYSCALE)

        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            img_h, img_w = label.shape

            pad_h = max(crop_h - img_h, 0)
            pad_w = max(crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0,
                    pad_w, cv2.BORDER_CONSTANT,
                    value=(0.0, 0.0, 0.0))
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                    pad_w, cv2.BORDER_CONSTANT,
                    value=(0,))
            else:
                img_pad, label_pad = img, label

            img_h, img_w = label_pad.shape
            h_off = random.randint(0, img_h - crop_h)
            w_off = random.randint(0, img_w - crop_w)
            img = img_pad[h_off : h_off + crop_h, w_off : w_off + crop_w]
            label = label_pad[h_off : h_off + crop_h, w_off : w_off + crop_w]

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        label = np.array(label, dtype=np.float32)
        label = label / np.max([label.max(), 1e-8])

        # multi object
        # ignore_label_mask = label == self.ignore_label
        # unique_labels = np.unique(label)
        if self.multi_object:
            if self.multi_object not in ['all', 'single_first', 'single_random']:
                raise NotImplementedError

            # all objects stacked in third axis
            unique_labels = np.unique(label)
            label = np.concatenate([np.expand_dims((label == l).astype(np.float32), axis=2)
                                    for l in np.unique(label)], axis=2)

            # single object from stack
            # if only one object on the frame this object is selected
            if self.multi_object == 'single_first':
                label = label[:, :, 0]
            elif self.multi_object == 'single_random':
                label = label[:, :, random.randint(0, len(unique_labels) - 1)]
        else:
            label = np.where(label != 0.0, label.max(), 0.0).astype(np.float32)
        # label = np.where(ignore_label_mask, self.ignore_label, label).astype(np.float32)

        # print(os.path.join(self.root_dir, self.imgs[idx]), img.shape, label.shape)
        return img, label
