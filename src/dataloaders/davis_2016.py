import os
import random
import numpy as np
import cv2
import torch
from scipy.misc import imresize
from collections import OrderedDict

from dataloaders.helpers import *
from torch.utils.data import Dataset


def listdir_nohidden(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]


class DAVIS2016(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self,
                 seqs='train_seqs',  # ['train_seqs', 'test_seqs', 'blackswan', ...]
                 frame_id=None,
                 crop_size=None,
                #  ignore_label=0,
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
        self.seqs = seqs
        self.frame_id = frame_id
        # self.ignore_label = ignore_label
        self.multi_object = multi_object

        seqs_dict = OrderedDict()
        img_list = []
        labels = []

        if '_' in seqs:
            with open(os.path.join(root_dir, f'{seqs}.txt')) as f:
                for seq in f.readlines():
                    seq = seq.strip()
                    seqs_dict[seq] = {}

                    images = np.sort(listdir_nohidden(os.path.join(root_dir, 'JPEGImages/480p/', seq)))
                    img_list_seq = list(map(lambda x: os.path.join('JPEGImages/480p/', seq, x), images))

                    lab = np.sort(listdir_nohidden(os.path.join(root_dir, 'Annotations/480p/', seq)))
                    labels_seq = list(map(lambda x: os.path.join('Annotations/480p/', seq, x), lab))

                    assert (len(labels_seq) == len(img_list_seq)), f'failure in: {seq}'

                    seqs_dict[seq]['img_list'] = img_list_seq
                    seqs_dict[seq]['labels'] = labels_seq

                    img_list.extend(img_list_seq)
                    labels.extend(labels_seq)
        else:
            # Initialize the per sequence images for online training
            names_img = np.sort(listdir_nohidden(os.path.join(root_dir, 'JPEGImages/480p/', seqs)))
            img_list_seq = list(map(lambda x: os.path.join('JPEGImages/480p/', seqs, x), names_img))

            names_label = np.sort(listdir_nohidden(os.path.join(root_dir, 'Annotations/480p/', seqs)))
            labels_seq = list(map(lambda x: os.path.join('Annotations/480p/', seqs, x), names_label))

            assert (len(labels_seq) == len(img_list_seq)), f'failure in: {seqs}'

            seqs_dict[seqs] = {}
            seqs_dict[seqs]['img_list'] = img_list_seq
            seqs_dict[seqs]['labels'] = labels_seq

            img_list.extend(img_list_seq)
            labels.extend(labels_seq)

        self.seqs_dict = seqs_dict
        self.img_list = img_list
        self.labels = labels

    def set_random_seq(self):
        rnd_key_idx = random.randint(0, len(self.seqs_dict.keys()) - 1)
        rnd_seq_name = list(self.seqs_dict.keys())[rnd_key_idx]

        self.set_seq(rnd_seq_name)

    def set_random_frame_id(self):
        self.frame_id = torch.randint(len(self.img_list), (1,)).item()

    def set_next_frame_id(self):
        if self.frame_id == 'middle':
            self.frame_id = len(self.img_list) // 2
        elif self.frame_id == 'random':
            self.frame_id = torch.randint(len(self.img_list), (1,)).item()

        if self.frame_id + 1 == len(self.img_list):
            self.frame_id = 0
        else:
            self.frame_id += 1

    def get_seq_id(self):
        return list(self.seqs_dict.keys()).index(self.seqs)

    def set_next_seq(self):
        if '_' in self.seqs:
            key_idx = 0
        else:
            key_idx = self.get_seq_id() + 1
            if key_idx == len(self.seqs_dict.keys()):
                key_idx = 0

        seq_name = list(self.seqs_dict.keys())[key_idx]

        self.set_seq(seq_name)

    def set_seq(self, seq_name):
        img_list = self.seqs_dict[seq_name]['img_list']
        labels = self.seqs_dict[seq_name]['labels']

        self.img_list = img_list
        self.labels = labels
        self.seqs = seq_name

    def __len__(self):
        if self.frame_id is not None:
            return 1
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.frame_id is not None:
            if self.frame_id == 'middle':
                idx = len(self.img_list) // 2
            elif self.frame_id == 'random':
                idx = torch.randint(len(self.img_list), (1,)).item()
            else:
                idx = self.frame_id
        img, gt = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt}

        if '_' not in self.seqs:
            sample['fname'] = os.path.join(self.seqs, "%05d" % idx)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.root_dir, self.img_list[idx]), cv2.IMREAD_COLOR)
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

        # print(os.path.join(self.root_dir, self.img_list[idx]), img.shape, label.shape)
        return img, label

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.root_dir, self.img_list[0]))

        return list(img.shape[:2])

