import os
import random
import numpy as np
import cv2
import torch
from PIL import Image

from .helpers import *
from torch.utils.data import Dataset


class VOSDataset(Dataset):
    """DAVIS dataset constructed using the PyTorch built-in functionalities"""

    meanval = None

    def __init__(self, seqs_key, root_dir, frame_id=None,
                 crop_size=None, transform=None, multi_object=False):
        """Loads image to label pairs.
        root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.seqs_key = seqs_key
        self.frame_id = frame_id
        self.crop_size = crop_size
        self.root_dir = root_dir
        self.transform = transform
        self.multi_object = multi_object
        self.multi_object_id = None

    @property
    def num_seqs(self):
        return len(self.seqs)

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
        im = Image.open(os.path.join(self.root_dir, self.labels[0]))
        label = np.atleast_3d(im)[...,0]
        # label = cv2.imread(os.path.join(self.root_dir, self.labels[0]), cv2.IMREAD_GRAYSCALE)
        # label = np.array(label, dtype=np.float32)
        # label = label / 255.0

        unique_labels = [l for l in np.unique(label)
                        #  if l != 0.0 and l != 1.0]
                         if l != 0.0]
        return len(unique_labels)

    @property
    def seqs_names(self):
        return list(self.seqs.keys())

    def set_random_seq(self):
        rnd_key_idx = random.randint(0, self.num_seqs - 1)
        rnd_seq_name = self.seqs_names[rnd_key_idx]

        self.set_seq(rnd_seq_name)
        return rnd_seq_name

    def set_random_frame_id(self):
        self.frame_id = torch.randint(len(self.imgs), (1,)).item()
        return self.frame_id
    
    def set_frame_id_with_biggest_label(self):
        num_labels = [np.count_nonzero(self.make_img_label_pair(idx)[1])
                      for idx in range(len(self.imgs))]
        self.frame_id = np.argmax(np.array(num_labels))

    def has_frame_object(self):
        assert self.frame_id is not None
        _, label = self.make_img_label_pair(self.frame_id)
        return len(np.unique(label)) > 1

    def set_random_frame_id_with_label(self):
        def _set_random_frame_id_with_label():
            self.set_random_frame_id()
            if self.has_frame_object():
                return
            else:
                _set_random_frame_id_with_label()

        _set_random_frame_id_with_label()
        return self.frame_id

    def set_next_frame_id(self):
        if self.frame_id == 'middle':
            self.frame_id = len(self.imgs) // 2
        elif self.frame_id == 'random':
            self.frame_id = torch.randint(len(self.imgs), (1,)).item()

        if self.frame_id + 1 == len(self.imgs):
            self.frame_id = 0
        else:
            self.frame_id += 1
        return self.frame_id

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

        img, label = self.make_img_label_pair(idx)

        sample = {'image': img, 'gt': label,
                  'file_name': os.path.splitext(os.path.basename(self.imgs[idx]))[0]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.root_dir, self.imgs[0]))

        return list(img.shape[:2])

    def make_img_label_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(
            self.root_dir, self.imgs[idx]), cv2.IMREAD_COLOR)
        
        im = Image.open(os.path.join(self.root_dir, self.labels[idx]))
        label = np.atleast_3d(im)[...,0]

        # label = cv2.imread(os.path.join(
        #     self.root_dir, self.labels[idx]), cv2.IMREAD_GRAYSCALE)

        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            img_h, img_w = label.shape

            if crop_h != img_h or crop_w != img_w:
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

                img = img_pad[h_off: h_off + crop_h, w_off: w_off + crop_w]
                label = label_pad[h_off: h_off + crop_h, w_off: w_off + crop_w]

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))
        # img = img / 255.0

        label = np.array(label, dtype=np.float32)
        # print(self.labels[idx], np.unique(label, return_counts=True))
        label = label
        
        assert len(
            img.shape) == 3, f"Image broken ({img.shape}): {self.root_dir, self.imgs[idx]}"
        assert len(
            label.shape) == 2, f"Label broken ({label.shape}): {self.root_dir, self.labels[idx]}"

        # multi object
        if self.multi_object:
            if self.multi_object not in ['all', 'single_id']:
                raise NotImplementedError

            unique_labels = [l for l in np.unique(label)
                            #  if l != 0.0 and l != 1.0]
                             if l != 0.0]
            
            if unique_labels:
                # all objects stacked in third axis
                label = np.concatenate([np.expand_dims((label == l).astype(np.float32), axis=2)
                                        for l in unique_labels], axis=2)

                # single object from stack
                # if only one object on the frame this object is selected
                if self.multi_object == 'single_id':
                    # if a frame does not include all objects
                    assert self.multi_object_id < self.num_objects
                    if self.num_objects > len(unique_labels) and self.multi_object_id >= len(unique_labels):
                        label = np.zeros((label.shape[0], label.shape[1]), dtype=np.float32)
                    else:
                        label = label[:, :, self.multi_object_id]
        else:
            label = np.where(label != 0.0, 1.0, 0.0).astype(np.float32)
        # label = np.where(ignore_label_mask, self.ignore_label, label).astype(np.float32)

        return img, label
