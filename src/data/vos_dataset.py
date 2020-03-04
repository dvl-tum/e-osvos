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
                 crop_size=None, transform=None, multi_object=False,
                 flip_label=False, no_label=False, normalize=True,
                 full_resolution=False):
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
        self.flip_label = flip_label
        self.normalize = normalize
        self.no_label = no_label
        self.seqs = None
        self.augment_with_single_obj_seq_key = None
        self._full_resolution = full_resolution
        self.test_mode = False
        self._label_id = None
        self._multi_object_id_to_label = None

        # self.preloaded_imgs = {}
        # self.preloaded_labels = {}

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
        im = Image.open(self.labels[0])
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
        self.imgs = self.seqs[seq_name]['imgs']
        self.labels = self.seqs[seq_name]['labels']
        self.seq_key = seq_name

    def set_gt_frame_id(self):
        self.frame_id = 0

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

        if self.augment_with_single_obj_seq_key is not None:
            assert self.num_objects == 1, f'{self.seq_key} is not a single object sequence.'

            prev_seq_key = self.seq_key
            prev_frame_id = self.frame_id

            self.set_seq(self.augment_with_single_obj_seq_key)

            def crop_center(img, crop_w, crop_h):
                w, h = img.shape[:2]
                start_w = w // 2 - (crop_w // 2)
                start_h = h // 2 - (crop_h // 2)
                return img[start_w:start_w + crop_w, start_h:start_h + crop_h]

            has_object = False
            while not has_object:
                self.set_random_frame_id_with_label()

                aug_img, aug_label = self.make_img_label_pair(self.frame_id)

                w, h, _ = img.shape
                w_a, h_a, _ = aug_img.shape

                pad_w = max(0, w - w_a)
                pad_h = max(0, h - h_a)
                aug_img = np.pad(aug_img,
                    [(0, pad_w), (0, pad_h), (0, 0)], mode='constant')
                aug_label = np.pad(aug_label,
                    [(0, pad_w), (0, pad_h)], mode='constant')

                aug_img = crop_center(aug_img, w, h)
                aug_label = crop_center(aug_label, w, h)

                obj_mask = aug_label == 1.0
                img[obj_mask] = aug_img[obj_mask]

                if not self.multi_object_id:
                    aug_label = np.copy(label)
                    aug_label[obj_mask] = 0

                if len(np.unique(aug_label)) > 1:
                    has_object = True

                    label = aug_label
                    # self.multi_object_id = 0

                    # import imageio
                    # # pred = np.transpose(img, (1, 2, 0))
                    # imageio.imsave(f"{prev_seq_key}_{prev_frame_id}_{self.seq_key}_{self.frame_id}_img.png", (img * 255).astype(np.uint8))
                    # imageio.imsave(
                    #     f"{prev_seq_key}_{prev_frame_id}_{self.seq_key}_{self.frame_id}_label.png", (label * 255).astype(np.uint8))
                    # print('AUGMENT')
                    # exit()

            self.set_seq(prev_seq_key)
            self.frame_id = prev_frame_id

        if self.flip_label:
            label = np.logical_not(label).astype(np.float32)

        if self.no_label:
            label[:] = 0.0

        sample = {'image': img,
                  'gt': label,
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
        # if self.imgs[idx] in self.preloaded_imgs:
        #     return self.preloaded_imgs[self.imgs[idx]], self.preloaded_labels[self.labels[idx]]

        img = cv2.imread(self.imgs[idx], cv2.IMREAD_COLOR)[..., ::-1]

        # load first frame GT as placeholder for test mode
        if self.test_mode:
            if self._label_id is not None:
                im = Image.open(self.labels[self._label_id])
            else:
                im = Image.open(self.labels[0])
        else:
            im = Image.open(self.labels[idx])
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
        if self.normalize:
            img = np.subtract(img, np.array(self.meanval, dtype=np.float32))
        img = img / 255.0

        label = np.array(label, dtype=np.float32)
        # print(self.labels[idx], np.unique(label, return_counts=True))
        label = label

        assert len(
            img.shape) == 3, f"Image broken ({img.shape}): {self.imgs[idx]}"
        assert len(
            label.shape) == 2, f"Label broken ({label.shape}): {self.labels[idx]}"

        # multi object
        if self.multi_object and self.num_objects > 1:
            # multi_object_id = max(self.num_objects - 1, self.multi_object_id)
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
                    # if a frame does not include all objects and in particular not
                    # the object with self.multi_object_id
                    assert self.multi_object_id < self.num_objects, f"{self.seq_key} {self.multi_object_id} {self.num_objects}"
                    # if self.num_objects > len(unique_labels) and self.multi_object_id > len(unique_labels):
                    #     label = np.zeros((label.shape[0], label.shape[1]), dtype=np.float32)

                    multi_object_id = self.multi_object_id + 1.0
                    if self._multi_object_id_to_label is not None:
                        multi_object_id = self._multi_object_id_to_label[self.multi_object_id]

                    if self.augment_with_single_obj_seq_key:
                        multi_object_id = 1.0

                    if multi_object_id in unique_labels:
                        label = label[:, :, unique_labels.index(multi_object_id)]
                    else:
                        label = np.zeros(
                            (label.shape[0], label.shape[1]), dtype=np.float32)
        else:
            label = np.where(label != 0.0, 1.0, 0.0).astype(np.float32)
        # label = np.where(ignore_label_mask, self.ignore_label, label).astype(np.float32)

        # self.preloaded_imgs[self.imgs[idx]] = img
        # self.preloaded_labels[self.labels[idx]] = label

        return img, label
