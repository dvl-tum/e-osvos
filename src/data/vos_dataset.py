import os
import random
import numpy as np
import cv2
import torch
from PIL import Image

from torch.utils.data import Dataset


class VOSDataset(Dataset):
    """DAVIS dataset constructed using the PyTorch built-in functionalities"""

    mean_val = None

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
        self._full_resolution = full_resolution
        self.test_mode = False
        self._label_id = None
        self._multi_object_id_to_label = []
        self.augment_with_single_obj_seq_dataset = None
        self.random_frame_id_epsilon = None
        self.random_frame_id_anchor_frame =None
        self._num_objects = None
        self._preload_buffer = []
        self.sub_group_ids = None
        self.all_frames = False
        self.propagate_frame_gt = None

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

        if self._num_objects is None:
            label = np.atleast_3d(Image.open(self.labels[0]))[...,0]

            unique_labels = [l for l in np.unique(label)
                            #  if l != 0.0 and l != 1.0]
                            if l != 0.0]
            self._num_objects = len(unique_labels)

        return self._num_objects

    @property
    def num_object_groups(self):
        if self.multi_object == 'all':
            return 1
        return self.num_objects

    @property
    def object_ids_in_group(self):
        object_ids = list(range(self.num_objects))
        if self.sub_group_ids is not None:
            object_ids = [object_ids[i] for i in self.sub_group_ids]
        return object_ids

    @property
    def num_objects_in_group(self):
        if self.multi_object == 'all':
            return len(self.object_ids_in_group)
        return 1

    @property
    def seqs_names(self):
        return list(self.seqs.keys())

    def set_random_seq(self):
        rnd_key_idx = random.randint(0, self.num_seqs - 1)
        rnd_seq_name = self.seqs_names[rnd_key_idx]

        self.set_seq(rnd_seq_name)
        return rnd_seq_name

    def get_random_frame_id(self):
        if self.random_frame_id_epsilon is not None:
            return torch.randint(max(0, self.random_frame_id_anchor_frame - self.random_frame_id_epsilon),
                                 min(self.random_frame_id_anchor_frame + self.random_frame_id_epsilon + 1, len(self.imgs)),
                                 (1,)).item()
        else:
            return torch.randint(len(self.imgs), (1,)).item()

    def set_random_frame_id(self):
        self.frame_id = self.get_random_frame_id()

    def set_frame_id_with_biggest_label(self):
        num_labels = [np.count_nonzero(self.make_img_label_pair(idx)[1])
                      for idx in range(len(self.imgs))]
        self.frame_id = np.argmax(np.array(num_labels))

    def has_frame_object(self):
        assert self.frame_id is not None
        _, label = self.make_img_label_pair(self.frame_id)

        return len([l for l in np.unique(label) if l != 0.0]) == self.num_objects_in_group

    def get_random_frame_id_with_label(self):
        prev_frame_id = self.frame_id
        def _set_random_frame_id_with_label():
            self.set_random_frame_id()

            if self.augment_with_single_obj_seq_dataset is not None:
                self.augment_with_single_obj_seq_dataset.set_random_frame_id()

            if self.has_frame_object():
                return
            else:
                _set_random_frame_id_with_label()

        _set_random_frame_id_with_label()

        random_frame_id_with_label = self.frame_id
        self.frame_id = prev_frame_id

        return random_frame_id_with_label

    def set_random_frame_id_with_label(self):
        self.frame_id = self.get_random_frame_id_with_label()

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
        self._num_objects = None
        self._preload_buffer = []

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

        if self.flip_label:
            label = np.logical_not(label).astype(np.float32)

        if self.no_label:
            label[:] = 0.0

        if self.propagate_frame_gt is not None:
            label = self.propagate_frame_gt

        sample = {'image': img,
                  'gt': label,
                  'file_name': os.path.splitext(os.path.basename(self.imgs[idx]))[0]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.root_dir, self.imgs[0]))

        return list(img.shape[:2])

    def fill_preload_buffer(self):
        self._preload_buffer = [self.make_img_label_pair(idx)
                                for idx in range(len(self.imgs))]

    def make_img_label_pair(self, idx):
        """
        Make the image-ground-truth pair
        """

        if self._preload_buffer:
            img, label = self._preload_buffer[idx]
        else:
            img_path = self.imgs[idx]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)[..., ::-1]

            if self._label_id is not None:
                label = Image.open(self.labels[self._label_id])
            else:
                if self.test_mode:
                    label = Image.open(self.labels[0])
                else:
                    label_path = self.labels[idx]
                    label = Image.open(label_path)

            label = np.atleast_3d(label)[..., 0]

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

                    crop_with_all_labels = False
                    while not crop_with_all_labels:
                        h_off = random.randint(0, img_h - crop_h)
                        w_off = random.randint(0, img_w - crop_w)

                        img = img_pad[h_off: h_off + crop_h, w_off: w_off + crop_w]
                        label = label_pad[h_off: h_off + crop_h, w_off: w_off + crop_w]

                        crop_with_all_labels = True # len(
                            # np.unique(label)) == num_unique_labels

            img = np.array(img, dtype=np.float32)
            if self.normalize:
                img = np.subtract(img, np.array(self.mean_val, dtype=np.float32))
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

                if self.multi_object == 'all':
                    for i in range(self.num_objects):
                        if i not in self.object_ids_in_group:

                            if self._multi_object_id_to_label:
                                group_mask = label == self._multi_object_id_to_label[i]
                            else:
                                group_mask = label == i + 1

                            label[group_mask] = 0.0

                    if self._multi_object_id_to_label:
                        group_masks = [label == self._multi_object_id_to_label[i]
                                    for i in self.object_ids_in_group]
                    else:
                        group_masks = [label == i + 1 for i in self.object_ids_in_group]

                    for i, m in enumerate(group_masks):
                        label[m] = i + 1

                if unique_labels:
                    # single object from stack
                    # if only one object on the frame this object is selected
                    if self.multi_object == 'single_id':
                        # all objects stacked in third axis
                        label = np.concatenate([np.expand_dims((label == l).astype(np.float32), axis=2)
                                                for l in unique_labels], axis=2)

                        # if a frame does not include all objects and in particular not
                        # the object with self.multi_object_id
                        assert self.multi_object_id < self.num_objects, f"{self.seq_key} {self.multi_object_id} {self.num_objects}"
                        # if self.num_objects > len(unique_labels) and self.multi_object_id > len(unique_labels):
                        #     label = np.zeros((label.shape[0], label.shape[1]), dtype=np.float32)

                        multi_object_id = self.multi_object_id + 1.0
                        if self._multi_object_id_to_label:
                            multi_object_id = self._multi_object_id_to_label[self.multi_object_id]

                        if multi_object_id in unique_labels:
                            label = label[:, :, unique_labels.index(multi_object_id)]
                        else:
                            label = np.zeros(
                                (label.shape[0], label.shape[1]), dtype=np.float32)
            else:
                label = np.where(label != 0.0, 1.0, 0.0).astype(np.float32)

        if self.augment_with_single_obj_seq_dataset is not None:
            assert self.num_objects_in_group == 1, f'{self.seq_key} is not a single object sequence.'

            def _crop_center(img, crop_w, crop_h):
                w, h = img.shape[:2]
                start_w = w // 2 - (crop_w // 2)
                start_h = h // 2 - (crop_h // 2)
                return img[start_w:start_w + crop_w, start_h:start_h + crop_h]

            aug_img, aug_label = self.augment_with_single_obj_seq_dataset.make_img_label_pair(self.augment_with_single_obj_seq_dataset.frame_id)

            w, h, _ = img.shape
            w_a, h_a, _ = aug_img.shape

            pad_w = max(0, w - w_a)
            pad_h = max(0, h - h_a)
            aug_img = np.pad(aug_img, [(0, pad_w), (0, pad_h), (0, 0)], mode='constant')
            aug_label = np.pad(aug_label, [(0, pad_w), (0, pad_h)], mode='constant')

            pad_w = max(0, w_a - w)
            pad_h = max(0, h_a - h)
            img = np.pad(img, [(0, pad_w), (0, pad_h), (0, 0)], mode='constant')
            label = np.pad(label, [(0, pad_w), (0, pad_h)], mode='constant')

            aug_obj_mask = aug_label == 1.0
            obj_mask = label == 1.0

            ###
            has_label = not (np.any(obj_mask) and np.any(aug_obj_mask))
            if has_label:
                label[...] = 0.0

            while not has_label:
                aug_obj_mask_tmp = np.copy(aug_obj_mask)
                aug_img_tmp = np.copy(aug_img)

                aug_obj_mask_x_min = np.min(np.where(aug_obj_mask_tmp)[0])
                aug_obj_mask_x_max = np.max(np.where(aug_obj_mask_tmp)[0]) + 1
                aug_obj_mask_y_min = np.min(np.where(aug_obj_mask_tmp)[1])
                aug_obj_mask_y_max = np.max(np.where(aug_obj_mask_tmp)[1]) + 1

                obj_mask_x_min = np.min(np.where(obj_mask)[0])
                obj_mask_x_max = np.max(np.where(obj_mask)[0]) + 1
                obj_mask_y_min = np.min(np.where(obj_mask)[1])
                obj_mask_y_max = np.max(np.where(obj_mask)[1]) + 1

                aug_obj_box_width = aug_obj_mask_x_max - aug_obj_mask_x_min
                aug_obj_box_crop_width = min(aug_obj_box_width,
                                             obj_mask_x_max - obj_mask_x_min)
                aug_obj_box_crop_width_pad = (aug_obj_box_width - aug_obj_box_crop_width) // 2
                aug_obj_box_height = aug_obj_mask_y_max - aug_obj_mask_y_min
                aug_obj_box_crop_height = min(aug_obj_box_height,
                                             obj_mask_y_max - obj_mask_y_min)
                aug_obj_box_crop_height_pad = (aug_obj_box_height - aug_obj_box_crop_height) // 2

                aug_obj_box = aug_img_tmp[aug_obj_mask_x_min + aug_obj_box_crop_width_pad:aug_obj_mask_x_max - aug_obj_box_crop_width_pad,
                                          aug_obj_mask_y_min + aug_obj_box_crop_height_pad:aug_obj_mask_y_max - aug_obj_box_crop_height_pad]
                aug_obj_mask_box = aug_obj_mask_tmp[aug_obj_mask_x_min + aug_obj_box_crop_width_pad:aug_obj_mask_x_max - aug_obj_box_crop_width_pad,
                                                    aug_obj_mask_y_min + aug_obj_box_crop_height_pad:aug_obj_mask_y_max - aug_obj_box_crop_height_pad].copy()

                random_obj_mask_x = obj_mask_x_min.item() + (obj_mask_x_max.item() - obj_mask_x_min.item()) // 2# + np.random.randint(0, obj_mask_x_max.item() - obj_mask_x_min.item())
                random_obj_mask_y = obj_mask_y_min.item() + (obj_mask_y_max.item() - obj_mask_y_min.item()) // 2# + np.random.randint(0, obj_mask_y_max.item() - obj_mask_y_min.item())

                aug_obj_box = aug_obj_box[0: min(img.shape[0] - random_obj_mask_x, aug_obj_box.shape[0]),
                                          0: min(img.shape[1] - random_obj_mask_y, aug_obj_box.shape[1])]
                aug_obj_mask_box = aug_obj_mask_box[0: min(img.shape[0] - random_obj_mask_x, aug_obj_mask_box.shape[0]),
                                                    0: min(img.shape[1] - random_obj_mask_y, aug_obj_mask_box.shape[1])]

                aug_img_tmp[random_obj_mask_x: random_obj_mask_x + aug_obj_box.shape[0],
                            random_obj_mask_y: random_obj_mask_y + aug_obj_box.shape[1]] = aug_obj_box
                aug_obj_mask_tmp[...] = False
                aug_obj_mask_tmp[random_obj_mask_x: random_obj_mask_x + aug_obj_mask_box.shape[0],
                                 random_obj_mask_y: random_obj_mask_y + aug_obj_mask_box.shape[1]] = aug_obj_mask_box

                aug_label = np.copy(label)
                aug_label[aug_obj_mask_tmp] = 0

                has_label = np.any(aug_label)

                if has_label:
                    aug_obj_mask = aug_obj_mask_tmp
                    aug_img = aug_img_tmp

                    img[aug_obj_mask] = aug_img[aug_obj_mask]
                    label = aug_label

        return img, label
