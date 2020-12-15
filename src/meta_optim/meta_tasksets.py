import copy
import random

import torch
from data import custom_transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MetaTaskset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_loader: DataLoader, test_loader: DataLoader, meta_loader: DataLoader,
                 random_frame_transform_per_task: bool, random_flip_label: bool,
                 random_no_label: bool, data_cfg: dict, single_obj_seq_mode: str,
                 random_box_coord_perm: bool, random_frame_epsilon: int,
                 random_object_id_sub_group: bool):
        """
        """
        self.train_loader_tmp = train_loader
        self.test_loader_tmp = test_loader
        self.meta_loader_tmp = meta_loader
        self.test_dataset = self.test_loader_tmp.dataset
        self.seqs_names = self.test_dataset.seqs_names
        self.random_frame_transform_per_task = random_frame_transform_per_task
        self.random_flip_label = random_flip_label
        self.random_no_label = random_no_label
        self.data_cfg = data_cfg
        self.single_obj_seq_mode = single_obj_seq_mode
        self.random_box_coord_perm = random_box_coord_perm
        self.random_frame_epsilon = random_frame_epsilon
        self.random_object_id_sub_group = random_object_id_sub_group

        self.object_groups = []

        self.single_obj_seqs = []
        for seq_name in self.seqs_names:
            self.test_dataset.set_seq(seq_name)

            if self.test_dataset.num_objects == 1:
                if self.single_obj_seq_mode == 'IGNORE':
                    continue
            else:
                if self.single_obj_seq_mode == 'ONLY':
                    continue

                self.single_obj_seqs.append(seq_name)

            for i in range(self.test_dataset.num_object_groups):
                self.object_groups.append((seq_name, i))

    def __len__(self):
        return len(self.object_groups)

    def __getitem__(self, idx):
        seq_name, obj_id = self.object_groups[idx]

        self.test_dataset.set_seq(seq_name)

        num_objects = self.test_dataset.num_objects

        train_loader = copy.deepcopy(self.train_loader_tmp)
        meta_loader = copy.deepcopy(self.meta_loader_tmp)

        train_loader.dataset.set_seq(seq_name)
        meta_loader.dataset.set_seq(seq_name)

        train_loader.dataset.multi_object_id = obj_id
        meta_loader.dataset.multi_object_id = obj_id

        if self.random_object_id_sub_group:
            sub_group_size = torch.randint(1, train_loader.dataset.num_objects_in_group + 1, (1,)).item()
            sub_group_ids = sorted([p.item()
                                    for p in torch.randperm(train_loader.dataset.num_objects_in_group)[:sub_group_size]])

            train_loader.dataset.sub_group_ids = sub_group_ids
            meta_loader.dataset.sub_group_ids = sub_group_ids

        single_augment = self.single_obj_seq_mode == 'AUGMENT_ALL' or (num_objects == 1 and self.single_obj_seq_mode == 'AUGMENT_SINGLE')
        if single_augment:
            assert self.data_cfg['batch_sizes']['meta'] == 1

            single_obj_seqs_ids = list(range(len(self.single_obj_seqs)))
            random_other_single_obj_seq = self.single_obj_seqs[random.choice(single_obj_seqs_ids)]

            train_loader_dataset = copy.deepcopy(self.train_loader_tmp).dataset
            meta_loader_dataset = copy.deepcopy(self.meta_loader_tmp).dataset

            train_loader_dataset.set_seq(random_other_single_obj_seq)
            meta_loader_dataset.set_seq(random_other_single_obj_seq)

            train_loader_dataset.multi_object_id = 0
            meta_loader_dataset.multi_object_id = 0

            train_loader.dataset.augment_with_single_obj_seq_dataset = train_loader_dataset
            meta_loader.dataset.augment_with_single_obj_seq_dataset = meta_loader_dataset

        train_loader.dataset.set_random_frame_id_with_label()

        if self.random_frame_epsilon is not None:
            meta_loader.dataset.random_frame_id_epsilon = self.random_frame_epsilon
            meta_loader.dataset.random_frame_id_anchor_frame = train_loader.dataset.frame_id

        meta_frame_ids = [meta_loader.dataset.get_random_frame_id_with_label()
                            for _ in range(self.data_cfg['batch_sizes']['meta'])]

        meta_loader.sampler.indices = meta_frame_ids

        if self.random_frame_transform_per_task:
            scales = (.5, 1.0)
            color_transform = custom_transforms.ColorJitter(brightness=.2,
                                                            contrast=.2,
                                                            hue=.1,
                                                            saturation=.2,
                                                            deterministic=True)
            flip_transform = custom_transforms.RandomHorizontalFlip(deterministic=True)
            scale_rotate_transform = custom_transforms.RandomScaleNRotate(
                rots=(-30, 30), scales=scales, deterministic=True)

            random_transform = [color_transform,
                                flip_transform,
                                scale_rotate_transform,
                                custom_transforms.ToTensor(),]

            train_loader.dataset.transform = transforms.Compose(random_transform)

            random_transform = [color_transform,
                                flip_transform,
                                scale_rotate_transform,
                                custom_transforms.ToTensor(),]

            meta_loader.dataset.transform = transforms.Compose(random_transform)

            # no random tran transform during meta training
            if self.data_cfg['random_train_transform']:
                raise NotImplementedError

        if self.random_flip_label:
            flip_label = bool(random.getrandbits(1))
            train_loader.dataset.flip_label = flip_label
            meta_loader.dataset.flip_label = flip_label

        if self.random_no_label:
            no_label = bool(random.getrandbits(1))
            train_loader.dataset.no_label = no_label
            meta_loader.dataset.no_label = no_label

        box_coord_perm = None
        if self.random_box_coord_perm:
            box_coord_perm = torch.randperm(4)

        return {'seq_name': seq_name,
                'box_coord_perm': box_coord_perm,
                'train_loader': train_loader,
                'meta_loader': meta_loader}
