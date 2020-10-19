import random
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image


class RandomScaleNRotate:
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25), deterministic=False):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales
        self.deterministic = deterministic
        self.deterministic_rot_sc = {}

    def _get_rot_and_sc(self):
        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        return rot, sc

    def _rot_and_sc(self, tmp, rot, sc, label=True):
        h, w = tmp.shape[:2]
        center = (w / 2, h / 2)
        assert(center != 0)  # Strange behaviour warpAffine
        M = cv2.getRotationMatrix2D(center, rot, sc)

        if label:
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC
        tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)
        return tmp

    def __call__(self, sample):
        still_has_object = False

        num_labels = len(np.unique(sample['gt']))
        while not still_has_object:
            if sample['file_name'] in self.deterministic_rot_sc:
                rot, sc = self.deterministic_rot_sc[sample['file_name']]['rot'], \
                    self.deterministic_rot_sc[sample['file_name']]['sc']
            else:
                rot, sc = self._get_rot_and_sc()

            aug_label = self._rot_and_sc(sample['gt'], rot, sc)

            # never had an object
            if not num_labels > 1:
                break

            still_has_object = len(np.unique(aug_label)) == num_labels

            if sample['file_name'] in self.deterministic_rot_sc:

                if not still_has_object:
                    import imageio
                    imageio.imsave("aug_img.png", (self._rot_and_sc(sample['image'], rot, sc, False) * 255).astype(np.uint8))
                    imageio.imsave("aug_label.png", (aug_label * 255).astype(np.uint8))

                assert still_has_object

        sample['gt'] = aug_label
        sample['image'] = self._rot_and_sc(sample['image'], rot, sc, False)

        if self.deterministic:
            self.deterministic_rot_sc[sample['file_name']] = {}
            self.deterministic_rot_sc[sample['file_name']]['rot'] = rot
            self.deterministic_rot_sc[sample['file_name']]['sc'] = sc

        return sample


class Resize:
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, scales=[0.5, 0.8, 1]):
        self.scales = scales

    def __call__(self, sample):

        # Fixed range of scales
        sc = self.scales[random.randint(0, len(self.scales) - 1)]

        for k in sample.keys():
            if 'file_name' in k:
                continue
            tmp = sample[k]

            if tmp.ndim == 2:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC

            tmp = cv2.resize(tmp, None, fx=sc, fy=sc, interpolation=flagval)

            sample[k] = tmp

        return sample


class ColorJitter:
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, deterministic=False):
        self.transform = torchvision.transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        self._deterministic = deterministic

    def __call__(self, sample):
        if self._deterministic:
            self.transform = self.transform.get_params(
                self.transform.brightness,
                self.transform.contrast,
                self.transform.saturation,
                self.transform.hue)
            self._deterministic = False

        # import imageio
        # imageio.imsave("img_pre.png",
        #                (sample['image'] * 255).astype(np.uint8))

        pil_image = Image.fromarray(np.uint8(sample['image'] * 255))
        sample['image'] = np.array(self.transform(
            pil_image), dtype=np.float32) / 255

        # imageio.imsave("img_after.png",
        #                (sample['image'] * 255).astype(np.uint8))

        return sample


# from .pyLucid import lucidDream, patchPaint


# class LucidDream:

#     # def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, deterministic=False):

#     def __call__(self, sample):

#         pil_image = (sample['image'] * 255).astype(np.uint8)
#         # Image.fromarray(np.uint8(sample['gt']))
#         pil_gt = sample['gt'].astype(np.uint8)

#         bg = patchPaint.paint(pil_image, pil_gt, False)
#         im_1, gt_1, _ = lucidDream.dreamData(
#             pil_image, pil_gt, bg, False)

#         sample['image'] = im_1.astype(np.float32) / 255.0
#         sample['gt'] = gt_1.astype(np.float32)

#         # import imageio
#         # imageio.imsave("img_after.png",
#         #                (sample['image'] * 255).astype(np.uint8))
#         # imageio.imsave("img_bg.png",
#         #                (bg).astype(np.uint8))
#         # imageio.imsave("label_after.png",
#         #                (sample['gt'] * 255).astype(np.uint8))
#         # exit()

#         return sample


class RandomHorizontalFlip:
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __init__(self, deterministic=False):
        self.deterministic = deterministic

        if deterministic:
            self.do_flip = random.random() < 0.5

    def __call__(self, sample):
        if self.deterministic:
            do_flip = self.do_flip
        else:
            do_flip = random.random() < 0.5

        if do_flip:
            for k in sample.keys():
                if 'file_name' in k:
                    continue
                tmp = sample[k]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[k] = tmp

        return sample


class RandomRemoveLabelRectangle:

    def __init__(self, size, deterministic=False):
        self.deterministic = deterministic
        self._size = size
        self._random_square = None

    def _get_random_square(self, label):
        h, w = label.shape[:2]
        th, tw = self._size

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return i, j, th, tw

    def __call__(self, sample):
        if self.deterministic:
            if self._random_square is None:
                self._random_square = self._get_random_square(sample['gt'])
            random_square = self._random_square
        else:
            random_square = self._get_random_square(sample['gt'])

        i, j, h, w = random_square

        # print(sample['gt'].sum())
        sample['gt'][i:i + h, j:j + w] = 0.0
        # print(sample['gt'].sum())

        # pred = sample['gt'].astype(np.uint8)
        # import os, imageio
        # pred_path = os.path.join(f"{sample['file_name']}.png")
        # imageio.imsave(pred_path, 20 * pred)
        # exit()
        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for k in sample.keys():
            if 'file_name' in k:
                continue
            tmp = sample[k]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W

            tmp = tmp.transpose((2, 0, 1))
            sample[k] = torch.from_numpy(tmp)

        return sample
