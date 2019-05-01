import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import *
import imgaug as ia
from imgaug import augmenters as iaa


logger = logging.getLogger(__name__)

def load_dataset(data_root, data_list):
    img_path_bbox_coords = []
    data_list = os.path.join(data_root, 'annotations', data_list)

    with open(data_list, 'r') as f:
        lines = f.read().split()

    for l in lines:
        img_name, c_x, c_y, w, h, k_x, k_y = l.split(',')
        c_x = float(c_x)
        c_y = float(c_y)
        w = float(w)
        h = float(h)
        k_x = float(k_x)
        k_y = float(k_y)
        img_path = os.path.join(data_root, 'images', img_name)

        img_path_bbox_coords.append([img_path, c_x, c_y, w, h, k_x, k_y])

    return img_path_bbox_coords


class MedicalDataset(Dataset):

    def __init__(self, config, transforms=None, is_train=True,
            img_size=(512, 512)):
        self.config = config
        self.transforms = transforms
        self.is_train = is_train
        self.w, self.h = img_size
        if is_train:
            self.path_bbox_keypoint = load_dataset(config.DATASET.ROOT,
                config.DATASET.TRAIN_SET)
            self.img_augmentor = ImgAugmentation(config)
        else:
            self.path_bbox_keypoint = load_dataset(config.DATASET.ROOT,
                config.DATASET.TEST_SET)

    def __len__(self):
        return len(self.path_bbox_keypoint)

    def _add_boundary_noise(self, x1, y1, x2, y2, w, h, kx_bbox, ky_bbox):
        if 20 < w//2:
            randint_w = (20, w//2)
        elif 20 == w//2:
            randint_w = (19, 20)
        else:
            randint_w = (w//2, 20)

        if 20 < h//2:
            randint_h = (20, h//2)
        elif 20 == h//2:
            randint_h = (19, 20)
        else:
            randint_h = (h//2, 20)

        dx1 = min(x1, np.random.randint(*randint_w))
        dx2 = min(self.w - x2, np.random.randint(*randint_w))
        dy1 = min(y1, np.random.randint(*randint_h))
        dy2 = min(self.h - y2, np.random.randint(*randint_h))

        new_w = w + dx1 + dx2
        new_h = h + dy1 + dy2
        new_x1 = x1 - dx1
        new_x2 = x2 + dx2
        new_kx_bbox = kx_bbox + dx1

        new_y1 = y1 - dy1
        new_y2 = y2 + dy2
        new_ky_bbox = ky_bbox + dy1
        return new_x1, new_x2, new_y1, new_y2, new_w, new_h, new_kx_bbox, new_ky_bbox

    def __getitem__(self, idx):
        img_path, cx, cy, w, h, kx, ky = self.path_bbox_keypoint[idx]
        img = np.array(Image.open(img_path))[:, :, 0]

        cx, cy, w, h = int(float(cx)), int(float(cy)), int(float(w)), int(float(h))
        kx, ky = int(kx), int(ky)
        x1, y1, x2, y2 = max(cx-w//2, 0), max(cy-h//2, 0), min(cx+w//2, self.w), min(cy+h//2, self.h)
        kx_bbox, ky_bbox = kx - x1, ky - y1  # keypoint in the bbox

        new_x1, new_x2, new_y1, new_y2, \
            new_w, new_h, new_kx_bbox, new_ky_bbox = self._add_boundary_noise(x1, y1, x2, y2, w, h, kx_bbox, ky_bbox)

        new_bbox = img[new_y1: new_y2, new_x1: new_x2]

        def _make_resized_img_heatmap(new_bbox, new_kx_bbox, new_ky_bbox):
            h, w = new_bbox.shape

            heatmap = np.zeros([64, 48])
            ratio_w = 48 / w
            ratio_h = 64 / h

            resized_new_kx_bbox = int(new_kx_bbox * ratio_w)
            resized_new_ky_bbox = int(new_ky_bbox * ratio_h)

            gaussian = make_gaussian([5, 5], 2)

            # heatmap coords adjusting gaussian
            hx1 = max(0, resized_new_kx_bbox - 2)
            hx2 = resized_new_kx_bbox + 3
            hy1 = max(0, resized_new_ky_bbox - 2)
            hy2 = resized_new_ky_bbox + 3

            # gaussian adjusting coords
            gx = max(2 - resized_new_kx_bbox, 0)
            gy = max(2 - resized_new_ky_bbox, 0)

            try:
                heatmap[hy1: hy2, hx1: hx2] = gaussian[gy:, gx:]
            except Exception as e:
                logger.info('Coords exceeded boundary [{}, {}, {}, {}, {}, {}]'.format(hx1,
                    hx2, hy1, hy2, gx, gy))
            resized_new_bbox = cv2.resize(new_bbox, (192, 256))
            return resized_new_bbox, heatmap

        resized_new_bbox, heatmap = _make_resized_img_heatmap(new_bbox, new_kx_bbox, new_ky_bbox)
        resized_new_bbox = Image.fromarray(resized_new_bbox)
        target_weight = np.ones([1, 1])

        if self.is_train:
            # augmentation to keypoint
            self.img_augmentor.to_deterministic()
            resized_new_bbox = self.img_augmentor(resized_new_bbox)
            heatmap = self.img_augmentor(heatmap, True)

        heatmap = heatmap.reshape(1, 64, 48).copy()

        if self.transforms is not None:
            resized_new_bbox = self.transforms(resized_new_bbox)

        return [resized_new_bbox, heatmap, target_weight]


def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)


class ImgAugmentation(object):
    """Overall Image Augmentation using imgaug framework"""
    def __init__(self, config):
        self.config = config
        self.seq = iaa.Sequential([
            iaa.Affine(
                scale=self.config.DATASET.SCALE,
                translate_percent=self.config.DATASET.TRANSLATE_PER,
                rotate=self.config.DATASET.ROTATE),
            iaa.Fliplr(p=self.config.DATASET.FLIP_PROB),
            iaa.ContrastNormalization(self.config.DATASET.CONTRASTNORM, per_channel=0.5),
            iaa.Sharpen(alpha=self.config.DATASET.SHARPEN_ALPHA,
                lightness=self.config.DATASET.SHARPEN_LIGHT)
            ])

        self.seq_det = self.seq.to_deterministic()

    def __call__(self, img, heatmap=False):
        # xywh to x1y1x2y2
        img = np.stack([img]*3, axis=-1)
        if heatmap:
            img_aug = img.copy()
            for aug in self.seq_det[:2]:
                img_aug = aug.augment_images([img_aug])[0]
#            img_aug = self.seq_det.augment_images([img])[0]
#            img_aug = iaa.Sequential(self.seq_det[:2]).augment_images([img])[0]
            return img_aug[:, :, 0]
        img_aug = self.seq_det.augment_images([img])[0]
        img = img_aug[:, :, 0]
        img = Image.fromarray(img)
        return img

    def to_deterministic(self):
        self.seq_det = self.seq.to_deterministic()
