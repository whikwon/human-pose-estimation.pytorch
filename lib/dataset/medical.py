import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import *


logger = logging.getLogger(__name__)

def load_dataset(data_root, data_list):
    img_path_bbox_coords = []
    data_list = os.path.join(data_root, data_list)

    with open(data_list, 'r') as f:
        lines = f.read().split()
        lines = [i.split(',') for i in lines]
    img_path_bbox_coords.extend(lines)

    # adjust file path
    for img_path_bbox_coord in img_path_bbox_coords:
        img_path_bbox_coord[0] = os.path.join(data_root,
            img_path_bbox_coord[0][8:])
    return img_path_bbox_coords


class MedicalDataset(Dataset):

    def __init__(self, config, transforms=None, is_train=True):
        self.config = config
        self.transforms = transforms
        self.is_train = is_train
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
        dx1 = min(x1, np.random.randint(20, w//2))
        dx2 = min(416 - x2, np.random.randint(20, w//2))
        dy1 = min(y1, np.random.randint(20, h//2))
        dy2 = min(416 - y2, np.random.randint(20, h//2))

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
        img = np.array(Image.open(img_path))

        cx, cy, w, h = int(float(cx)), int(float(cy)), int(float(w)), int(float(h))
        kx, ky = int(kx), int(ky)
        x1, y1, x2, y2 = max(cx-w//2, 0), max(cy-h//2, 0), min(cx+w//2, 416), min(cy+h//2, 416)
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
            heatmap[resized_new_ky_bbox-2: resized_new_ky_bbox+3,
                    resized_new_kx_bbox-2: resized_new_kx_bbox+3] = gaussian

            resized_new_bbox = cv2.resize(new_bbox, (192, 256))
            return resized_new_bbox, heatmap

        resized_new_bbox, heatmap = _make_resized_img_heatmap(new_bbox, new_kx_bbox, new_ky_bbox)
        resized_new_bbox = Image.fromarray(resized_new_bbox)
        heatmap = heatmap.reshape(1, 64, 48)

        if self.is_train:
            # augmentation to keypoint
            pass

        if self.transforms is not None:
            resized_new_bbox = self.transforms(resized_new_bbox)


        sample = {'imgs': resized_new_bbox, 'heatmap': heatmap}
        return sample


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

    def __call__(self, img, label):
        # xywh to x1y1x2y2
        x, y, w, h = label
        x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
        assert x2 > x1, f"ValueError: {x2} shoulde be larger than {x1}"
        assert y2 > y1, f"ValueError: {y2} shoulde be larger than {y1}"

        img = np.stack([img]*3, axis=-1)

        bbs = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)],
            shape=img.shape)

        seq = iaa.Sequential([
            iaa.Affine(
                scale=self.config.scale,
                translate_px=self.config.translate_px,),
            iaa.Fliplr(p=self.config.flip_prob),
            iaa.ContrastNormalization(self.config.contrastnorm, per_channel=0.5),
            iaa.Sharpen(alpha=self.config.sharpen_alpha,
                lightness=self.config.sharpen_lightness)
            ])

        seq_det = seq.to_deterministic()

        img_aug = seq_det.augment_images([img])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        # get new coords
        x1 = max(bbs_aug.bounding_boxes[0].x1, 0)
        y1 = max(bbs_aug.bounding_boxes[0].y1, 0)
        x2 = max(bbs_aug.bounding_boxes[0].x2, 0)
        y2 = max(bbs_aug.bounding_boxes[0].y2, 0)

        # coordination convert
        x = (x1 + x2)/2
        y = (y1 + y2)/2
        w = x2 - x1
        h = y2 - y1

        label = np.array([x, y, w, h])
        img = img_aug[:, :, 0]

        img = Image.fromarray(img)
        return img, label
