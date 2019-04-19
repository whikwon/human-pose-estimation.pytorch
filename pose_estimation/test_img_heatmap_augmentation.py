import os

import torchvision.transforms as transforms
import torch

from core.config import update_config
import _init_paths

import matplotlib.pyplot as plt


update_config('./experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3_medi.yaml')
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
train_dataset = eval('dataset.'+ config.DATASET.DATASET)(
    config,
    transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]),
    True)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.TRAIN.BATCH_SIZE,
    shuffle=config.TRAIN.SHUFFLE,
    num_workers=config.WORKERS,
    pin_memory=True)

cnt = 0
save_dir = '../debug'

os.makedirs(save_dir, exists_ok=True)

for sample in train_loader:
    imgs = sample[0]
    heatmaps = sample[1]

    for i in range(config.DATASET.BATCH_SIZE):
        img = imgs[i][0]
        heatmap = heatmaps[i][0]

        plt.imsave(os.path.join(save_dir, f'{cnt}_img.png', img))
        plt.imsave(os.path.join(save_dir, f'{cnt}_heatmap.png', heatmap))

        cnt += 1
