# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os
import math

import numpy as np
import torch
import cv2

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input.cuda())
        target = target.cuda()
        target_weight = target_weight.cuda()

        loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
#            save_debug_images(config, input, meta, target, pred*4, output,
#                              prefix)


def validate(config, val_loader, model, criterion, output_dir,
             tb_log_dir, epoch, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_loader)
#    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
#                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight) in enumerate(val_loader):
            # compute output
            output = model(input.cuda())
#            if config.TEST.FLIP_TEST:
#                # this part is ugly, because pytorch has not supported negative index
#                # input_flipped = model(input[:, :, :, ::-1])
#                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
#                input_flipped = torch.from_numpy(input_flipped).cuda()
#                output_flipped = model(input_flipped)
#                output_flipped = flip_back(output_flipped.cpu().numpy(),
#                                           val_dataset.flip_pairs)
#                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
#
#                # feature is not aligned, shift flipped heatmap for higher accuracy
#                if config.TEST.SHIFT_HEATMAP:
#                    output_flipped[:, :, :, 1:] = \
#                        output_flipped.clone()[:, :, :, 0:-1]
#                    # output_flipped[:, :, :, 0] = 0
#
#                output = (output + output_flipped) * 0.5

            target = target.cuda()
            target_weight = target_weight.cuda()

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

#            c = meta['center'].numpy()
#            s = meta['scale'].numpy()
#            score = meta['score'].numpy()

#            preds, maxvals = get_final_preds(
#                config, output.clone().cpu().numpy(), c, s)

#            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
#            all_preds[idx:idx + num_images, :, 2:3] = maxvals
#            # double check this all_boxes parts
#            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
#            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
#            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
#            all_boxes[idx:idx + num_images, 5] = score
#            image_path.extend(meta['image'])
#            if config.DATASET.DATASET == 'posetrack':
#                filenames.extend(meta['filename'])
#                imgnums.extend(meta['imgnum'].numpy())

            idx += num_images

        msg = 'Test: [{0}/{1}]\t' \
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
              'Loss {loss.avg:.4f}\t' \
              'Accuracy {acc.avg:.3f}'.format(
                  i, len(val_loader), batch_time=batch_time,
                  loss=losses, acc=acc)
        logger.info(msg)

        save_interm_result(input, output, target, output_dir, epoch)
#            save_debug_images(config, input, target, pred*4, output,
#                              prefix)

#        name_values, perf_indicator = val_dataset.evaluate(
#            config, all_preds, output_dir, all_boxes, image_path,
#            filenames, imgnums)

#        _, full_arch_name = get_model_name(config)
#        if isinstance(name_values, list):
#            for name_value in name_values:
#                _print_name_value(name_value, full_arch_name)
#        else:
#            _print_name_value(name_values, full_arch_name)

#        if writer_dict:
#            writer = writer_dict['writer']
#            global_steps = writer_dict['valid_global_steps']
#            writer.add_scalar('valid_loss', losses.avg, global_steps)
#            writer.add_scalar('valid_acc', acc.avg, global_steps)
#            if isinstance(name_values, list):
#                for name_value in name_values:
#                    writer.add_scalars('valid', dict(name_value), global_steps)
#            else:
#                writer.add_scalars('valid', dict(name_values), global_steps)
#            writer_dict['valid_global_steps'] = global_steps + 1


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def save_interm_result(imgs, outputs, targets, save_dir, epoch):
    """Save intermediate result during training

    Args:
        imgs (torch.Tensor): [batch_size, n_channel, width, height]
        targets (torch.Tensor): [batch_size, n_bbox, 5]
        save_dir (str): directory saving intermediate result
        curr_epoch (int): current epoch number
    """
    def _denormalize_img(img):
        return (img*127.5) + 127.5

    num_imgs = len(imgs)
    img_size = imgs.shape[-2], imgs.shape[-1]
    output_size = outputs.shape[-2], outputs.shape[-1]

    num_rows = math.ceil(num_imgs**0.5)
    num_cols = round(num_imgs**0.5)

    img_mask = np.zeros([img_size[0]*num_rows, img_size[1]*num_cols])
    output_mask = np.zeros([output_size[0]*num_rows, output_size[1]*num_cols])
    target_mask = np.zeros([output_size[0]*num_rows, output_size[1]*num_cols])

    imgs_arr = imgs.cpu().data.numpy()[:, 0, :, :]
    targets_arr = targets.cpu().data.numpy()[:, 0, :, :]
    outputs_arr = outputs.cpu().data.numpy()[:, 0, :, :]

    for i, (img, target, output) in enumerate(zip(imgs_arr, targets_arr, outputs_arr)):
            row = i // num_cols
            col = i % num_cols
            img_mask[row*img_size[0]: (row+1)*img_size[0], col*img_size[1]:(col+1)*img_size[1]] = img.copy()
            output_mask[row*output_size[0]: (row+1)*output_size[0], col*output_size[1]:(col+1)*output_size[1]] = output.copy()
            target_mask[row*output_size[0]: (row+1)*output_size[0], col*output_size[1]:(col+1)*output_size[1]] = target.copy()


    for i, mask in enumerate([img_mask, output_mask, target_mask]):
        save_path = os.path.join(save_dir, f'{epoch}_{i}.jpg')
        cv2.imwrite(save_path, _denormalize_img(mask))
    return img_mask, output_mask, target_mask
