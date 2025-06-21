#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import torch
import torch.nn as nn
import numpy as np
import cv2

__all__=('crossEntropyLoss', 'binaryCrossEntropyLogitLoss', 'lovaszLoss')

def crossEntropyLoss(input_mask, target_mask):
    batch_size = input_mask.size(0)
    num_channels = input_mask.size(1)
    
    if num_channels > 1:
        loss = nn.CrossEntropyLoss()
        target_mask = target_mask.type(torch.LongTensor)
    else:
        loss = nn.BCELoss()
    loss = loss(input_mask, target_mask)/batch_size
    return loss

def binaryCrossEntropyLogitLoss(input_mask, target_mask):
    batch_size = input_mask[0].data[0].cpu().numpy().shape[0]
    loss = nn.BCEWithLogitsLoss()
    loss = loss(input_mask, target_mask)/batch_size
    return loss

def lovaszLoss(input_mask, target_mask):
    return True

def iou(pred, target):
    ious = []

    pred = pred.view(-1)
    target = target.view(-1)

    pred = (pred > 0.5).data.cpu()
    target = (target > 0.5).data.cpu()

    # Ignore IoU for background class ("0")
    cls = 1
    pred_inds = np.where(pred == cls)
    target_inds = np.where(target == cls)
    intersection = (pred[target_inds]).long().sum()  # Cast to long to prevent overflows
    union = pred[pred_inds].long().sum() + target[target_inds].long().sum() - intersection
    if union == 0:
        ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
        ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)