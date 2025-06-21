#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import torch.utils.data as data
import os
import os.path
import numpy as np
import cv2
from marvin_image_segmentation_dl_model.util.data_processing.common import split2list, recursive_glob

cv2.setNumThreads(0)

def segmentation_10k_loader(path_imgs, path_mask):
    img_original = path_imgs[0]
    img_background = path_imgs[1]
    img_mask = path_mask
    mask = cv2.imread(img_mask)

    if mask is None:
        print(img_mask)

    return [np.float32(cv2.imread(img_original)), np.float32(cv2.imread(img_background))], np.float32(mask)


def segmentation_10k(dir, is_train=True, split=None):
    images = []

    pattern = '*_mask.jpg'
    matches = recursive_glob(dir, pattern)

    for mask in matches:
        mask = os.path.basename(mask)
        root_filename = mask[:-9]
        img_org = root_filename + '_original.jpg'
        img_bg = root_filename + '_background.jpg'

        mask = os.path.join(os.path.join(dir, 'mask'), mask)
        img_org = os.path.join(os.path.join(dir, 'original'), img_org)
        img_bg = os.path.join(os.path.join(dir, 'background'), img_bg)

        if not (os.path.isfile(img_org) or os.path.isfile(img_bg)):
            continue

        images.append([[img_org, img_bg], mask])

    if is_train:
        return split2list(images, '', default_split=0.97)
    else:
        return images
