#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import torch.utils.data as data
import os
import os.path
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from marvin_image_segmentation_dl_model.util.data_processing.common import split2list, recursive_glob

cv2.setNumThreads(0)

def marvin_loader(path_imgs, name):
    img_obj = cv2.imread(path_imgs[0])
    img_bk = cv2.imread(path_imgs[1])
    return [np.float32(img_obj), np.float32(img_bk)], name

def marvin(dir, is_train=True, split=None):
    images = []

    dir_img = os.path.join(dir, 'all_manual_images')
    dir_bk = os.path.join(dir, 'all_manual_backgrounds')

    img_paths = [f for f in listdir(dir_img) if isfile(join(dir_img, f))]

    for path in img_paths:
        basename = os.path.basename(path)
        path_img = os.path.join(dir_img, basename)
        path_bk = os.path.join(dir_bk, basename[:-4] + '_background' + basename[-4:])

        images.append([[path_img, path_bk], basename])

    if is_train:
        return split2list(images, split, default_split=0.97)
    else:
        return images
