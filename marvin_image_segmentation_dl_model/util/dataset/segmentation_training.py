#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import torch.utils.data as data
import os
import os.path
import numpy as np
import cv2
from marvin_image_segmentation_dl_model.util.data_processing.common import split2list, recursive_glob

# cv2.setNumThreads(0)

def segmentation_training_loader(path_imgs, path_mask):
    img_obj = cv2.imread(path_imgs[0])
    img_bk = cv2.imread(path_imgs[1])
    img_mask = cv2.imread(path_mask)
    if img_mask is None:
        print(img_mask)

    return [np.float32(img_obj), np.float32(img_bk)], np.float32(img_mask)


def segmentation_training(dir = None, manifest_file = None, is_train=True, split=None):
    if manifest_file is not None:
        images_train = []
        images_test = []
        with open(manifest_file) as fp:

            # errors = []

            for cnt, line in enumerate(fp):
                # print(cnt)
                parts = line.strip().split('\t')

                # for i in range(3):
                #     tmp = cv2.imread(parts[i])
                #     if tmp is None:
                #         errors.append(parts[i])
                #         print("Error reading: {} {}".format(parts[i], parts[3]))


                if parts[3] == 'Train':
                    sample = [[parts[0], parts[1]], parts[2]]
                    images_train.append(sample)
                if parts[3] == 'Test':
                    sample = [[parts[0], parts[1]], parts[2]]
                    images_test.append(sample)

            # for e in errors:
            #     print(e)
        return images_train, images_test
    else:
        images = []
        pattern = '*_mask.png'
        matches = recursive_glob(dir, pattern)

        for mask in matches:
            mask = os.path.basename(mask)
            root_filename = mask[:-9]
            img_org = root_filename + '.jpg'
            img_bg = root_filename + '_background.jpg'

            mask = os.path.join(os.path.join(dir, 'masks'), mask)
            img_org = os.path.join(os.path.join(dir, 'original'), img_org)
            img_bg = os.path.join(os.path.join(dir, 'backgrounds'), img_bg)

            if not (os.path.isfile(img_org) or os.path.isfile(img_bg)):
                continue

            images.append([[img_org, img_bg], mask])

        if is_train:
            return split2list(images, split, default_split=0.97)
        else:
            return images
