#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import torch
import random
import numpy as np
import numbers
import cv2
import fnmatch
import os
import torch.utils.data as data

'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''

cv2.setNumThreads(0)

class ComposeIndividual(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class CoCompose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input,target = t(input,target)
        return input,target


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()

class Normalize(object):
    """Normalize image to range [0, 1]"""
    def __call__(self, array):
        if torch.max(array)<=1:
            return array

        array = array/255.0

        return array

class RgbToBinary(object):
    """Convert rgb image to binary"""
    def __call__(self, img):
        img = np.mean(img, axis = 2)
        img = img[:, :, np.newaxis]
        img = (img > 177).astype(np.uint8) * 255.0
        return np.float32(img)

class RgbToGrey(object):
    """convert rgb image to gray scale"""
    def __call__(self, img):
        img = np.mean(img, axis = 2)
        img = img[:, :, np.newaxis]
        return np.float32(img)

class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target):
        h1, w1, _ = inputs[0].shape
        h2, w2, _ = inputs[1].shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
        x2 = int(round((w2 - tw) / 2.))
        y2 = int(round((h2 - th) / 2.))

        inputs[0] = inputs[0][y1: y1 + th, x1: x1 + tw]
        inputs[1] = inputs[1][y2: y2 + th, x2: x2 + tw]
        target = target[y1: y1 + th, x1: x1 + tw]
        return inputs,target

class FixScale(object):
    def __init__(self, scale):
        self.scale = scale
    def __call__(self, inputs, target):
        h, w = target.shape[0], target.shape[1]
        inputs[0] = cv2.resize(inputs[0], (np.int16(w*self.scale), np.int16(h*self.scale)), interpolation=cv2.INTER_AREA)
        inputs[1] = cv2.resize(inputs[1], (np.int16(w*self.scale), np.int16(h*self.scale)), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (np.int16(w*self.scale), np.int16(h*self.scale)), interpolation=cv2.INTER_AREA)

        return inputs, target

class CoFixResolution(object):
    def __init__(self, w, h):
        self.width = w
        self.height = h
    def __call__(self, inputs, target):
        inputs[0] = cv2.resize(inputs[0], (np.int16(self.width), np.int16(self.height)), interpolation=cv2.INTER_AREA)
        inputs[1] = cv2.resize(inputs[1], (np.int16(self.width), np.int16(self.height)), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (np.int16(self.width), np.int16(self.height)), interpolation=cv2.INTER_AREA)

        return inputs, target

class FixResolution(object):
    def __init__(self, w, h):
        self.width = w
        self.height = h
    def __call__(self, inputs):
        inputs = cv2.resize(inputs, (np.int16(self.width), np.int16(self.height)), interpolation=cv2.INTER_AREA)

        return inputs

class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs,target):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs,target

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs[0] = inputs[0][y1: y1 + th,x1: x1 + tw]
        inputs[1] = inputs[1][y1: y1 + th,x1: x1 + tw]
        return inputs, target[y1: y1 + th,x1: x1 + tw]


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            inputs[0] = np.copy(np.fliplr(inputs[0]))
            inputs[1] = np.copy(np.fliplr(inputs[1]))
            target = np.copy(np.fliplr(target))
        return inputs,target


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            inputs[0] = np.copy(np.flipud(inputs[0]))
            inputs[1] = np.copy(np.flipud(inputs[1]))
            target = np.copy(np.flipud(target))
        return inputs,target


class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order

    def rotate(self, image, angle, center=None, scale=1.0):
        # grab the dimensions of the image
        (h, w) = image.shape[:2]

        # if the center is None, initialize it as the center of
        # the image
        if center is None:
            center = (w // 2, h // 2)

        # perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        # return the rotated image
        return rotated

    def __call__(self, inputs,target):
        applied_angle = random.uniform(-self.angle,self.angle)
        # inputs[0] = ndimage.interpolation.rotate(inputs[0], applied_angle, reshape=self.reshape, order=self.order)
        # inputs[1] = ndimage.interpolation.rotate(inputs[1], applied_angle, reshape=self.reshape, order=self.order)
        # target = ndimage.interpolation.rotate(target, applied_angle, reshape=self.reshape, order=self.order)
        inputs[0] = self.rotate(inputs[0], applied_angle)
        inputs[1] = self.rotate(inputs[1], applied_angle)
        target = self.rotate(target, applied_angle)
        return inputs,target

class RandomScale(object):
    def __init__(self, scale):
        self.scale = scale
    def __call__(self, inputs, target):
        applied_scale = random.uniform(1.0/self.scale, self.scale)
        h, w = target.shape[0], target.shape[1]

        _w = np.int16(applied_scale * w)
        _h = np.int16(applied_scale * h)

        _w = _w+1 if _w%2!=0 else _w
        _h = _h+1 if _h%2!=0 else _h

        _w = np.min([_w, w])
        _h = np.min([_h, h])

        tmp_inputs0 = cv2.resize(inputs[0], (_w, _h), interpolation=cv2.INTER_AREA)
        tmp_inputs1 = cv2.resize(inputs[1], (_w, _h), interpolation=cv2.INTER_AREA)
        tmp_target = cv2.resize(target, (_w, _h), interpolation=cv2.INTER_AREA)

        canvas0 = np.zeros([h*2, w*2, 3])
        canvas1 = np.zeros([h*2, w*2, 3])
        canvas_target = np.zeros([h*2, w*2, 3])
        canvas0[np.int16(h-_h/2):np.int16(h+_h/2), np.int16(w-_w/2):np.int16(w+_w/2), :] = tmp_inputs0
        canvas1[np.int16(h-_h/2):np.int16(h+_h/2), np.int16(w-_w/2):np.int16(w+_w/2), :] = tmp_inputs1
        canvas_target[np.int16(h-_h/2):np.int16(h+_h/2), np.int16(w-_w/2):np.int16(w+_w/2), :] = tmp_target

        inputs[0] = canvas0[np.int16(h-h/2):np.int16(h+h/2), np.int16(w-w/2):np.int16(w+w/2), :]
        inputs[1] = canvas1[np.int16(h-h/2):np.int16(h+h/2), np.int16(w-w/2):np.int16(w+w/2), :]
        target = canvas_target[np.int16(h-h/2):np.int16(h+h/2), np.int16(w-w/2):np.int16(w+w/2), :]

        return inputs, target

class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def translate(self, image, x, y):
        # define the translation matrix and perform the translation
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # return the translated image
        return shifted

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        # if tw == 0 and th == 0:
        #     return inputs, target
        # # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        # x1,x2,x3,x4 = max(0,tw), min(w+tw,w), max(0,-tw), min(w-tw,w)
        # y1,y2,y3,y4 = max(0,th), min(h+th,h), max(0,-th), min(h-th,h)
        #
        # x1, x2 = max(0, tw), min(w+tw, w)
        # y1, y2 = max(0, th), min(h + th, h)
        #
        # newh = y2-y1+1
        # neww = x2-x1+1
        #
        # inputs[0] = inputs[0][y1:y2,x1:x2]
        # inputs[1] = inputs[1][y1:y2,x1:x2]
        # target = target[y1:y2,x1:x2]

        inputs[0] = self.translate(inputs[0], tw, th)
        inputs[1] = self.translate(inputs[1], tw, th)
        target = self.translate(target, tw, th)

        return inputs, target


class RandomColorWarp(object):
    def __init__(self, mean_range=0, std_range=0):
        self.mean_range = mean_range
        self.std_range = std_range

    def __call__(self, inputs, target):
        random_std = np.random.uniform(-self.std_range, self.std_range, 3)
        random_mean = np.random.uniform(-self.mean_range, self.mean_range, 3)
        random_order = np.random.permutation(3)

        inputs[0] *= (1 + random_std)
        inputs[0] += random_mean

        inputs[1] *= (1 + random_std)
        inputs[1] += random_mean

        inputs[0] = inputs[0][:,:,random_order]
        inputs[1] = inputs[1][:,:,random_order]

        return inputs, target


def split2list(images, split, default_split=0.9):
    if isinstance(split, str):
        id_train = []
        id_val = []

        with open(split) as f:
            for line in f.readlines():
                items = line.split('\t')
                basename = os.path.basename(items[0])
                basename, ext = os.path.splitext(basename)

                if 'Train' in items[3]:
                    id_train.append(basename)
                else:
                    id_val.append(basename)

        split_values = []

        for image in images:
            image = image[1]
            id = image[:-9]
            split_values.append(id in id_train)

        assert(len(images) == len(split_values))
    elif isinstance(split, float):
        split_values = np.random.uniform(0,1,len(images)) < split
    else:
        split_values = np.random.uniform(0,1,len(images)) < default_split
    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    return train_samples, test_samples

def recursive_glob(rootdir='.', pattern='*'):
    """Search recursively for files matching a specified pattern.

    Adapted from http://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
    """

    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))

    return matches

class DataGenerator(data.Dataset):
    def __init__(self, root, path_list, input_transform=None,
                 target_transform=None, co_transform=None, loader=None):

        self.root = root
        self.path_list = path_list
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        if loader is None:
            raise Exception("Loader can not be None.")
        self.loader = loader

    def __getitem__(self, index):
        inputs, target = self.path_list[index]

        try:
            inputs_img, target_img = self.loader(inputs, target)

            if self.co_transform is not None:
                inputs_img, target_img = self.co_transform(inputs_img, target_img)
            if self.input_transform is not None:
                for i in range(len(inputs_img)):
                    inputs_img[i]=self.input_transform(inputs_img[i])
            if self.target_transform is not None:
                target_img = self.target_transform(target_img)
        except:
            print("Loading error: {} {}".format(inputs_img[0], inputs_img[1]))

            inputs, target = self.path_list[0]

            inputs_img, target_img = self.loader(inputs, target)

            if self.co_transform is not None:
                inputs_img, target_img = self.co_transform(inputs_img, target_img)
            if self.input_transform is not None:
                for i in range(len(inputs_img)):
                    inputs_img[i] = self.input_transform(inputs_img[i])
            if self.target_transform is not None:
                target_img = self.target_transform(target_img)

        return inputs_img, target_img

    def __len__(self):
        return len(self.path_list)