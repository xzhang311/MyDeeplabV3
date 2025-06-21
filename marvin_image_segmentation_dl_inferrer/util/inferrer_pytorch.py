#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import argparse
import os
import shutil
import time

import cv2

import torch
import torch.optim
import torch.utils.data
import numpy as np

import torch
import torch.backends.cudnn as cudnn

import marvin_image_segmentation_dl_inferrer.util.inferrer as inferrer
from marvin_image_segmentation_dl_inferrer.util.config_parser_pytorch import ConfigParserPyTorch
from marvin_image_segmentation_dl_inferrer.util.system_helpers import get_total_memory_gb

import marvin_image_segmentation_dl_model.model as models
import marvin_image_segmentation_dl_model.util.data_processing.common as common
from marvin_image_segmentation_dl_model.util.data_processing.data_preparer_inferrer import get_data_loader

import pickle

from collections import OrderedDict #### in order to load from dataparallel saved models

cv2.setNumThreads(0)

class NNInferrerPyTorch(inferrer.NNInferrer):
    def __init__(self, input_path, output_path, config_path, weights_path, net_arch, mask_suffix, use_gpu=False):
        super(NNInferrerPyTorch, self).__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.net_arch = net_arch
        self.mask_suffix = mask_suffix
        # load config from file
        self.config_parser = ConfigParserPyTorch(config_path)
        if not self.config_parser.validate_params():
            raise ValueError('Invalid config file contents')

        self.batch_size = self.get_batch_size()
        self.pretrained_weights_path = weights_path
        self.use_gpu = use_gpu

    def get_batch_size(self):
        mem = get_total_memory_gb()

        batch_size = int(self.config_parser.params['batch_size_large']) if mem > int(self.config_parser.params['mem_threshold_gb']) \
            else int(self.config_parser.params['batch_size_small'])
        print("Using batch_size set to be {}".format(batch_size))
        print("The inferred mem size is {} GB".format(mem))
        return batch_size

    def load_model(self):
        if self.net_arch in models.__dict__:
            network_data = torch.load(self.pretrained_weights_path, map_location=lambda storage, loc: storage)

            new_state_dict = OrderedDict()
            for k, v in network_data['state_dict'].items():
                if k[-7:]=='tracked':
                    continue

                name = k
                new_state_dict[name] = v

            network_data['state_dict'] = new_state_dict

            self.model = models.__dict__[network_data['arch']](data=network_data,
                                                               img_height=self.config_parser.params['image_height'],
                                                               img_width=self.config_parser.params['image_width'])
            if not self.pretrained_weights_path == 'None':
                self.model.load_state_dict(network_data['state_dict'])
            if self.use_gpu is True or self.use_gpu == 'true':
                self.model = self.model.cuda()
                self.model = torch.nn.DataParallel(self.model).cuda()
                cudnn.benchmark = True
            return self.model
        else:
            raise Exception("Specified model does not match pretrained model")

    def infer(self):
        self.model.eval()
        end = time.time()
        stime = time.time()

        for i, (input, name) in enumerate(self.data_loader):
            input_var = torch.autograd.Variable(torch.cat(input, 1), volatile=True)
            print('Start inferring for batch {}'.format(i))
            # compute output
            output = self.model(input_var)
            for j in range(output[0].data.size(0)):
                basename, ext = os.path.splitext(name[j])
                feature = {}
                if self.use_gpu:
                    img = output[0].data[j].cpu().numpy().transpose(1, 2, 0)
                    feature['BSx256x32x40'] = output[1].data[j].cpu().numpy()
                    feature['BSx64x128x160'] = output[2].data[j].cpu().numpy()
                    feature['BSx1x512x640'] = output[3].data[j].cpu().numpy()
                else:
                    img = output[0].data[j].numpy().transpose(1, 2, 0)
                    feature['BSx256x32x40'] = output[1].data[j]
                    feature['BSx64x128x160'] = output[2].data[j]
                    feature['BSx1x512x640'] = output[3].data[j]
                img = img * 255
                file = os.path.join(self.output_path, basename + self.mask_suffix + '.png')
                feature_file = os.path.join(self.output_path, basename + '_features.pkl')
                img = cv2.resize(img[:, :, 0], (self.config_parser.params['image_width'], self.config_parser.params['image_height']), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(file, np.uint8(img))
                pickle.dump(feature, open(feature_file, 'wb'))

            timelast = time.time() - stime
            stime = time.time()
            print('Batch {} complete in: {} sec'.format(i, timelast))
            print('By average, each image takes {} sec'.format(timelast/(1.0*output[0].data.size(0))))
        print('All inference complete')

    def run(self):
        # load input tensor
        input_tensor = common.ComposeIndividual([
            common.FixResolution(int(self.config_parser.params['image_width']),
                                 int(self.config_parser.params['image_height'])),
            common.ArrayToTensor(),
            common.Normalize()
        ])

        self.data_loader = get_data_loader(dataset=self.config_parser.params['dataset'],
                                           dataset_root_folder=self.input_path,
                                           batch_size = self.batch_size,
                                           workers = self.config_parser.params['num_workers'],
                                           input_transform=input_tensor)

        self.load_model()

        self.infer()
