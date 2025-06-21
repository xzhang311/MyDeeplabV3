#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import os
import shutil
import time
import datetime

import marvin_image_segmentation_dl_model.model as models 
import marvin_image_segmentation_dl_model.util.data_processing.common as common 
from marvin_image_segmentation_dl_trainer.util.config_parser_pytorch import ConfigParserPyTorch

import torch
import torch.backends.cudnn as cudnn

from marvin_image_segmentation_dl_model.util.data_processing.data_preparer_trainer import get_data_loader 

import marvin_image_segmentation_dl_trainer.util.average_meter as avergae_meter
import marvin_image_segmentation_dl_trainer.util.trainer as trainer
import marvin_image_segmentation_dl_trainer.util.config_parser_pytorch as config_parser_pytorch
from marvin_image_segmentation_dl_trainer.util.loss_functions import crossEntropyLoss
import marvin_image_segmentation_dl_trainer.util.loss_functions as loss_func
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from collections import OrderedDict #### in order to load from dataparallel saved models

train_n_iter = 0
test_n_iter = 0

class NNTrainerPyTorch(trainer.NNTrainer):
    def __init__(self,
                 arch=None,
                 dataset_root=None,
                 manifest_file=None,
                 output_model_path=None,
                 config_path=None,
                 split=None,
                 pretrained_path=None,
                 epochs=None,
                 learning_rate=None,
                 decay=None,
                 loss_func=None,
                 optimizer=None,
                 batch_size=None,
                 num_workers=None,
                 image_width=None,
                 image_height=None):
        super(NNTrainerPyTorch, self).__init__()
        self.config_path = config_path
        self.config_parser = ConfigParserPyTorch(self.config_path)
        if not self.config_parser.validate_params():
            raise ValueError('Invalid config file contents')
        self.params = self.config_parser.params

        self.params['dataset_root'] = None
        self.params['manifest_file'] = None
        self.params['n_output_imgs'] = 64

        if arch is not None: self.params['arch']=arch
        if dataset_root is not None: self.params['dataset_root']=dataset_root
        if manifest_file is not None: self.params['manifest_file']=manifest_file
        if split is not None: self.params['split']=split
        if output_model_path is not None: self.params['output_model_path']=output_model_path
        if pretrained_path is not None: self.params['pretrained_path']=pretrained_path
        if epochs is not None: self.params['epochs']=epochs
        if learning_rate is not None: self.params['learning_rate']=learning_rate
        if decay is not None: self.params['decay']=decay
        if loss_func is not None: self.params['loss']=loss_func
        if optimizer is not None: self.params['optimizer']=optimizer
        if batch_size is not None: self.params['batch_size']=batch_size
        if num_workers is not None: self.params['num_workers']=num_workers
        if image_width is not None: self.params['image_width']=image_width
        if image_height is not None: self.params['image_height']=image_height

    def build_savepath(self):
        save_path = 'arch_{}-epochs_{}-lr_{}-decay_{}-optimizer_{}-batchsize_{}-workers'.format(
            self.params['arch'],
            self.params['epochs'],
            self.params['learning_rate'],
            self.params['decay'],
            self.params['optimizer'],
            self.params['batch_size'],
            self.params['num_workers']
        )

        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
        save_path = os.path.join(timestamp, save_path)
        save_path = os.path.join(self.params['output_model_path'], save_path)
        print('=> will save everything to {}'.format(save_path))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        return save_path

    def get_data_transform(self): 
        input_tensor = common.ComposeIndividual([
            common.ArrayToTensor(),
            common.Normalize()
        ])

        target_tensor = common.ComposeIndividual([
            common.RgbToGrey(),
            common.ArrayToTensor(),
            common.Normalize()
        ])

        co_transform_train = common.CoCompose([
            common.CoFixResolution(int(self.params['image_width']), int(self.params['image_height'])),
            common.RandomTranslate(10),
            common.RandomRotate(10),
            common.RandomScale(1.5),
            common.RandomHorizontalFlip()
        ])

        co_transform_validation = common.CoCompose([
            common.CoFixResolution(int(self.params['image_width']), int(self.params['image_height']))
        ])

        return input_tensor, target_tensor, co_transform_train, co_transform_validation

    def load_model(self):
        if self.params['arch'] in models.__dict__:

            self.model = models.__dict__[self.params['arch']](img_height=self.params['image_height'],
                                                               img_width=self.params['image_width'])
            if not self.params['pretrained_path'] == "":
                network_data = torch.load(self.params['pretrained_path'], map_location=lambda storage, loc: storage)

                # new_state_dict = OrderedDict()
                # for k, v in network_data['state_dict'].items():
                #     if k[-7:] == 'tracked':
                #         continue
                #
                #     name = k
                #     new_state_dict[name] = v
                #
                # network_data['state_dict'] = new_state_dict

                self.model.load_state_dict(network_data['state_dict'])

                # new_state_dict = OrderedDict()
                # for k, v in network_data['state_dict'].items():
                #     name = k[7:]  # remove `module.`
                #     new_state_dict[name] = v
                # # load params
                #
                # self.model.load_state_dict(new_state_dict)
                #
                newsavepath = '/mnt/ebs_xizhn/Projects/DeeplabV3/marvin_image_segmentation_dl_trainer/pretrained_weights'
                self.save_checkpoint(newsavepath, {
                    'epoch': 893,
                    'arch': self.params['arch'],
                    'state_dict': network_data['state_dict'],
                    'best_val_iou': network_data['best_val_iou'],
                }, False)

            if self.params['gpu_support']=='Yes':
                self.model = self.model.cuda()
                self.model = torch.nn.DataParallel(self.model).cuda()
                cudnn.benchmark = True
            return self.model
        else:
            raise Exception("Specified model does not match pretrained model")

    def load_optimizer(self, model):
        assert (self.params['optimizer'] in ['adam', 'sgd'])

        param_groups = model.parameters()
        if self.params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(param_groups, self.params['learning_rate'],
                                         betas=(0.9, 0.999), eps=1e-8)
        elif self.params['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(param_groups, self.params['learning_rate'],
                                        momentum=0.9)

        return optimizer

    def save_checkpoint(self, save_path, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, os.path.join(save_path, filename))
        print('Saved checkpoint model to {}'.format(os.path.join(save_path, filename)))
        if is_best: 
            cmd = 'cp' + ' ' + os.path.join(save_path, filename) + ' ' + os.path.join(save_path, 'model_best.pth.tar')
            os.system(cmd)
            print('New best model at {}'.format(os.path.join(save_path, 'model_best.pth.tar')))

    def train(self, train_loader, model, optimizer, epoch, train_writer):
        global train_n_iter

        batch_time = avergae_meter.AverageMeter()
        data_time = avergae_meter.AverageMeter()
        losses = avergae_meter.AverageMeter()
        ious = avergae_meter.AverageMeter()

        epoch_size = len(train_loader) if self.params['epochs'] == 0 else min(len(train_loader), self.params['epochs'])

        # switch to train mode
        model.train()

        end = time.time()

        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if self.params['gpu_support']=='Yes':
                target = target.cuda(async=True)
                input = [j.cuda() for j in input]
            input_var = torch.autograd.Variable(torch.cat(input, 1))
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)

            loss = loss_func.__dict__[self.params['loss']](output[0], target_var)
            iou = loss_func.__dict__['iou'](output[0], target_var)

            # record loss and BCE
            losses.update(loss.data, target.size(0))
            ious.update(iou[0], target.size(0))

            # compute gradient and do optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            train_writer.add_scalar('iteration loss', loss.data, train_n_iter)
            train_writer.add_scalar('iteration IOU', iou[0], train_n_iter)

            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t Batch Process Time {3}\t Data Loading Time{4}\t Loss {5} IOU {6}'
                      .format(epoch, i, epoch_size, batch_time,
                              data_time, losses, ious))

            train_n_iter = train_n_iter + 1

            if i >= epoch_size:
                break

        return losses.avg, ious.avg

    def validate(self, val_loader, model, epoch, test_writer, output_img_writers):
        global test_n_iter

        print_frequency = 10


        batch_time = avergae_meter.AverageMeter()
        losses = avergae_meter.AverageMeter()
        ious = avergae_meter.AverageMeter()

        with torch.no_grad():
            # switch to evaluate mode
            model.eval()

            end = time.time()
            for i, (input, target) in enumerate(val_loader):
                if self.params['gpu_support']=='Yes':
                    target = target.cuda(async=True)
                    input = [j.cuda() for j in input]
                input_var = torch.autograd.Variable(torch.cat(input, 1), volatile=True)
                target_var = torch.autograd.Variable(target, volatile=True)

                # compute output
                output = model(input_var)

                loss = crossEntropyLoss(output[0], target_var)
                iou = loss_func.__dict__['iou'](output[0], target_var)

                # record loss and BCE
                losses.update(loss.data, target.size(0))
                ious.update(iou[0], target.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                test_writer.add_scalar('iteration loss', loss.data, test_n_iter)
                test_writer.add_scalar('iteration IOU', iou[0], test_n_iter)

                if i % print_frequency == 0:
                    print('Test: [{0}/{1}]\t Batch Inference Time {2}\t Loss {3} IOU {4}'
                          .format(i, len(val_loader), batch_time, losses, ious))

                if i < len(output_img_writers):
                    if epoch == 0:
                        gt_img = target[0].data.cpu().numpy().transpose(1, 2, 0)[:, :, 0]
                        gt_img = cv2.resize(gt_img, (512, 512), interpolation=cv2.INTER_AREA)
                        img_container = np.zeros([512, 512, 3])
                        img_container[:, :, 0] = gt_img
                        img_container[:, :, 1] = gt_img
                        img_container[:, :, 2] = gt_img
                        output_img_writers[i].add_image('GT', img_container, 0)

                        obj_img = input[0][0].data.cpu().numpy().transpose(1, 2, 0)
                        obj_img = cv2.resize(obj_img, (512, 512), interpolation=cv2.INTER_AREA)
                        output_img_writers[i].add_image('Obj', obj_img, 0)

                    out_img = output[0][0].data.cpu().numpy().transpose(1, 2, 0)[:, :, 0]
                    out_img = cv2.resize(out_img, (512, 512), interpolation=cv2.INTER_AREA)
                    img_container = np.zeros([512, 512, 3])
                    img_container[:, :, 0] = out_img
                    img_container[:, :, 1] = out_img
                    img_container[:, :, 2] = out_img
                    output_img_writers[i].add_image('Output', img_container, epoch)


                test_n_iter = test_n_iter + 1

        return losses.avg, ious.avg

    def run(self):
        savepath = self.build_savepath()

        train_writer = SummaryWriter(os.path.join(savepath, 'Train'))
        val_writer = SummaryWriter(os.path.join(savepath, 'Validation'))

        output_img_writers = []
        for i in range(self.params['n_output_imgs']):
            output_img_writers.append(SummaryWriter(os.path.join(savepath, 'Imgs', str(i))))

        input_tensor, target_tensor, co_transform_train, co_transform_validation = self.get_data_transform()

        train_loader, val_loader = get_data_loader(dataset = self.params['dataset'],
                                                   dataset_root_folder = self.params['dataset_root'],
                                                   manifest_file = self.params['manifest_file'],
                                                   split = self.params['split'],
                                                   batch_size = self.params['batch_size'],
                                                   workers = self.params['num_workers'],
                                                   input_transform = input_tensor,
                                                   target_transform = target_tensor,
                                                   co_transform_train = co_transform_train,
                                                   co_transform_validation = co_transform_validation)

        model = self.load_model()

        optimizer = self.load_optimizer(model)
        best_val_iou = 0 #initial invalid value
        for epoch in range(self.params['epochs']):
            train_loss, train_iou = self.train(train_loader, model, optimizer, epoch, train_writer)
            val_loss, val_iou = self.validate(val_loader, model, epoch, val_writer, output_img_writers)

            train_writer.add_scalar('epoch loss', train_loss, epoch)
            train_writer.add_scalar('epoch IOU', train_iou, epoch)

            print('Epoch: {}    Train loss: {}'.format(epoch, train_loss))
            print('Epoch: {}    Train IOU: {}'.format(epoch, train_iou))

            val_writer.add_scalar('epoch loss', val_loss, epoch)
            val_writer.add_scalar('epoch IOU', val_iou, epoch)

            print('Epoch: {}    Val loss: {}'.format(epoch, val_loss))
            print('Epoch: {}    Val IOU: {}'.format(epoch, val_iou))

            is_best_val_iou = val_iou > best_val_iou
            best_val_iou = max(val_iou, best_val_iou)

            self.save_checkpoint(savepath,{
                'epoch': epoch + 1,
                'arch': self.params['arch'],
                'state_dict': model.state_dict(),
                'best_val_iou': best_val_iou,
            }, is_best_val_iou)

        return True
