#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import argparse
import marvin_image_segmentation_dl_model.model as model
from marvin_image_segmentation_dl_trainer.util.trainer_pytorch import NNTrainerPyTorch
import os

#asdfasdfsdafqassafasdf

def main():
    model_names = sorted(name for name in model.__dict__
                         if name.islower() and not name.startswith("__"))

    parser = argparse.ArgumentParser(description='Image segmentation deep learning inference tool', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--arch', metavar='ARCH', choices=model_names, help='model architecture' + ' | '.join(model_names), required=True)
    parser.add_argument('--dataset_root', metavar='DATASETROOT', help='root of training data including images and backgrounds', required=False)
    parser.add_argument('--manifest_file', metavar='MANIFEST', help='file that split the data to train and test', required=False)
    parser.add_argument('--output_model_path', metavar='OUTDIR', help='folder of saving output masks', required=True)
    parser.add_argument('--config_path', metavar='CFGFILE', help='Path of config JSON file', required=True)
    parser.add_argument('--split', metavar='SPLIT', default=0.7, help='Manifest file split data to trian/test/val', required=False)
    parser.add_argument('--pretrained_path', metavar='PRETRAINED', default=None, help = 'Path of pretrained weights file', required=False)
    parser.add_argument('--epochs', metavar='EPOCHS', type=int, default=1000, help='epochs', required=False)
    parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.00005, help='learning rate', required=False)
    parser.add_argument('--decay', metavar='DECAY', type=float, default=0, help='decay', required=False)
    parser.add_argument('--loss_func', metavar='LOSSFUNC', type=str, default='crossEntropyLoss', help='loss function', required=False)
    parser.add_argument('--optimizer', metavar='OPTIMIZER', type=str, default='adam', help='optimizer in sgd', required=False)
    parser.add_argument('--batch_size', metavar='BATCHSIZE', type=int, default=2, help='batch size', required=False)
    parser.add_argument('--num_workers', metavar='NWORKERS', type=int, default=16, help='num of workers in dataloader', required=False)
    parser.add_argument('--image_width', metavar='IMAGEWIDTH', type=int, default=640, help='image width', required=False)
    parser.add_argument('--image_height', metavar='IMAGEHEIGHT', type=int, default=512, help='image height', required=False)
    args = parser.parse_args()

    if args.manifest_file is None and args.dataset_root is None:
        raise('Error. manifest_file aned dataset_root can not be both empty.')

    if not os.path.isdir(args.output_model_path):
        os.makedirs(args.output_model_path)

    config_path = ''
    if args.config_path == '':
        env_root = os.environ['APOLLO_ENVIRONMENT_ROOT']
        config_root_path = os.path.join(env_root, 'config_trainer')
        config_path = os.path.join(config_root_path, args.net_arch + '.json')
    else:
        config_path = args.config_path
    print('Using config: ', config_path)

    trainer = NNTrainerPyTorch(arch = args.arch,
                               dataset_root = args.dataset_root,
                               manifest_file = args.manifest_file,
                               output_model_path = args.output_model_path,
                               config_path = args.config_path,
                               split = args.split,
                               pretrained_path = args.pretrained_path,
                               epochs = args.epochs,
                               learning_rate = args.learning_rate,
                               decay = args.decay,
                               loss_func = args.loss_func,
                               optimizer = args.optimizer,
                               batch_size = args.batch_size,
                               num_workers = args.num_workers,
                               image_width = args.image_width,
                               image_height = args.image_height)

    trainer.run()

if __name__ == '__main__':
    main()
