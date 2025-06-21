#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import argparse
from marvin_image_segmentation_dl_inferrer.util.inferrer_pytorch import NNInferrerPyTorch
import marvin_image_segmentation_dl_model.model as model
import os

def main():
    model_names = sorted(name for name in model.__dict__
                         if name.islower() and not name.startswith("__"))

    parser = argparse.ArgumentParser(description='Image segmentation deep learning inference tool', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', metavar='INPUTDIR', help='root of working path including images and backgrounds', required=True)
    parser.add_argument('--output_path', metavar='OUTDIR', help='folder of saving output masks', required=True)
    parser.add_argument('--config_path', metavar='CFGFILE', default='', help= 'Path of config JSON file', required=False)
    parser.add_argument('--weights_path', metavar='BINFILE', default='', help = 'Path of pretrained weights file', required=False)
    parser.add_argument('--net_arch', metavar='ARCH', default='deeplabv3plus_dualinput_resnet50', choices=model_names, help='model architecture, overwritten if pretrained is specified: ' + ' | '.join(model_names))
    parser.add_argument('--use_gpu', metavar='USEGPU', required=True)
    parser.add_argument('--mask_suffix', metavar='SUFFIX', default = '_mask', help='suffix to append to input filenames whne writing a mask. For example, image.png with mask_suffix = _mask will result in image_mask.png')
    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    
    config_path = ''
    if args.config_path == '':
        env_root = os.environ['APOLLO_ENVIRONMENT_ROOT']
        config_root_path = os.path.join(env_root, 'config_inferrer')
        config_path = os.path.join(config_root_path, args.net_arch+'.json')
    else:
        config_path = args.config_path
    print('Using config: ', config_path)

    weights_path = ''
    if args.weights_path == '':
        env_root = os.environ['APOLLO_ENVIRONMENT_ROOT']
        weight_root_path = os.path.join(env_root, 'model')
        weights_path = os.path.join(weight_root_path, args.net_arch)
        weights_path = os.path.join(weights_path, args.net_arch+'.pth')       
    else:
        weights_path = args.weights_path
    print('Using weights: ', weights_path)
    
    #initialize a inferrer
    inferrer = NNInferrerPyTorch(input_path = args.input_path,
                                 output_path = args.output_path,
                                 config_path = config_path,
                                 weights_path = weights_path,
                                 net_arch = args.net_arch,
                                 mask_suffix = args.mask_suffix,
                                 use_gpu = args.use_gpu)

    inferrer.run()


if __name__ == '__main__':
    main()
