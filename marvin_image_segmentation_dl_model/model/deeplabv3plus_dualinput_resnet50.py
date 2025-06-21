#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal

from marvin_image_segmentation_dl_model.util.blocks import conv_bn_relu, conv_relu, deconv_bn_relu, \
    AtrousResNetBlock


class DeepLabV3PlusDualInputResNet50(nn.Module):
    def __init__(self, isASPP, img_height, img_width):
        super(DeepLabV3PlusDualInputResNet50, self).__init__()

        self.isASPP = isASPP

        if self.isASPP:
            multi_grid = (1, 2, 4)
        else:
            multi_grid = (1, 2, 1)

        self.conv_input = conv_bn_relu(6, 64, kernel_size = 7, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.resnset_module = nn.Sequential(
            AtrousResNetBlock(baseplanes=64, inplanes=64, num_units=3, stride_first_block=2),
            AtrousResNetBlock(baseplanes=128, inplanes=256, num_units=4, stride_first_block=2),
            AtrousResNetBlock(baseplanes=256, inplanes=512, num_units=6),
            AtrousResNetBlock(baseplanes=512, inplanes=1024, num_units=3, base_rate=2, multi_grid=multi_grid)
        )

        # for Going deeper model only
        self.going_deeper_module = nn.Sequential(
            AtrousResNetBlock(baseplanes=512, inplanes=2048, num_units=3, base_rate=4, multi_grid=multi_grid),
            AtrousResNetBlock(baseplanes=512, inplanes=2048, num_units=3, base_rate=8, multi_grid=multi_grid),
            AtrousResNetBlock(baseplanes=512, inplanes=2048, num_units=3, base_rate=16, multi_grid=multi_grid)
        )

        # for ASPP
        self.aspp_1x1         = conv_bn_relu(inplanes=2048, outplanes=256, kernel_size=1, stride=1, rate=1)
        self.aspp_3x3_rate_6  = conv_bn_relu(inplanes=2048, outplanes=256, kernel_size=3, stride=1, rate=6)
        self.aspp_3x3_rate_12 = conv_bn_relu(inplanes=2048, outplanes=256, kernel_size=3, stride=1, rate=12)
        self.aspp_3x3_rate_18 = conv_bn_relu(inplanes=2048, outplanes=256, kernel_size=3, stride=1, rate=18)

        # image pooling
        self.img_pooling = nn.Sequential(
            nn.AvgPool2d((img_height // 16, img_width // 16), stride = (1, 1)),
            conv_relu(2048, 256, kernel_size=1, stride=1),
            nn.Upsample((img_height // 16, img_width // 16), mode='bilinear')
        )

        # output layer of deeplabV3
        self.conv_output_aspp = nn.Conv2d(1280, 256, kernel_size=1, stride=1, bias=True)
        self.conv_output_deeper = nn.Conv2d(2048, 1, kernel_size=1, stride=1, bias=True)
        self.tanh_output = nn.Tanh()

        # additional part similar to ones in deeplab V3+
        self.v3p_1x1 = conv_bn_relu(inplanes=2048, outplanes=256, kernel_size=1, stride=1)
        self.v3p_deconv1 = deconv_bn_relu(inplanes=256, outplanes=128, kernel_size = 4, stride = 2)
        self.v3p_deconv2 = deconv_bn_relu(inplanes=128, outplanes=64, kernel_size = 4, stride = 2)
        self.v3p_deconv3 = deconv_bn_relu(inplanes=256, outplanes=128, kernel_size=4, stride=2)
        self.v3p_deconv4 = deconv_bn_relu(inplanes=128, outplanes=64, kernel_size=4, stride=2)
        self.v3p_deconv5 = deconv_bn_relu(inplanes=128, outplanes=64, kernel_size=4, stride=2)
        self.v3p_deconv6 = deconv_bn_relu(inplanes=64, outplanes=32, kernel_size=4, stride=2)

        self.v3p_conv_output = nn.Conv2d(32, 1, kernel_size = 3, stride = 1, padding = 1, bias = True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x): # 6 x H x W
        x1 = self.conv_input(x) # x: batchSize x 6 x 512 x 640, x1: batchSize x 64 x 256 x 320
        x2 = self.maxpool(x1) # x2: batchSize x 64 x 128 x 160
        x3 = self.resnset_module(x2) # x3: batchSize x 2048 x 32 x 40

        if self.isASPP:
            x1_aspp = self.aspp_1x1(x3)
            x2_aspp = self.aspp_3x3_rate_6(x3)
            x3_aspp = self.aspp_3x3_rate_12(x3)
            x4_aspp = self.aspp_3x3_rate_18(x3)
            x5_aspp = self.img_pooling(x3)
            x6_aspp = torch.cat((x1_aspp, x2_aspp, x3_aspp, x4_aspp, x5_aspp), dim = 1)
            x4 = self.conv_output_aspp(x6_aspp) # x4: batchSize x 256 x 32 x 40
        else:
            x1_deeper=self.going_deeper_module(x3)
            x4=self.conv_output_deeper(x1_deeper)

        x5 = self.v3p_1x1(x3) # x5: batchSize x 256 x 32 x 40
        x6 = self.v3p_deconv1(x5) # x6: batchSize x 128 x 64 x 80
        x7 = self.v3p_deconv2(x6) # x7 batchSize x 64 x 128 x 160


        # x4 256 x h x w
        x8 = self.v3p_deconv3(x4) # x8: batchSize x 128 x 64 x 80
        x9 = self.v3p_deconv4(x8) # x9: batchSize x 64 x 128 x 160

        x10 = torch.cat((x7, x9), dim = 1) # x10: batchSize x 128 x 128 x 160

        x11 = self.v3p_deconv5(x10) # x11: batchSize x 64 x 256 x 320
        x12 = self.v3p_deconv6(x11) # x12: batchSize x 32 x 512 x 640

        x13 = self.v3p_conv_output(x12) # x13: batchSize x 1 x 512 x 640

        net_output = (self.tanh_output(x13)+1)/2

        return [net_output, x4, x9, x13]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

def deeplabv3plus_dualinput_resnet50(data=None, img_height=512, img_width=640):
    model=DeepLabV3PlusDualInputResNet50(isASPP=True, img_height=img_height, img_width=img_width)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
