# -*- coding: utf-8 -*-
"""
Build network

Created on Feb 2023

@author: Xing-Yi Zhang (zxy20004182@163.com)

"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeismicRecordDownSampling(nn.Module):
    '''
    Downsampling module for seismic records
    '''
    def __init__(self, shot_num):
        super().__init__()

        self.pre_dim_reducer1 = ConvBlock(shot_num, 8, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer2 = ConvBlock(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pre_dim_reducer3 = ConvBlock(8, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer4 = ConvBlock(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pre_dim_reducer5 = ConvBlock(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer6 = ConvBlock(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):

        width = x.shape[3]
        new_size = [width * 8, width]
        dimred0 = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)

        dimred1 = self.pre_dim_reducer1(dimred0)
        dimred2 = self.pre_dim_reducer2(dimred1)
        dimred3 = self.pre_dim_reducer3(dimred2)
        dimred4 = self.pre_dim_reducer4(dimred3)
        dimred5 = self.pre_dim_reducer5(dimred4)
        dimred6 = self.pre_dim_reducer6(dimred5)

        return dimred6

###############################################
#         Conventional Network Unit           #
# (The red arrow shown in Fig 1 of the paper) #
###############################################

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, activ_fuc = nn.ReLU(inplace=True)):
        '''
        Conventional Network Unit
        (The red arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_batchnorm: Whether to use BN
        :param activ_fuc:    Activation function
        '''
        super(unetConv2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       activ_fuc)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       activ_fuc)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       activ_fuc)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       activ_fuc)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

##################################################
#             Downsampling Unit                  #
# (The purple arrow shown in Fig 1 of the paper) #
##################################################

class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, activ_fuc = nn.ReLU(inplace=True)):
        '''
        Downsampling Unit
        (The purple arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_batchnorm: Whether to use BN
        :param activ_fuc:    Activation function
        '''
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm, activ_fuc)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        skip_output = self.conv(inputs)
        outputs = self.down(skip_output)
        return outputs

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, output_lim, is_deconv, activ_fuc=nn.ReLU(inplace=True)):
        '''
        Upsampling Unit
        (The yellow arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        :param activ_fuc:    Activation function
        '''
        super(unetUp, self).__init__()
        self.output_lim = output_lim
        self.conv = unetConv2(in_size, out_size, True, activ_fuc)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input1, input2):
        input2 = self.up(input2)
        input2 = F.interpolate(input2, size=self.output_lim, mode='bilinear', align_corners=False)
        return self.conv(torch.cat([input1, input2], 1))

class netUp(nn.Module):
    def __init__(self, in_size, out_size, output_lim, is_deconv):
        '''
        Upsampling Unit
        (The yellow arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        :param activ_fuc:    Activation function
        '''
        super(netUp, self).__init__()
        self.output_lim = output_lim
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input):
        input = self.up(input)
        output = F.interpolate(input, size=self.output_lim, mode='bilinear', align_corners=False)
        return output

###################################################
# Non-square convolution with flexible definition #
#            Similar to InversionNet              #
###################################################

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, activ_fuc = nn.ReLU(inplace=True)):
        '''
        Non-square convolution with flexible definition
        (Similar to InversionNet)

        :param in_fea:       Number of channels for convolution layer input
        :param out_fea:      Number of channels for convolution layer output
        :param kernel_size:  Size of the convolution kernel
        :param stride:       Convolution stride
        :param padding:      Convolution padding
        :param activ_fuc:    Activation function
        '''
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(activ_fuc)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

###################################################
#    Convolution at the end for normalization     #
#            Similar to InversionNet              #
###################################################

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        '''
        Convolution at the end for normalization
        (Similar to InversionNet)

        :param in_fea:       Number of channels for convolution layer input
        :param out_fea:      Number of channels for convolution layer output
        :param kernel_size:  Size of the convolution kernel
        :param stride:       Convolution stride
        :param padding:      Convolution padding
        '''
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LossDDNet:
    def __init__(self, weights=[1, 1], entropy_weight=[1, 1]):
        '''
        Define the loss function of DDNet

        :param weights:         The weights of the two decoders in the calculation of the loss value.
        :param entropy_weight:  The weights of the two output channels in the second decoder.
        '''

        self.criterion1 = nn.MSELoss()
        ew = torch.from_numpy(np.array(entropy_weight).astype(np.float32)).cuda()
        self.criterion2 = nn.CrossEntropyLoss(weight = ew)    # For multi-classification, the current issue is a binary problem (either black or white).
        self.weights = weights

    def __call__(self, outputs1, outputs2, targets1, targets2):
        '''

        :param outputs1: Output of the first decoder
        :param outputs2: Velocity model
        :param targets1: Output of the second decoder
        :param targets2: Profile of the speed model
        :return:
        '''
        mse = self.criterion1(outputs1, targets1)
        cross = self.criterion2(outputs2, torch.squeeze(targets2).long())

        criterion = (self.weights[0] * mse + self.weights[1] * cross)

        return criterion

############################################
#          DD-Net70 Architecture           #
############################################

class DDNet70Model(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        '''
        DD-Net70 Architecture

        :param n_classes:    Number of channels of output (any single decoder)
        :param in_channels:  Number of channels of network input
        :param is_deconv:    Whether to use deconvolution
        :param is_batchnorm: Whether to use BN
        '''
        super(DDNet70Model, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        self.pre_seis_conv = SeismicRecordDownSampling(self.in_channels)

        # Intrinsic UNet section
        self.down3 = unetDown(32, 64, self.is_batchnorm)
        self.down4 = unetDown(64, 128, self.is_batchnorm)
        self.down5 = unetDown(128, 256, self.is_batchnorm)

        self.center = unetDown(256, 512, self.is_batchnorm)

        self.dc1_up5 = unetUp1(512, 256, output_lim=[9, 9], is_deconv=self.is_deconv)
        self.dc1_up4 = unetUp1(256, 128, output_lim=[18, 18], is_deconv=self.is_deconv)
        self.dc1_up3 = netUp1(128, 64, output_lim=[35, 35], is_deconv=self.is_deconv)
        self.dc1_up2 = netUp1(64, 32, output_lim=[70, 70], is_deconv=self.is_deconv)

        self.dc2_up5 = unetUp2(512, 256, output_lim=[9, 9], is_deconv=self.is_deconv)
        self.dc2_up4 = unetUp2(256, 128, output_lim=[18, 18], is_deconv=self.is_deconv)
        self.dc2_up3 = netUp2(128, 64, output_lim=[35, 35], is_deconv=self.is_deconv)
        self.dc2_up2 = netUp2(64, 32, output_lim=[70, 70], is_deconv=self.is_deconv)

        self.dc1_final = ConvBlock_Tanh(32, self.n_classes)
        self.dc2_final = ConvBlock_Tanh(32, 2)

    def forward(self, inputs, _=None):
        '''

        :param inputs:      Input Image
        :param _:           Variables for filling, for alignment with DD-Net
        :return:
        '''

        compress_seis = self.pre_seis_conv(inputs)

        down3 = self.down3(compress_seis)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        center = self.center(down5)

        # 16*18*512
        decoder1_image = center
        decoder2_image = center

        #################
        ###  Decoder1 ###
        #################
        dc1_up5 = self.dc1_up5(down5, decoder1_image)
        dc1_up4 = self.dc1_up4(down4, dc1_up5)
        dc1_up3 = self.dc1_up3(dc1_up4)
        dc1_up2 = self.dc1_up2(dc1_up3)

        #################
        ###  Decoder2 ###
        #################
        dc2_up5 = self.dc2_up5(down5, decoder2_image)
        dc2_up4 = self.dc2_up4(down4, dc2_up5)
        dc2_up3 = self.dc2_up3(dc2_up4)
        dc2_up2 = self.dc2_up2(dc2_up3)

        return [self.dc1_final(dc1_up2), self.dc2_final(dc2_up2)]

############################################
#          SD-Net70 Architecture           #
############################################

class SDNet70Model(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        '''
        DD-Net70 Architecture

        :param n_classes:    Number of channels of output (any single decoder)
        :param in_channels:  Number of channels of network input
        :param is_deconv:    Whether to use deconvolution
        :param is_batchnorm: Whether to use BN
        '''
        super(SDNet70Model, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        self.pre_seis_conv = SeismicRecordDownSampling(self.in_channels)

        # Intrinsic UNet section
        self.down3 = unetDown(32, 64, self.is_batchnorm)
        self.down4 = unetDown(64, 128, self.is_batchnorm)
        self.down5 = unetDown(128, 256, self.is_batchnorm)

        self.center = unetDown(256, 512, self.is_batchnorm)

        self.up5 = unetUp(512, 256, output_lim=[9, 9], is_deconv=self.is_deconv)
        self.up4 = unetUp(256, 128, output_lim=[18, 18], is_deconv=self.is_deconv)
        self.up3 = netUp(128, 64, output_lim=[35, 35], is_deconv=self.is_deconv)
        self.up2 = netUp(64, 32, output_lim=[70, 70], is_deconv=self.is_deconv)

        self.dc1_final = ConvBlock_Tanh(32, self.n_classes)
        self.dc2_final = ConvBlock_Tanh(32, 2)

    def forward(self, inputs, _=None):
        '''

        :param inputs:      Input Image
        :param _:           Variables for filling, for alignment with DD-Net
        :return:
        '''

        compress_seis = self.pre_seis_conv(inputs)

        down3 = self.down3(compress_seis)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        center = self.center(down5)

        # 16*18*512
        decoder1_image = center
        decoder2_image = center

        #################
        ###  Decoder1 ###
        #################
        dc1_up5 = self.up5(down5, decoder1_image)
        dc1_up4 = self.up4(down4, dc1_up5)
        dc1_up3 = self.up3(dc1_up4)
        dc1_up2 = self.up2(dc1_up3)

        return self.dc1_final(dc1_up2)

if __name__ == '__main__':

    # # Model output size test (for DD-Net70)
    #
    # x = torch.zeros((30, 5, 1000, 70))
    # model = DDNet70Model(n_classes = 1,
    #                      in_channels = 5,
    #                      is_deconv = True,
    #                      is_batchnorm = True)
    # out = model(x)
    # print("out1: {}, out2: {}".format(str(out[0].size()), str(out[1].size())))

    model = DDNet70Model(n_classes=1, in_channels=7, is_deconv=True, is_batchnorm=True)
    device = torch.device('cuda:0')
    model.to(device)
    from torchsummary import summary
    summary(model, input_size=[(7, 1000, 70)])
