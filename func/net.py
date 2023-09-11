# -*- coding: utf-8 -*-
"""
Build network

Created on Feb 2023

@author: Xing-Yi Zhang (Zhangzxy20004182@163.com)

"""

from lib_config import *


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
        outputs = self.conv(inputs)
        outputs = self.down(outputs)
        return outputs

##################################################
#               Upsampling Unit                  #
# (The yellow arrow shown in Fig 1 of the paper) #
##################################################

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, activ_fuc = nn.ReLU(inplace=True)):
        '''
        Upsampling Unit
        (The yellow arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        :param activ_fuc:    Activation function
        '''
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, True, activ_fuc)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset1 = (outputs2.size()[2] - inputs1.size()[2])
        offset2 = (outputs2.size()[3] - inputs1.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]

        # Skip and concatenate
        outputs1 = F.pad(inputs1, padding)

        return self.conv(torch.cat([outputs1, outputs2], 1))

class unetUp_nonSqConv(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, activ_fuc = nn.ReLU(inplace=True)):
        '''
        Non-square convolution with flexible definition

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        :param activ_fuc:    Activation function
        '''
        super(unetUp_nonSqConv, self).__init__()
        self.conv = unetConv2(in_size, out_size, True, activ_fuc)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size = (2, 1), stride = (2, 1))
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset1 = (outputs2.size()[2] - inputs1.size()[2])
        offset2 = (outputs2.size()[3] - inputs1.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]
        # Skip and concatenate
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

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

############################################
#            DD-Net Architecture           #
############################################

class DDNetModel(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        '''
        DD-Net Architecture

        :param n_classes:    Number of channels of output (any single decoder)
        :param in_channels:  Number of channels of network input
        :param is_deconv:    Whether to use deconvolution
        :param is_batchnorm: Whether to use BN
        '''
        super(DDNetModel, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        filters = [64, 128, 256, 512, 1024]

        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)

        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.dc1_final = nn.Conv2d(filters[0], self.n_classes, 1)
        self.dc2_final = nn.Conv2d(filters[0], 2, 1)

    def forward(self, inputs, label_dsp_dim):
        '''

        :param inputs:          Input Image
        :param label_dsp_dim:   Size of the network output image (velocity model size)
        :return:
        '''
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        center = self.center(down4)

        # 25*19*1024
        decoder1_image = center
        decoder2_image = center

        #################
        ###  Decoder1 ###
        #################
        dc1_up4 = self.up4(down4, decoder1_image)
        dc1_up3 = self.up3(down3, dc1_up4)
        dc1_up2 = self.up2(down2, dc1_up3)
        dc1_up1 = self.up1(down1, dc1_up2)

        dc1_capture = dc1_up1[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()

        #################
        ###  Decoder2 ###
        #################
        dc2_up4 = self.up4(down4, decoder2_image)
        dc2_up3 = self.up3(down3, dc2_up4)
        dc2_up2 = self.up2(down2, dc2_up3)
        dc2_up1 = self.up1(down1, dc2_up2)

        dc2_capture = dc2_up1[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()

        return [self.dc1_final(dc1_capture), self.dc2_final(dc2_capture)]

class LossDDNet:
    def __init__(self, weights = [1, 1], entropy_weight = [1, 1]):
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
        print('MSELoss:{:.12f}'.format(mse.item()), end='\t')
        cross = self.criterion2(outputs2, torch.squeeze(targets2).long())
        print('CROSS:{:.12f}'.format(cross.item()))

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

        filters = [32, 64, 128, 256, 512]


        # This is a pre-network dimension reducer, which will not be learned by subsequent layers in the UNet way

        self.pre1   = ConvBlock(self.in_channels, 8, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.pre2_1 = ConvBlock(8, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre2_2 = ConvBlock(16, 16, kernel_size=(3, 1), padding=(1, 0))
        self.pre3_1 = ConvBlock(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre3_2 = ConvBlock(32, 32, kernel_size=(3, 1), padding=(1, 0))

        self.ready_down1 = ConvBlock(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.ready_down2 = ConvBlock(64, 64, kernel_size=(3, 1), padding=(1, 0))

        # Intrinsic UNet section
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up3 = unetUp(filters[3], filters[2], self.is_deconv)

        self.ready_up2 = unetUp_nonSqConv(filters[2], filters[1], self.is_deconv)

        self.dc1_final = ConvBlock_Tanh(filters[1], self.n_classes)
        self.dc2_final = ConvBlock_Tanh(filters[1], 2)

    def forward(self, inputs, _ = None):
        '''

        :param inputs:      Input Image
        :param _:           Variables for filling, for alignment with DD-Net
        :return:
        '''
        prelayer = self.pre1(inputs)
        prelayer = self.pre2_1(prelayer)
        prelayer = self.pre2_2(prelayer)
        prelayer = self.pre3_1(prelayer)
        prelayer = self.pre3_2(prelayer)

        lay64 = self.ready_down1(prelayer)
        input_lay64 = self.ready_down2(lay64)

        down3 = self.down3(input_lay64)
        down4 = self.down4(down3)
        center = self.center(down4)

        # 16*18*512
        decoder1_image = center
        decoder2_image = center

        #################
        ###  Decoder1 ###
        #################
        dc1_up4 = self.up4(down4, decoder1_image)
        dc1_up3 = self.up3(down3, dc1_up4)
        dc1_up2 = self.ready_up2(input_lay64, dc1_up3)

        dc1_capture = F.pad(dc1_up2, [-1, -1, -29, -29], mode="constant", value=0)

        #################
        ###  Decoder2 ###
        #################
        dc2_up4 = self.up4(down4, decoder2_image)
        dc2_up3 = self.up3(down3, dc2_up4)
        dc2_up2 = self.ready_up2(input_lay64, dc2_up3)

        dc2_capture = F.pad(dc2_up2, [-1, -1, -29, -29], mode="constant", value=0)

        return [self.dc1_final(dc1_capture), self.dc2_final(dc2_capture)]

if __name__ == '__main__':

    # Model output size test (for DD-Net70)

    x = torch.zeros((30, 5, 1000, 70))
    model = DDNet70Model(n_classes = 1,
                         in_channels = 5,
                         is_deconv = True,
                         is_batchnorm = True)
    out = model(x)
    print("out1: {}, out2: {}".format(str(out[0].size()), str(out[1].size())))

    # Model output size test (for DD-Net)

    x = torch.zeros((2, 29, 400, 301))
    model = DDNetModel(n_classes = 1,
                       in_channels = 29,
                       is_deconv = True,
                       is_batchnorm = True)
    out = model(x, label_dsp_dim = [201, 301])
    print("out1: {}, out2: {}".format(str(out[0].size()), str(out[1].size())))
