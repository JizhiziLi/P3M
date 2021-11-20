"""
Privacy-Preserving Portrait Matting [ACM MM-21]
Main test file.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au) and Sihan Ma (sima7436@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/P3M
Paper link : https://dl.acm.org/doi/10.1145/3474085.3475512

"""
import torch
import torch.nn as nn
from torchvision import models
from config import *
import torch.nn.functional as F
from util import *
from network.resnet_mp import *


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv_up_psp(in_channels, out_channels, up_sample):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=up_sample, mode='bilinear'))



class RefineModule(nn.Module):
    expansion = 1

    def __init__(self, planes,stride=1):
        super(RefineModule, self).__init__()
        middle_planes = int(planes/2)
        self.transform = conv1x1(planes, middle_planes)
        self.conv1 = conv3x3(middle_planes*3, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, input_s_guidance, input_m_decoder, input_m_encoder):
        input_s_guidance_transform = self.transform(input_s_guidance)
        input_m_decoder_transform = self.transform(input_m_decoder)
        input_m_encoder_transform = self.transform(input_m_encoder)

        x = torch.cat((input_s_guidance_transform,input_m_decoder_transform,input_m_encoder_transform),1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class ResidualModule(nn.Module):

    def __init__(self, planes,stride=1):
        super(ResidualModule, self).__init__()
        self.stride = stride

        self.transform1 = conv1x1(planes, int(planes/2))
        self.transform2 = conv1x1(64, int(planes/2))

        self.maxpool = nn.MaxPool2d(2, stride=stride)

        self.conv1 = conv3x3(planes, planes, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, input_m_decoder,e0):
        input_m_decoder_transform = self.transform1(input_m_decoder)
        e0_maxpool = self.maxpool(e0)
        e0_transform = self.transform2(e0_maxpool)
        x = torch.cat((input_m_decoder_transform,e0_transform),1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out+input_m_decoder
        return out




class ResidualModuleSemantic(nn.Module):

    def __init__(self, planes,stride=1):
        super(ResidualModuleSemantic, self).__init__()
        self.stride = stride

        self.transform1 = conv1x1(planes, int(planes/2))
        self.transform2 = conv1x1(512, int(planes/2))
        self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')

        self.conv1 = conv3x3(planes, planes, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, 3, 1)
        self.upsample2 = nn.Upsample(scale_factor=int(32/stride), mode='bilinear')
        

    def forward(self, input_s_decoder,e4):

        input_s_decoder_transform = self.transform1(input_s_decoder)
        e4_transform = self.transform2(e4)
        e4_upsample = self.upsample(e4_transform)
        x = torch.cat((input_s_decoder_transform,e4_upsample),1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out+input_s_decoder

        out_side = self.conv2(out)
        out_side = self.upsample2(out_side)        

        return out, out_side


###################
### prm v21: v20 with side loss for semantic 
### final result for mm
###################


class P3mNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.resnet = resnet34_mp()
        ##########################
        ### Encoder part - RESNET
        ##########################
        #stage 0
        self.encoder0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            )

        self.mp0 = self.resnet.maxpool1
        #stage 1
        self.encoder1 = nn.Sequential(
            # self.resnet.maxpool,
            self.resnet.layer1
            )

        self.mp1 = self.resnet.maxpool2
        
        #stage 2
        self.encoder2 = self.resnet.layer2

        self.mp2 = self.resnet.maxpool3
        #stage 3
        self.encoder3 = self.resnet.layer3
        self.mp3 = self.resnet.maxpool4
        #stage 4
        self.encoder4 = self.resnet.layer4
        self.mp4 = self.resnet.maxpool5


        self.refinemodule3 = RefineModule(256)
        self.refinemodule2 = RefineModule(128)
        self.refinemodule1 = RefineModule(64)
        self.refinemodule0 = RefineModule(64)

        self.residualmodule2 = ResidualModule(128, 8)
        self.residualmodule1 = ResidualModule(64, 4)
        self.residualmodule0 = ResidualModule(64, 2)


        self.residualmodule2_s = ResidualModuleSemantic(128, 4)
        self.residualmodule1_s = ResidualModuleSemantic(64, 8)
        self.residualmodule0_s = ResidualModuleSemantic(64, 16)




       
        ##########################
        ### Decoder part - GLOBAL
        ##########################

        #stage 4d
        self.decoder4_g = nn.Sequential(
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear') )

        #stage 3d
        self.decoder3_g = nn.Sequential(
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear') )
        
        #stage 2d
        self.decoder2_g = nn.Sequential(
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))

        #stage 1d
        self.decoder1_g = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        
        #stage 0d
        self.decoder0_g = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,3,3,padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'))

        ##########################
        ### Decoder part - LOCAL
        ##########################

        #stage 4l
        self.decoder4_l = nn.Sequential(
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))


        #stage 3l
        self.decoder3_l = nn.Sequential(
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        
        #stage 2l
        self.decoder2_l = nn.Sequential(
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))


        #stage 1l
        self.decoder1_l = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        
        #stage 0l
        self.decoder0_l = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.decoder_final_l = nn.Conv2d(64,1,3,padding=1)

        
    def forward(self, input):


        ##########################
        ### Encoder part - RESNET
        ##########################
        e0 = self.encoder0(input)
        #e0: N, 64, H, W
        e0p, id0 = self.mp0(e0)
        #e0p: N, 64, H/2, W/2

        e1p, id1 = self.mp1(e0p)
        #e1p: N, 64, H/4, W/4
        e1 = self.encoder1(e1p)
        #e1: N, 64, H/4, W/5
        e2p, id2 = self.mp2(e1)
        #e2p: N, 64, H/8, W/8
        e2 = self.encoder2(e2p)
        #e2: N, 128, H/4, W/4
        e3p, id3 = self.mp3(e2)
        #e3p: N, 128, H/16, W/16
        e3 = self.encoder3(e3p)
        #e3: N, 256, H/8, W/8
        e4p, id4 = self.mp4(e3)
        #e4p: N, 256, H/32, W/32
        e4 = self.encoder4(e4p)
        #e4p: N, 512, H/16, W/16

        ##########################
        ### Decoder part - GLOBAL 
        ##########################
        d4_g = self.decoder4_g(e4)
        #d4_g: N, 256, H/16, W/16
        d3_g = self.decoder3_g(d4_g)
        #d3_g: N, 128, H/8, W/8


        d2_g, global_sigmoid_side2 = self.residualmodule2_s(d3_g, e4)
        d2_g = self.decoder2_g(d2_g)
        #d2_g: N, 64, H/4, W/4

        d1_g, global_sigmoid_side1 = self.residualmodule1_s(d2_g, e4)
        d1_g = self.decoder1_g(d1_g)
        #d1_g: N, 64, H/2, W/2

        d0_g, global_sigmoid_side0 = self.residualmodule0_s(d1_g, e4)
        d0_g = self.decoder0_g(d0_g)
        #d0_g: N, 3, H, W
        # global_sigmoid = F.sigmoid(d0_g)
        global_sigmoid = d0_g
        #global_sigmoid: N, 3, H, W


        ##########################
        ### Decoder part - LOCAL
        ##########################

        d4_l = self.decoder4_l(e4)

        d4_l = F.max_unpool2d(d4_l, id4, kernel_size=2, stride=2)
        #d3_l: N, 256, H/16, W/16

        d3_l = self.refinemodule3(d4_g, d4_l, e3)
        d3_l = self.decoder3_l(d3_l)
        #d3_l: N, 128, H/16, W/16
        d3_l = F.max_unpool2d(d3_l, id3, kernel_size=2, stride=2)


        d2_l = self.refinemodule2(d3_g, d3_l, e2)

        d2_l = self.residualmodule2(d2_l, e0)

        d2_l = self.decoder2_l(d2_l)
        d2_l  = F.max_unpool2d(d2_l, id2, kernel_size=2, stride=2)

        #d1_l: N, 64, H/4, W/4
        d1_l = self.refinemodule1(d2_g, d2_l, e1)
        d1_l = self.residualmodule1(d1_l, e0)

        d1_l = self.decoder1_l(d1_l)
        d1_l  = F.max_unpool2d(d1_l, id1, kernel_size=2, stride=2)
        #d0_l: N, 64, H/2, W/2
        d0_l = self.refinemodule0(d1_g, d1_l, e0p)
        d0_l = self.residualmodule0(d0_l, e0)
        d0_l = self.decoder0_l(d0_l)


        d0_l  = F.max_unpool2d(d0_l, id0, kernel_size=2, stride=2)
        
        #d0_l: N, 64, H, W
        d0_l = self.decoder_final_l(d0_l)

        local_sigmoid = F.sigmoid(d0_l)
   

        ##########################
        ### Fusion net - G/L
        ##########################

        fusion_sigmoid = get_masked_local_from_global(global_sigmoid, local_sigmoid)
        

        return global_sigmoid, local_sigmoid, fusion_sigmoid, global_sigmoid_side2, global_sigmoid_side1, global_sigmoid_side0
        