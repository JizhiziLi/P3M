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


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class TFI(nn.Module):
    expansion = 1
    def __init__(self, planes,stride=1):
        super(TFI, self).__init__()
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
class SBFI(nn.Module):
    def __init__(self, planes,stride=1):
        super(SBFI, self).__init__()
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
class DBFI(nn.Module):
    def __init__(self, planes,stride=1):
        super(DBFI, self).__init__()
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
class P3mNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34_mp()
        ############################
        ### Encoder part - RESNETMP
        ############################
        self.encoder0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            )
        self.mp0 = self.resnet.maxpool1
        self.encoder1 = nn.Sequential(
            self.resnet.layer1)
        self.mp1 = self.resnet.maxpool2
        self.encoder2 = self.resnet.layer2
        self.mp2 = self.resnet.maxpool3
        self.encoder3 = self.resnet.layer3
        self.mp3 = self.resnet.maxpool4
        self.encoder4 = self.resnet.layer4
        self.mp4 = self.resnet.maxpool5

        self.tfi_3 = TFI(256)
        self.tfi_2 = TFI(128)
        self.tfi_1 = TFI(64)
        self.tfi_0 = TFI(64)

        self.sbfi_2 = SBFI(128, 8)
        self.sbfi_1 = SBFI(64, 4)
        self.sbfi_0 = SBFI(64, 2)

        self.dbfi_2 = DBFI(128, 4)
        self.dbfi_1 = DBFI(64, 8)
        self.dbfi_0 = DBFI(64, 16)

        ##########################
        ### Decoder part - GLOBAL
        ##########################
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
        e0p, id0 = self.mp0(e0)
        e1p, id1 = self.mp1(e0p)
        e1 = self.encoder1(e1p)
        e2p, id2 = self.mp2(e1)
        e2 = self.encoder2(e2p)
        e3p, id3 = self.mp3(e2)
        e3 = self.encoder3(e3p)
        e4p, id4 = self.mp4(e3)
        e4 = self.encoder4(e4p)
        ###########################
        ### Decoder part - Global
        ###########################
        d4_g = self.decoder4_g(e4)
        d3_g = self.decoder3_g(d4_g)
        d2_g, global_sigmoid_side2 = self.dbfi_2(d3_g, e4)
        d2_g = self.decoder2_g(d2_g)
        d1_g, global_sigmoid_side1 = self.dbfi_1(d2_g, e4)
        d1_g = self.decoder1_g(d1_g)
        d0_g, global_sigmoid_side0 = self.dbfi_0(d1_g, e4)
        d0_g = self.decoder0_g(d0_g)
        global_sigmoid = d0_g
        ###########################
        ### Decoder part - Local
        ###########################
        d4_l = self.decoder4_l(e4)
        d4_l = F.max_unpool2d(d4_l, id4, kernel_size=2, stride=2)
        d3_l = self.tfi_3(d4_g, d4_l, e3)
        d3_l = self.decoder3_l(d3_l)
        d3_l = F.max_unpool2d(d3_l, id3, kernel_size=2, stride=2)
        d2_l = self.tfi_2(d3_g, d3_l, e2)
        d2_l = self.sbfi_2(d2_l, e0)
        d2_l = self.decoder2_l(d2_l)
        d2_l  = F.max_unpool2d(d2_l, id2, kernel_size=2, stride=2)
        d1_l = self.tfi_1(d2_g, d2_l, e1)
        d1_l = self.sbfi_1(d1_l, e0)
        d1_l = self.decoder1_l(d1_l)
        d1_l  = F.max_unpool2d(d1_l, id1, kernel_size=2, stride=2)
        d0_l = self.tfi_0(d1_g, d1_l, e0p)
        d0_l = self.sbfi_0(d0_l, e0)
        d0_l = self.decoder0_l(d0_l)
        d0_l  = F.max_unpool2d(d0_l, id0, kernel_size=2, stride=2)
        d0_l = self.decoder_final_l(d0_l)
        local_sigmoid = F.sigmoid(d0_l)
        ##########################
        ### Fusion net - G/L
        ##########################
        fusion_sigmoid = get_masked_local_from_global(global_sigmoid, local_sigmoid)
        return global_sigmoid, local_sigmoid, fusion_sigmoid, global_sigmoid_side2, global_sigmoid_side1, global_sigmoid_side0
        