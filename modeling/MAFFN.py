# -*- coding: utf-8 -*-


import torch
from torch import nn
from torch.nn import functional as F
import collections

from torch.nn import ModuleList

from modeling.backbones.resnet import ResNet, Bottleneck
from modeling.backbones.resnet_nl import ResNetNL
from modeling.layer import CrossEntropyLabelSmooth, TripletLoss, WeightedRegularizedTriplet, CenterLoss, \
    GeneralizedMeanPooling, GeneralizedMeanPoolingP, ChannelAttention, SpatialAttention


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, model_name, gem_pool, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50_nl':
            self.base = ResNetNL(last_stride=last_stride,
                                 block=Bottleneck,
                                 layers=[3, 4, 6, 3],
                                 non_layers=[0, 2, 3, 0])

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.num_classes = num_classes

        if gem_pool == 'on':
            print("Generalized Mean Pooling")
            self.global_pool = GeneralizedMeanPoolingP()
        else:
            print("Global Adaptive Pooling")
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        # attention
        self.attention_C = ChannelAttention(in_planes=self.in_planes)
        self.attention_S = SpatialAttention()

        # local
        self.local_conv = nn.Conv2d(in_channels=self.in_planes, out_channels=self.in_planes, kernel_size=1, stride=1,
                                    padding=(1, 0))
        self.parts_avgpool = nn.AdaptiveAvgPool2d((6, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.conv5 = DimReduceLayer(self.in_planes, 1024, nonlinear='relu')
        self.feature_dim = 1024
        self.pcb_classifier = nn.ModuleList([
            nn.Linear(self.feature_dim, num_classes)
            for _ in range(6)])

        self.conv = nn.Conv2d(in_channels=self.in_planes * 2, out_channels=self.in_planes, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.base(x)

        # global
        global__feat = self.global_pool(x)  # (b, 2048, 1, 1)

        # attention
        attention_c_feat = x * self.attention_C(x)
        attention_s_feat = x * self.attention_S(x)
        attention_final_feat = x + attention_c_feat + attention_s_feat
        attention_pool_feat = self.pool(attention_final_feat)

        # local
        pcb_x = self.local_conv(x)
        v_g = self.parts_avgpool(pcb_x)
        v_g = self.dropout(v_g)
        v_h = self.conv5(v_g)
        y = []
        for i in range(6):
            v_h_i = v_h[:, :, i, :]
            v_h_i = v_h_i.view(v_h_i.size(0), -1)
            y_i = self.pcb_classifier[i](v_h_i)
            y.append(y_i)

        # fusion
        cat_feat = torch.cat((global__feat, attention_pool_feat), dim=1)
        final_feat = self.conv(cat_feat)

        _feat = final_feat.view(final_feat.shape[0], -1)  # flatten to (bs, 2048)

        feat = self.bottleneck(_feat)  # normalize for angular softmax

        if not self.training:
            return feat

        score = self.classifier(feat)
        return score, feat, y
