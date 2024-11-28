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


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def split_feature_map_n_parts(feature_map, n_parts):
    # 获取特征图的形状
    batch_size, channels, height, width = feature_map.shape

    # 计算每一份的高度
    part_height = height // n_parts

    # 存储分割后的特征图部分
    feature_parts = []

    # 分割特征图
    for i in range(n_parts):
        # 计算每个部分的起始和结束索引
        start_idx = i * part_height
        end_idx = start_idx + part_height
        # 提取对应的部分
        part = feature_map[:, :, start_idx:end_idx, :]
        feature_parts.append(part)

    return feature_parts


class DimReduceLayer(nn.Module):

    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, 1, stride=1, padding=0, bias=False
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))

        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if not isinstance(param_dict, collections.OrderedDict):
            param_dict = param_dict.state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def get_optimizer(self, cfg, criterion):
        optimizer = {}
        params = []
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        for key, value in self.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
            optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
        else:
            optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        if cfg.MODEL.CENTER_LOSS == 'on':
            optimizer['center'] = torch.optim.SGD(criterion['center'].parameters(), lr=cfg.SOLVER.CENTER_LR)
        return optimizer

    def get_creterion(self, cfg, num_classes):
        criterion = {}
        criterion['xent'] = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo

        print("Weighted Regularized Triplet:", cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET)
        if cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET == 'on':
            criterion['triplet'] = WeightedRegularizedTriplet()
        else:
            criterion['triplet'] = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss

        if cfg.MODEL.CENTER_LOSS == 'on':
            criterion['center'] = CenterLoss(num_classes=num_classes, feat_dim=cfg.MODEL.CENTER_FEAT_DIM,
                                             use_gpu=True)

        def criterion_total(score, feat, target):
            loss = criterion['xent'](score, target) + criterion['triplet'](feat, target)[0]
            if cfg.MODEL.CENTER_LOSS == 'on':
                loss = 0.6 * loss + cfg.SOLVER.CENTER_LOSS_WEIGHT * criterion['center'](feat, target)
            return loss

        criterion['total'] = criterion_total

        return criterion


