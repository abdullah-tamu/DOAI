from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from . import pretrained_networks as pn
import torch.nn

import lpips
import cv2
from paths import *


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    scale_factor_H, scale_factor_W = 1. * out_HW[0] / in_H, 1. * out_HW[1] / in_W

    return nn.Upsample(scale_factor=(scale_factor_H, scale_factor_W), mode='bilinear', align_corners=False)(in_tens)


# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(self, pretrained=True, net='alex', version='0.1', lpips=True, spatial=False,
                 pnet_rand=False, pnet_tune=False, use_dropout=True, model_path=None, eval_mode=True, verbose=True):
        # lpips - [True] means with linear calibration on top of base network
        # pretrained - [True] means load linear weights

        super(LPIPS, self).__init__()
        if (verbose):
            print('Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]' %
                  ('LPIPS' if lpips else 'baseline', net, version, 'on' if spatial else 'off'))

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips  # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if (self.pnet_type in ['vgg', 'vgg16']):
            net_type = pn.vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif (self.pnet_type == 'alex'):
            net_type = pn.alexnet
            net_type_f = pn.alexnet_f
            # net_type = Alexnet_inv(pretrained=True)
            self.chns = [64, 192, 384, 256, 256]
        elif (self.pnet_type == 'squeeze'):
            net_type = pn.squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
        self.net_flipped = net_type_f(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if (lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)

            # for the other unflipped network
            self.lin7 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin8 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin9 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin10 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin11 = NetLinLayer(self.chns[4], use_dropout=use_dropout)

            # self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4, self.lin7, self.lin8, self.lin9,
                         self.lin10, self.lin11]
            if (self.pnet_type == 'squeeze'):  # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]
            self.lins = nn.ModuleList(self.lins)

            if (pretrained):
                if (model_path is None):
                    import inspect
                    import os
                    model_path = os.path.abspath(
                        os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth' % (version, net)))
                    # model_path = 'C:\\ML\\Second_PhD_Part\\Face_Anomaly_Appraisal\\checkpoints\\tmp\\latest_net_.pth'

                if (verbose):
                    print('Loading model from: %s' % model_path)
                self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

        if (eval_mode):
            self.eval()

    def mask_img(self, img):
        m_path = mask_path + '\\' + 'mask.png'
        mask_img = cv2.imread(m_path, 0)

        if (len(img) < 3):
            e = 0

        for i in range(len(img)):
            img[i][0][mask_img[:, :] == 0] = 0
        return img

    def forward(self, in0, in1, retPerLayer=True, normalize=False):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (
            in0, in1)
        outs0_flipped, outs1_flipped = self.net_flipped.forward(in0_input), self.net_flipped.forward(in1_input)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        feats0_f, feats1_f, diffs_f = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = lpips.normalize_tensor(outs0[kk]), lpips.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        for kk in range(self.L):
            feats0_f[kk], feats1_f[kk] = lpips.normalize_tensor(outs0_flipped[kk]), lpips.normalize_tensor(
                outs1_flipped[kk])
            diffs[self.L + kk] = (feats0_f[kk] - feats1_f[kk]) ** 2

            # diffs[kk] = np.abs(feats0[kk]-feats1[kk])

        if (self.lpips):
            if (self.spatial):
                res = [upsample(self.lins[kk].model(diffs[kk]), out_HW=in0.shape[2:]) for kk in range(2 * self.L)]

                # res = [self.mask_img(res[kk]) for kk in range(2*self.L)]
                # res = [upsample(self.lins[kk].model(diffs[kk]), out_HW=in0.shape[2:]) for kk in range(2,3)]

                # res = [upsample(torch.sum(diffs[kk]/20, keepdim=True, dim=1), out_HW=in0.shape[2:]) for kk in range(self.L)]

            else:
                # res = [spatial_average(self.lins[kk].model(diffs[kk]), keepdim=True) for kk in range(2*self.L)]
                res = [upsample(self.lins[kk].model(diffs[kk]), out_HW=in0.shape[2:]) for kk in range(2 * self.L)]

                res = [self.mask_img(res[kk]) for kk in range(2 * self.L)]
                res = [spatial_average(res[kk], keepdim=True) for kk in range(2 * self.L)]



        else:
            if (self.spatial):
                res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1, 2 * self.L):
            # for l in range(1,len(res)):
            val += res[l]

        if (retPerLayer):
            return (val, res)
        else:
            return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2, True), ]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2, True), ]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True), ]
        if (use_sigmoid):
            layers += [nn.Sigmoid(), ]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge + 1.) / 2.
        self.logit = self.net.forward(d0, d1)
        return self.loss(self.logit, per)


# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='RGB'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace


class L2(FakeNet):
    def forward(self, in0, in1, retPerLayer=None):
        # assert (in0.size()[0] == 1)  # currently only supports batchSize 1

        if (self.colorspace == 'RGB'):
            (N, C, X, Y) = in0.size()
            # value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y),
            #                    dim=3).view(N)

            value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1)))
            value = -torch.log10(value)
            return value
        elif (self.colorspace == 'Lab'):
            value = lpips.l2(lpips.tensor2np(lpips.tensor2tensorlab(in0.data, to_norm=False)),
                             lpips.tensor2np(lpips.tensor2tensorlab(in1.data, to_norm=False)), range=100.).astype(
                'float')
            ret_var = Variable(torch.Tensor((value,)))
            if (self.use_gpu):
                ret_var = ret_var.cuda()
            return ret_var


class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert (in0.size()[0] == 1)  # currently only supports batchSize 1

        if (self.colorspace == 'RGB'):
            value = lpips.dssim(1. * lpips.tensor2im(in0.data), 1. * lpips.tensor2im(in1.data), range=255.).astype(
                'float')
        elif (self.colorspace == 'Lab'):
            value = lpips.dssim(lpips.tensor2np(lpips.tensor2tensorlab(in0.data, to_norm=False)),
                                lpips.tensor2np(lpips.tensor2tensorlab(in1.data, to_norm=False)), range=100.).astype(
                'float')
        ret_var = Variable(torch.Tensor((value,)))
        if (self.use_gpu):
            ret_var = ret_var.cuda()
        return ret_var


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network', net)
    print('Total number of parameters: %d' % num_params)
