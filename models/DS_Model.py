#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 07:32:18 2020

@author: user1
"""


import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss

logger = logging.getLogger('base')

class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=13, stride=1, padding=4):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)

class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=9, stride=1, padding=True, include_pad=True, gaussian=True):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img

class FilterHigh(nn.Module):
    def __init__(self, recursions=1, kernel_size=9, stride=1, include_pad=True, normalize=True, gaussian=False):
        super(FilterHigh, self).__init__()
        self.filter_low = FilterLow(recursions=1, kernel_size=kernel_size, stride=stride, include_pad=include_pad,
                                    gaussian=gaussian)
        self.recursions = recursions
        self.normalize = normalize

    def forward(self, img):
        if self.recursions > 1:
            for i in range(self.recursions - 1):
                img = self.filter_low(img)
        img = img - self.filter_low(img)
        if self.normalize:
            return 0.5 + img * 0.5
        else:
            return img

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss,self).__init__()

    
    def forward(self, input,target):
        G1=gram_matrix(input)
        G2=gram_matrix(target)
        loss=F.mse_loss(G1,G2)

        
        return loss
class BTVLoss(nn.Module):
    def __init__(self, weight, neighborhood_size=5, epsilon=1e-6):
        super(BTVLoss, self).__init__()
        self.weight = weight
        self.neighborhood_size = neighborhood_size
        self.epsilon = epsilon

    def forward(self, x):
        batch_size, c, h, w = x.size()
        btv_loss = 0.0
        for k in range(-self.neighborhood_size, self.neighborhood_size + 1):
            for l in range(-self.neighborhood_size, self.neighborhood_size + 1):
                if k == 0 and l == 0:
                    continue
                shifted_x = torch.roll(x, shifts=(k, l), dims=(2, 3))
                diff = torch.abs(x - shifted_x)
                btv_loss += torch.sqrt(diff * diff + self.epsilon).sum()
        return self.weight * btv_loss / (batch_size * c * h * w)
    
def tv_loss(Y_hat):
    return 0.5 * torch.sqrt((torch.square(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.square(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean()))


class DS_Model(BaseModel):
    def __init__(self, opt):
        super(DS_Model, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G6(opt).to(self.device)  # G1
        if self.is_train:
            self.netD = networks.define_D2(opt).to(self.device)  # G1
            self.netD2 = networks.define_D2(opt).to(self.device) 
            #self.vgg = networks.define_F(opt, use_bn=False,Rlu=True).to(self.device)
            #self.netQ = networks.define_Q(opt).to(self.device)
            self.netG.train()
            self.netD.train()
            self.netD2.train()
        self.load()  # load G and D if needed
        self.n1 = torch.nn.Upsample(scale_factor=4,align_corners=True,mode='bicubic').to(self.device) 

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None
            #self.weight_kl = 1e-2
            #self.weight_D = 1e-3
            self.l_gan_w = train_opt['gan_weight']
            #self.qa_w = train_opt['QA_weight']
            self.color_filter = FilterLow(recursions=1,kernel_size=9,gaussian=True).to(self.device)
            self.high_filter = (FilterHigh(recursions=1,kernel_size=9,gaussian=True).to(self.device))

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False,Rlu=True).to(self.device)   #Rlu=True if feature taken before relu, else false

            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)

            #D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
            self.optimizers.append(self.optimizer_D)

            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
            self.optimizers.append(self.optimizer_D2)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()
    def calculate_gradient_penalty(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates.requires_grad_(True)

        d_interpolates = netD(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size()).to(self.device),
            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data['LR'].to(self.device)
        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)

    def optimize_parameters(self, step):

        if (step<=10000):

        # G
            self.optimizer_G.zero_grad()
            self.SR = self.netG(self.var_L)
            
            #SR_low = self.color_filter(self.SR)
            #HR_low = self.color_filter(self.var_H).detach()
            
            self.SR_Encoded = self.netD(self.SR)
            self.SR_Encoded2 = self.netD2(self.SR - self.color_filter(self.SR))
            self.texure=StyleLoss().to(self.device)
            self.tv=BTVLoss(1e-4)

            #self.SR_Encoded2 = self.netD2(self.vgg(self.SR))
            #Quality_loss = self.qa_w * torch.exp(-0.5*(torch.mean(self.netQ(self.SR).detach())-5))

            #n1 = torch.nn.Upsample(scale_factor=4,align_corners=True,mode='bicubic')

            l_g_total = 0
            btv_loss=self.tv(self.SR)
            tv_og=tv_loss(self.SR)
            
            #l_g_pix = self.l_pix_w * self.cri_pix(self.color_filter(self.SR), self.color_filter(n1(self.var_L)))
            l_g_pix = self.l_pix_w * self.cri_pix(self.SR, self.n1(self.var_L))
            l_g_total += l_g_pix
            l__texure=self.texure(self.SR,self.var_H)
            l_g_total+=1e4*l__texure
            l_g_total+=btv_loss
            l_g_total+=1e-1*tv_og
            # l_g_dis = self.l_gan_w * self.cri_gan(self.SR_Encoded, True)
            # l_g_total += l_g_dis
            # l_g_dis2 = self.l_gan_w * self.cri_gan(self.SR_Encoded2, True)
            # l_g_total += l_g_dis2

            #sr_g= (torch.pow((self.SR[:, :, 1:, :-1] - self.SR[:, :, 1:, 1:]),2) + torch.pow((self.SR[:, :, :-1, 1:] - self.SR[:, :, 1:, 1:]),2)) / (0.8)**2
            #l_grad = 2e-7 * torch.sum(torch.min(sr_g,torch.ones_like(sr_g).to(self.device)))
            #l_g_total +=l_g_tv

            #l_g_total += Quality_loss
            #l_g_total += l_grad

            l_g_total.backward()
            self.optimizer_G.step()
            self.log_dict['l_g_pix'] = l_g_pix.item()
            self.log_dict['total_g_loss']=l_g_total.item()
            self.log_dict['BTV']=btv_loss
            self.log_dict['TV']=1e-3*tv_og
            self.log_dict['textute']=l__texure
        
        else:
            self.optimizer_G.zero_grad()
            self.SR = self.netG(self.var_L)
            
            #SR_low = self.color_filter(self.SR)
            #HR_low = self.color_filter(self.var_H).detach()
            
            self.SR_Encoded = self.netD(self.SR)
            self.SR_Encoded2 = self.netD2(self.SR - self.color_filter(self.SR))
            self.texure=StyleLoss().to(self.device)
            self.tv=BTVLoss(1e-4)

            #self.SR_Encoded2 = self.netD2(self.vgg(self.SR))
            #Quality_loss = self.qa_w * torch.exp(-0.5*(torch.mean(self.netQ(self.SR).detach())-5))

            #n1 = torch.nn.Upsample(scale_factor=4,align_corners=True,mode='bicubic')

            l_g_total = 0
            btv_loss=self.tv(self.SR)
            tv_og=tv_loss(self.SR)
          
            #l_g_pix = self.l_pix_w * self.cri_pix(self.color_filter(self.SR), self.color_filter(n1(self.var_L)))
            l_g_pix = self.l_pix_w * self.cri_pix(self.SR, self.n1(self.var_L))
            l_g_total += l_g_pix
            l__texure=self.texure(self.SR,self.var_H)
            l_g_total+=1e4*l__texure
            l_g_dis = self.l_gan_w * self.cri_gan(self.SR_Encoded, True)
            l_g_total += l_g_dis
            l_g_dis2 = self.l_gan_w * self.cri_gan(self.SR_Encoded2, True)
            l_g_total += l_g_dis2
            l_g_total+=btv_loss
            l_g_total+=1e-1*tv_og

            #sr_g= (torch.pow((self.SR[:, :, 1:, :-1] - self.SR[:, :, 1:, 1:]),2) + torch.pow((self.SR[:, :, :-1, 1:] - self.SR[:, :, 1:, 1:]),2)) / (0.8)**2
            #l_grad = 2e-7 * torch.sum(torch.min(sr_g,torch.ones_like(sr_g).to(self.device)))
            #l_g_total +=l_g_tv

            #l_g_total += Quality_loss
            #l_g_total += l_grad

            l_g_total.backward()
            self.optimizer_G.step()

            self.optimizer_D.zero_grad()
            log_d_total = 0
            self.SR = self.netG(self.var_L)
            
            #SR_low = self.color_filter(self.SR)
            #HR_low = self.color_filter(self.var_H)
            self.HR_Encoded = self.netD(self.var_H)
            self.SR_Encoded = self.netD(self.SR)
            
            g1 = self.l_gan_w * self.cri_gan(self.HR_Encoded, True)
            g2 = self.l_gan_w * self.cri_gan(self.SR_Encoded, False)
            gp = self.calculate_gradient_penalty(self.netD, self.var_H, self.SR)
        
            log_d_total += (g1 + g2)*0.5 +1e-3*gp


            log_d_total.backward()
            self.optimizer_D.step()

            self.optimizer_D2.zero_grad()
            log_d2_total = 0
            self.SR = self.netG(self.var_L)
            
            #SR_low = self.color_filter(self.SR)
            #HR_low = self.color_filter(self.var_H)
            self.HR_Encoded2 = self.netD2(self.var_H - self.color_filter(self.var_H))
            self.SR_Encoded2 = self.netD2(self.SR - self.color_filter(self.SR))
            #self.HR_Encoded2 = self.netD2(self.vgg(self.var_H))
            #self.SR_Encoded2 = self.netD2(self.vgg(self.SR))
            
            g1 = self.l_gan_w * self.cri_gan(self.HR_Encoded2, True)
            g2 = self.l_gan_w * self.cri_gan(self.SR_Encoded2, False)
            gp_D2 = self.calculate_gradient_penalty(self.netD2, self.var_H - self.color_filter(self.var_H),self.SR - self.color_filter(self.SR) )
            log_d2_total += (g1 + g2)*0.5+1e-3*gp_D2


            log_d2_total.backward()
            self.optimizer_D2.step()

            # set log
            self.log_dict['l_g_pix'] = l_g_pix.item()
            self.log_dict['total_g_loss']=l_g_total.item()
            self.log_dict['l_g_d'] = l_g_dis.item()
            self.log_dict['l_g_d2'] = l_g_dis2.item()
            #self.log_dict['l_grad'] = l_grad.item()
            #self.log_dict['l_g_qa'] = Quality_loss.item()
            self.log_dict['d_total'] = log_d_total.item()
            self.log_dict['d2_total'] = log_d2_total.item()
            self.log_dict['WGAN_D']=gp
            self.log_dict['WGAN_D2']=gp_D2
            self.log_dict['BTV']=btv_loss
            self.log_dict['TV']=tv_og
            self.log_dict['texture']=l__texure.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.SR = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['SR'] = self.SR.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD)
        load_path_D2 = self.opt['path']['pretrain_model_D2']
        if self.opt['is_train'] and load_path_D2 is not None:
            logger.info('Loading pretrained model for D2 [{:s}] ...'.format(load_path_D2))
            self.load_network(load_path_D2, self.netD2)
        load_path_Q = self.opt['path']['pretrain_model_Q']
        #if self.opt['is_train'] and load_path_Q is not None:
            #load_path_Q = "/home/user1/Documents/Kalpesh/NTIRE2_Code/latest_G.pth"
            #logger.info('Loading pretrained model for Q [{:s}] ...'.format(load_path_Q))
            #self.load_network(load_path_Q, self.netQ)
    

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
        self.save_network(self.netD2, 'D2', iter_step)
    
    
    
