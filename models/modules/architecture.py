import math
import torch
import torch.nn as nn
import torchvision
from . import block as B
from . import spectral_norm as SN
from . import dbpn
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_

from einops import rearrange
####################
# Generator
####################

class VGGGAPQualifierModel(nn.Module):
    def __init__(self, in_nc, nf, height=1024, width=768):
        super(VGGGAPQualifierModel, self).__init__()
        
        self.model = B.VGGGAPQualifier()
    def forward(self,x):
        x = self.model(x)
        return x

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x

class RGT_1(nn.Module):
    def __init__(self):
        super(RGT_1,self).__init__()
        self.model=B.RGT()
    def forward(self,x):
        x1=self.model(x)
        return x1
    
class HLNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf,nb=20, kernel_size=3, norm_type=None, act_type='leakyrelu'):
        super(HLNet, self).__init__()
        self.model = B.high_low_network(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, kernel_size=kernel_size,norm_type=norm_type, act_type=act_type)
    def forward(self,x):
        x1=self.model(x)
        return x1
        
# class HAT_1(nn.module):
    # def __init__(self, nf,nb=20):
        # super(HAT_1,self).__init__()
        # self.model=B.HAT()
    # def forward(self,x):
        # x1=self.model(x)
        # return x1
class RRDBNet2(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(RRDBNet2, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        ks=5
        fea_conv = B.conv_block(in_nc, nf, kernel_size=ks, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=ks, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(int(nb*0.5))]
        rb_blocks2 = [B.RRDB(nf, kernel_size=ks-2, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(int(nb*0.5))]

        LR_conv = B.conv_block(nf, nf, kernel_size=1, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks,*rb_blocks2, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)
        self.ups = torch.nn.Upsample(scale_factor = 4, mode = 'bicubic')

    def forward(self, x):
        x = self.model(x) + self.ups(x)
        return x
class ParNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(ParNet, self).__init__()
        self.net1 = RRDBNet2(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, upscale=upscale, norm_type=norm_type,act_type=act_type, mode=mode, upsample_mode=upsample_mode)
        self.net2 = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, upscale=upscale, norm_type=norm_type,act_type=act_type, mode=mode, upsample_mode=upsample_mode)
        self.conv = B.conv_block(out_nc, out_nc, kernel_size=3, norm_type=None, act_type='sigm')

    def forward(self, x):
        x = self.conv(self.net2(x) + self.net1(x))
        return x

class DegNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(DegNet, self).__init__()
        n_dscale = int(math.log(upscale, 2))
        
        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        dsampler = [B.downconv_blcok(nf, nf, downscale_factor=2, act_type=act_type) for _ in range(n_dscale)]
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(8)]
        conv2 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        conv3 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, *dsampler, B.ShortcutBlock(B.sequential(*rb_blocks, conv2)),conv3)

    def forward(self, x):
        x = self.model(x)
        return x


####################
# Discriminator
####################
class Discriminator(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA',out_feat=256):
        super(Discriminator, self).__init__()
        # 192, 64 (12,512)
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=3, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 96, 64 (6,64)
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=3, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 128 (3,128)
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=3, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 256 (2,256)
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 512 (1,512)
        
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(base_nf*8, 512), nn.LeakyReLU(0.2, True), nn.Linear(512, out_feat))

    def forward(self, x):
        x = self.gap(self.features(x))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Patch_Discriminator(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA',out_feat=256):
        super(Patch_Discriminator, self).__init__()
        # 192, 64 (12,512)
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=4, stride=2, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, 2*base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 96, 64 (6,64)
        conv2 = B.conv_block(2*base_nf, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*4, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 128 (3,128)
        conv4 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*8, 1, kernel_size=4, norm_type=None, \
            act_type='sigm', mode=mode)
        
        
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5)

        #self.gap = nn.AdaptiveAvgPool2d((1,1))
        #self.classifier = nn.Sequential(
        #    nn.Linear(base_nf*8, 512), nn.LeakyReLU(0.2, True), nn.Linear(512, out_feat))

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x


####################
# Perceptual Network
####################


# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

class Discriminator_VGG_128(nn.Module):
    def __init__(self, nf):
        super(Discriminator_VGG_128, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)

        self.linear1 = nn.Linear(512 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out

