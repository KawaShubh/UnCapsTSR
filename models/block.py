from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from einops.layers.torch import Rearrange

####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2, n_selu=1):
    # helper selecting activation
    # neg_slope: for selu and init of selu
    # n_selu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU()
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(0.2,inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU()
    elif act_type == 'sigm':
        layer = nn.Sigmoid()
    elif act_type == 'elu':
        layer = nn.ELU()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, eps=1e-5, momentum=0.01, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)



def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

def trans_conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    #padding = 1
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups,output_padding=1)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

####################
# Useful blocks
####################



class UP_Sample(nn.Module):
    def __init__(self,in_nc, nc, kernel_size=3, stride=1, bias=True, pad_type='zero', \
            act_type=None, mode='CNA',upscale_factor=2):
        super(UP_Sample, self).__init__()
        self.U1 = pixelshuffle_block(in_nc, nc, upscale_factor=upscale_factor, kernel_size=3, norm_type = 'batch')
        self.co1 = conv_block(nc, 16, kernel_size=1, norm_type=None, act_type='leakyrelu', mode='CNA')
        self.co2 = conv_block(16, 3, kernel_size=3, norm_type=None, act_type='leakyrelu', mode='CNA')

    def forward(self, x):
        out1 = self.U1(x)
        return self.co2(self.co1(out1))

class ChannelAttention(nn.Module):
    def __init__(self, nf,reduction=16):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(nf,nf//reduction , kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf//reduction, nf, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x.mul(self.module(x))

class HFM(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        
        self.k = k

        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size = self.k, stride = self.k),
            nn.Upsample(scale_factor = self.k, mode = 'nearest'),
        )

    def forward(self, tL):
        assert tL.shape[2] % self.k == 0, 'h, w must divisible by k'
        return tL - self.net(tL)        

class ResidualUnit(nn.Module):
    def __init__(self, in_nc, out_nc, reScale=1, kernel_size=1, bias=True,norm_type='batch', act_type='leakyrelu'):
        super(ResidualUnit,self).__init__()

        self.reduction =conv_block(in_nc, in_nc//2, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
 
        self.expansion = conv_block(in_nc//2, in_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.lamRes = reScale
        self.lamX = reScale

    def forward(self, x):
        res = self.reduction(x)
        res1 = self.expansion(res)
        res2 = self.lamRes * res1
        x = self.lamX * x + res2

        return x



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = conv_block(2, 1, kernel_size=kernel_size)
        #BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale



class ARFB(nn.Module):
    def __init__(self, in_nc, out_nc, reScale=1,norm_type='batch', act_type='leakyrelu',kernel_size=1):
        super().__init__()
        self.RU1 = ResidualUnit(in_nc, out_nc, reScale)
        self.RU2 = ResidualUnit(in_nc, out_nc, reScale)
        self.conv1 =conv_block(2*in_nc, 2*in_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type) 
        self.conv3 =conv_block(2*in_nc, in_nc, kernel_size=3, norm_type=norm_type, act_type=act_type) 
        self.lamRes = reScale
        self.lamX = reScale

    def forward(self, x):

        x_ru1 = self.RU1(x)
        x_ru2 = self.RU2(x_ru1)
        x_ru = torch.cat((x_ru1, x_ru2), 1)
        x_ru = self.conv1(x_ru)
        x_ru = self.conv3(x_ru)
        x_ru = self.lamRes * x_ru
        x = x*self.lamX + x_ru
        return x

class high_block(nn.Module):
    def __init__(self, in_nc, kernel_size=3,norm_type='batch', act_type='leakyrelu'):
        super(high_block, self).__init__()

        self.conv0 = conv_block(in_nc, in_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.conv1 = conv_block(in_nc, in_nc, kernel_size=kernel_size, norm_type=norm_type,act_type=act_type)
        self.conv2 = conv_block(in_nc, in_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        #self.gap = nn.AdaptiveAvgPool2d((1,1))
        #self.conv3 = conv_block(in_nc, 16, kernel_size=1, norm_type=None, act_type='prelu')
        #self.conv4 = conv_block(16, in_nc, kernel_size=1, norm_type=None, act_type='sigm')

    def forward(self,x):
        x1 = self.conv2(self.conv1(self.conv0(x)))
        #m = self.conv4(self.conv3(self.gap(x1)))
        #x2 = x1.mul(m)
        return x1.mul(0.2) + x


class BB(nn.Module) :
  def __init__(self, nf) :
    super(BB, self).__init__()
    #self.k = k
    # self.uk3_1 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    # self.uk3_2 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    # self.uk3_3 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    #self.arf1=ARFB(in_nc=nf, out_nc=nf, reScale=1,norm_type='batch', act_type='leakyrelu',kernel_size=1)
    self.hfm1=HFM(k=2)
    self.uk3_1 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    self.channel1=ChannelAttention(nf)
    self.hfm4=HFM(k=2)
    self.uk3_2 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    self.channel2=ChannelAttention(nf)
    self.hfm5=HFM(k=2)
    self.uk3_3 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    self.channel3=ChannelAttention(nf)
    self.hfm6=HFM(k=2)
    self.uk3_4 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    self.channel10=ChannelAttention(nf)

    
    #self.arf2=ARFB(in_nc=nf, out_nc=nf, reScale=1,norm_type='batch', act_type='leakyrelu',kernel_size=1)
    self.hfm2=HFM(k=2)
    self.lk5_1 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 5)
    self.channel4=ChannelAttention(nf)
    self.hfm7=HFM(k=2)
    self.lk5_2 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 5)
    self.channel5=ChannelAttention(nf)
    self.hfm8=HFM(k=2)
    self.lk5_3 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 5)
    self.channel6=ChannelAttention(nf)
    self.hfm9=HFM(k=2)
    self.lk5_4 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 5)
    self.channel11=ChannelAttention(nf)

    # self.lk7_1 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 7)
    # self.lk7_2 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 7)
    # self.lk7_3 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 7)
    
    #self.arf3=ARFB(in_nc=nf, out_nc=nf, reScale=1,norm_type='batch', act_type='leakyrelu',kernel_size=1)
    self.hfm3=HFM(k=2)
    self.lk1_1 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 1)
    self.channel7=ChannelAttention(nf)
    self.hfm10=HFM(k=2)
    self.lk1_2 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 1)
    self.channel8=ChannelAttention(nf)
    self.hfm11=HFM(k=2)
    self.lk1_3 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 1)
    self.channel9=ChannelAttention(nf)
    self.hfm12=HFM(k=2)
    self.lk1_4 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 1)
    self.channel12=ChannelAttention(nf)
    #self.lk9_1 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 9)
    #self.lk9_2 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 9)
   # self.lk9_3 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 9)
    self.k1 = conv_block(in_nc=4*nf, out_nc=4*nf, kernel_size= 1)
    self.k2 = conv_block(in_nc=4*nf, out_nc=nf, kernel_size= 3)

    
    # self.emha = EMHA(nf*k*k, splitfactors, heads)
    # self.norm = nn.LayerNorm(nf*k*k)
    # self.unFold = nn.Unfold(kernel_size=(k, k), padding=1)
  
  def forward(self,x):
    _, _, h, w = x.shape

    #upper path
    #xarf1=self.arf1(x)
    xu1_1hfm=self.hfm1(x)
    xu1_1= self.uk3_1(xu1_1hfm)
    xc1=self.channel1(xu1_1)
    xc1=xc1+x
    xc1=self.hfm4(xc1)
    xu2_1= self.uk3_2(xc1)
    xc2=self.channel2(xu2_1)
    xc2=xc2+xc1
    xc2=self.hfm5(xc2)
    xu3_1= self.uk3_3(xc2)
    xc3=self.channel3(xu3_1)
    xc3=xc3+xc2
    xc3=self.hfm6(xc3)
    xu4_1=self.uk3_4(xc3)
    xc10=self.channel10(xu4_1)


    #xarf2=self.arf2(x)
    xl0hfm=self.hfm2(x)
    xl1= self.lk5_1(xl0hfm)
    xc4=self.channel4(xl1)
    xc4=xc4+x
    xc4=self.hfm7(xc4)
    xl2= self.lk5_2(xc4)
    xc5=self.channel5(xl2)
    xc5=xc5+xc4
    xc5=self.hfm8(xc5)
    xl3= self.lk5_3(xc5)
    xc6=self.channel6(xl3)
    xc6=xc6+xc5
    xc6=self.hfm9(xc6)
    xl4=self.lk5_4(xc6)
    xc11=self.channel11(xl4)
    

    #lower part
    #xarf3=self.arf3(x)
    xu5_1hfm=self.hfm3(x)
    xl5_1= self.lk1_1(xu5_1hfm)
    xc7=self.channel7(xl5_1)
    xc7=xc7+x
    xc7=self.hfm10(xc7)
    xl5_2= self.lk1_2(xc7)
    xc8=self.channel8(xl5_2)
    xc8=xc8+xc7
    xc8=self.hfm11(xc8)
    xl5_3= self.lk1_3(xc8)
    xc9=self.channel9(xl5_3)
    xc9=xc9+xc8
    xc9=self.hfm12(xc9)
    xl5_4=self.lk1_4(xc9)
    xc12=self.channel12(xl5_4)

    # #transformer
    
    out= torch.cat((xc10,xc11,xc12,x),1)
    out1=self.k1(out)
    out2=self.k2(out1)

    return out2



class Residual(nn.Module):

    def __init__(self, nf,act_type='relu'):
        super(Residual,self).__init__()
        #self.conv1 = conv_block(nf, nf, kernel_size=1, norm_type=None, act_type=act_type)
        self.bb1 = BB(nf)
        self.bb2 = BB(nf)
        self.bb3 = BB(nf)
        self.bb4 = BB(nf)
        self.bb5 = BB(nf)
        # self.k1 = conv_block(in_nc=5*nf, out_nc=nf, kernel_size= 3)
        # self.k2 = conv_block(in_nc=2*nf, out_nc=nf, kernel_size= 3)
        # self.k3 = conv_block(in_nc=3*nf, out_nc=nf, kernel_size= 3)
        # self.k4 = conv_block(in_nc=4*nf, out_nc=nf, kernel_size= 3)
        
        
    def forward(self, x):
        #x1= self.bb1(x)
        xu1_1= self.bb1(x)
        
        xu2_1= self.bb2(xu1_1)
        
        # # # # out2 = self.channel2(xu2_2)
        xu3_1= self.bb3(xu2_1)
        
        
        # # # out3 = self.channel3(xu3_2)
        xu3_3= self.bb4(xu3_1)
       
        # # # out4 = self.channel4(xu3_4)
        xu3_5= self.bb5(xu3_3)
       
   
        return xu3_5+x


class EMHA(nn.Module):
  
    def __init__(self, in_nc, splitfactors=4, heads=8):
        super().__init__()
        dimHead = in_nc // (2*heads)

        self.heads = heads
        self.splitfactors = splitfactors
        self.scale = dimHead ** -0.5

        self.reduction = nn.Conv1d(
            in_channels=in_nc, out_channels=in_nc//2, kernel_size=1)
        self.attend = nn.Softmax(dim=-1)
        self.toQKV = nn.Linear(
            in_nc // 2, in_nc // 2 * 3, bias=False)
        self.expansion = nn.Conv1d(
            in_channels=in_nc//2, out_channels=in_nc, kernel_size=1)

    def forward(self, x):
        x = self.reduction(x)
        x = x.transpose(-1, -2)

        qkv = self.toQKV(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        qs, ks, vs = map(lambda t: t.chunk(
            self.splitfactors, dim=2), [q, k, v])

        pool = []
        for qi, ki, vi in zip(qs, ks, vs):
            tmp = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale
            attn = self.attend(tmp)
            out = torch.matmul(attn, vi)
            out = rearrange(out, 'b h n d -> b n (h d)')
            pool.append(out)

        out = torch.cat(tuple(pool), dim=1)
        out = out.transpose(-1, -2)
        out = self.expansion(out)
        return out        

#Residual transformer block
class Transformer1(nn.Module):
    def __init__(self, nf, splitfactors=4, heads=8, k=3):
        super(Transformer1, self).__init__()
        self.k = k
        self.unFold1 = nn.Unfold(kernel_size=(k, k), padding=1)
        self.emha1 = EMHA(nf*k*k, splitfactors, heads)
        self.norm1 = nn.LayerNorm(nf*k*k)
        self.conv1 = conv_block(nf, nf, kernel_size=3)
        self.unFold2 = nn.Unfold(kernel_size=(k, k), padding=1)
        self.norm2 = nn.LayerNorm(nf*k*k)
        self.emha2 = EMHA(nf*k*k, splitfactors, heads)
        self.conv2 = conv_block(nf, nf, kernel_size=3)
        self.unFold3 = nn.Unfold(kernel_size=(k, k), padding=1)
        self.norm3 = nn.LayerNorm(nf*k*k)
        self.emha3 = EMHA(nf*k*k, splitfactors, heads)
        self.conv3 = conv_block(nf, nf, kernel_size=3)
    def forward(self,x):

        _, _, h, w = x.shape
        #rt1 = self.emha1(self.unFold1(x))
        #residual block 1
        rt1 = self.unFold1(x)
        rt13 = rt1.transpose(-2, -1)
        rt13 = self.norm1(rt13)
        rt13 = rt13.transpose(-2, -1)
        rt13 = self.emha1(rt13)+rt1
        rt11 = F.fold(rt13, output_size=(h, w), kernel_size=(self.k, self.k), padding=(1, 1))
        rt11 = rt11+x
        #print(rt11.shape)
        rt12 = self.conv1(rt11)
        #residual block 2
        rt2 = self.unFold2(rt12)
        rt23 = rt2.transpose(-2, -1)
        rt23 = self.norm2(rt23)
        rt23 = rt23.transpose(-2, -1)
        rt23 =self.emha2(rt23)+rt2
        rt21 = F.fold(rt23, output_size=(h, w), kernel_size=(self.k, self.k), padding=(1, 1))
        rt21 = rt21+rt12
        #print(rt21.shape)
        rt22 = self.conv2(rt21)
        #residual block 3
        rt3 = self.unFold3(rt22)
        rt33 = rt3.transpose(-2, -1)
        rt33 = self.norm3(rt33)
        rt33 = rt33.transpose(-2, -1)
        rt33 = self.emha3(rt33)+rt3
        rt31 = F.fold(rt33, output_size=(h, w), kernel_size=(self.k, self.k), padding=(1, 1))
        rt31 = rt31+rt22
        #print(rt31.shape)
        rt32 = self.conv3(rt31)
        rt32 = rt32+x
        return rt32


class VIT(nn.Module):
    def __init__(self, in_nc, nf, splitfactors=4, heads=8,norm_type='batch', act_type='elu'):
        super(VIT, self).__init__()
        self.FUP = nn.Upsample(scale_factor=4, mode='bicubic')

        
        #self.conv1 = conv_block(nf,nf,kernel_size=3,norm_type=norm_type,act_type=act_type)
        self.sfe = conv_block(in_nc, nf, kernel_size=3, act_type='elu')
        self.sfe0 = base_block(nf, kernel_size=3, norm_type=norm_type,act_type=act_type)
        self.RT1 = Transformer1(nf, splitfactors, heads)
        self.RT2 = Transformer1(nf, splitfactors, heads)
        self.RT3 = Transformer1(nf, splitfactors, heads)
        self.RT4 = Transformer1(nf, splitfactors, heads)
        self.RT5 = Transformer1(nf, splitfactors, heads)
      
        self.c1 = conv_block(nf, nf, kernel_size=3, act_type=None,norm_type=None)
        self.up = UP_Sample(nf,nf, kernel_size=3, act_type='elu',upscale_factor=4)
        #self.up1 = upconv_blcok(nf, nf, upscale_factor=4)
        self.c2 = conv_block(nf, 3, kernel_size=3)
    def forward(self, x):
       # xconv=self.conv1(x)
        
        
        x1 = self.sfe(x)
        xFUP = self.FUP(x)
        xbase=self.sfe0(x1)
        x2 = self.RT1(xbase)
        x3 = self.RT2(x2)
        x4 = self.RT3(x3)
        x5 = self.RT4(x4)
        x6 = self.RT5(x5)
      
        xa = self.c1(x6)
        xa = xa+xbase
        xu1 = self.up(xa)
        xout = xu1+xFUP
        return xout



class high_low_network(nn.Module):
    def __init__(self, in_nc, out_nc,nb=20, nf=32,splitfactors=4, heads=8, kernel_size=3,norm_type='batch', act_type='leakyrelu'):
        super(high_low_network, self).__init__()

        self.FUP = nn.Upsample(scale_factor=4, mode='bicubic')

        
        #self.conv1 = conv_block(nf,nf,kernel_size=3,norm_type=norm_type,act_type=act_type)
        self.sfe = conv_block(in_nc, nf, kernel_size=3, act_type='leakyrelu')
        #self.hfm=HFM(k=2)
        self.SA= SpatialGate()
        self.sfe0 = Residual(nf)
        #self.RT1 = Transformer1(nf, splitfactors, heads)
        #self.RT2 = Transformer1(nf, splitfactors, heads)
        #self.RT3 = Transformer1(nf, splitfactors, heads)
        #self.RT4 = Transformer1(nf, splitfactors, heads)
        #self.RT5 = Transformer1(nf, splitfactors, heads)
        #self.RT6 = Transformer1(nf, splitfactors, heads)
      
        self.c1 = conv_block(nf, nf, kernel_size=3, act_type=None,norm_type=None)
        self.up = UP_Sample(nf,nf, kernel_size=3, act_type='leakyrelu',upscale_factor=4)
        #self.up1 = upconv_blcok(nf, nf, upscale_factor=4)
        self.c2 = conv_block(nf, 3, kernel_size=3)
        #self.CA= ChannelAttention(nf)
    def forward(self, x):
       # xconv=self.conv1(x)
        
        
        x1 = self.sfe(x)
        xSA=self.SA(x1)
        #xhfm=self.hfm(x1)
        #x8=self.CA(x1)
        xFUP = self.FUP(x)
        xbase=self.sfe0(xSA)
        #x2 = self.RT1(xbase)
        #x3 = self.RT2(x2)
        #x4 = self.RT3(x3)
        #x5 = self.RT4(x4)
        #x6 = self.RT5(x5)
        #x7 = self.RT6(x6)
      
        xa = self.c1(xbase)
        xu1 = self.up(xa)
        xout = xu1+xFUP
        return xout


class VGG_Block(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3,norm_type='batch', act_type='leakyrelu'):
        super(VGG_Block, self).__init__()

        self.conv0 = conv_block(in_nc, out_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.conv1 = conv_block(out_nc, out_nc, kernel_size=kernel_size, stride=2, norm_type=None,act_type=act_type)

    def forward(self, x):
        x1 = self.conv0(x)
        out = self.conv1(x1)
        
        return out


class VGGGAPQualifier(nn.Module):
    def __init__(self, in_nc=3, base_nf=32, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(VGGGAPQualifier, self).__init__()
        # 1024,768,3

        B11 = VGG_Block(in_nc,base_nf,norm_type=norm_type,act_type=act_type)
        # 512,384,32
        B12 = VGG_Block(base_nf,base_nf,norm_type=norm_type,act_type=act_type)
        # 256,192,32
        B13 = VGG_Block(base_nf,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 128,96,64
        B14 = VGG_Block(base_nf*2,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 64,48,64

        # 1024,768,3
        B21 = VGG_Block(in_nc,base_nf,norm_type=norm_type,act_type=act_type)
        # 512,384,32
        B22 = VGG_Block(base_nf,base_nf,norm_type=norm_type,act_type=act_type)
        # 256,192,32
        B23 = VGG_Block(base_nf,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 128,96,64
        B24 = VGG_Block(base_nf*2,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 64,48,64


        B3 = VGG_Block(base_nf*2,base_nf*4,norm_type=norm_type,act_type=act_type)
        # 32,24,128
        B4 = VGG_Block(base_nf*4,base_nf*8,norm_type=norm_type,act_type=act_type)
        # 16,12,256
        B5 = VGG_Block(base_nf*8,base_nf*16,norm_type=norm_type,act_type=act_type)
        
        self.feature1 = sequential(B11,B12,B13,B14)
        self.feature2 = sequential(B21,B22,B23,B24)

        self.combine = sequential(B3,B4,B5)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # classifie
        self.classifier = nn.Sequential(
            nn.Linear(base_nf*16, 512), nn.LeakyReLU(0.2, True), nn.Dropout(0.25), nn.Linear(512,256),nn.LeakyReLU(0.2, True), nn.Dropout(0.5), nn.Linear(256, 1), nn.LeakyReLU(0.2, True))

    def forward(self, x):

        f1 = self.feature1(x)
        f2 = self.feature2(x)
        x = self.gap(self.combine(f1-f2))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res

class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x




####################
# Upsampler
####################


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='leakyrelu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='leakyrelu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)

def downconv_blcok(in_nc, out_nc, downscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='leakyrelu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    f = 0.5
    upsample = nn.Upsample(scale_factor=f)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)
