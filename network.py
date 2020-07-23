import torch
import torch.nn as nn
from network_module import *
import torch.nn.functional as F
import numpy as np
import pixel_unshuffle
import cv2

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):

        classname = m.__class__.__name__

        if hasattr(m, 'weight') and classname.find('Conv') != -1:

            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
             torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
             torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
             torch.nn.init.normal_(m.weight, 0, init_gain)
             # torch.nn.init.constant_(m.bias, 0)


    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)




class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,opt, in_dim, reduction = 8):
        super(Self_Attn,self).__init__()

        self.reduction = reduction
        self.chanel_in = in_dim
        
        self.query_conv = Conv2dLayer(in_dim, in_dim//reduction, 1, 1, 0, pad_type = opt.pad, norm = 'none', activation = 'none')
        self.key_conv = Conv2dLayer(in_dim, in_dim//reduction, 1, 1, 0, pad_type = opt.pad, norm = 'none', activation = 'none')
        self.value_conv = Conv2dLayer(in_dim, in_dim//reduction, 1, 1, 0, pad_type = opt.pad, norm = 'none', activation = 'none')

        self.conv = Conv2dLayer(in_dim//reduction, in_dim, 1, 1, 0, pad_type = opt.pad, norm = 'none', activation = 'none')
        #self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X N X C
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x N
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # B X N X N 

        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)# B X N X C
        out =  torch.bmm(attention,proj_value) #B X N X C
        
        out = out.permute(0,2,1).view(m_batchsize,self.chanel_in,width,height)
        out = self.conv(out)

        #out = self.gamma*out + x
        return out


class Channel_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,opt, channel):
        super(Channel_Attn,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel, bias=False)
        self.fc2 = nn.Linear(channel, channel, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace = True)
        self.sigmoid  = nn.Sigmoid()

    
    def forward(self,x):

        batch_size,channel,height ,width = x.size()
        y = self.avg_pool(x).view(batch_size, channel)
        y = self.fc1(y)
        y = self.lrelu(y)
        y = self.fc2(y)

        y = self.sigmoid(y).view(batch_size, channel, 1, 1)
        out =  x * y.expand_as(x)
        return out


##########################################################################
##---------- Spatial Attention ----------
class SA(nn.Module):
    def __init__(self, opt, kernel_size = 5, bias = False):
        super(SA, self).__init__()
        self.conv = Conv2dLayer(2, 1, kernel_size, 1, (kernel_size-1) // 2, pad_type = opt.pad, norm = 'none', bias = bias, activation = 'sigmoid')
    
    def forward(self, x):

        GMP = torch.max(x,1)[0].unsqueeze(1)
        GAP = torch.mean(x,1).unsqueeze(1)
        out = torch.cat([GMP, GAP], dim=1)
        out = self.conv(out)
        return x * out

##########################################################################
## ------ Channel Attention --------------
class CA(nn.Module):
    def __init__(self, opt, channel, reduction=8, bias=False):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            Conv2dLayer(channel, channel // reduction, 1, 1, 0, norm = 'none', bias = bias),
            Conv2dLayer(channel // reduction, channel, 1, 1, 0, norm = 'none', bias = bias, activation='sigmoid'),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y

##########################################################################
##---------- Dual Attention ----------
class DA(nn.Module):
    def __init__(self, opt, channel, kernel_size=3, reduction=8, bias=False):
        super(DA, self).__init__()

        modules_body = [Conv2dLayer(channel, channel, kernel_size, 1, 1, pad_type = opt.pad, norm = opt.norm, bias = bias),
                        Conv2dLayer(channel, channel, kernel_size, 1, 1, pad_type = opt.pad, norm = opt.norm, activation = 'none', bias = bias)]
        self.body = nn.Sequential(*modules_body)
        
        ## Spatial Attention
        self.SA = SA(opt)

        ## Channel Attention        
        self.CA = CA(opt, channel,reduction, bias=bias)

        self.conv = Conv2dLayer(channel*2, channel, 1, 1, 0, pad_type = opt.pad, norm = opt.norm, bias = bias, activation = 'none')

    def forward(self, x):
        res = self.body(x)
        SA = self.SA(res)
        CA = self.CA(res)
        res = torch.cat([SA, CA], dim=1)
        res = self.conv(res)
        res += x
        return res


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        #Level 3
        self.lv3_layer1 = Conv2dLayer(opt.in_channels * (4 ** 3), opt.latent_channels * (2 ** 3), 3, 1, 1, pad_type = opt.pad, norm = 'none')
        self.lv3_layer2 = Conv2dLayer(opt.latent_channels * (2 ** 3), opt.latent_channels * (2 ** 3), 3, 1, 1, pad_type = opt.pad, norm = opt.norm)

        # Bottle neck
        self.BottleNeck = nn.Sequential(
            ResConv2dLayer(opt.latent_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, dilation = 1),
            ResConv2dLayer(opt.latent_channels * 8, 3, 1, 2, pad_type = opt.pad, norm = opt.norm, dilation = 2),
            ResConv2dLayer(opt.latent_channels * 8, 3, 1, 4, pad_type = opt.pad, norm = opt.norm, dilation = 4),
            ResConv2dLayer(opt.latent_channels * 8, 3, 1, 8, pad_type = opt.pad, norm = opt.norm, dilation = 8),
            ResConv2dLayer(opt.latent_channels * 8, 3, 1, 16, pad_type = opt.pad, norm = opt.norm, dilation = 16),
        )

        #local feature branch
        self.FL1 = Conv2dLayer(opt.latent_channels * (2 ** 3), opt.latent_channels * (2 ** 3), 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.FL2 = Conv2dLayer(opt.latent_channels * (2 ** 3), opt.latent_channels * (2 ** 3), 3, 1, 1, pad_type = opt.pad, norm = opt.norm)

        #Global feature branch
        self.GL1 = Conv2dLayer(opt.latent_channels * (2 ** 3), opt.latent_channels * (2 ** 3), 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.GL2 = Conv2dLayer(opt.latent_channels * (2 ** 3), opt.latent_channels * (2 ** 3), 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.GL_FC1 = nn.Linear(opt.latent_channels * (2 ** 3), opt.latent_channels * (2 ** 3))
        self.GL_FC2 = nn.Linear(opt.latent_channels * (2 ** 3), opt.latent_channels * (2 ** 3))

        self.lv3_layer3 = DA(opt, opt.latent_channels * (2 ** 3))
        
        #Level 2
        self.lv2_layer1 = Conv2dLayer(opt.in_channels * (4 ** 2), opt.latent_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, norm = 'none')
        self.lv2_layer2 = Conv2dLayer(int(opt.latent_channels * (2 ** 2 + 2 ** 3 / 4)), opt.latent_channels * (2 ** 2), 1, 1, 0, pad_type = opt.pad, norm = opt.norm)
        self.lv2_layer3 = DA(opt, opt.latent_channels * (2 ** 2))
        
        #Level 1
        self.lv1_layer1 = Conv2dLayer(opt.in_channels * (4 ** 1), opt.latent_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, norm = 'none')
        self.lv1_layer2 = Conv2dLayer(int(opt.latent_channels * (2 ** 1 + 2 ** 2 / 4)), opt.latent_channels * (2 ** 1), 1, 1, 0, pad_type = opt.pad, norm = opt.norm)
        self.lv1_layer3 = DA(opt, opt.latent_channels * (2 ** 1))

        #Level 0
        self.lv0_layer1 = Conv2dLayer(opt.in_channels, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none')
        self.lv0_layer2 = Conv2dLayer(int(opt.latent_channels * (2 ** 0 + 2 ** 1 / 4)), opt.latent_channels, 1, 1, 0, pad_type = opt.pad, norm = opt.norm)
        self.lv0_layer3 = DA(opt, opt.latent_channels)

        self.upsample = TransposeConv2dLayer(opt.latent_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, scale_factor = 2, activation = 'tanh', norm = 'none')
        self.avg_pool = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):


        x1 = pixel_unshuffle.pixel_unshuffle(x, 2)
        x2 = pixel_unshuffle.pixel_unshuffle(x, 4)
        x3 = pixel_unshuffle.pixel_unshuffle(x, 8)

        x3 = self.lv3_layer1(x3)
        x3 = self.lv3_layer2(x3)
        x3 = self.BottleNeck(x3)
        
        FL1 = self.FL1(x3)
        FL2 = self.FL2(FL1)

        GL1 = self.GL1(x3)
        GL2 = self.GL2(GL1)
        out = self.avg_pool(GL2).view(GL2.shape[0], GL2.shape[1])
        GL_FC1 = self.GL_FC1(out)
        GL_FC2 = self.GL_FC2(GL_FC1)

        out = GL_FC2.view(GL_FC2.shape[0], GL_FC2.shape[1], 1, 1)
        GL = out.expand_as(FL2)
        out = FL2 + GL

        x3 = F.pixel_shuffle(x3, 2)
        x2 = self.lv2_layer1(x2)
        x2 = torch.cat([x2, x3], dim = 1)
        x2 = self.lv2_layer2(x2)
        x2 = self.lv2_layer3(x2)

        x2 = F.pixel_shuffle(x2, 2)
        x1 = self.lv1_layer1(x1)
        x1 = torch.cat([x1, x2], dim = 1)
        x1 = self.lv1_layer2(x1)
        x1 = self.lv1_layer3(x1)

        x1 = F.pixel_shuffle(x1, 2)
        x = self.lv0_layer1(x) 
        x = torch.cat([x, x1], dim = 1)
        x = self.lv0_layer2(x)
        x = self.lv0_layer3(x)
        x = self.upsample(x)

        return x
