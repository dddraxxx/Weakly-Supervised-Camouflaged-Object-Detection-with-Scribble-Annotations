#v46
import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: '+n)
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.GELU) or isinstance(m, nn.LeakyReLU) or isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.ReLU6) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Softmax):
            pass
        elif isinstance(m, nn.ModuleList):
            weight_init(m)
        else:
            m.initialize()

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

# Feature Fusion Module
class FFM(nn.Module):
    def __init__(self, channel):
        super(FFM, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        out = x_1 * x_2
        #out = torch.cat((x_1, x_2), dim=1)
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)

# Cross Aggregation Module
class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.down = nn.Sequential(
            conv3x3(channel, channel, stride=2),
            nn.BatchNorm2d(channel)
        )
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.mul = FFM(channel)

    def forward(self, x_high, x_low):
        left_1 = x_low
        left_2 = F.relu(self.down(x_low), inplace=True)
        right_1 = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        right_2 = x_high
        left = F.relu(self.bn_1(self.conv_1(left_1 * right_1)), inplace=True)
        right = F.relu(self.bn_2(self.conv_2(left_2 * right_2)), inplace=True)
        # left = F.relu(left_1 * right_1, inplace=True)
        # right = F.relu(left_2 * right_2, inplace=True)
        right = F.interpolate(right, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        out = self.mul(left, right)
        return out

    def initialize(self):
        weight_init(self)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

# Boundary Refinement Module
class BRM(nn.Module):
    def __init__(self, channel):
        super(BRM, self).__init__()
        self.conv_atten = conv1x1(channel, channel)
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_edge):
        # x = torch.cat([x_1, x_edge], dim=1)
        x = x_1 + x_edge
        atten = F.avg_pool2d(x, x.size()[2:])
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten) + x
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)
 
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            basicConv(in_channel, out_channel, 1, relu=False),
        )
        self.branch1 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch2 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch3 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.conv_cat = basicConv(4*out_channel, out_channel, 3, p=1, relu=False)
        self.conv_res = basicConv(in_channel, out_channel, 1, relu=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
        
    def initialize(self):
        weight_init(self)

############################################# ResNet50 #############################################
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)
    
    def initialize(self):
        weight_init(self)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('./assets/resnet50-19c8e357.pth'), strict=False)


############################################# Pooling ##############################################
class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(basicConv, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
            #conv.append(nn.LayerNorm(out_channel, eps=1e-6))
        if relu:
            conv.append(nn.GELU())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
    
    def initialize(self):
        weight_init(self)

class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = basicConv(in_channel*2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = F.interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = F.interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x
        
    def initialize(self):
        weight_init(self)

########################################### CoordAttention #########################################
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

    def initialize(self):
        weight_init(self)

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

    def initialize(self):
        weight_init(self)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    
    def initialize(self):
        weight_init(self)

####################################### Contrast Texture ###########################################
class Contrast_Block_Deep(nn.Module):
    def __init__(self, planes, d1=4, d2=8):
        super(Contrast_Block_Deep, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(planes / 2)

        self.local_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d1, dilation=d1)

        self.local_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d2, dilation=d2)

        self.bn1 = nn.BatchNorm2d(self.outplanes)
        self.bn2 = nn.BatchNorm2d(self.outplanes)


        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        self.ca = nn.ModuleList([
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes)
        ])


    def forward(self, x):
        local_1 = self.local_1(x)
        local_1 = self.ca[0](local_1)
        context_1 = self.context_1(x)
        context_1 = self.ca[1](context_1)
        ccl_1 = local_1 - context_1
        ccl_1 = self.bn1(ccl_1)
        ccl_1 = self.relu1(ccl_1)

        local_2 = self.local_2(x)
        local_2 = self.ca[2](local_2)
        context_2 = self.context_2(x)
        context_2 = self.ca[3](context_2)
        ccl_2 = local_2 - context_2
        ccl_2 = self.bn2(ccl_2)
        ccl_2 = self.relu2(ccl_2)

        out = torch.cat((ccl_1, ccl_2), 1)

        return out

    def initialize(self):
        weight_init(self)
    
class RW_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, shrink_factor):
        super(RW_Module, self).__init__()
        self.chanel_in = in_dim
        self.shrink_factor = shrink_factor

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        
    def own_softmax1(self, x):
    
        maxes1 = torch.max(x, 1, keepdim=True)[0]
        maxes2 = torch.max(x, 2, keepdim=True)[0]
        x_exp = torch.exp(x-0.5*maxes1-0.5*maxes2)
        x_exp_sum_sqrt = torch.sqrt(torch.sum(x_exp, 2, keepdim=True))

        return (x_exp/x_exp_sum_sqrt)/torch.transpose(x_exp_sum_sqrt, 1, 2)
    
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        x_shrink = x
        m_batchsize, C, height, width = x.size()
        if self.shrink_factor != 1:
            x_shrink = F.interpolate(x_shrink, scale_factor=self.shrink_factor, mode='bilinear', align_corners=True)
            height = x_shrink.size(-2)
            width = x_shrink.size(-1)
            
        
        proj_query = self.query_conv(x_shrink).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x_shrink).view(m_batchsize, -1, width*height)
        
        energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)

        proj_value = self.value_conv(x_shrink).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        
        # if self.shrink_factor != 1:
        #     height = (height - 1) * self.shrink_factor + 1
        #     width = (width - 1) * self.shrink_factor + 1
        #     out = F.interpolate(out, size=(height, width), mode='bilinear', align_corners=True)

        out = self.gamma*out + (1-self.gamma)*x_shrink
        return out #,energy
    
    def initialize(self):
        weight_init(self)

################################################ Net ###############################################
class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.bkbone = ResNet()

        # new added
        self.rw = RW_Module(64, 2)

        self.pyramid_pooling = PyramidPooling(2048, 64)

        self.conv1 = nn.ModuleList([
            basicConv(64, 64, k=1, s=1, p=0),
            basicConv(256, 64, k=1, s=1, p=0),
            basicConv(512, 64, k=1, s=1, p=0),
            basicConv(1024, 64, k=1, s=1, p=0),
            basicConv(2048, 64, k=1, s=1, p=0),
            #basicConv(2048, 2048, k=1, s=1, p=0)
        ])

        
        self.rfb = nn.ModuleList([
             RFB_modified(1024, 64),
             RFB_modified(2048, 64)
        ])

        self.contrast = nn.ModuleList([
            Contrast_Block_Deep(64),
            Contrast_Block_Deep(64)
        ])
        
        self.ca = nn.ModuleList([
            CoordAtt(64, 64),
            CoordAtt(64, 64),
            CoordAtt(64, 64),
            CoordAtt(64, 64),
            CoordAtt(64, 64)
        ])
        

        self.fusion = nn.ModuleList([
            FFM(64),
            FFM(64),
            FFM(64),
            FFM(64)
        ])

        self.aggregation = nn.ModuleList([
            CAM(64),
            CAM(64)
        ])

        self.edge_extract = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), 
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(), 
                                          nn.Conv2d(64, 64, 1, 1, 0), 
                                          nn.BatchNorm2d(64),
                                          nn.ReLU())

        self.refine = BRM(64)
        #self.conv2 = basicConv(128, 64, k=1, s=1, p=0)

        self.edge_head = nn.Conv2d(64, 1, 3, 1, 1)
        self.head = nn.ModuleList([
                conv3x3(64, 1, bias=True),
                conv3x3(64, 1, bias=True),
                conv3x3(64, 1, bias=True),
                conv3x3(64, 1, bias=True),
                conv3x3(64, 1, bias=True),
        ])

        self.feature_head = nn.ModuleList([
            basicConv(64, 32, relu=False)
            # conv3x3(64, 64, bias=True)
        ])
    
        self.initialize()


    def forward(self, x, shape=None, epoch=None):
        shape = x.size()[2:] if shape is None else shape
        bk_stage1, bk_stage2, bk_stage3, bk_stage4, bk_stage5 = self.bkbone(x)
        f_c3 = self.pyramid_pooling(bk_stage5)
        
        f_c2 = self.rfb[1](bk_stage5)
        #f_c2 = self.ca[1](f5)#512
        fused3 = F.interpolate(f_c3, size=f_c2.size()[2:], mode='bilinear', align_corners=True)
        fused3 = self.fusion[2](f_c2, fused3)

        f_c1 = self.rfb[0](bk_stage4)
        fused2 = F.interpolate(f_c2, size=f_c1.size()[2:], mode='bilinear', align_corners=True)
        fused2 = self.fusion[1](f_c1, fused2)

        f_t2 = self.conv1[2](bk_stage3)
        #f_t2 = self.tem[1](f3)
        f_t2 = self.contrast[1](f_t2)
        #f_t2 = self.ulsam[1](f_t2)
        #f_t2 = self.ca[3](f_t2)

        a2 = F.interpolate(fused3, size=[f_t2.size(2)//2, f_t2.size(3)//2], mode='bilinear', align_corners=True)
        a2 = self.aggregation[1](a2, f_t2)

        f_t1 = self.conv1[1](bk_stage2)
        f_t1 = self.contrast[0](f_t1)

        a1 = F.interpolate(fused2, size=[f_t1.size(2)//2, f_t1.size(3)//2], mode='bilinear', align_corners=True)
        a1 = self.aggregation[0](a1, f_t1)
        
        a2 = F.interpolate(a2, size=a1.size()[2:], mode='bilinear', align_corners=True)
        out0 = self.fusion[0](a1, a2)
        
        out0 = F.interpolate(self.head[0](out0), size=shape, mode='bilinear', align_corners=False)

        if self.cfg.mode == 'train':
            out1 = F.interpolate(self.head[1](a1), size=shape, mode='bilinear', align_corners=False)
            # print(not x.isnan().any())
            out2 = F.interpolate(self.head[2](a2), size=shape, mode='bilinear', align_corners=False)
            out3 = F.interpolate(self.head[3](fused2), size=shape, mode='bilinear', align_corners=False)
            out4 = F.interpolate(self.head[4](fused3), size=shape, mode='bilinear', align_corners=False)
            return out0, None, out1, out2, out3, out4
        else:
            return out0, None

    def initialize(self):
        print('initialize net')
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot), strict=False)
        else:
            weight_init(self)