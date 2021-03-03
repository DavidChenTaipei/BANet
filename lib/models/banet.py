import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo
from torch.nn import BatchNorm2d

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_chan),
                )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        self.init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x) # channel: 1/8
        feat16 = self.layer3(feat8) # channel : 1/4
        feat32 = self.layer4(feat16) # channel : 1/2
        
        return feat8, feat16, feat32
    
    def init_weights(self):
        state_dict = modelzoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=1, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat
    
class PFB(nn.Module):
    def __init__(self,channel):
        super(PFB, self).__init__()
        self.avgpooling = nn.AvgPool2d(3,stride=1,padding=1)
        self.cbr = ConvBNReLU(channel*2, 128, ks=1, stride=1,padding=0)

    def forward(self, low,high):
        high = self.avgpooling(high)
        mix = torch.cat([low, high], dim=1)
        out = self.cbr(mix)
        return out
    
class CA(nn.Module):
    def __init__(self,channel):
        super(CA, self).__init__()
        ## Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channel,channel,kernel_size=(1,1))
        self.bn = nn.BatchNorm2d(channel)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        feat = self.gap(x)
        feat = self.conv(feat)
        feat = self.bn(feat)
        feat = self.sigmoid(feat)
        upsampled_feat = F.interpolate(feat,size=((x.shape)[-2],x.shape[-1]), mode='nearest')
        out = torch.mul(x,upsampled_feat)
        return out

class SA(nn.Module):
    def __init__(self,channel):
        super(SA, self).__init__()
        self.cbr = ConvBNReLU(channel, channel//2, ks=1, stride=1,padding=0)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(2,channel,kernel_size=1,stride=1,padding=0)
        self.bn = nn.BatchNorm2d(channel)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        feat = self.cbr(x)
        avg_out = torch.mean(feat, dim=1, keepdim=True) ##pixel-wise average pooling
        max_out, _ = torch.max(feat, dim=1, keepdim=True) ##pixel-wise max pooling
        feat = torch.cat((avg_out, max_out), dim=1)
        feat = self.conv(feat)
        feat = self.bn(feat)
        feat = self.sigmoid(feat)
        out = torch.mul(x,feat)
        return out
    
class BA(nn.Module):
    def __init__(self,channel):
        super(BA, self).__init__()
        self.sa = SA(channel=channel)
        self.conv = nn.Conv2d(channel,channel,kernel_size=(3,3),padding=1)
        self.ca = CA(channel=channel)
        self.pfb = PFB(channel=channel)
    def forward(self,low,high):
        refined_low = self.sa(low)
        refined_low = self.conv(refined_low)
        refined_high = self.ca(high)
        upsampled_high= F.interpolate(refined_high,size=((refined_low.shape)[-2],(refined_low.shape)[-1]), mode='nearest')
        out = self.pfb(low = refined_low,high = upsampled_high)
        return out

class Output_Block(nn.Module):
    def __init__(self,channel):
        super(Output_Block, self).__init__() 
        self.cbr = ConvBNReLU(channel, channel , ks=3, stride=1,padding=1)
        self.conv = nn.Conv2d(channel,19,kernel_size=1,padding=1)
    def forward(self,x):
        feat = self.cbr(x)
        #print('feat',feat.shape)
        feat = self.conv(feat)
        #print('ready out',feat.shape)
        out = F.interpolate(feat,size=(input_h,input_w), mode='nearest')
        return out

class decoder(nn.Module):
    def __init__(self,channel):
        super(decoder, self).__init__() 
        self.conv_for_block4 = ConvBNReLU(channel//2, channel//4 , ks=3, stride=1,padding=1)
        self.ba_block3 = BA(channel//4)
        self.conv_for_ba = ConvBNReLU(128, 128 , ks=3, stride=1,padding=1)
        self.ba_block2 = BA(channel//8)
        
        self.output_block3 = Output_Block(128)
        self.output_block4 = Output_Block(128)
        self.init_weights()
    def forward(self,block2,block3,block4):
        conv_block4 = self.conv_for_block4(block4)
        ba_block3 = self.ba_block3(low = block3,high = conv_block4)
        
        conv_block3 = self.conv_for_ba(ba_block3)
        ba_block2 = self.ba_block2(low = block2,high = conv_block3)
        
        out_middle = self.output_block3(ba_block3)
        out_final = self.output_block4(ba_block2)
        return out_middle, out_final
    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
class model(nn.Module):
    def __init__(self,channel):
        super(model, self).__init__()
        self.resnet18 = Resnet18()
        self.decoder = decoder(channel)
    def forward(self,x):
        global input_h,input_w
        input_h = x.shape[-2]
        input_w = x.shape[-1]
        block2,block3,block4 = self.resnet18(x)
        out_middle,out_findal = self.decoder(block2 = block2, block3 = block3,block4 = block4)
        return out_middle,out_findal
