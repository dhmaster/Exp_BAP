import math
import os
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchinfo import summary
from utils import BAP1,ResizeCat

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResidualBlock(nn.Module):
    """
    3*3 64
    3*3 64
    """
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride,padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        # if self.right is None:
        #     residual = x
        # else:
        #     residual = self.right(x)
        out += residual
        return F.relu(out)

class Bottleneck(nn.Module):
    """
    1*1 64
    3*3 64
    1*1 256
    """
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_places, places, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(places, places, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(places, places*self.expansion, kernel_size=1,
                      stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion)
        )
        if self.downsampling:
            self.downsampling = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsampling(x)
        out += residual
        out = self.relu(out)
        return out
"""
# ResNet-18 and ResNet-34

class ResNet(nn.Module):
    # Using Residual block

    def __init__(self, blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.mdoel_name = 'ResNet34'
        # 前几层，图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # 重复 layer：3, 4, 6, 3
        self.layer1 = self._make_layer(64, 64, blocks[0])
        self.layer2 = self._make_layer(64, 128, blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, blocks[3],stride=2)

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks_num, stride=1):
        
        # build layers which consist of several residual blocks
        
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, shortcut))
        for i in range(1, blocks_num):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)

        return self.fc(x)

def ResNet18():
    return ResNet([2, 2, 2, 2])

def ResNet34():
    return ResNet([3, 4, 6, 3])

# ResNet-50、 ResNet-101 and ResNet-152
"""

def Conv1(in_channels, out_channels, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1,ceil_mode=True)
    )


# num_features:CNN输出值
# M：M个attentions map
# attentions = BasicConv2d(num_features, M, kernel_size=1)
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

# self.spp = SPPLayer(pool_size=[1, 2, 4])
# self.fc = nn.Linear(512 * block.expansion * self.spp.out_length, num_classes)
class SPPLayer(nn.Module):
    def __init__(self, pool_size, pool=nn.MaxPool2d):
        super(SPPLayer, self).__init__()
        self.pool_size = pool_size
        self.pool = pool
        self.out_length = np.sum(np.array(self.pool_size) ** 2)

    def forward(self, x):
        B, C, H, W = x.size()
        for i in range(len(self.pool_size)):
            h_wid = int(math.ceil(H / self.pool_size[i]))
            w_wid = int(math.ceil(W / self.pool_size[i]))
            h_pad = (h_wid * self.pool_size[i] - H + 1) / 2
            w_pad = (w_wid * self.pool_size[i] - W + 1) / 2
            out = self.pool((h_wid, w_wid), stride=(h_wid, w_wid), padding=int(h_pad))(x)
            if i == 0:
                spp = out.view(B, -1)
            else:
                spp = torch.cat([spp, out.view(B, -1)], dim=1)
        return spp

class ResNetx(nn.Module):
    def __init__(self, blocks, expansion=4):
        super(ResNetx, self).__init__()
        self.expansion = expansion
        self.Conv1 = Conv1(in_channels=3, out_channels=64)
        self.layer1 = self.make_layer(64, 64, blocks=blocks[0], stride=1)
        self.layer2 = self.make_layer(256, 128, blocks= blocks[1], stride=2)
        self.layer3 = self.make_layer(512, 256, blocks=blocks[2],stride=2)
        self.layer4 = self.make_layer(1024, 512, blocks=blocks[3],stride=2)
        self.att = BasicConv2d(2048, 1024, kernel_size=1)
        self.upsample = ResizeCat()
        self.bap = BAP1()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(4096),
            nn.ELU(inplace=True),
            nn.Linear(4096, 200)
        )
        self.spp = SPPLayer(pool_size=[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.Conv1(x)
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)

        att3 = self.att(x3)
        fea3 = self.bap(x3, att3)
        in_2 = self.upsample(x2,fea3) # N*2048*28*28

        att2 = self.att(in_2)
        fea2 = self.bap(in_2, att2)
        in_1 = self.upsample(x1, fea2) # N*2048*56*56

        att1 = self.att(in_1)
        fea1 = self.bap(in_1, att1)
        fc = self.spp(fea1)
        fc = torch.flatten(fc, 1)
        pre_raw = self.classifier(fc)
        print(fc.shape,fea1.shape)

        return pre_raw

    def make_layer(self, in_places, places, blocks, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, blocks):
            layers.append(Bottleneck(places*self.expansion, places))
        return nn.Sequential(*layers)

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        if len(pretrained_dict) == len(state_dict):
            print('%s: All params loaded' % type(self).__name__)
        else:
            print('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
        model_dict.update(pretrained_dict)
        super(ResNetx, self).load_state_dict(model_dict)


def ResNet50(pretrained=True):
    model = ResNetx([3, 4, 6, 3])
    if pretrained:
        model_weight_path = "./resnet50-19c8e357.pth"
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        pre_state_dict = torch.load(model_weight_path)
        model.load_state_dict(pre_state_dict, True)
    return model

def ResNet101():
    return ResNetx([3, 4, 23, 3])

def ResNet152():
    return ResNetx([3, 8, 36, 3])

if __name__ == '__main__':
    net = ResNet50()
    # a = torch.rand([4, 3, 448, 448])
    # b = net(a)
    # print(b[0].shape)
    print(net)
    # summary(net, input_size=(1, 3, 448, 448))
