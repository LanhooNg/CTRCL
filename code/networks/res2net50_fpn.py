import torch
import torch.nn as nn
from networks.Res2Net_v1b import res2net50_v1b_26w_4s


class CBR(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CR(nn.Module):

    def __init__(self, in_channels=256, out_channels=64):
        super(CR, self).__init__()
        self.proj = CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.proj(x)
        return out


class FB(nn.Module):
    """
    Fusion block
    """

    def __init__(self, in_channels=128, out_channels=64):
        super(FB, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.proj = CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x)
        return x


class Res2Net(nn.Module):

    def __init__(self, num_class):
        super(Res2Net, self).__init__()

        self.backbone = res2net50_v1b_26w_4s()

        self.cr_1 = CR(in_channels=256, out_channels=64)
        self.cr_2 = CR(in_channels=512, out_channels=64)
        self.cr_3 = CR(in_channels=1024, out_channels=64)
        self.cr_4 = CR(in_channels=2048, out_channels=64)

        self.fb_1 = FB(in_channels=128, out_channels=64)
        self.fb_2 = FB(in_channels=128, out_channels=64)
        self.fb_3 = FB(in_channels=128, out_channels=64)

        self.out_conv = nn.Conv2d(in_channels=64, out_channels=num_class, kernel_size=3, stride=1, padding=1)
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self._init_weights()

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        f0 = self.backbone.maxpool(x)

        f1 = self.backbone.layer1(f0)
        f2 = self.backbone.layer2(f1)
        f3 = self.backbone.layer3(f2)
        f4 = self.backbone.layer4(f3)

        s1 = self.cr_1(f1)
        s2 = self.cr_2(f2)
        s3 = self.cr_3(f3)
        s4 = self.cr_4(f4)

        fuse_34 = self.fb_3(s3, s4)
        fuse_234 = self.fb_2(s2, fuse_34)
        fuse_1234 = self.fb_1(s1, fuse_234)

        feature = self.out_conv(fuse_1234)
        out = self.final_up(feature)

        return out

    def _init_weights(self):
        pretrained_dict = torch.load('/home/ubuntu/codes/wlh/code/pretrained_ckpt/res2net50_v1b_26w_4s.pth')
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("res2net50_v1b_26w_4s.pth is successfully loaded!!!!")




