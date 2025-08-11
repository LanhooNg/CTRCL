import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from ResNet import ResNet50

ResNet50 = models.resnet50(pretrained=False)
# print(ResNet)
ResNet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# print(ResNet50)


class CGR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CGR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.gn = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, num_class):
        super(ResNet, self).__init__()

        self.resnet = ResNet50

        self.cr_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.cr_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.cr_3 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.cr_4 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.smooth_4 = nn.Sequential(
            CGR(256, 128, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            CGR(128, 128, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            CGR(128, 128, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.smooth_3 = nn.Sequential(
            CGR(256, 128, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            CGR(128, 128, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.smooth_2 = nn.Sequential(
            CGR(256, 128, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.smooth_1 = CGR(256, 128, 3, 1, 1)

        # self.fuse_conv = CBR(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.out_conv = nn.Conv2d(in_channels=128, out_channels=num_class, kernel_size=1, stride=1, padding=0)
        self._init_weights()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        s4 = self.cr_4(x4)
        s3 = self.cr_3(x3) + F.interpolate(s4, scale_factor=2, mode='bilinear')
        s2 = self.cr_2(x2) + F.interpolate(s3, scale_factor=2, mode='bilinear')
        s1 = self.cr_1(x1) + F.interpolate(s2, scale_factor=2, mode='bilinear')

        r4 = self.smooth_4(s4)
        r3 = self.smooth_3(s3)
        r2 = self.smooth_2(s2)
        r1 = self.smooth_1(s1)

        out_feature = r1 + r2 + r3 + r4

        out = self.out_conv(out_feature)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)

        return out, x4, out_feature

    def _init_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)
        print("resnet50.pth is successfully loaded!")





