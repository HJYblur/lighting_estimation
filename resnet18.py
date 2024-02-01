import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, identity_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # 让identity与经过conv后的特征维数相同，所以需要降采样
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x = x + identity
        x = self.relu(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, image_channels, num_classes) -> None:
        super().__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.layer1 = self.__make_layer(64, 64, 1)
        self.layer2 = self.__make_layer(64, 128, 2)
        self.layer3 = self.__make_layer(128, 256, 2)
        self.layer4 = self.__make_layer(256, 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def __make_layer(self, in_channel, out_channel, stride):
        identity_downsample = None
        if in_channel != out_channel:
            identity_downsample = self.identity_downsample(in_channel, out_channel)
        return nn.Sequential(
            BasicBlock(in_channel, out_channel, identity_downsample, stride),
            BasicBlock(out_channel, out_channel),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.dropout(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    def identity_downsample(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(out_channel),
        )
