import torch
import torch.nn as nn

EXPANSION = 4


class Resnet(nn.Module):

    def __init__(self, layers, image_channels, num_classes):
        super(Resnet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=image_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.bn1 = nn.BatchNorm2d(num_features=self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1)
        )

        self.layers1 = self._make_layers(layers[0], 64, 1)
        self.layers2 = self._make_layers(layers[1], 128, 2)
        self.layers3 = self._make_layers(layers[2], 256, 2)
        self.layers4 = self._make_layers(layers[3], 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*EXPANSION, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layers(self, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * EXPANSION:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=out_channels*EXPANSION,
                    kernel_size=(1, 1),
                    stride=stride
                ),
                nn.BatchNorm2d(num_features=out_channels*EXPANSION)
            )
        
        layers.append(Block(
            in_channels=self.in_channels,
            out_channels=out_channels,
            identity_downsample=identity_downsample,
            stride=stride
        ))
        self.in_channels = out_channels * EXPANSION

        for _ in range(num_residual_blocks - 1):
            layers.append(Block(in_channels=self.in_channels, out_channels=out_channels))

        return nn.Sequential(*layers)


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=(1, 1)
        )
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * EXPANSION,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )
        self.batchnorm3 = nn.BatchNorm2d(num_features=out_channels * EXPANSION)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        x = self.relu(self.batchnorm1(x))
        x = self.conv2(x)
        x = self.relu(self.batchnorm2(x))
        x = self.conv3(x)
        x = self.relu(self.batchnorm3(x))

        if self.identity_downsample:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x


def resnet50(img_channels=3, num_classes=1000):
    return Resnet(layers=[3, 4, 6, 3], image_channels=img_channels, num_classes=num_classes)


def resnet101(img_channels=3, num_classes=1000):
    return Resnet(layers=[3, 4, 23, 3], image_channels=img_channels, num_classes=num_classes)


def resnet152(img_channels=3, num_classes=1000):
    return Resnet(layers=[3, 8, 36, 3], image_channels=img_channels, num_classes=num_classes)


x = torch.rand(2, 3, 224, 224)
model = resnet152()
out = model(x)
print(out.shape)
