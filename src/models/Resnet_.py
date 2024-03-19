import torch.nn as nn
from src.models.Model import Model

__all__ = ["resnet20", "resnet32", "resnet44", "resnet56", "resnet110", "resnet1202"]


class Resnet8(Model):
    def __init__(self, num_classes, KD=False, projection=False, *args, **kwargs):
        super(Resnet8, self).__init__()
        self.input_require_shape = [3, -1, -1]
        self.target_require_shape = []
        self.KD = KD
        self.projection = projection
        self.num_classes = num_classes
        self.generate_net()

    def generate_net(self):
        self.name = "Resnet8"
        self.model = ResNet(
            depth=8,
            num_classes=self.num_classes,
            KD=self.KD,
            projection=self.projection,
        )
        self.create_Loc_reshape_list()

    def forward(self, x):
        return self.model(x)


class Resnet32(Model):
    def __init__(self, num_classes, KD=False, projection=False, *args, **kwargs):
        super(Resnet8, self).__init__()
        self.input_require_shape = [3, -1, -1]
        self.target_require_shape = []
        self.KD = KD
        self.projection = projection
        self.num_classes = num_classes
        self.generate_net()

    def generate_net(self):
        self.name = "Resnet32"
        self.model = ResNet(
            depth=32,
            num_classes=self.num_classes,
            KD=self.KD,
            projection=self.projection,
        )
        self.create_Loc_reshape_list()

    def forward(self, x):
        return self.model(x)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv_1 = conv3x3(inplanes, planes, stride)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = conv3x3(planes, planes)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.conv_2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn_2 = nn.BatchNorm2d(planes)
        self.conv_3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu(out)

        out = self.conv_3(out)
        out = self.bn_3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self, depth, num_classes, block_name="BasicBlock", KD=False, projection=False
    ):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name == "BasicBlock":
            assert (
                depth - 2
            ) % 6 == 0, "depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name == "Bottleneck":
            assert (
                depth - 2
            ) % 9 == 0, "depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199"
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError("block_name shoule be Basicblock or Bottleneck")

        self.inplanes = 16
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.stage_1 = self._make_layer(block, 16, n)
        self.stage_2 = self._make_layer(block, 32, n, stride=2)
        self.stage_3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.KD = KD
        self.projection = projection
        if self.projection:
            self.projection_layer = nn.Sequential(
                nn.Linear(64 * block.expansion, 64 * block.expansion),
                nn.Linear(64 * block.expansion, 64 * block.expansion),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal(m.weight.data)
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)  # 32x32

        x = self.stage_1(x)  # 32x32
        x = self.stage_2(x)  # 16x16
        x = self.stage_3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        if self.KD:
            if self.projection:
                x = self.projection_layer(x)
                logits = self.fc(x)
            return x, logits
        return logits


def resnet8(num_classes, KD=False, projection=False):
    return ResNet(depth=8, num_classes=num_classes, KD=KD, projection=projection)


def resnet20(num_classes, KD=False, projection=False):
    return ResNet(depth=20, num_classes=num_classes, KD=KD, projection=projection)


def resnet32(num_classes, KD=False, projection=False):
    return ResNet(depth=32, num_classes=num_classes, KD=KD, projection=projection)


def resnet44(num_classes):
    return ResNet(depth=44, num_classes=num_classes)


def resnet56(num_classes):
    return ResNet(depth=56, num_classes=num_classes)


def resnet110(num_classes):
    return ResNet(depth=110, num_classes=num_classes)


def resnet1202(num_classes):
    return ResNet(depth=1202, num_classes=num_classes)
