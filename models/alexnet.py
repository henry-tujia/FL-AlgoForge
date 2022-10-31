# -*-coding:utf-8-*-
import torch.nn as nn

__all__ = ["alexnet"]


class AlexNet(nn.Module):
    def __init__(self, num_classes,KD = False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(256, num_classes)
        self.KD = KD

    def forward(self, x):
        x = self.features(x)
        feature = x.view(x.size(0), -1)
        x = self.fc(feature)
        if self.KD:
            return feature,x
        return x


def alexnet(num_classes,KD  =False):
    return AlexNet(num_classes=num_classes,KD= KD)

#feature