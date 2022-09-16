from models.resnet_fednonlocal import EncoderInner
import torch
from torch import nn

class resnet(nn.Module):
    def __init__(self, blocks=2, input_nc=3, feature_dim=784, net_mode='resnet', in_channels=4, numclass=10,KD = False) -> None:
        super(resnet, self).__init__()

        self.encoder_new = EncoderInner(net_mode, n_blocks=int(blocks), input_nc=input_nc)
        self.block = nn.Sequential(
            nn.Tanh()
        )

        self.projection_new = nn.Linear(feature_dim, 256)

        self.fc = nn.Linear(256, numclass, bias=False)
        self.KD = KD

    def forward(self, x):

        x_new = self.encoder_new(x)
        x_new = self.block(x_new)
        x_new = torch.flatten(x_new, start_dim=1)
        h_new = self.projection_new(x_new)
        logits = self.fc(h_new)

        if self.KD:
            return h_new,logits

        return logits